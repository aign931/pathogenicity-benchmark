from __future__ import print_function

import argparse
import json

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from xgboost import XGBClassifier

from loop05_corrected_common import (
    ensure_scenario_dir,
    fold_local_median_impute,
    model_features,
    read_scenario,
    write_run_metadata,
)


MODEL_NAME = "VARITY"
PARAMS = {
    "base_score": 0.5,
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.7,
    "gamma": 0.9,
    "learning_rate": 0.3,
    "max_delta_step": 0,
    "max_depth": 5,
    "min_child_weight": 2,
    "n_estimators": 109,
    "n_jobs": 1,
    "objective": "binary:logistic",
    "random_state": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "scale_pos_weight": 1,
    "subsample": 0.65,
}


def extract_binary_shap(values):
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="full519")
    args = parser.parse_args()

    df, folds, config = read_scenario(args.scenario)
    root = ensure_scenario_dir(args.scenario)
    features = model_features(MODEL_NAME, config["panel"])
    fold_ids = sorted(folds["fold"].astype(int).unique().tolist())
    metric_rows = []

    for fold_id in fold_ids:
        train_mask = folds["fold"].astype(int).values != fold_id
        valid_mask = ~train_mask
        X_train, X_valid, _ = fold_local_median_impute(
            df.loc[train_mask],
            df.loc[valid_mask],
            features,
            root / "preprocess/varity_fold{}_median.csv".format(fold_id),
        )
        y_train = df.loc[train_mask, "label"].astype(int)
        y_valid = df.loc[valid_mask, "label"].astype(int)

        model = XGBClassifier(**PARAMS)
        # Match the local VARITY framework's weighted_example=1 behavior.
        n_negative = int((y_train.values == 0).sum())
        n_positive = int((y_train.values == 1).sum())
        if n_negative == 0 or n_positive == 0:
            raise ValueError("VARITY fold {} has a single-class training set".format(fold_id))
        positive_weight_ratio = float(n_negative) / float(n_positive)
        sample_weight = np.ones(len(y_train), dtype=float)
        sample_weight[y_train.values == 1] *= positive_weight_ratio
        model.fit(X_train, y_train, sample_weight=sample_weight)
        probability = model.predict_proba(X_valid)[:, 1]
        predicted = (probability >= 0.5).astype(int)

        model.save_model(str(root / "models/varity_fold{}.model".format(fold_id)))
        explainer = shap.TreeExplainer(model)
        shap_matrix = extract_binary_shap(explainer.shap_values(X_valid))

        prediction = folds.loc[valid_mask, [c for c in ["row_id", "Uploaded_variation", "Gene", "fold", "label"] if c in folds.columns]].copy()
        prediction["model"] = MODEL_NAME
        prediction["prob_pathogenic"] = probability
        prediction.to_csv(str(root / "predictions/varity_fold{}_predictions.csv".format(fold_id)), index=False)

        shap_df = pd.DataFrame(shap_matrix, columns=features)
        shap_df.insert(0, "row_id", folds.loc[valid_mask, "row_id"].astype(int).values)
        shap_df.insert(1, "fold", fold_id)
        shap_df.to_csv(str(root / "shap/varity_fold{}_shap_values.csv".format(fold_id)), index=False)

        importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_matrix).mean(axis=0)})
        importance["model"] = MODEL_NAME
        importance["fold"] = fold_id
        importance.sort_values("mean_abs_shap", ascending=False).to_csv(
            str(root / "tables/varity_fold{}_shap_importance.csv".format(fold_id)), index=False
        )

        metric_rows.append(
            {
                "model": MODEL_NAME,
                "fold": fold_id,
                "n_outer_train": int(train_mask.sum()),
                "n_fit": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "n_train_negative": n_negative,
                "n_train_positive": n_positive,
                "positive_sample_weight": positive_weight_ratio,
                "auc": roc_auc_score(y_valid, probability),
                "average_precision": average_precision_score(y_valid, probability),
                "accuracy": accuracy_score(y_valid, predicted),
                "balanced_accuracy": balanced_accuracy_score(y_valid, predicted),
                "f1": f1_score(y_valid, predicted),
                "brier": brier_score_loss(y_valid, probability),
            }
        )
        print("{} {} fold {} done".format(MODEL_NAME, args.scenario, fold_id), flush=True)

    pd.DataFrame(metric_rows).to_csv(str(root / "tables/varity_metrics.csv"), index=False)
    write_run_metadata(
        root,
        MODEL_NAME,
        args.scenario,
        {
            "implementation": "XGBClassifier matching the benchmark VARITY-adapted XGBoost recipe.",
            "hyperparameters": PARAMS,
            "hyperparameter_source": "benchmark_varity_train_hp_tuning_MYVARITY_R_tf0_best_hps.csv",
            "hyperparameter_search_repeated_inside_outer_folds": False,
            "sample_weighting": "Positive rows weighted n_negative/n_positive, matching weighted_example=1 in alm_es.py.",
        },
    )


if __name__ == "__main__":
    main()

