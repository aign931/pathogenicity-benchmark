from __future__ import print_function

import argparse

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

from common import (
    ensure_scenario_dir,
    fold_local_median_impute,
    model_features,
    read_scenario,
    write_run_metadata,
)


MODEL_NAME = "MutScore"
PARAMS = {"n_estimators": 1000, "random_state": 2005, "n_jobs": 1}


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
            root / "preprocess/mutscore_fold{}_median.csv".format(fold_id),
        )
        y_train = df.loc[train_mask, "label"].astype(int)
        y_valid = df.loc[valid_mask, "label"].astype(int)

        model = RandomForestClassifier(**PARAMS)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_valid)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        joblib.dump(model, str(root / "models/mutscore_fold{}.pkl".format(fold_id)))

        explainer = shap.TreeExplainer(model)
        shap_matrix = extract_binary_shap(explainer.shap_values(X_valid))

        prediction = folds.loc[valid_mask, [c for c in ["row_id", "Uploaded_variation", "Gene", "fold", "label"] if c in folds.columns]].copy()
        prediction["model"] = MODEL_NAME
        prediction["prob_pathogenic"] = probability
        prediction.to_csv(str(root / "predictions/mutscore_fold{}_predictions.csv".format(fold_id)), index=False)

        shap_df = pd.DataFrame(shap_matrix, columns=features)
        shap_df.insert(0, "row_id", folds.loc[valid_mask, "row_id"].astype(int).values)
        shap_df.insert(1, "fold", fold_id)
        shap_df.to_csv(str(root / "shap/mutscore_fold{}_shap_values.csv".format(fold_id)), index=False)

        importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_matrix).mean(axis=0)})
        importance["model"] = MODEL_NAME
        importance["fold"] = fold_id
        importance.sort_values("mean_abs_shap", ascending=False).to_csv(
            str(root / "tables/mutscore_fold{}_shap_importance.csv".format(fold_id)), index=False
        )

        metric_rows.append(
            {
                "model": MODEL_NAME,
                "fold": fold_id,
                "n_outer_train": int(train_mask.sum()),
                "n_fit": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "auc": roc_auc_score(y_valid, probability),
                "average_precision": average_precision_score(y_valid, probability),
                "accuracy": accuracy_score(y_valid, predicted),
                "balanced_accuracy": balanced_accuracy_score(y_valid, predicted),
                "f1": f1_score(y_valid, predicted),
                "brier": brier_score_loss(y_valid, probability),
            }
        )
        print("{} {} fold {} done".format(MODEL_NAME, args.scenario, fold_id), flush=True)

    pd.DataFrame(metric_rows).to_csv(str(root / "tables/mutscore_metrics.csv"), index=False)
    write_run_metadata(
        root,
        MODEL_NAME,
        args.scenario,
        {
            "implementation": "RandomForestClassifier matching the local benchmark MutScore recipe.",
            "hyperparameters": PARAMS,
        },
    )


if __name__ == "__main__":
    main()

