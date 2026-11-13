from __future__ import print_function
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from loop05_corrected_common import (
    ensure_scenario_dir,
    fold_local_median_impute,
    model_features,
    read_scenario,
    write_run_metadata,
)


MODEL_NAME = "MVP"
CODE_DIR = workspace_path('thesis/models/missense/code')
TRAIN_SEED = 2017
SHAP_BACKGROUND_SEED = 2025
EPOCHS = 20
BATCH_SIZE = 16

sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)
from models import CNN_Model  # noqa: E402


def extract_binary_shap(values):
    arr = values[0] if isinstance(values, list) else values
    arr = np.asarray(arr)
    if arr.ndim == 4:
        return arr.reshape((arr.shape[0], arr.shape[1]))
    if arr.ndim == 3:
        return arr[:, :, 0]
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
        K.clear_session()
        # The same initialization seed is used for every outer fold so that
        # fold-to-fold variation is not deliberately confounded with a seed change.
        np.random.seed(TRAIN_SEED)
        tf.set_random_seed(TRAIN_SEED)

        train_mask = folds["fold"].astype(int).values != fold_id
        valid_mask = ~train_mask
        X_train_df, X_valid_df, _ = fold_local_median_impute(
            df.loc[train_mask],
            df.loc[valid_mask],
            features,
            root / "preprocess/mvp_fold{}_median.csv".format(fold_id),
        )
        y_train = df.loc[train_mask, "label"].astype(int).values
        y_valid = df.loc[valid_mask, "label"].astype(int).values

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_df.values).reshape((-1, len(features), 1, 1))
        X_valid = scaler.transform(X_valid_df.values).reshape((-1, len(features), 1, 1))
        with open(str(root / "preprocess/mvp_fold{}_minmax_scaler.pkl".format(fold_id)), "wb") as handle:
            pickle.dump(scaler, handle)
        scaler_sidecar = pd.DataFrame(
            {
                "feature": features,
                "data_min": scaler.data_min_,
                "data_max": scaler.data_max_,
                "data_range": scaler.data_range_,
                "scale": scaler.scale_,
                "min_offset": scaler.min_,
            }
        )
        scaler_sidecar.to_csv(str(root / "preprocess/mvp_fold{}_minmax_scaler.csv".format(fold_id)), index=False)

        model = CNN_Model(
            weights_path=None,
            input_shape=(len(features), 1, 1),
            train_flag=True,
            nb_epoch=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            exclude_cols=set(),
            fname="",
            f_out="",
        )
        model._init_model(verbose=False)
        # No unused inner validation holdout: all outer-training observations
        # contribute to the fixed 20-epoch fit.
        model.model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
            verbose=0,
        )
        probability = model.model.predict(X_valid, batch_size=BATCH_SIZE)[:, 0]
        predicted = (probability >= 0.5).astype(int)

        weights_path = root / "models/mvp_fold{}.h5".format(fold_id)
        model.model.save_weights(str(weights_path))
        K.get_session().run(tf.global_variables_initializer())
        model.model.load_weights(str(weights_path))

        rng = np.random.RandomState(SHAP_BACKGROUND_SEED)
        bg_n = min(100, X_train.shape[0])
        bg_idx = rng.choice(X_train.shape[0], bg_n, replace=False)
        background_rows = pd.DataFrame(
            {
                "training_array_index": bg_idx.astype(int),
                "row_id": folds.loc[train_mask, "row_id"].astype(int).values[bg_idx],
            }
        )
        if "Uploaded_variation" in folds.columns:
            background_rows["Uploaded_variation"] = folds.loc[train_mask, "Uploaded_variation"].astype(str).values[bg_idx]
        background_rows.to_csv(str(root / "preprocess/mvp_fold{}_shap_background_rows.csv".format(fold_id)), index=False)
        explainer = shap.DeepExplainer(model.model, X_train[bg_idx], session=K.get_session())
        shap_matrix = extract_binary_shap(explainer.shap_values(X_valid))

        prediction = folds.loc[valid_mask, [c for c in ["row_id", "Uploaded_variation", "Gene", "fold", "label"] if c in folds.columns]].copy()
        prediction["model"] = MODEL_NAME
        prediction["prob_pathogenic"] = probability
        prediction.to_csv(str(root / "predictions/mvp_fold{}_predictions.csv".format(fold_id)), index=False)

        shap_df = pd.DataFrame(shap_matrix, columns=features)
        shap_df.insert(0, "row_id", folds.loc[valid_mask, "row_id"].astype(int).values)
        shap_df.insert(1, "fold", fold_id)
        shap_df.to_csv(str(root / "shap/mvp_fold{}_shap_values.csv".format(fold_id)), index=False)

        importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_matrix).mean(axis=0)})
        importance["model"] = MODEL_NAME
        importance["fold"] = fold_id
        importance.sort_values("mean_abs_shap", ascending=False).to_csv(
            str(root / "tables/mvp_fold{}_shap_importance.csv".format(fold_id)), index=False
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

    pd.DataFrame(metric_rows).to_csv(str(root / "tables/mvp_metrics.csv"), index=False)
    write_run_metadata(
        root,
        MODEL_NAME,
        args.scenario,
        {
            "implementation": "Local CNN_Model residual architecture adapted to the benchmark feature panel.",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_seed_fixed_across_folds": TRAIN_SEED,
            "shap_background_seed": SHAP_BACKGROUND_SEED,
            "validation_split": 0.0,
            "scaling": "Per-fold MinMaxScaler fitted after per-fold median imputation.",
        },
    )


if __name__ == "__main__":
    main()

