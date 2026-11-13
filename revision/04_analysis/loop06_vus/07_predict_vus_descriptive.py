from __future__ import print_function
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

"""Apply one corrected full519 model family to VUS using its five saved folds.

This script is intentionally not run by the VUS eligibility pipeline. It emits
descriptive probabilities only and never computes predicted classes or performance
metrics because these exact ClinVar VUS have no binary ground-truth labels here.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REVISION = Path(workspace_path('revision'))
ANALYSIS = REVISION / "04_analysis"
VUS_DIR = ANALYSIS / "loop06_vus"
VUS_INPUT = VUS_DIR / "vus_full43_aligned_unimputed.csv"
MODEL_ROOT = ANALYSIS / "loop05_corrected_common5fold/full519"
OUTPUT_ROOT = VUS_DIR / "descriptive_predictions_not_primary"
CODE_DIR = workspace_path('thesis/models/missense/code')

sys.path.insert(0, str(ANALYSIS))
from loop05_corrected_common import model_features  # noqa: E402


def load_fold_medians(model_key, fold_id, features):
    path = MODEL_ROOT / "preprocess/{}_fold{}_median.csv".format(model_key, fold_id)
    table = pd.read_csv(str(path))
    if table["feature"].astype(str).tolist() != features:
        raise ValueError("Median feature order mismatch: {}".format(path))
    medians = pd.Series(table["train_median"].astype(float).values, index=features)
    if medians.isna().any():
        raise ValueError("Saved fold median contains NaN: {}".format(path))
    return medians, path


def numeric_unimputed(df, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError("VUS input missing model features: {}".format(missing))
    result = df.loc[:, features].copy()
    for feature in features:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result


def predict_varity(X, fold_id):
    from xgboost import XGBClassifier

    model_path = MODEL_ROOT / "models/varity_fold{}.model".format(fold_id)
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model.predict_proba(X)[:, 1], model_path, None


def predict_mutscore(X, fold_id):
    import joblib

    model_path = MODEL_ROOT / "models/mutscore_fold{}.pkl".format(fold_id)
    model = joblib.load(str(model_path))
    return model.predict_proba(X)[:, 1], model_path, None


def predict_mvp(X, fold_id, features):
    import tensorflow as tf
    from keras import backend as K

    sys.path.insert(0, CODE_DIR)
    os.chdir(CODE_DIR)
    from models import CNN_Model

    K.clear_session()
    scaler_path = MODEL_ROOT / "preprocess/mvp_fold{}_minmax_scaler.pkl".format(fold_id)
    with open(str(scaler_path), "rb") as handle:
        scaler = pickle.load(handle)
    scaled = scaler.transform(X.values).reshape((-1, len(features), 1, 1))

    model_path = MODEL_ROOT / "models/mvp_fold{}.h5".format(fold_id)
    model = CNN_Model(
        weights_path=None,
        input_shape=(len(features), 1, 1),
        train_flag=True,
        nb_epoch=20,
        batch_size=16,
        verbose=0,
        exclude_cols=set(),
        fname="",
        f_out="",
    )
    model._init_model(verbose=False)
    K.get_session().run(tf.global_variables_initializer())
    model.model.load_weights(str(model_path))
    probability = model.model.predict(scaled, batch_size=16)[:, 0]
    return probability, model_path, scaler_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["VARITY", "MutScore", "MVP"])
    args = parser.parse_args()

    if not VUS_INPUT.exists():
        raise IOError("Aligned VUS input missing: {}".format(VUS_INPUT))
    df = pd.read_csv(str(VUS_INPUT), dtype={"Uploaded_variation": str})
    if "label" in df.columns:
        raise ValueError("VUS input must not contain a model label")
    features = model_features(args.model, "full")
    model_key = args.model.lower()
    raw = numeric_unimputed(df, features)

    output = df.loc[:, ["row_id", "Uploaded_variation", "source_gene"]].copy()
    artifacts = []
    for fold_id in range(5):
        medians, median_path = load_fold_medians(model_key, fold_id, features)
        X = raw.fillna(medians)
        if X.isna().any().any():
            raise ValueError("Fold-local median application left missing cells")

        if args.model == "VARITY":
            probability, model_path, scaler_path = predict_varity(X, fold_id)
        elif args.model == "MutScore":
            probability, model_path, scaler_path = predict_mutscore(X, fold_id)
        else:
            probability, model_path, scaler_path = predict_mvp(X, fold_id, features)

        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Invalid probability values in fold {}".format(fold_id))
        output["fold{}_prob_pathogenic".format(fold_id)] = probability
        artifacts.append(
            {
                "fold": fold_id,
                "model": str(model_path),
                "median": str(median_path),
                "scaler": str(scaler_path) if scaler_path is not None else None,
            }
        )

    probability_columns = ["fold{}_prob_pathogenic".format(fold_id) for fold_id in range(5)]
    output["fivefold_mean_prob_pathogenic"] = output[probability_columns].mean(axis=1)
    output["fivefold_sd_prob_pathogenic"] = output[probability_columns].std(axis=1, ddof=1)
    output["fivefold_min_prob_pathogenic"] = output[probability_columns].min(axis=1)
    output["fivefold_max_prob_pathogenic"] = output[probability_columns].max(axis=1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "vus_{}_fivefold_descriptive_probabilities.csv".format(model_key)
    output.to_csv(str(output_path), index=False)
    metadata = {
        "model_family": args.model,
        "input": str(VUS_INPUT),
        "n_vus": int(len(output)),
        "model_features": features,
        "fold_artifacts": artifacts,
        "output": str(output_path),
        "interpretation": "Descriptive five-fold ensemble probabilities only.",
        "ground_truth_binary_labels_available": False,
        "predicted_class_or_threshold_applied": False,
        "performance_metrics_computed": False,
        "warning": "Do not describe these VUS probabilities as accuracy, validation, reclassification, or clinical performance.",
    }
    metadata_path = OUTPUT_ROOT / "vus_{}_inference_metadata.json".format(model_key)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
