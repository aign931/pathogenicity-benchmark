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
from keras import backend as K

from loop05_corrected_common import model_features, numeric_frame, read_scenario, scenario_dir


CODE_DIR = workspace_path('thesis/models/missense/code')
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)
from models import CNN_Model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    df, folds, config = read_scenario(args.scenario)
    root = scenario_dir(args.scenario)
    validation_dir = root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    features = model_features("MVP", config["panel"])
    rows = []

    for fold_id in sorted(folds["fold"].astype(int).unique().tolist()):
        K.clear_session()
        valid_mask = folds["fold"].astype(int).values == fold_id
        median_table = pd.read_csv(str(root / "preprocess/mvp_fold{}_median.csv".format(fold_id))).set_index("feature")
        medians = median_table["train_median"].reindex(features)
        if medians.isna().any():
            raise ValueError("Missing persisted MVP medians")
        X_valid_df = numeric_frame(df.loc[valid_mask], features).fillna(medians)

        scaler_path = root / "preprocess/mvp_fold{}_minmax_scaler.pkl".format(fold_id)
        with open(str(scaler_path), "rb") as handle:
            scaler = pickle.load(handle)
        sidecar = pd.read_csv(str(root / "preprocess/mvp_fold{}_minmax_scaler.csv".format(fold_id)))
        if sidecar["feature"].astype(str).tolist() != features:
            raise ValueError("MVP fold {} sidecar feature order mismatch".format(fold_id))
        sidecar_checks = [
            np.max(np.abs(sidecar["data_min"].values - scaler.data_min_)),
            np.max(np.abs(sidecar["data_max"].values - scaler.data_max_)),
            np.max(np.abs(sidecar["scale"].values - scaler.scale_)),
            np.max(np.abs(sidecar["min_offset"].values - scaler.min_)),
        ]
        if max(sidecar_checks) >= 1e-12:
            raise ValueError("MVP fold {} scaler sidecar mismatch".format(fold_id))

        X_valid = scaler.transform(X_valid_df.values).reshape((-1, len(features), 1, 1))
        model = CNN_Model(weights_path=None, input_shape=(len(features), 1, 1),
                          train_flag=False, nb_epoch=20, batch_size=16, verbose=0,
                          exclude_cols=set(), fname="", f_out="")
        model._init_model(verbose=False)
        model.model.load_weights(str(root / "models/mvp_fold{}.h5".format(fold_id)))
        probability = model.model.predict(X_valid, batch_size=16)[:, 0]

        saved = pd.read_csv(str(root / "predictions/mvp_fold{}_predictions.csv".format(fold_id)))
        expected_rows = folds.loc[valid_mask, "row_id"].astype(int).values
        if not np.array_equal(saved["row_id"].astype(int).values, expected_rows):
            raise ValueError("MVP fold {} row order mismatch".format(fold_id))
        difference = np.abs(probability - saved["prob_pathogenic"].astype(float).values)
        rows.append({"model": "MVP", "scenario": args.scenario, "fold": fold_id,
                     "n": len(saved), "max_abs_probability_difference": float(difference.max()),
                     "mean_abs_probability_difference": float(difference.mean()),
                     "max_scaler_sidecar_difference": float(max(sidecar_checks)),
                     "pass_1e_7": bool(difference.max() < 1e-7)})

    result = pd.DataFrame(rows)
    result.to_csv(str(validation_dir / "mvp_reload_validation.csv"), index=False)
    if not result["pass_1e_7"].all():
        raise RuntimeError("MVP reload validation failed")
    print("MVP {} reload validation PASS".format(args.scenario))


if __name__ == "__main__":
    main()

