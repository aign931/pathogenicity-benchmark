from __future__ import print_function

import argparse

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from loop05_corrected_common import model_features, numeric_frame, read_scenario, scenario_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    df, folds, config = read_scenario(args.scenario)
    root = scenario_dir(args.scenario)
    validation_dir = root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    features = model_features("VARITY", config["panel"])
    rows = []

    for fold_id in sorted(folds["fold"].astype(int).unique().tolist()):
        valid_mask = folds["fold"].astype(int).values == fold_id
        median_table = pd.read_csv(str(root / "preprocess/varity_fold{}_median.csv".format(fold_id))).set_index("feature")
        medians = median_table["train_median"].reindex(features)
        if medians.isna().any():
            raise ValueError("Missing persisted VARITY medians")
        X_valid = numeric_frame(df.loc[valid_mask], features).fillna(medians)

        model = XGBClassifier()
        model.load_model(str(root / "models/varity_fold{}.model".format(fold_id)))
        probability = model.predict_proba(X_valid)[:, 1]

        saved = pd.read_csv(str(root / "predictions/varity_fold{}_predictions.csv".format(fold_id)))
        expected_rows = folds.loc[valid_mask, "row_id"].astype(int).values
        if not np.array_equal(saved["row_id"].astype(int).values, expected_rows):
            raise ValueError("VARITY fold {} row order mismatch".format(fold_id))
        difference = np.abs(probability - saved["prob_pathogenic"].astype(float).values)
        rows.append({"model": "VARITY", "scenario": args.scenario, "fold": fold_id,
                     "n": len(saved), "max_abs_probability_difference": float(difference.max()),
                     "mean_abs_probability_difference": float(difference.mean()),
                     "pass_1e_7": bool(difference.max() < 1e-7)})

    result = pd.DataFrame(rows)
    result.to_csv(str(validation_dir / "varity_reload_validation.csv"), index=False)
    if not result["pass_1e_7"].all():
        raise RuntimeError("VARITY reload validation failed")
    print("VARITY {} reload validation PASS".format(args.scenario))


if __name__ == "__main__":
    main()

