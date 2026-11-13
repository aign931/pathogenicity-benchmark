import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path
"""Summarize descriptive VUS probabilities across the three persisted model families.

No binary labels, thresholds, clinical classifications, or performance metrics are
created here. The outputs describe score distributions and cross-model rank
agreement only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(workspace_path('revision/04_analysis/loop06_vus'))
PRED_DIR = ROOT / "descriptive_predictions_not_primary"
MODELS = ["varity", "mutscore", "mvp"]
KEYS = ["row_id", "Uploaded_variation", "source_gene"]


def read_model(model):
    path = PRED_DIR / "vus_{}_fivefold_descriptive_probabilities.csv".format(model)
    frame = pd.read_csv(str(path), dtype={"Uploaded_variation": str})
    needed = KEYS + [
        "fivefold_mean_prob_pathogenic",
        "fivefold_sd_prob_pathogenic",
        "fivefold_min_prob_pathogenic",
        "fivefold_max_prob_pathogenic",
    ]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError("{} missing columns {}".format(path, missing))
    if frame[KEYS].duplicated().any():
        raise ValueError("Duplicate VUS key in {}".format(path))
    return frame[needed].rename(
        columns={
            "fivefold_mean_prob_pathogenic": "{}_mean_probability".format(model),
            "fivefold_sd_prob_pathogenic": "{}_fold_sd".format(model),
            "fivefold_min_prob_pathogenic": "{}_fold_min".format(model),
            "fivefold_max_prob_pathogenic": "{}_fold_max".format(model),
        }
    )


def describe(series):
    values = pd.to_numeric(series, errors="raise")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q05": float(values.quantile(0.05)),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def main():
    merged = read_model(MODELS[0])
    for model in MODELS[1:]:
        candidate = read_model(model)
        merged = merged.merge(candidate, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != 3228:
        raise ValueError("Expected 3228 aligned VUS, found {}".format(len(merged)))

    mean_columns = ["{}_mean_probability".format(model) for model in MODELS]
    merged["across_model_mean_probability"] = merged[mean_columns].mean(axis=1)
    merged["across_model_sd"] = merged[mean_columns].std(axis=1, ddof=1)
    merged["across_model_range"] = merged[mean_columns].max(axis=1) - merged[mean_columns].min(axis=1)
    combined_path = PRED_DIR / "vus_three_model_descriptive_probabilities.csv"
    merged.to_csv(str(combined_path), index=False)

    distribution_rows = []
    for model in MODELS:
        row = {"model": model.upper() if model != "mutscore" else "MutScore"}
        row.update(describe(merged["{}_mean_probability".format(model)]))
        row["median_within_model_fold_sd"] = float(merged["{}_fold_sd".format(model)].median())
        row["q95_within_model_fold_sd"] = float(merged["{}_fold_sd".format(model)].quantile(0.95))
        distribution_rows.append(row)
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(str(PRED_DIR / "vus_model_score_distributions.csv"), index=False)

    pairwise_rows = []
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            x = merged["{}_mean_probability".format(left)]
            y = merged["{}_mean_probability".format(right)]
            pairwise_rows.append(
                {
                    "model_1": left.upper() if left != "mutscore" else "MutScore",
                    "model_2": right.upper() if right != "mutscore" else "MutScore",
                    "spearman_rank_correlation": float(x.corr(y, method="spearman")),
                    "pearson_correlation": float(x.corr(y, method="pearson")),
                    "mean_absolute_probability_difference": float((x - y).abs().mean()),
                    "median_absolute_probability_difference": float((x - y).abs().median()),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(str(PRED_DIR / "vus_pairwise_model_agreement.csv"), index=False)

    gene_rows = []
    for gene, group in merged.groupby("source_gene", sort=True):
        for model in MODELS:
            values = group["{}_mean_probability".format(model)]
            gene_rows.append(
                {
                    "source_gene": gene,
                    "model": model.upper() if model != "mutscore" else "MutScore",
                    "n_vus": int(len(group)),
                    "mean_probability": float(values.mean()),
                    "median_probability": float(values.median()),
                    "q25_probability": float(values.quantile(0.25)),
                    "q75_probability": float(values.quantile(0.75)),
                }
            )
    gene_summary = pd.DataFrame(gene_rows)
    gene_summary.to_csv(str(PRED_DIR / "vus_gene_model_score_summary.csv"), index=False)

    dispersion = pd.DataFrame(
        [
            {"quantity": "across_model_sd", **describe(merged["across_model_sd"])},
            {"quantity": "across_model_range", **describe(merged["across_model_range"])},
        ]
    )
    dispersion.to_csv(str(PRED_DIR / "vus_across_model_dispersion.csv"), index=False)

    report_path = PRED_DIR / "VUS_DESCRIPTIVE_PREDICTION_REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Descriptive VUS Model-Score Analysis\n\n")
        handle.write("## Scope and interpretation boundary\n\n")
        handle.write(
            "The 3,228 exact ClinVar VUS have no binary ground-truth label in this analysis. "
            "The three study-specific model families were applied using all five persisted full519 "
            "fold models and each fold's training-derived preprocessing. Values below are descriptive "
            "pathogenic-class probabilities and rank-agreement summaries only. They are not accuracy, "
            "calibration, clinical validation, or VUS reclassification. No threshold was applied.\n\n"
        )
        handle.write("## Model-score distributions\n\n```text\n")
        handle.write(distributions.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n## Pairwise cross-model agreement\n\n```text\n")
        handle.write(pairwise.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n## Across-model dispersion\n\n```text\n")
        handle.write(dispersion.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n")
        handle.write(
            "These outputs can support a response explaining why VUS were excluded from supervised "
            "benchmarking while still showing that the trained pipelines can generate exploratory "
            "scores. They must not be used to infer correctness or clinical utility.\n"
        )
    print(report_path)


if __name__ == "__main__":
    main()
