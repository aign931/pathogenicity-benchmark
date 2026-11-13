import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path
"""Consolidate corrected Loop05/Loop06 evidence into manuscript-ready audit tables.

This script does not retrain models. It joins independently generated outputs,
adds explicit scope labels, and creates compact supplement-ready figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REVISION = Path(workspace_path('revision'))
LOOP05 = REVISION / "04_analysis/loop05_corrected_common5fold"
LOOP06 = REVISION / "04_analysis/loop06_vus"
OUT = REVISION / "04_analysis/loop07_consolidated_evidence"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

SCENARIOS = ["full519", "reduced519", "palb2_526", "expert_panel", "leave_one_gene_out"]
SCENARIO_LABELS = {
    "full519": "Canonical 519, full 43-feature panel",
    "reduced519": "Canonical 519, predefined 19-feature panel",
    "palb2_526": "Pooled 526 including seven PALB2 variants",
    "expert_panel": "Expert-panel subset (n=279)",
    "leave_one_gene_out": "Leave-one-gene-out transfer stress test",
}
MODEL_ORDER = ["VARITY", "MutScore", "MVP"]
MODEL_COLORS = {"VARITY": "#2B6CB0", "MutScore": "#DD6B20", "MVP": "#2F855A"}


def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def auc_summary():
    ci = pd.read_csv(str(LOOP05 / "statistical_comparison/all_scenarios_auc_bootstrap_ci.csv"))
    fold_rows = []
    for scenario in SCENARIOS:
        table = pd.read_csv(str(LOOP05 / scenario / "tables/model_metric_summary.csv"))
        table.insert(0, "scenario", scenario)
        fold_rows.append(table[["scenario", "model", "auc_mean", "auc_std", "auc_min", "auc_max"]])
    fold = pd.concat(fold_rows, ignore_index=True)
    result = ci.merge(fold, on=["scenario", "model"], how="left", validate="one_to_one")
    result.insert(1, "scenario_description", result["scenario"].map(SCENARIO_LABELS))
    result["evaluation_scope"] = np.where(
        result["scenario"] == "leave_one_gene_out",
        "gene-transfer stress test; pooled OOF AUC with heterogeneous held-out gene sizes",
        "post-development fixed-recipe sensitivity analysis; not nested model-development CV",
    )
    result.to_csv(str(TABLES / "scenario_auc_summary.csv"), index=False)
    return result


def stability_summary():
    individual_rows = []
    grouped_rows = []
    for scenario in SCENARIOS:
        individual = pd.read_csv(str(LOOP05 / scenario / "tables/individual_stability_summary.csv"))
        individual.insert(0, "scenario", scenario)
        individual.insert(1, "scenario_description", SCENARIO_LABELS[scenario])
        individual_rows.append(individual)

        grouped = pd.read_csv(str(LOOP05 / scenario / "tables/grouped_stability_summary.csv"))
        grouped.insert(0, "scenario", scenario)
        grouped.insert(1, "scenario_description", SCENARIO_LABELS[scenario])
        grouped_rows.append(grouped)
    individual_all = pd.concat(individual_rows, ignore_index=True)
    grouped_all = pd.concat(grouped_rows, ignore_index=True)
    individual_all.to_csv(str(TABLES / "scenario_individual_shap_stability.csv"), index=False)
    grouped_all.to_csv(str(TABLES / "scenario_grouped_shap_stability.csv"), index=False)
    return individual_all, grouped_all


def palb2_delta(auc):
    indexed = auc.set_index(["scenario", "model"])
    rows = []
    for model in MODEL_ORDER:
        base = indexed.loc[("full519", model)]
        pooled = indexed.loc[("palb2_526", model)]
        rows.append(
            {
                "model": model,
                "full519_pooled_auc": base["auc"],
                "palb2_526_pooled_auc": pooled["auc"],
                "absolute_auc_change": pooled["auc"] - base["auc"],
                "palb2_eligible_n": 7,
                "palb2_benign_n": 6,
                "palb2_pathogenic_n": 1,
                "interpretation": "pooled sensitivity only; no PALB2-specific performance estimate",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(str(TABLES / "palb2_pooled_sensitivity_delta.csv"), index=False)
    return result


def logo_gene_table():
    assignments = pd.read_csv(str(LOOP05 / "inputs/fold_assignments_leave_one_gene_out.csv"))
    mapping = assignments.groupby("fold", as_index=False).agg(
        held_out_gene=("source_gene", "first"),
        n_valid=("label", "size"),
        n_valid_negative=("label", lambda values: int((values == 0).sum())),
        n_valid_positive=("label", lambda values: int((values == 1).sum())),
    )
    gene_per_fold = assignments.groupby("fold")["source_gene"].nunique()
    if not (gene_per_fold == 1).all():
        raise ValueError("Each LOGO fold must contain one held-out gene")
    metrics = pd.read_csv(str(LOOP05 / "leave_one_gene_out/tables/all_model_fold_metrics.csv"))
    result = metrics.merge(mapping, on=["fold", "n_valid"], how="left", validate="many_to_one")
    result["interpretation_flag"] = np.where(
        result["n_valid"] < 30,
        "very small held-out gene; AUC is unstable and descriptive only",
        "gene-transfer stress estimate",
    )
    ordered = [
        "held_out_gene", "fold", "model", "n_valid", "n_valid_negative", "n_valid_positive",
        "auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "brier",
        "n_outer_train", "n_fit", "interpretation_flag",
    ]
    result = result[ordered].sort_values(["fold", "model"])
    result.to_csv(str(TABLES / "leave_one_gene_out_metrics_by_gene.csv"), index=False)
    return result


def delong_table():
    result = pd.read_csv(str(LOOP05 / "statistical_comparison/all_scenarios_delong_pairwise.csv"))
    result.insert(1, "scenario_description", result["scenario"].map(SCENARIO_LABELS))
    result.to_csv(str(TABLES / "scenario_pairwise_delong_holm.csv"), index=False)
    return result


def vus_tables():
    source_dir = LOOP06 / "descriptive_predictions_not_primary"
    distributions = pd.read_csv(str(source_dir / "vus_model_score_distributions.csv"))
    agreement = pd.read_csv(str(source_dir / "vus_pairwise_model_agreement.csv"))
    dispersion = pd.read_csv(str(source_dir / "vus_across_model_dispersion.csv"))
    distributions.to_csv(str(TABLES / "vus_model_score_distributions.csv"), index=False)
    agreement.to_csv(str(TABLES / "vus_pairwise_model_agreement.csv"), index=False)
    dispersion.to_csv(str(TABLES / "vus_across_model_dispersion.csv"), index=False)
    return distributions, agreement, dispersion


def plot_auc(auc):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(SCENARIOS), dtype=float)
    offsets = {"VARITY": -0.22, "MutScore": 0.0, "MVP": 0.22}
    for model in MODEL_ORDER:
        frame = auc[auc["model"] == model].set_index("scenario").loc[SCENARIOS]
        y = frame["auc"].to_numpy()
        lower = y - frame["ci_low"].to_numpy()
        upper = frame["ci_high"].to_numpy() - y
        ax.errorbar(
            x + offsets[model], y, yerr=np.vstack([lower, upper]), fmt="o", capsize=3,
            markersize=5, linewidth=1.3, color=MODEL_COLORS[model], label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Full 519", "Reduced 19", "+ PALB2", "Expert panel", "LOGO"])
    ax.set_ylim(0.945, 1.002)
    ax.set_ylabel("Pooled out-of-fold AUC (95% bootstrap CI)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(str(FIGURES / "scenario_auc_with_bootstrap_ci.png"), dpi=300)
    fig.savefig(str(FIGURES / "scenario_auc_with_bootstrap_ci.pdf"))
    plt.close(fig)


def plot_full_stability(individual, grouped):
    individual = individual[individual["scenario"] == "full519"]
    grouped = grouped[grouped["scenario"] == "full519"]
    rows = [
        ("Individual top-5\nJaccard", individual, 5, "jaccard_topk_mean"),
        ("Individual rank\nSpearman", individual, 5, "spearman_importance_mean"),
        ("Grouped top-3\nJaccard", grouped, 3, "jaccard_topk_mean"),
        ("Grouped rank\nSpearman", grouped, 3, "spearman_importance_mean"),
    ]
    intra = []
    inter = []
    labels = []
    for label, frame, k, column in rows:
        selected = frame[frame["k"] == k].set_index("comparison")
        intra.append(selected.loc["intra_model_across_folds", column])
        inter.append(selected.loc["inter_model_same_fold", column])
        labels.append(label)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - width / 2, intra, width, label="Within model, across folds", color="#3182CE")
    ax.bar(x + width / 2, inter, width, label="Between models, same fold", color="#ED8936")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(FIGURES / "full519_intra_inter_shap_stability.png"), dpi=300)
    fig.savefig(str(FIGURES / "full519_intra_inter_shap_stability.pdf"))
    plt.close(fig)


def report(auc, individual, grouped, palb2, logo, delong, vus):
    full_i = individual[(individual["scenario"] == "full519") & (individual["k"] == 5)]
    full_g = grouped[(grouped["scenario"] == "full519") & (grouped["k"] == 3)]
    reduced_auc = auc[auc["scenario"] == "reduced519"][["model", "auc", "ci_low", "ci_high"]]
    full_auc = auc[auc["scenario"] == "full519"][["model", "auc", "ci_low", "ci_high"]]
    vus_dist, vus_agree, vus_disp = vus
    path = OUT / "CONSOLIDATED_EVIDENCE_REPORT.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Consolidated Corrected Evidence\n\n")
        handle.write("## Evidence boundary\n\n")
        handle.write(
            "The five labeled scenarios use persisted study-specific model implementations with "
            "outer-fold-local preprocessing. Hyperparameters were fixed after original development; "
            "these are post-development sensitivity analyses, not nested estimates of a de novo "
            "model-development pipeline. VUS results are label-free descriptive scores and are never "
            "mixed with performance estimates.\n\n"
        )
        handle.write("## Full 519 pooled OOF AUC\n\n```text\n")
        handle.write(full_auc.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n## Reduced 19-feature pooled OOF AUC\n\n```text\n")
        handle.write(reduced_auc.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n")
        handle.write(
            "The reduced panel lowers AUC for all models and removes the near-perfect separation of "
            "the full feature set. It is the primary non-saturated sensitivity analysis.\n\n"
        )
        handle.write("## Full 519 SHAP stability\n\n```text\n")
        handle.write(full_i.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n```text\n")
        handle.write(full_g.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n")
        handle.write(
            "Within-model similarity exceeds between-model similarity on average, but this pattern "
            "is not uniform by model. The defensible interpretation is a shared high-ranking core "
            "with model-specific secondary attribution tendencies, not fundamentally disjoint feature sets.\n\n"
        )
        handle.write("## PALB2 pooled sensitivity\n\n```text\n")
        handle.write(palb2.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
        handle.write("\n```\n\n")
        handle.write(
            "Only seven PALB2 variants (six benign, one pathogenic) survive the same pipeline. The "
            "pooled 526-variant analysis is therefore a robustness check, not a PALB2 performance study.\n\n"
        )
        handle.write("## Leave-one-gene-out scope\n\n")
        handle.write(
            "Pooled LOGO AUC remains high, but gene-specific fold sizes range from 6 to 242. CHEK2 "
            "and other small held-out sets cannot support precise gene-specific estimates; see the "
            "machine-readable gene table.\n\n"
        )
        handle.write("## VUS descriptive scores\n\n```text\n")
        handle.write(vus_dist.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n```text\n")
        handle.write(vus_agree.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
        handle.write("\n```\n\n")
        handle.write(
            "The 3,228 exact VUS show positive rank agreement across models, but raw probability "
            "distributions differ. Without binary ground truth or clinical follow-up, these values "
            "cannot establish correctness, calibration, or reclassification.\n\n"
        )
        handle.write("## Statistical comparison\n\n")
        handle.write(
            "Paired DeLong tests and Holm correction are recorded in "
            "`tables/scenario_pairwise_delong_holm.csv`; stratified 5,000-replicate bootstrap CIs "
            "are recorded in `tables/scenario_auc_summary.csv`.\n"
        )
    return path


def main():
    ensure_dirs()
    auc = auc_summary()
    individual, grouped = stability_summary()
    palb2 = palb2_delta(auc)
    logo = logo_gene_table()
    delong = delong_table()
    vus = vus_tables()
    plot_auc(auc)
    plot_full_stability(individual, grouped)
    path = report(auc, individual, grouped, palb2, logo, delong, vus)
    print(path)


if __name__ == "__main__":
    main()
