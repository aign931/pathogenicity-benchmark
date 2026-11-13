from __future__ import print_function

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from loop05_corrected_common import (
    common_features,
    ensure_scenario_dir,
    model_features,
    numeric_frame,
    read_scenario,
)


MODELS = ["VARITY", "MutScore", "MVP"]
MODEL_STEMS = {"VARITY": "varity", "MutScore": "mutscore", "MVP": "mvp"}


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).correlation)


def jaccard(a, b):
    a = set(a)
    b = set(b)
    return float(len(a & b)) / float(len(a | b)) if (a | b) else np.nan


def load_importances(root, scenario, panel):
    records = []
    by_key = {}
    for model in MODELS:
        stem = MODEL_STEMS[model]
        for path in sorted((root / "tables").glob("{}_fold*_shap_importance.csv".format(stem))):
            table = pd.read_csv(str(path))
            fold = int(table["fold"].iloc[0])
            native = model_features(model, panel)
            table = table.set_index("feature").reindex(native)
            if table["mean_abs_shap"].isna().any():
                raise ValueError("Missing importance values in {}".format(path))
            values = table["mean_abs_shap"].astype(float)
            by_key[(model, fold)] = values
            total = float(values.sum())
            ranked = values.sort_values(ascending=False)
            for rank, (feature, value) in enumerate(ranked.items(), start=1):
                records.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "fold": fold,
                        "feature": feature,
                        "rank_native": rank,
                        "mean_abs_shap": value,
                        "native_share": value / total if total else np.nan,
                    }
                )
    expected = 3 * len(set(k[1] for k in by_key))
    if len(by_key) != expected:
        raise ValueError("Incomplete importance files: {} keys, expected {}".format(len(by_key), expected))
    return by_key, pd.DataFrame(records)


def individual_feature_stability(by_key, common, top_ks):
    rows = []
    folds = sorted(set(fold for _, fold in by_key))
    for model in MODELS:
        for fold_a, fold_b in itertools.combinations(folds, 2):
            a = by_key[(model, fold_a)].reindex(common)
            b = by_key[(model, fold_b)].reindex(common)
            for k in top_ks:
                rows.append(
                    {
                        "comparison": "intra_model_across_folds",
                        "scope": "common_{}_features".format(len(common)),
                        "model_a": model,
                        "model_b": model,
                        "fold_a": fold_a,
                        "fold_b": fold_b,
                        "k": k,
                        "jaccard_topk": jaccard(a.nlargest(k).index, b.nlargest(k).index),
                        "spearman_importance": safe_spearman(a.values, b.values),
                    }
                )
    for fold in folds:
        for model_a, model_b in itertools.combinations(MODELS, 2):
            a = by_key[(model_a, fold)].reindex(common)
            b = by_key[(model_b, fold)].reindex(common)
            for k in top_ks:
                rows.append(
                    {
                        "comparison": "inter_model_same_fold",
                        "scope": "common_{}_features".format(len(common)),
                        "model_a": model_a,
                        "model_b": model_b,
                        "fold_a": fold,
                        "fold_b": fold,
                        "k": k,
                        "jaccard_topk": jaccard(a.nlargest(k).index, b.nlargest(k).index),
                        "spearman_importance": safe_spearman(a.values, b.values),
                    }
                )
    return pd.DataFrame(rows)


def correlation_components(df, features, threshold=0.8):
    numeric = numeric_frame(df, features)
    numeric = numeric.fillna(numeric.median(axis=0, skipna=True))
    corr = numeric.corr(method="spearman")
    edge_rows = []
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = float(corr.loc[left, right])
            if np.isfinite(value) and abs(value) > threshold:
                edge_rows.append({"feature_a": left, "feature_b": right, "spearman": value, "abs_spearman": abs(value)})

    # Complete linkage avoids transitive chains that merge features which are
    # not themselves highly correlated. Distance is defined as 1-|rho|.
    distance = 1.0 - corr.abs().clip(0.0, 1.0)
    np.fill_diagonal(distance.values, 0.0)
    condensed = squareform(distance.values, checks=False)
    cluster_labels = fcluster(linkage(condensed, method="complete"), t=1.0 - threshold, criterion="distance")
    components = []
    for label in sorted(set(cluster_labels)):
        members = [feature for feature, assigned in zip(features, cluster_labels) if assigned == label]
        components.append(sorted(members, key=features.index))
    components.sort(key=lambda members: min(features.index(feature) for feature in members))

    group_rows = []
    mapping = {}
    for index, members in enumerate(components, start=1):
        group_id = "G{:02d}".format(index)
        for feature in members:
            mapping[feature] = group_id
        group_rows.append(
            {
                "group_id": group_id,
                "n_features": len(members),
                "members": " | ".join(members),
                "is_correlated_group": len(members) > 1,
            }
        )
    return corr, pd.DataFrame(edge_rows), pd.DataFrame(group_rows), mapping


def grouped_attribution(by_key, common, groups, mapping):
    group_sizes = groups.set_index("group_id")["n_features"].to_dict()
    rows = []
    for (model, fold), importance in sorted(by_key.items()):
        values = importance.reindex(common).astype(float)
        total = float(values.sum())
        shares = values / total if total else values * np.nan
        for group_id in groups["group_id"]:
            members = [feature for feature in common if mapping[feature] == group_id]
            group_share = float(shares.loc[members].sum())
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "group_id": group_id,
                    "n_features": int(group_sizes[group_id]),
                    "group_share": group_share,
                    "mean_share_per_feature": group_share / float(group_sizes[group_id]),
                }
            )
    return pd.DataFrame(rows)


def grouped_stability(grouped, top_ks=(3, 5)):
    rows = []
    folds = sorted(grouped["fold"].unique())

    def vector(model, fold):
        return grouped[(grouped["model"] == model) & (grouped["fold"] == fold)].set_index("group_id")["group_share"].sort_index()

    for model in MODELS:
        for fold_a, fold_b in itertools.combinations(folds, 2):
            a = vector(model, fold_a)
            b = vector(model, fold_b)
            for k in top_ks:
                rows.append(
                    {
                        "comparison": "intra_model_across_folds",
                        "model_a": model,
                        "model_b": model,
                        "fold_a": fold_a,
                        "fold_b": fold_b,
                        "k": k,
                        "jaccard_topk": jaccard(a.nlargest(k).index, b.nlargest(k).index),
                        "spearman_importance": safe_spearman(a.values, b.values),
                    }
                )
    for fold in folds:
        for model_a, model_b in itertools.combinations(MODELS, 2):
            a = vector(model_a, fold)
            b = vector(model_b, fold)
            for k in top_ks:
                rows.append(
                    {
                        "comparison": "inter_model_same_fold",
                        "model_a": model_a,
                        "model_b": model_b,
                        "fold_a": fold,
                        "fold_b": fold,
                        "k": k,
                        "jaccard_topk": jaccard(a.nlargest(k).index, b.nlargest(k).index),
                        "spearman_importance": safe_spearman(a.values, b.values),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairwise(pairwise, group_cols):
    metrics = ["jaccard_topk", "spearman_importance"]
    summary = pairwise.groupby(group_cols)[metrics].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in summary.columns]
    return summary


def collect_metrics_and_predictions(root):
    metrics = []
    predictions = []
    for model in MODELS:
        stem = MODEL_STEMS[model]
        metrics.append(pd.read_csv(str(root / "tables/{}_metrics.csv".format(stem))))
        for path in sorted((root / "predictions").glob("{}_fold*_predictions.csv".format(stem))):
            predictions.append(pd.read_csv(str(path), dtype={"Uploaded_variation": str}))
    metrics = pd.concat(metrics, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    return metrics, predictions


def performance_summary(metrics, predictions):
    numeric = ["auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "brier"]
    fold_summary = metrics.groupby("model")[numeric].agg(["mean", "std", "min", "max"]).reset_index()
    fold_summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in fold_summary.columns]
    pooled_rows = []
    for model, group in predictions.groupby("model"):
        if group["row_id"].duplicated().any():
            raise ValueError("Duplicate OOF row_id for {}".format(model))
        pooled_rows.append(
            {
                "model": model,
                "n_oof": len(group),
                "pooled_auc": roc_auc_score(group["label"], group["prob_pathogenic"]),
                "pooled_average_precision": average_precision_score(group["label"], group["prob_pathogenic"]),
            }
        )
    return fold_summary, pd.DataFrame(pooled_rows)


def plot_stability(individual, grouped, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    colors = {"intra_model_across_folds": "#2C7FB8", "inter_model_same_fold": "#E34A33"}
    labels = ["Intra-model\nacross folds", "Inter-model\nwithin fold"]
    comparisons = ["intra_model_across_folds", "inter_model_same_fold"]
    for ax, metric, title in zip(axes, ["jaccard_topk", "spearman_importance"], ["Top-feature overlap", "Full-profile rank correlation"]):
        datasets = []
        positions = []
        facecolors = []
        tick_positions = []
        tick_labels = []
        cursor = 1
        for level_name, table, k in [("Individual", individual, 5), ("Grouped", grouped, 3)]:
            for comparison in comparisons:
                values = table[(table["comparison"] == comparison) & (table["k"] == k)][metric].dropna().values
                datasets.append(values)
                positions.append(cursor)
                facecolors.append(colors[comparison])
                cursor += 1
            tick_positions.append(cursor - 1.5)
            tick_labels.append(level_name)
            cursor += 0.7
        bp = ax.boxplot(datasets, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
        for box, color in zip(bp["boxes"], facecolors):
            box.set_facecolor(color)
            box.set_alpha(0.75)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Similarity")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=colors[c], lw=8, alpha=0.75) for c in comparisons]
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=[0, 0.13, 1, 1])
    figure.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_grouped_heatmap(grouped, groups, output_path):
    mean_group = grouped.groupby(["model", "group_id"])["group_share"].mean().unstack(fill_value=0).reindex(MODELS)
    ordered = mean_group.mean(axis=0).sort_values(ascending=False).index.tolist()
    mean_group = mean_group.loc[:, ordered]
    figure, ax = plt.subplots(figsize=(max(8.5, 0.48 * len(ordered)), 3.2))
    image = ax.imshow(mean_group.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered, rotation=55, ha="right")
    ax.set_xlabel("Correlation group (membership in correlation_groups.csv)")
    ax.set_title("Mean normalized grouped SHAP share across folds")
    figure.colorbar(image, ax=ax, label="Share of common-feature mean |SHAP|")
    figure.tight_layout()
    figure.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def to_text(table, columns=None, n=None):
    view = table if columns is None else table.loc[:, columns]
    if n is not None:
        view = view.head(n)
    return view.to_string(index=False)


def write_report(root, scenario, config, df, folds, common, metrics_summary, pooled, individual_summary, groups, grouped_summary):
    missing_cells = int(numeric_frame(df, common).isna().sum().sum())
    n_folds = int(folds["fold"].nunique())
    lines = [
        "# Corrected common-fold analysis: {}".format(scenario),
        "",
        "## Scope",
        "",
        "- {}".format(config["description"]),
        "- N = {}; folds = {}; common comparison space = {} features.".format(len(df), n_folds, len(common)),
        "- Missing cells in the common feature matrix before fold-local imputation: {}.".format(missing_cells),
        "- Every median and MVP MinMax scaler was fitted within the outer-training fold only.",
        "- The same fold assignment and common feature space were used for intra- and inter-model SHAP comparisons.",
        "",
        "## Interpretive boundary",
        "",
        "This is a post-development sensitivity analysis of fold-related attribution stability. It is not presented as a strictly unbiased estimate of de-novo model-development performance. In particular, the fixed VARITY hyperparameters were selected in the original benchmark workflow and were not re-optimized within each outer fold.",
        "",
        "## Fold-wise metrics",
        "",
        "```text",
        to_text(metrics_summary, ["model", "auc_mean", "auc_std", "average_precision_mean", "average_precision_std", "balanced_accuracy_mean", "f1_mean", "brier_mean"]),
        "```",
        "",
        "## Pooled out-of-fold predictions",
        "",
        "```text",
        to_text(pooled),
        "```",
        "",
        "These performance values are descriptive for this fixed post-development recipe; the attribution-stability comparison is the inferential target.",
        "",
        "## Individual-feature SHAP stability on the common space",
        "",
        "```text",
        to_text(individual_summary),
        "```",
        "",
        "## Correlation-grouped SHAP stability",
        "",
        "Groups were defined once by complete-linkage hierarchical clustering of distance 1-|rho| over the common feature matrix, cut at 0.20 so every within-group pair had |rho| > 0.80. Mean absolute SHAP values were normalized to shares within each model/fold before group summation; raw SHAP magnitudes were not compared across explainers.",
        "",
        "```text",
        to_text(grouped_summary),
        "```",
        "",
        "Correlation group membership is recorded in `tables/correlation_groups.csv` ({} groups; {} multi-feature groups).".format(len(groups), int((groups["n_features"] > 1).sum())),
        "",
        "## Files",
        "",
        "- `tables/all_model_fold_metrics.csv` and `tables/oof_predictions.csv`",
        "- `tables/individual_stability_pairwise.csv` and `tables/individual_stability_summary.csv`",
        "- `tables/correlation_groups.csv`, `tables/correlation_edges_absrho_gt_0.8.csv`",
        "- `tables/grouped_shap_by_model_fold.csv` and `tables/grouped_stability_pairwise.csv`",
        "- `figures/stability_individual_and_grouped.png` and `figures/grouped_shap_heatmap.png`",
        "- per-fold models, preprocessing records, predictions, and SHAP matrices in sibling directories",
        "",
    ]
    (root / "corrected_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="full519")
    args = parser.parse_args()

    df, folds, config = read_scenario(args.scenario)
    root = ensure_scenario_dir(args.scenario)
    common = common_features(config["panel"])
    by_key, native_ranks = load_importances(root, args.scenario, config["panel"])
    native_ranks.to_csv(str(root / "tables/native_feature_ranks.csv"), index=False)

    common_rank_rows = []
    for (model, fold), values in sorted(by_key.items()):
        common_values = values.reindex(common)
        total = float(common_values.sum())
        for rank, (feature, value) in enumerate(common_values.sort_values(ascending=False).items(), start=1):
            common_rank_rows.append(
                {"model": model, "fold": fold, "feature": feature, "rank_common": rank, "mean_abs_shap": value, "common_share": value / total if total else np.nan}
            )
    pd.DataFrame(common_rank_rows).to_csv(str(root / "tables/common_feature_ranks.csv"), index=False)

    top_ks = tuple(sorted(set([min(5, len(common)), min(10, len(common))])))
    individual = individual_feature_stability(by_key, common, top_ks)
    individual.to_csv(str(root / "tables/individual_stability_pairwise.csv"), index=False)
    individual_summary = summarize_pairwise(individual, ["comparison", "scope", "k"])
    individual_summary.to_csv(str(root / "tables/individual_stability_summary.csv"), index=False)
    individual_model_summary = summarize_pairwise(individual, ["comparison", "model_a", "model_b", "k"])
    individual_model_summary.to_csv(str(root / "tables/individual_stability_by_model_or_pair.csv"), index=False)

    corr, edges, groups, mapping = correlation_components(df, common, threshold=0.8)
    corr.to_csv(str(root / "tables/common_feature_spearman_correlation.csv"))
    edges.to_csv(str(root / "tables/correlation_edges_absrho_gt_0.8.csv"), index=False)
    groups.to_csv(str(root / "tables/correlation_groups.csv"), index=False)
    grouped = grouped_attribution(by_key, common, groups, mapping)
    grouped.to_csv(str(root / "tables/grouped_shap_by_model_fold.csv"), index=False)
    grouped_summary = grouped.groupby(["model", "group_id"])[["group_share", "mean_share_per_feature"]].agg(["mean", "std", "min", "max"]).reset_index()
    grouped_summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in grouped_summary.columns]
    grouped_summary.to_csv(str(root / "tables/grouped_shap_summary.csv"), index=False)

    group_pairwise = grouped_stability(grouped, top_ks=tuple(sorted(set([min(3, len(groups)), min(5, len(groups))]))))
    group_pairwise.to_csv(str(root / "tables/grouped_stability_pairwise.csv"), index=False)
    group_stability_summary = summarize_pairwise(group_pairwise, ["comparison", "k"])
    group_stability_summary.to_csv(str(root / "tables/grouped_stability_summary.csv"), index=False)
    group_model_summary = summarize_pairwise(group_pairwise, ["comparison", "model_a", "model_b", "k"])
    group_model_summary.to_csv(str(root / "tables/grouped_stability_by_model_or_pair.csv"), index=False)

    metrics, predictions = collect_metrics_and_predictions(root)
    metrics.to_csv(str(root / "tables/all_model_fold_metrics.csv"), index=False)
    predictions.to_csv(str(root / "tables/oof_predictions.csv"), index=False)
    metric_summary, pooled = performance_summary(metrics, predictions)
    metric_summary.to_csv(str(root / "tables/model_metric_summary.csv"), index=False)
    pooled.to_csv(str(root / "tables/pooled_oof_metrics.csv"), index=False)

    plot_stability(individual, group_pairwise, root / "figures/stability_individual_and_grouped.png")
    plot_grouped_heatmap(grouped, groups, root / "figures/grouped_shap_heatmap.png")
    write_report(root, args.scenario, config, df, folds, common, metric_summary, pooled, individual_summary, groups, group_stability_summary)
    print("aggregated {}".format(args.scenario), flush=True)


if __name__ == "__main__":
    main()
