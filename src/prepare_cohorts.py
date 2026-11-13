from __future__ import annotations

import ast
import os
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 2025
N_SPLITS = 5

from paths import DATA_DIR, FOLDS_DIR, SOURCE_DIR
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = DATA_DIR / 'processed'
CANONICAL_CLEANED_PATH = OUTPUT_DIR / 'canonical519.csv'
PALB2_CLEANED_PATH = OUTPUT_DIR / 'palb2.csv'
CANONICAL_VCF_PATH = DATA_DIR / 'raw/clinvar_bc_variants_for_vep.vcf'
PALB2_VCF_PATH = DATA_DIR / 'raw/clinvar_palb2_binary_breast_snv_for_vep.vcf'
ORIGINAL_FOLDS_PATH = FOLDS_DIR / 'main.csv'
FEATURE_SOURCE = SOURCE_DIR / 'common.py'

METADATA_FROM_CLEANED = [
    "Uploaded_variation",
    "Location",
    "Allele",
    "Gene",
    "Feature",
    "Feature_type",
    "Consequence",
    "cDNA_position",
    "CDS_position",
    "Protein_position",
    "Amino_acids",
    "Codons",
    "Existing_variation",
    "IMPACT",
    "DISTANCE",
    "STRAND",
    "FLAGS",
    "CANONICAL",
    "ClinicalSignificance",
    "label",
]

OUTPUT_METADATA = [
    "row_id",
    "source_row_id",
    "source_dataset",
    "Uploaded_variation",
    "Location",
    "Allele",
    "Gene",
    "source_gene",
    "is_palb2",
    "Feature",
    "Feature_type",
    "Consequence",
    "cDNA_position",
    "CDS_position",
    "Protein_position",
    "Amino_acids",
    "Codons",
    "Existing_variation",
    "IMPACT",
    "DISTANCE",
    "STRAND",
    "FLAGS",
    "CANONICAL",
    "ClinicalSignificance",
    "label",
    "CLNSIG_vcf",
    "CLNREVSTAT",
    "review_stars",
    "is_expert_panel",
    "CLNVC",
    "CLNDN",
    "vcf_gene_symbols",
]

ENSEMBL_TO_SYMBOL = {
    "ENSG00000012048": "BRCA1",
    "ENSG00000139618": "BRCA2",
    "ENSG00000141510": "TP53",
    "ENSG00000149311": "ATM",
    "ENSG00000183765": "CHEK2",
    "ENSG00000083093": "PALB2",
}

REVIEW_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "no_assertion_criteria_provided": 0,
    "no_assertion_provided": 0,
    "no_interpretation_for_the_single_variant": 0,
}

PATHOGENIC_LABELS = {
    "Pathogenic",
    "Likely_pathogenic",
    "Pathogenic/Likely_pathogenic",
}
BENIGN_LABELS = {
    "Benign",
    "Likely_benign",
    "Benign/Likely_benign",
}


def parse_features(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "FULL_FEATURES" for target in node.targets):
            features = ast.literal_eval(node.value)
            if not isinstance(features, list) or not all(
                isinstance(item, str) for item in features
            ):
                raise ValueError("FULL_FEATURES is not a literal list of strings")
            return features
    raise ValueError("FULL_FEATURES was not found in {}".format(path))


def parse_info(info_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in info_text.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = unquote(value)
        else:
            result[item] = "1"
    return result


def read_vcf_for_ids(path: Path, wanted_ids: set[str]) -> pd.DataFrame:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            variation_id = str(fields[2])
            if variation_id not in wanted_ids:
                continue
            if variation_id in records:
                raise ValueError(
                    "Duplicate requested VCF ID {} in {}".format(variation_id, path)
                )
            info = parse_info(fields[7])
            geneinfo = info.get("GENEINFO", "")
            gene_symbols = [
                pair.split(":", 1)[0]
                for pair in geneinfo.split("|")
                if pair and ":" in pair
            ]
            records[variation_id] = {
                "Uploaded_variation": variation_id,
                "CLNSIG_vcf": info.get("CLNSIG"),
                "CLNREVSTAT": info.get("CLNREVSTAT"),
                "CLNVC": info.get("CLNVC"),
                "CLNDN": info.get("CLNDN"),
                "vcf_gene_symbols": "|".join(gene_symbols),
            }

    missing = sorted(wanted_ids - set(records))
    if missing:
        raise ValueError(
            "{} requested IDs were absent from {}: {}".format(
                len(missing), path, missing[:20]
            )
        )
    return pd.DataFrame([records[key] for key in sorted(records)])


def binarize_clnsig(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value)
    if text in PATHOGENIC_LABELS:
        return 1
    if text in BENIGN_LABELS:
        return 0
    return None


def normalize_input_frame(
    source_path: Path,
    vcf_path: Path,
    features: list[str],
    source_dataset: str,
    is_palb2: int,
    combined_start_row: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    raw = pd.read_csv(source_path, dtype={"Uploaded_variation": str}, low_memory=False)
    required = METADATA_FROM_CLEANED + features
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise ValueError(
            "{} is missing required columns: {}".format(source_path, missing_columns)
        )

    raw["Uploaded_variation"] = raw["Uploaded_variation"].astype(str)
    if raw["Uploaded_variation"].duplicated().any():
        raise ValueError("Duplicate Uploaded_variation values in {}".format(source_path))

    numeric_failures: list[dict[str, object]] = []
    for feature in features:
        before = raw[feature].copy()
        converted = pd.to_numeric(before, errors="coerce")
        bad_mask = before.notna() & converted.isna()
        if bad_mask.any():
            values = sorted(before[bad_mask].astype(str).unique().tolist())
            numeric_failures.append(
                {
                    "source_dataset": source_dataset,
                    "feature": feature,
                    "failure_count": int(bad_mask.sum()),
                    "example_values": "|".join(values[:10]),
                }
            )
        raw[feature] = converted
    if numeric_failures:
        raise ValueError(
            "Non-numeric values remained in cleaned feature columns: {}".format(
                numeric_failures[:5]
            )
        )

    raw["label"] = pd.to_numeric(raw["label"], errors="raise").astype(int)
    if not set(raw["label"].unique()).issubset({0, 1}):
        raise ValueError("Non-binary label in {}".format(source_path))

    vcf = read_vcf_for_ids(vcf_path, set(raw["Uploaded_variation"]))
    selected = raw[METADATA_FROM_CLEANED + features].copy()
    selected.insert(0, "source_row_id", np.arange(len(selected), dtype=int))
    selected.insert(0, "row_id", np.arange(combined_start_row, combined_start_row + len(selected), dtype=int))
    selected.insert(2, "source_dataset", source_dataset)
    selected = selected.merge(
        vcf,
        on="Uploaded_variation",
        how="left",
        validate="one_to_one",
    )

    selected["source_gene"] = selected["Gene"].map(ENSEMBL_TO_SYMBOL)
    if selected["source_gene"].isna().any():
        unknown = selected.loc[selected["source_gene"].isna(), "Gene"].unique()
        raise ValueError("Unknown Ensembl genes: {}".format(unknown.tolist()))
    selected["is_palb2"] = int(is_palb2)
    selected["review_stars"] = selected["CLNREVSTAT"].map(REVIEW_STARS).astype("Int64")
    selected["is_expert_panel"] = (
        selected["CLNREVSTAT"].eq("reviewed_by_expert_panel").astype(int)
    )

    gene_membership_ok = selected.apply(
        lambda row: row["source_gene"] in str(row["vcf_gene_symbols"]).split("|"),
        axis=1,
    )
    gene_validation = selected[
        ["Uploaded_variation", "Gene", "source_gene", "vcf_gene_symbols"]
    ].copy()
    gene_validation.insert(0, "source_dataset", source_dataset)
    gene_validation["expected_gene_present_in_vcf_geneinfo"] = gene_membership_ok.values
    if not gene_membership_ok.all():
        bad = gene_validation.loc[
            ~gene_validation["expected_gene_present_in_vcf_geneinfo"]
        ]
        raise ValueError(
            "Gene mapping failed for {} records: {}".format(len(bad), bad.head().to_dict("records"))
        )

    derived_label = selected["CLNSIG_vcf"].map(binarize_clnsig)
    label_match = derived_label.eq(selected["label"])
    if not label_match.all():
        bad = selected.loc[
            ~label_match,
            ["Uploaded_variation", "ClinicalSignificance", "CLNSIG_vcf", "label"],
        ]
        raise ValueError(
            "VCF CLNSIG/label mismatch for {} records: {}".format(
                len(bad), bad.head().to_dict("records")
            )
        )

    clinical_match = selected["ClinicalSignificance"].astype(str).eq(
        selected["CLNSIG_vcf"].astype(str)
    )
    if not clinical_match.all():
        bad = selected.loc[
            ~clinical_match,
            ["Uploaded_variation", "ClinicalSignificance", "CLNSIG_vcf"],
        ]
        raise ValueError(
            "Cleaned ClinicalSignificance/VCF CLNSIG mismatch for {} records: {}".format(
                len(bad), bad.head().to_dict("records")
            )
        )

    if selected["CLNREVSTAT"].isna().any():
        raise ValueError("CLNREVSTAT failed to map for one or more records")
    if selected["review_stars"].isna().any():
        unknown = selected.loc[selected["review_stars"].isna(), "CLNREVSTAT"].unique()
        raise ValueError("Unknown CLNREVSTAT values: {}".format(unknown.tolist()))

    aligned = selected[OUTPUT_METADATA + features].copy()
    return aligned, gene_validation, numeric_failures


def missingness_table(
    frames: dict[str, pd.DataFrame], features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, frame in frames.items():
        for feature_index, feature in enumerate(features, start=1):
            missing_count = int(frame[feature].isna().sum())
            rows.append(
                {
                    "dataset_scope": scope,
                    "feature_index": feature_index,
                    "feature": feature,
                    "n_rows": len(frame),
                    "nonmissing_count": int(frame[feature].notna().sum()),
                    "missing_count": missing_count,
                    "missing_fraction": missing_count / len(frame) if len(frame) else np.nan,
                    "all_missing": int(missing_count == len(frame)),
                }
            )
    return pd.DataFrame(rows)


def fold_label_counts(folds: pd.DataFrame, scope: str) -> pd.DataFrame:
    counts = (
        folds.groupby(["fold", "label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts.insert(0, "dataset_scope", scope)
    return counts


def assign_palb2_folds(
    original_folds: pd.DataFrame, palb2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_counts: dict[tuple[int, int], int] = defaultdict(int)
    total_counts: dict[int, int] = defaultdict(int)
    for (fold, label), count in original_folds.groupby(["fold", "label"]).size().items():
        label_counts[(int(fold), int(label))] = int(count)
    for fold, count in original_folds.groupby("fold").size().items():
        total_counts[int(fold)] = int(count)

    work = palb2.copy()
    work["_id_numeric"] = pd.to_numeric(work["Uploaded_variation"], errors="coerce")
    work = work.sort_values(
        ["label", "_id_numeric", "Uploaded_variation"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    assignments: list[dict[str, object]] = []
    for assignment_order, (_, row) in enumerate(work.iterrows(), start=1):
        label = int(row["label"])
        selected_fold = min(
            range(N_SPLITS),
            key=lambda fold: (
                label_counts[(fold, label)],
                total_counts[fold],
                fold,
            ),
        )
        assignments.append(
            {
                "assignment_order": assignment_order,
                "Uploaded_variation": str(row["Uploaded_variation"]),
                "label": label,
                "selected_fold": selected_fold,
                "same_label_count_before": label_counts[(selected_fold, label)],
                "total_fold_count_before": total_counts[selected_fold],
                "selection_rule": "min(current_same_label_count, current_total_count, fold_id); pathogenic rows assigned first, then numeric Uploaded_variation",
            }
        )
        label_counts[(selected_fold, label)] += 1
        total_counts[selected_fold] += 1

    return pd.DataFrame(assignments), work.drop(columns=["_id_numeric"])


def make_expert_folds(expert: pd.DataFrame) -> pd.DataFrame:
    label_counts = expert["label"].value_counts().to_dict()
    if len(label_counts) != 2 or min(label_counts.values()) < N_SPLITS:
        return pd.DataFrame()

    fold = np.full(len(expert), -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold_id, (_, valid_idx) in enumerate(
        splitter.split(expert, expert["label"].astype(int))
    ):
        fold[valid_idx] = fold_id

    result = expert[
        [
            "row_id",
            "Uploaded_variation",
            "Gene",
            "source_gene",
            "Feature",
            "ClinicalSignificance",
            "label",
            "CLNREVSTAT",
            "review_stars",
        ]
    ].copy()
    result = result.rename(columns={"row_id": "canonical519_row_id"})
    result.insert(0, "expert_row_id", np.arange(len(result), dtype=int))
    result["fold"] = fold
    return result


def logo_feasibility(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, frame in scopes.items():
        for gene in sorted(frame["source_gene"].dropna().unique()):
            heldout = frame[frame["source_gene"] == gene]
            training = frame[frame["source_gene"] != gene]
            held_counts = heldout["label"].value_counts().to_dict()
            train_counts = training["label"].value_counts().to_dict()
            held0, held1 = int(held_counts.get(0, 0)), int(held_counts.get(1, 0))
            train0, train1 = int(train_counts.get(0, 0)), int(train_counts.get(1, 0))
            both_held = held0 > 0 and held1 > 0
            min5_held = min(held0, held1) >= 5
            both_train = train0 > 0 and train1 > 0
            train_5fold = min(train0, train1) >= N_SPLITS
            if not both_held:
                interpretation = "not feasible for held-out ROC AUC: one label absent"
            elif not min5_held:
                interpretation = "mathematically feasible but highly unstable: fewer than 5 in one held-out label"
            else:
                interpretation = "feasible; interpret gene-specific uncertainty from sample size"
            rows.append(
                {
                    "dataset_scope": scope,
                    "held_out_gene": gene,
                    "heldout_n": len(heldout),
                    "heldout_label_0": held0,
                    "heldout_label_1": held1,
                    "training_n": len(training),
                    "training_label_0": train0,
                    "training_label_1": train1,
                    "heldout_has_both_labels": int(both_held),
                    "heldout_minimum_5_per_label": int(min5_held),
                    "training_has_both_labels": int(both_train),
                    "training_supports_stratified_5fold": int(train_5fold),
                    "feasible_for_logo_roc_auc": int(both_held and both_train),
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []

    features = parse_features(FEATURE_SOURCE)
    if len(features) != 43 or len(set(features)) != 43:
        raise ValueError("Expected exactly 43 unique benchmark features, got {}".format(len(features)))
    checks.append({"check": "feature_schema", "status": "PASS", "details": "43 unique features parsed from benchmark source"})

    canonical, gene_check_519, failures_519 = normalize_input_frame(
        CANONICAL_CLEANED_PATH,
        CANONICAL_VCF_PATH,
        features,
        "canonical519",
        0,
        0,
    )
    palb2, gene_check_7, failures_7 = normalize_input_frame(
        PALB2_CLEANED_PATH,
        PALB2_VCF_PATH,
        features,
        "palb2_same_pipeline",
        1,
        len(canonical),
    )
    if len(canonical) != 519 or len(palb2) != 7:
        raise ValueError("Expected 519 canonical plus 7 PALB2 rows")
    if set(canonical["Uploaded_variation"]) & set(palb2["Uploaded_variation"]):
        raise ValueError("Canonical and PALB2 IDs overlap")
    checks.append({"check": "input_row_counts", "status": "PASS", "details": "519 canonical + 7 PALB2 unique, non-overlapping IDs"})
    checks.append({"check": "vcf_gene_and_label_mapping", "status": "PASS", "details": "all 526 IDs matched expected gene, CLNSIG-derived label, and CLNREVSTAT"})

    combined = pd.concat([canonical, palb2], ignore_index=True)
    if combined["row_id"].tolist() != list(range(526)):
        raise ValueError("Combined row_id is not contiguous 0..525")
    if combined[features].shape != (526, 43):
        raise ValueError("Combined feature matrix is not 526 x 43")

    feature_schema = pd.DataFrame(
        {
            "feature_index": np.arange(1, len(features) + 1),
            "feature": features,
            "VARITY_uses_feature": 1,
            "MutScore_uses_feature": [int(feature != "MutScore_score") for feature in features],
            "MVP_uses_feature": [int(feature != "MVP_score") for feature in features],
            "source_definition": str(FEATURE_SOURCE),
        }
    )
    missingness = missingness_table(
        {
            "canonical519": canonical,
            "palb2_7": palb2,
            "canonical526": combined,
        },
        features,
    )
    row_missingness = combined[
        ["row_id", "Uploaded_variation", "source_gene", "is_palb2", "label"]
    ].copy()
    row_missingness["missing_feature_count"] = combined[features].isna().sum(axis=1)
    row_missingness["missing_feature_fraction"] = (
        row_missingness["missing_feature_count"] / len(features)
    )

    write_csv(feature_schema, "feature_schema_43.csv")
    write_csv(canonical, "canonical519_aligned.csv")
    write_csv(palb2, "palb2_7_cleaned_unimputed_aligned.csv")
    write_csv(combined, "pooled526.csv")
    write_csv(missingness, "feature_missingness_by_dataset.csv")
    write_csv(row_missingness, "row_feature_missingness.csv")
    write_csv(
        pd.concat([gene_check_519, gene_check_7], ignore_index=True),
        "vcf_gene_mapping_validation.csv",
    )
    write_csv(
        pd.DataFrame(failures_519 + failures_7, columns=["source_dataset", "feature", "failure_count", "example_values"]),
        "numeric_coercion_failures.csv",
    )

    original_fold_copy = OUTPUT_DIR / "fold_assignments_519_original_copy.csv"
    shutil.copyfile(ORIGINAL_FOLDS_PATH, original_fold_copy)
    if original_fold_copy.read_bytes() != ORIGINAL_FOLDS_PATH.read_bytes():
        raise ValueError("Original 519 fold copy is not byte-identical")
    original_folds = pd.read_csv(
        original_fold_copy, dtype={"Uploaded_variation": str}
    )
    if len(original_folds) != 519 or original_folds["Uploaded_variation"].duplicated().any():
        raise ValueError("Original fold file is not 519 unique variants")
    if set(original_folds["Uploaded_variation"]) != set(canonical["Uploaded_variation"]):
        raise ValueError("Original fold IDs do not match canonical cleaned IDs")
    if not set(original_folds["fold"].unique()) == set(range(N_SPLITS)):
        raise ValueError("Original fold IDs are not 0..4")
    checks.append({"check": "original_folds_preserved", "status": "PASS", "details": "519 copy is byte-identical and IDs/labels map to canonical cleaned data"})

    palb2_log, palb2_assignment_order = assign_palb2_folds(original_folds, palb2)
    palb2_fold_map = palb2_log.set_index("Uploaded_variation")["selected_fold"]
    canonical_fold_map = original_folds.set_index("Uploaded_variation")["fold"]

    combined_folds = combined[
        [
            "row_id",
            "Uploaded_variation",
            "Gene",
            "source_gene",
            "Feature",
            "ClinicalSignificance",
            "label",
            "is_palb2",
            "source_dataset",
        ]
    ].copy()
    combined_folds["fold"] = combined_folds["Uploaded_variation"].map(canonical_fold_map)
    palb2_mask = combined_folds["is_palb2"].eq(1)
    combined_folds.loc[palb2_mask, "fold"] = combined_folds.loc[
        palb2_mask, "Uploaded_variation"
    ].map(palb2_fold_map)
    if combined_folds["fold"].isna().any():
        raise ValueError("One or more combined rows lacks a fold")
    combined_folds["fold"] = combined_folds["fold"].astype(int)

    mapped_original = combined_folds.loc[~palb2_mask].set_index("Uploaded_variation")["fold"]
    original_expected = original_folds.set_index("Uploaded_variation")["fold"].astype(int)
    if not mapped_original.sort_index().equals(original_expected.sort_index()):
        raise ValueError("An original 519 fold assignment changed")
    checks.append({"check": "palb2_fold_extension", "status": "PASS", "details": "all original folds unchanged; 7 PALB2 rows assigned deterministically"})

    write_csv(combined_folds, "fold_assignments_526_with_palb2.csv")
    write_csv(palb2_log, "palb2_fold_assignment_log.csv")
    fold_counts = pd.concat(
        [
            fold_label_counts(original_folds, "canonical519_original"),
            fold_label_counts(combined_folds, "canonical526_with_palb2"),
        ],
        ignore_index=True,
    )
    write_csv(fold_counts, "fold_label_counts_519_and_526.csv")

    review_status = canonical[
        [
            "row_id",
            "Uploaded_variation",
            "Gene",
            "source_gene",
            "ClinicalSignificance",
            "label",
            "CLNSIG_vcf",
            "CLNREVSTAT",
            "review_stars",
            "is_expert_panel",
        ]
    ].copy()
    review_counts = (
        review_status.groupby(
            ["CLNREVSTAT", "review_stars", "is_expert_panel", "label"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["review_stars", "CLNREVSTAT", "label"], ascending=[False, True, True])
    )
    write_csv(review_status, "final519_clinvar_review_status.csv")
    write_csv(review_counts, "review_status_label_counts.csv")

    expert = canonical[canonical["is_expert_panel"].eq(1)].copy()
    expert.insert(0, "expert_row_id", np.arange(len(expert), dtype=int))
    expert_folds = make_expert_folds(expert)
    expert_missingness = missingness_table({"expert_panel_only_519": expert}, features)
    missingness = pd.concat([missingness, expert_missingness], ignore_index=True)
    write_csv(missingness, "feature_missingness_by_dataset.csv")
    write_csv(expert, "expert_panel.csv")
    write_csv(expert_folds, "expert_panel_only_5fold_assignments.csv")
    if not expert_folds.empty:
        expert_fold_counts = fold_label_counts(
            expert_folds, "expert_panel_only_519"
        )
    else:
        expert_fold_counts = pd.DataFrame(
            columns=["dataset_scope", "fold", "label", "count"]
        )
    write_csv(expert_fold_counts, "expert_panel_only_fold_label_counts.csv")
    if len(expert) != 279 or expert["label"].value_counts().to_dict() != {0: 225, 1: 54}:
        raise ValueError("Unexpected strict expert-panel subset size/labels")
    if expert_folds.empty:
        raise ValueError("Expert-panel labels should support stratified 5-fold")
    checks.append({"check": "expert_panel_subset", "status": "PASS", "details": "strict status yielded 279 rows (225 label 0, 54 label 1) and supports stratified 5-fold"})

    logo = logo_feasibility(
        {
            "canonical519": canonical,
            "canonical526": combined,
            "expert_panel_only_519": expert,
        }
    )
    write_csv(logo, "leave_one_gene_out_label_counts_and_feasibility.csv")

    shape_summary = []
    for scope, frame in {
        "canonical519": canonical,
        "palb2_7": palb2,
        "canonical526": combined,
        "expert_panel_only_519": expert,
    }.items():
        shape_summary.append(
            {
                "dataset_scope": scope,
                "rows": len(frame),
                "unique_variants": frame["Uploaded_variation"].nunique(),
                "label_0": int(frame["label"].eq(0).sum()),
                "label_1": int(frame["label"].eq(1).sum()),
                "genes": frame["source_gene"].nunique(),
                "feature_columns": len(features),
                "missing_feature_cells": int(frame[features].isna().sum().sum()),
            }
        )
    write_csv(pd.DataFrame(shape_summary), "dataset_shape_summary.csv")

    checks.append({"check": "no_imputation", "status": "PASS", "details": "feature NaNs retained; aligned outputs were not median-filled"})
    checks.append({"check": "logo_is_input_summary_only", "status": "PASS", "details": "label counts/feasibility computed; no model trained"})
    summary_checks_path = write_csv(pd.DataFrame(checks), "summary_checks.csv")

    # Re-read the principal outputs so the summary covers serialized CSVs rather than
    # only in-memory frames.
    reread_526 = pd.read_csv(
        OUTPUT_DIR / "pooled526.csv",
        dtype={"Uploaded_variation": str},
        low_memory=False,
    )
    reread_folds = pd.read_csv(
        OUTPUT_DIR / "fold_assignments_526_with_palb2.csv",
        dtype={"Uploaded_variation": str},
    )
    if reread_526.shape != combined.shape or reread_folds.shape != combined_folds.shape:
        raise ValueError("Serialized principal output shape mismatch")
    if int(reread_526[features].isna().sum().sum()) != int(
        combined[features].isna().sum().sum()
    ):
        raise ValueError("Serialized missingness changed")

    print("Prepared cohorts:", len(canonical), len(combined), len(expert))


if __name__ == "__main__":
    main()
