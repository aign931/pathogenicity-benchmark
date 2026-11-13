from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import gzip
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


THESIS_DATA = Path(workspace_path('thesis/data'))
OUTDIR = Path(workspace_path('revision/04_analysis/vep_palb2'))

ORIGINAL_VEP = THESIS_DATA / "clinvar_bc_annotated_canonical.txt"
ORIGINAL_VCF = THESIS_DATA / "clinvar_bc_variants_for_vep.vcf"
ORIGINAL_FINAL = THESIS_DATA / "benchmark_dataset.csv"

PALB2_VEP = OUTDIR / "clinvar_palb2_annotated_canonical.txt"
PALB2_VCF = OUTDIR / "clinvar_palb2_binary_breast_snv_for_vep.vcf"

AF_THRESHOLD = 0.01
PALB2_ENSEMBL = "ENSG00000083093"

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

SCORE_COLUMNS = [
    "SIFT_score",
    "SIFT4G_score",
    "PROVEAN_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "MutationTaster_score",
    "MutationAssessor_score",
    "fathmm-XF_coding_score",
    "CADD_raw",
    "VEST4_score",
    "DANN_score",
    "MetaSVM_score",
    "MetaLR_score",
    "MetaRNN_score",
    "Eigen-raw_coding",
    "Eigen-PC-raw_coding",
    "M-CAP_score",
    "REVEL_score",
    "MutPred_score",
    "MVP_score",
    "gMVP_score",
    "MPC_score",
    "PrimateAI_score",
    "DEOGEN2_score",
    "BayesDel_addAF_score",
    "BayesDel_noAF_score",
    "ClinPred_score",
    "LIST-S2_score",
    "VARITY_R_score",
    "VARITY_ER_score",
    "VARITY_R_LOO_score",
    "VARITY_ER_LOO_score",
    "ESM1b_score",
    "AlphaMissense_score",
    "PHACTboost_score",
    "MutFormer_score",
    "MutScore_score",
    "phyloP100way_vertebrate",
    "phyloP470way_mammalian",
    "phyloP17way_primate",
    "phastCons100way_vertebrate",
    "phastCons470way_mammalian",
    "phastCons17way_primate",
    "bStatistic",
    "GERP++_RS",
    "GERP_91_mammals",
    "1000Gp3_AF",
    "gnomAD4.1_joint_AF",
    "TOPMed_frz8_AF",
    "AllofUs_ALL_AF",
    "RegeneronME_ALL_AF",
    "dbNSFP_POPMAX_AF",
]

HIGH_MISSING_DROP_COLS = [
    "1000Gp3_AF",
    "MutPred_score",
    "TOPMed_frz8_AF",
    "VARITY_ER_score",
    "VARITY_R_score",
    "VARITY_R_LOO_score",
    "VARITY_ER_LOO_score",
    "PHACTboost_score",
    "LIST-S2_score",
]


@dataclass
class PipelineResult:
    name: str
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    final: pd.DataFrame
    stage_counts: dict[str, int]
    missing_before_drop: pd.Series
    label_distribution: dict[object, int]
    source_gene_distribution: dict[str, int]
    vep_gene_distribution: dict[str, int]
    duplicate_rare_groups: int
    rare_missense_unique_variants: int


def parse_info(info: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in info.split(";"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = unquote(value)
        else:
            result[part] = "1"
    return result


def first_gene(geneinfo: str | None) -> str | None:
    if not geneinfo:
        return None
    return geneinfo.split("|", 1)[0].split(":", 1)[0]


def read_vcf_labels(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, str | None]] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            info = parse_info(fields[7])
            records.append(
                {
                    "Uploaded_variation": fields[2],
                    "ClinicalSignificance": info.get("CLNSIG"),
                    "source_gene": first_gene(info.get("GENEINFO")),
                    "CLNVC": info.get("CLNVC"),
                    "CLNDN": info.get("CLNDN"),
                }
            )
    labels = pd.DataFrame(records)
    if not labels.empty:
        labels["Uploaded_variation"] = labels["Uploaded_variation"].astype(str)
    return labels


def load_vep_tab(path: Path) -> pd.DataFrame:
    header_line = None
    colnames: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if line.startswith("#Uploaded_variation"):
                header_line = idx
                colnames = line.lstrip("#").rstrip("\n").split("\t")
                break
    if header_line is None or colnames is None:
        raise ValueError(f"Could not find VEP header in {path}")
    return pd.read_csv(
        path,
        sep="\t",
        skiprows=header_line + 1,
        header=None,
        names=colnames,
        dtype=str,
        low_memory=False,
    )


def binarize_label(value: object) -> int | None:
    if pd.isna(value):
        return None
    label = str(value)
    if label in PATHOGENIC_LABELS:
        return 1
    if label in BENIGN_LABELS:
        return 0
    return None


def clean_score(value: object) -> float | None:
    if pd.isna(value):
        return None
    items = str(value).split(",")
    numbers: list[float] = []
    for item in items:
        if item not in [".", "-", ""]:
            try:
                numbers.append(float(item))
            except ValueError:
                continue
    if numbers:
        return numbers[0]
    return None


def apply_canonical_pipeline(name: str, vep_path: Path, vcf_path: Path) -> PipelineResult:
    df = load_vep_tab(vep_path)
    labels = read_vcf_labels(vcf_path)
    missing_vcf_ids = set(labels["Uploaded_variation"]) - set(df["Uploaded_variation"])
    stage_counts: dict[str, int] = {
        "input_vcf_records": len(labels),
        "input_vcf_unique_uploaded_variations": labels["Uploaded_variation"].nunique(
            dropna=False
        ),
        "vep_rows": len(df),
        "vep_unique_uploaded_variations": df["Uploaded_variation"].nunique(dropna=False),
        "input_vcf_ids_without_vep_rows": len(missing_vcf_ids),
    }

    canonical = df[df["CANONICAL"] == "YES"].copy()
    stage_counts["canonical_yes_rows"] = len(canonical)
    stage_counts["canonical_yes_unique_uploaded_variations"] = canonical[
        "Uploaded_variation"
    ].nunique(dropna=False)

    canonical["dbNSFP_POPMAX_AF"] = pd.to_numeric(
        canonical["dbNSFP_POPMAX_AF"], errors="coerce"
    )
    stage_counts["canonical_rows_with_numeric_popmax"] = int(
        canonical["dbNSFP_POPMAX_AF"].notna().sum()
    )

    canonical_missense = canonical[
        canonical["Consequence"].str.contains("missense_variant", na=False)
    ].copy()
    stage_counts["canonical_missense_rows"] = len(canonical_missense)
    stage_counts["canonical_missense_unique_uploaded_variations"] = canonical_missense[
        "Uploaded_variation"
    ].nunique(dropna=False)
    stage_counts["canonical_missense_rows_with_numeric_popmax"] = int(
        canonical_missense["dbNSFP_POPMAX_AF"].notna().sum()
    )
    stage_counts["canonical_missense_rows_popmax_lt_0_01"] = int(
        (canonical_missense["dbNSFP_POPMAX_AF"] < AF_THRESHOLD).sum()
    )

    rare = canonical[canonical["dbNSFP_POPMAX_AF"] < AF_THRESHOLD].copy()
    stage_counts["rare_rows_popmax_lt_0_01"] = len(rare)
    stage_counts["rare_unique_uploaded_variations"] = rare[
        "Uploaded_variation"
    ].nunique(dropna=False)

    duplicate_rare_groups = int((rare.groupby("Uploaded_variation").size() > 1).sum())
    stage_counts["rare_uploaded_variation_groups_with_multiple_rows"] = duplicate_rare_groups

    # The original notebook deduplicated before filtering to missense. In the
    # canonical 519 branch there were no duplicate rare rows, so row selection
    # did not affect counts. Here we use first row deterministically and report
    # duplicate groups separately.
    rare_unique = rare.drop_duplicates(subset=["Uploaded_variation"], keep="first").copy()
    stage_counts["rare_rows_after_uploaded_variation_dedup"] = len(rare_unique)

    missense_mask = rare_unique["Consequence"].str.contains(
        "missense_variant", na=False
    )
    rare_unique_missense = rare_unique[missense_mask].copy()
    stage_counts["missense_rows_after_original_order"] = len(rare_unique_missense)

    rare_missense_unique_variants = int(
        rare[rare["Consequence"].str.contains("missense_variant", na=False)][
            "Uploaded_variation"
        ].nunique(dropna=False)
    )
    stage_counts["rare_missense_unique_variants_without_pre_missense_dedup"] = (
        rare_missense_unique_variants
    )

    rare_unique_missense["Uploaded_variation"] = rare_unique_missense[
        "Uploaded_variation"
    ].astype(str)
    merged = pd.merge(
        rare_unique_missense,
        labels,
        on="Uploaded_variation",
        how="left",
        validate="one_to_one",
    )
    stage_counts["rows_after_label_merge"] = len(merged)
    stage_counts["rows_without_clinvar_label_after_merge"] = int(
        merged["ClinicalSignificance"].isna().sum()
    )

    merged["label"] = merged["ClinicalSignificance"].apply(binarize_label)
    raw = merged.dropna(subset=["label"]).copy()
    raw["label"] = raw["label"].astype(int)
    stage_counts["rows_with_binary_label"] = len(raw)

    cleaned = raw.copy()
    for col in SCORE_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(clean_score)

    missing_before_drop = cleaned.isna().sum().sort_values(ascending=False)
    missing_before_drop = missing_before_drop[missing_before_drop > 0]

    final = cleaned.drop(
        columns=[col for col in HIGH_MISSING_DROP_COLS if col in cleaned.columns]
    ).copy()
    fill_failures: list[str] = []
    for col in [c for c in final.columns if final[c].isna().sum() > 0]:
        try:
            median_value = final[col].median()
        except Exception:
            fill_failures.append(col)
            continue
        final[col] = final[col].fillna(median_value)

    stage_counts["columns_raw"] = raw.shape[1]
    stage_counts["columns_cleaned_before_drop"] = cleaned.shape[1]
    stage_counts["columns_final_after_feature_drop"] = final.shape[1]
    stage_counts["rows_final_after_feature_drop_and_median_impute"] = len(final)
    stage_counts["median_impute_non_numeric_failures"] = len(fill_failures)

    return PipelineResult(
        name=name,
        raw=raw,
        cleaned=cleaned,
        final=final,
        stage_counts=stage_counts,
        missing_before_drop=missing_before_drop,
        label_distribution=raw["label"].value_counts(dropna=False).to_dict(),
        source_gene_distribution=raw["source_gene"].value_counts(dropna=False).to_dict(),
        vep_gene_distribution=raw["Gene"].value_counts(dropna=False).to_dict(),
        duplicate_rare_groups=duplicate_rare_groups,
        rare_missense_unique_variants=rare_missense_unique_variants,
    )


def write_counts_csv(result: PipelineResult, path: Path) -> None:
    pd.DataFrame(
        [{"stage": key, "count": value} for key, value in result.stage_counts.items()]
    ).to_csv(path, index=False)


def md_dict(d: dict[object, int]) -> str:
    if not d:
        return "`{}`"
    return "`" + str(d) + "`"


def write_report(original: PipelineResult, palb2: PipelineResult) -> Path:
    original_final_shape = pd.read_csv(ORIGINAL_FINAL).shape
    report = OUTDIR / "palb2_canonical_processing_audit.md"
    lines: list[str] = [
        "# PALB2 Canonical Processing Audit",
        "",
        "Date: 2026-07-02",
        "",
        "Purpose: test whether PALB2 would survive the same row-level filters used for the canonical 519-variant manuscript dataset.",
        "",
        "## Pipeline Interpreted From `Processing.ipynb`",
        "",
        "Row-level filters:",
        "",
        "1. Start from a ClinVar VCF already restricted to target genes, single-nucleotide variants, breast-related disease terms, and binary ClinVar labels.",
        "2. Annotate with VEP 113 using `--canonical` and dbNSFP.",
        "3. Retain rows with `CANONICAL == YES`.",
        "4. Convert `dbNSFP_POPMAX_AF` to numeric and retain variants with `dbNSFP_POPMAX_AF < 0.01`.",
        "5. Deduplicate by `Uploaded_variation`.",
        "6. Retain rows whose VEP `Consequence` contains `missense_variant`.",
        "7. Merge ClinVar labels and retain only binary labels.",
        "",
        "Feature-level processing after row selection:",
        "",
        "- Clean dbNSFP score columns by keeping the first comma-separated numeric value.",
        "- Drop high-missingness feature columns: `"
        + "`, `".join(HIGH_MISSING_DROP_COLS)
        + "`.",
        "- Median-impute the remaining missing feature values.",
        "",
        "Important interpretation: the high-missingness step removes columns/features, not variant rows.",
        "",
        "## 519-Variant Pipeline Replication Check",
        "",
        f"- Original final benchmark file shape: `{original_final_shape}`.",
        f"- Recomputed rows after canonical+MAF+dedup+missense+binary label: `{original.stage_counts['rows_with_binary_label']}`.",
        f"- Recomputed final rows after feature drop/imputation: `{original.stage_counts['rows_final_after_feature_drop_and_median_impute']}`.",
        f"- Rare duplicate groups before deduplication: `{original.duplicate_rare_groups}`.",
        f"- Original VCF input IDs without VEP rows: `{original.stage_counts['input_vcf_ids_without_vep_rows']}`.",
        "",
        "## PALB2 Same-Pipeline Count",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
    ]
    for key, value in palb2.stage_counts.items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "PALB2 distributions after row-level selection:",
            "",
            f"- Source VCF gene distribution: {md_dict(palb2.source_gene_distribution)}",
            f"- VEP Ensembl gene distribution: {md_dict(palb2.vep_gene_distribution)}",
            f"- Binary label distribution: {md_dict(palb2.label_distribution)}",
            f"- PALB2 VCF input IDs without VEP rows: `{palb2.stage_counts['input_vcf_ids_without_vep_rows']}`.",
            f"- Canonical PALB2 missense variants before MAF filtering: `{palb2.stage_counts['canonical_missense_unique_uploaded_variations']}`.",
            f"- Canonical PALB2 missense variants with numeric `dbNSFP_POPMAX_AF`: `{palb2.stage_counts['canonical_missense_rows_with_numeric_popmax']}`.",
            f"- Canonical PALB2 missense variants with `dbNSFP_POPMAX_AF < 0.01`: `{palb2.stage_counts['canonical_missense_rows_popmax_lt_0_01']}`.",
            f"- Rare missense unique variants if missense is checked before deduplication: `{palb2.rare_missense_unique_variants}`.",
            "",
            "## Missingness Before Feature Drop",
            "",
            "| Column | Missing values |",
            "| --- | ---: |",
        ]
    )
    for col, value in palb2.missing_before_drop.head(30).items():
        lines.append(f"| `{col}` | {int(value)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PALB2 was not removed by the high-missingness feature-drop step; that step occurs after row selection and removes columns only.",
            "- PALB2 is absent from the manuscript 519 set because the original VEP input was restricted to five genes. In the counterfactual PALB2-inclusive run, the canonical/MAF/missense row-level filters further reduce PALB2 to 7 binary-labeled variants.",
            "- If PALB2 is added to the same binary-label, breast-related SNV, canonical, rare, missense pipeline, the final PALB2 row count is the `rows_with_binary_label` value above.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    original = apply_canonical_pipeline("original_5_gene", ORIGINAL_VEP, ORIGINAL_VCF)
    palb2 = apply_canonical_pipeline("palb2_only", PALB2_VEP, PALB2_VCF)

    original.raw.to_csv(OUTDIR / "canonical519_replication_raw.csv", index=False)
    original.final.to_csv(OUTDIR / "canonical519_replication_final.csv", index=False)
    write_counts_csv(
        original, OUTDIR / "canonical519_replication_stage_counts.csv"
    )

    palb2.raw.to_csv(OUTDIR / "palb2_benchmark_canonical_raw.csv", index=False)
    palb2.cleaned.to_csv(
        OUTDIR / "palb2_benchmark_canonical_cleaned.csv", index=False
    )
    palb2.final.to_csv(OUTDIR / "palb2_benchmark_canonical_final.csv", index=False)
    write_counts_csv(palb2, OUTDIR / "palb2_processing_stage_counts.csv")
    palb2.missing_before_drop.rename("missing_values").to_csv(
        OUTDIR / "palb2_missingness_before_feature_drop.csv", header=True
    )

    report = write_report(original, palb2)
    print(report)
    print(f"original_rows_with_binary_label={original.stage_counts['rows_with_binary_label']}")
    print(f"palb2_rows_with_binary_label={palb2.stage_counts['rows_with_binary_label']}")


if __name__ == "__main__":
    main()
