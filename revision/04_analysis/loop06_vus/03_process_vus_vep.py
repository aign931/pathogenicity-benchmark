from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import csv
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


OUTDIR = Path(workspace_path('revision/04_analysis/loop06_vus'))
VUS_VCF = OUTDIR / "clinvar_5gene_vus_breast_snv_for_vep.vcf"
VEP_OUTPUT = OUTDIR / "clinvar_5gene_vus_annotated_canonical.txt"
VUS_LABEL = "Uncertain_significance"
AF_THRESHOLD = 0.01

SCORE_COLUMNS = [
    "SIFT_score", "SIFT4G_score", "PROVEAN_score", "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score", "MutationTaster_score", "MutationAssessor_score",
    "fathmm-XF_coding_score", "CADD_raw", "VEST4_score", "DANN_score",
    "MetaSVM_score", "MetaLR_score", "MetaRNN_score", "Eigen-raw_coding",
    "Eigen-PC-raw_coding", "M-CAP_score", "REVEL_score", "MutPred_score",
    "MVP_score", "gMVP_score", "MPC_score", "PrimateAI_score", "DEOGEN2_score",
    "BayesDel_addAF_score", "BayesDel_noAF_score", "ClinPred_score", "LIST-S2_score",
    "VARITY_R_score", "VARITY_ER_score", "VARITY_R_LOO_score", "VARITY_ER_LOO_score",
    "ESM1b_score", "AlphaMissense_score", "PHACTboost_score", "MutFormer_score",
    "MutScore_score", "phyloP100way_vertebrate", "phyloP470way_mammalian",
    "phyloP17way_primate", "phastCons100way_vertebrate",
    "phastCons470way_mammalian", "phastCons17way_primate", "bStatistic", "GERP++_RS",
    "GERP_91_mammals", "1000Gp3_AF", "gnomAD4.1_joint_AF", "TOPMed_frz8_AF",
    "AllofUs_ALL_AF", "RegeneronME_ALL_AF", "dbNSFP_POPMAX_AF",
]


def parse_info(info: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in info.split(";"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            parsed[key] = unquote(value)
        else:
            parsed[part] = "1"
    return parsed


def first_gene(geneinfo: str | None) -> str | None:
    if not geneinfo:
        return None
    return geneinfo.split("|", 1)[0].split(":", 1)[0]


def read_vcf_metadata(path: Path) -> pd.DataFrame:
    records: list[dict[str, str | None]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
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
                    "source_CLNVC": info.get("CLNVC"),
                    "source_CLNDN": info.get("CLNDN"),
                    "source_CHROM": fields[0],
                    "source_POS": fields[1],
                    "source_REF": fields[3],
                    "source_ALT": fields[4],
                }
            )
    result = pd.DataFrame(records)
    if not result.empty:
        result["Uploaded_variation"] = result["Uploaded_variation"].astype(str)
    return result


def load_vep_tab(path: Path) -> pd.DataFrame:
    header_line: int | None = None
    colnames: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if line.startswith("#Uploaded_variation"):
                header_line = idx
                colnames = line.lstrip("#").rstrip("\n").split("\t")
                break
    if header_line is None or colnames is None:
        raise ValueError(f"Could not find VEP tab header in {path}")
    return pd.read_csv(
        path,
        sep="\t",
        skiprows=header_line + 1,
        header=None,
        names=colnames,
        dtype=str,
        low_memory=False,
    )


def clean_score(value: object) -> float | None:
    """Match Processing.ipynb: keep the first comma-separated numeric value."""
    if pd.isna(value):
        return None
    for item in str(value).split(","):
        if item in {".", "-", ""}:
            continue
        try:
            return float(item)
        except ValueError:
            continue
    return None


def write_counts(path: Path, counts: dict[str, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "count"])
        writer.writeheader()
        writer.writerows({"stage": stage, "count": count} for stage, count in counts.items())


def main() -> None:
    metadata = read_vcf_metadata(VUS_VCF)
    vep = load_vep_tab(VEP_OUTPUT)

    if metadata.empty:
        raise ValueError("The exact-VUS candidate VCF contains no records.")
    if vep.empty:
        raise ValueError("The VEP output contains no annotation rows.")

    metadata_id_counts = metadata["Uploaded_variation"].value_counts(dropna=False)
    duplicate_input_ids = int((metadata_id_counts > 1).sum())
    unique_metadata = metadata.drop_duplicates("Uploaded_variation", keep="first").copy()

    input_ids = set(unique_metadata["Uploaded_variation"])
    vep_ids = set(vep["Uploaded_variation"].astype(str))
    missing_ids = sorted(input_ids - vep_ids)

    counts: dict[str, int] = {
        "input_vcf_records": len(metadata),
        "input_vcf_unique_uploaded_variations": metadata["Uploaded_variation"].nunique(dropna=False),
        "input_vcf_duplicate_uploaded_variation_groups": duplicate_input_ids,
        "vep_rows": len(vep),
        "vep_unique_uploaded_variations": vep["Uploaded_variation"].nunique(dropna=False),
        "input_vcf_ids_without_vep_rows": len(missing_ids),
    }

    canonical = vep[vep["CANONICAL"] == "YES"].copy()
    counts["canonical_yes_rows"] = len(canonical)
    counts["canonical_yes_unique_uploaded_variations"] = canonical[
        "Uploaded_variation"
    ].nunique(dropna=False)

    canonical["dbNSFP_POPMAX_AF"] = pd.to_numeric(
        canonical["dbNSFP_POPMAX_AF"], errors="coerce"
    )
    counts["canonical_rows_with_numeric_popmax"] = int(
        canonical["dbNSFP_POPMAX_AF"].notna().sum()
    )

    canonical_missense = canonical[
        canonical["Consequence"].str.contains("missense_variant", na=False)
    ].copy()
    counts["canonical_missense_rows"] = len(canonical_missense)
    counts["canonical_missense_unique_uploaded_variations"] = canonical_missense[
        "Uploaded_variation"
    ].nunique(dropna=False)
    counts["canonical_missense_rows_with_numeric_popmax"] = int(
        canonical_missense["dbNSFP_POPMAX_AF"].notna().sum()
    )
    counts["canonical_missense_rows_popmax_lt_0_01"] = int(
        (canonical_missense["dbNSFP_POPMAX_AF"] < AF_THRESHOLD).sum()
    )

    rare = canonical[canonical["dbNSFP_POPMAX_AF"] < AF_THRESHOLD].copy()
    counts["rare_rows_popmax_lt_0_01"] = len(rare)
    counts["rare_unique_uploaded_variations"] = rare["Uploaded_variation"].nunique(dropna=False)
    counts["rare_uploaded_variation_groups_with_multiple_rows"] = int(
        (rare.groupby("Uploaded_variation").size() > 1).sum()
    )

    # Exact original order: numeric rare filter -> Uploaded_variation dedup -> missense.
    rare_unique = rare.drop_duplicates(subset=["Uploaded_variation"], keep="first").copy()
    counts["rare_rows_after_uploaded_variation_dedup"] = len(rare_unique)

    eligible_vep = rare_unique[
        rare_unique["Consequence"].str.contains("missense_variant", na=False)
    ].copy()
    counts["missense_rows_after_original_order"] = len(eligible_vep)
    counts["rare_missense_unique_variants_without_pre_missense_dedup"] = int(
        rare[rare["Consequence"].str.contains("missense_variant", na=False)][
            "Uploaded_variation"
        ].nunique(dropna=False)
    )

    eligible_vep["Uploaded_variation"] = eligible_vep["Uploaded_variation"].astype(str)
    eligible_raw = eligible_vep.merge(
        unique_metadata,
        on="Uploaded_variation",
        how="left",
        validate="one_to_one",
    )
    counts["rows_after_vcf_metadata_merge"] = len(eligible_raw)
    counts["rows_without_vcf_metadata_after_merge"] = int(
        eligible_raw["ClinicalSignificance"].isna().sum()
    )
    counts["rows_with_exact_vus_label_after_merge"] = int(
        (eligible_raw["ClinicalSignificance"] == VUS_LABEL).sum()
    )

    if not eligible_raw["ClinicalSignificance"].eq(VUS_LABEL).all():
        bad = Counter(eligible_raw["ClinicalSignificance"].fillna("<missing>"))
        raise ValueError(f"Non-VUS labels appeared after metadata merge: {dict(bad)}")

    eligible_cleaned = eligible_raw.copy()
    present_score_columns = [column for column in SCORE_COLUMNS if column in eligible_cleaned.columns]
    missing_score_columns = [column for column in SCORE_COLUMNS if column not in eligible_cleaned.columns]
    for column in present_score_columns:
        eligible_cleaned[column] = eligible_cleaned[column].apply(clean_score)

    counts["eligible_vus_rows"] = len(eligible_cleaned)
    counts["eligible_vus_columns"] = eligible_cleaned.shape[1]
    counts["requested_score_columns_present"] = len(present_score_columns)
    counts["requested_score_columns_missing"] = len(missing_score_columns)

    eligible_raw.to_csv(OUTDIR / "eligible_vus_raw.csv", index=False)
    eligible_cleaned.to_csv(OUTDIR / "eligible_vus.csv", index=False)
    write_counts(OUTDIR / "vus_processing_stage_counts.csv", counts)

    missingness = eligible_cleaned[present_score_columns].isna().sum().rename("missing_values")
    missingness = missingness.to_frame()
    missingness["non_missing_values"] = len(eligible_cleaned) - missingness["missing_values"]
    missingness["missing_fraction"] = (
        missingness["missing_values"] / len(eligible_cleaned) if len(eligible_cleaned) else float("nan")
    )
    missingness.index.name = "feature"
    missingness.reset_index().to_csv(OUTDIR / "eligible_vus_score_missingness.csv", index=False)

    pd.DataFrame({"Uploaded_variation": missing_ids}).to_csv(
        OUTDIR / "vus_input_ids_without_vep_rows.csv", index=False
    )

    gene_counts = (
        eligible_cleaned["source_gene"]
        .value_counts(dropna=False)
        .rename_axis("gene")
        .rename("count")
        .reset_index()
    )
    gene_counts.to_csv(OUTDIR / "eligible_vus_gene_counts.csv", index=False)

    report_lines = [
        "# Exact ClinVar VUS Same-Pipeline Audit",
        "",
        "## Definition",
        "",
        "This analysis includes only ClinVar records whose `CLNSIG` is exactly "
        "`Uncertain_significance`. `Conflicting_classifications_of_pathogenicity` records are "
        "kept in a separate count/VCF and are not part of this VUS cohort.",
        "",
        "## Same-Pipeline Eligibility Rules",
        "",
        "1. Five original genes: BRCA1, BRCA2, TP53, ATM, CHEK2.",
        "2. `CLNVC == single_nucleotide_variant`.",
        "3. `CLNDN` contains `breast` (case-insensitive).",
        "4. Exact `CLNSIG == Uncertain_significance`.",
        "5. VEP 113 offline annotation with `--canonical` and dbNSFP 5.1a.",
        "6. Retain `CANONICAL == YES`.",
        "7. Parse `dbNSFP_POPMAX_AF` numerically and retain `< 0.01`.",
        "8. Deduplicate by `Uploaded_variation` (first row), matching the original order.",
        "9. Retain consequences containing `missense_variant`.",
        "10. Clean requested score columns by keeping the first comma-separated numeric value.",
        "",
        "## Stage Counts",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
    ]
    report_lines.extend(f"| `{stage}` | {count} |" for stage, count in counts.items())
    report_lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            f"The same-pipeline eligible VUS cohort contains **{len(eligible_cleaned)} variants**.",
            "These variants have no binary ground-truth label in this analysis. No model prediction "
            "was run here, and AUC, accuracy, sensitivity, specificity, F1, or calibration against "
            "ClinVar cannot be calculated for this VUS-only cohort.",
            "",
            "`eligible_vus.csv` contains score-cleaned eligible variants but deliberately does not "
            "perform VUS-cohort median imputation. Any later model inference must reuse the fitted "
            "training-fold preprocessing and feature schema, not derive preprocessing parameters from VUS.",
            "",
        ]
    )
    (OUTDIR / "vus_same_pipeline_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"eligible_vus_rows={len(eligible_cleaned)}")
    print(f"missing_vep_ids={len(missing_ids)}")
    print(OUTDIR / "eligible_vus.csv")


if __name__ == "__main__":
    main()
