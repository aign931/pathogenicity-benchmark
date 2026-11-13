from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


from paths import DATA_DIR, FOLDS_DIR
RAW_DIR = DATA_DIR / 'raw'
OUTDIR = DATA_DIR / 'processed'
ORIGINAL_VEP = RAW_DIR / 'clinvar_bc_annotated_canonical.txt'
ORIGINAL_VCF = RAW_DIR / 'clinvar_bc_variants_for_vep.vcf'
PALB2_VEP = RAW_DIR / 'clinvar_palb2_annotated_canonical.txt'
PALB2_VCF = RAW_DIR / 'clinvar_palb2_binary_breast_snv_for_vep.vcf'

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


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    original = apply_canonical_pipeline('canonical519', ORIGINAL_VEP, ORIGINAL_VCF)
    palb2 = apply_canonical_pipeline('palb2', PALB2_VEP, PALB2_VCF)
    # Preserve the row order used for the manuscript's fixed folds.
    main_folds = pd.read_csv(FOLDS_DIR / 'main.csv', dtype={'Uploaded_variation': str})
    pooled_folds = pd.read_csv(FOLDS_DIR / 'palb2.csv', dtype={'Uploaded_variation': str})
    main_ids = main_folds.sort_values('row_id')['Uploaded_variation'].tolist()
    palb2_ids = pooled_folds.sort_values('row_id')['Uploaded_variation'].tolist()
    palb2_ids = [variant for variant in palb2_ids if variant not in set(main_ids)]
    for frame, ids, filename in [(original.cleaned, main_ids, 'canonical519.csv'),
                                  (palb2.cleaned, palb2_ids, 'palb2.csv')]:
        if frame['Uploaded_variation'].duplicated().any() or set(frame['Uploaded_variation']) != set(ids):
            raise ValueError('Processed variants do not match fixed folds: ' + filename)
        frame.set_index('Uploaded_variation').loc[ids].reset_index().to_csv(OUTDIR / filename, index=False)
    write_counts_csv(original, OUTDIR / 'canonical_selection_counts.csv')
    write_counts_csv(palb2, OUTDIR / 'palb2_selection_counts.csv')
    print('Saved processed annotations to', OUTDIR)

if __name__ == '__main__':
    main()
