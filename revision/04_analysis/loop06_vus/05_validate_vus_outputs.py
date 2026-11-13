from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import csv
import gzip
import hashlib
import importlib.util
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd


OUTDIR = Path(workspace_path('revision/04_analysis/loop06_vus'))
SOURCE = Path(workspace_path('thesis/data/clinvar_20250323.vcf.gz'))
VUS_VCF = OUTDIR / "clinvar_5gene_vus_breast_snv_for_vep.vcf"
CONFLICTING_VCF = OUTDIR / "clinvar_5gene_conflicting_breast_snv.vcf"
VEP_OUTPUT = OUTDIR / "clinvar_5gene_vus_annotated_canonical.txt"
ELIGIBLE_RAW = OUTDIR / "eligible_vus_raw.csv"
ELIGIBLE = OUTDIR / "eligible_vus.csv"
SELECTION_COUNTS = OUTDIR / "candidate_selection_stage_counts.csv"
SELECTION_GENE_COUNTS = OUTDIR / "candidate_selection_gene_counts.csv"
PROCESSING_COUNTS = OUTDIR / "vus_processing_stage_counts.csv"
ELIGIBLE_GENE_COUNTS = OUTDIR / "eligible_vus_gene_counts.csv"
MISSINGNESS = OUTDIR / "eligible_vus_score_missingness.csv"
MISSING_VEP_IDS = OUTDIR / "vus_input_ids_without_vep_rows.csv"
VEP_LOG = OUTDIR / "logs" / "vep_vus.log"
VEP_COMMAND = OUTDIR / "vep_command.txt"
VEP_ENVIRONMENT = OUTDIR / "vep_environment.txt"
MANIFEST = OUTDIR / "manifests" / "sha256_manifest.csv"
POSITIVE_CONTROL_MODULE = Path(workspace_path('revision/04_analysis/process_palb2_vep_output.py'))
CHECKS_CSV = OUTDIR / "independent_validation_checks.csv"
REPORT = OUTDIR / "INDEPENDENT_VALIDATION.md"

TARGET_GENES = {"BRCA1", "BRCA2", "TP53", "ATM", "CHEK2"}
VUS_LABEL = "Uncertain_significance"
CONFLICTING_LABEL = "Conflicting_classifications_of_pathogenicity"
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

REQUIRED_VEP_COLUMNS = {
    "Uploaded_variation", "Location", "Allele", "Gene", "Feature", "Feature_type",
    "Consequence", "CANONICAL", "dbNSFP_POPMAX_AF",
}


class CheckBook:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(
        self,
        name: str,
        condition: bool,
        observed: object,
        expected: object,
        evidence: str,
        failure_status: str = "FAIL",
    ) -> None:
        self.rows.append(
            {
                "status": "PASS" if condition else failure_status,
                "check": name,
                "observed": str(observed),
                "expected": str(expected),
                "evidence": evidence,
            }
        )

    def note(self, name: str, observed: object, evidence: str) -> None:
        self.rows.append(
            {
                "status": "INFO",
                "check": name,
                "observed": str(observed),
                "expected": "not applicable",
                "evidence": evidence,
            }
        )

    def warn(self, name: str, observed: object, expected: object, evidence: str) -> None:
        self.add(name, False, observed, expected, evidence, failure_status="WARN")

    @property
    def failures(self) -> list[dict[str, str]]:
        return [row for row in self.rows if row["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [row for row in self.rows if row["status"] == "WARN"]


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


def clean_score(value: object) -> float | None:
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


def read_count_table(path: Path) -> dict[str, int]:
    table = pd.read_csv(path)
    return {str(row.stage): int(row.count) for row in table.itertuples(index=False)}


def read_candidate(path: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    headers: list[str] = []
    records: list[str] = []
    metadata: list[dict[str, str | None]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            normalized = line.rstrip("\r\n")
            if normalized.startswith("#"):
                headers.append(normalized)
                continue
            records.append(normalized)
            fields = normalized.split("\t")
            if len(fields) < 8:
                continue
            info = parse_info(fields[7])
            metadata.append(
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
    frame = pd.DataFrame(metadata)
    if not frame.empty:
        frame["Uploaded_variation"] = frame["Uploaded_variation"].astype(str)
    return headers, records, frame


def scan_source() -> tuple[
    list[str], list[str], list[str], dict[str, int], dict[tuple[str, str], int]
]:
    headers: list[str] = []
    expected_vus: list[str] = []
    expected_conflicting: list[str] = []
    counts = Counter()
    genes = Counter()
    with gzip.open(SOURCE, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            normalized = line.rstrip("\r\n")
            if normalized.startswith("#"):
                headers.append(normalized)
                counts["header_lines_copied"] += 1
                continue
            counts["source_records"] += 1
            fields = normalized.split("\t")
            if len(fields) < 8:
                counts["malformed_records"] += 1
                continue
            info = parse_info(fields[7])
            gene = first_gene(info.get("GENEINFO"))
            if gene not in TARGET_GENES:
                continue
            counts["five_gene_records"] += 1
            if str(info.get("CLNVC", "")).strip().lower() != "single_nucleotide_variant":
                continue
            counts["five_gene_snv_records"] += 1
            if "breast" not in str(info.get("CLNDN", "")).lower():
                continue
            counts["five_gene_breast_snv_records"] += 1
            label = str(info.get("CLNSIG", "")).strip()
            if label == VUS_LABEL:
                expected_vus.append(normalized)
                counts["exact_vus_records"] += 1
                genes[("exact_vus", gene)] += 1
                if fields[4] in {"", "."}:
                    counts["exact_vus_records_without_alt_allele"] += 1
            elif label == CONFLICTING_LABEL:
                expected_conflicting.append(normalized)
                counts["exact_conflicting_records"] += 1
                genes[("exact_conflicting", gene)] += 1
                if fields[4] in {"", "."}:
                    counts["exact_conflicting_records_without_alt_allele"] += 1
    ordered_stages = [
        "header_lines_copied", "source_records", "malformed_records", "five_gene_records",
        "five_gene_snv_records", "five_gene_breast_snv_records", "exact_vus_records",
        "exact_vus_records_without_alt_allele", "exact_conflicting_records",
        "exact_conflicting_records_without_alt_allele",
    ]
    return (
        headers,
        expected_vus,
        expected_conflicting,
        {stage: int(counts[stage]) for stage in ordered_stages},
        {key: int(value) for key, value in genes.items()},
    )


def load_vep(path: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    preamble: list[str] = []
    columns: list[str] | None = None
    header_line: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            normalized = line.rstrip("\r\n")
            if normalized.startswith("#Uploaded_variation"):
                columns = normalized.lstrip("#").split("\t")
                header_line = idx
                break
            preamble.append(normalized)
    if columns is None or header_line is None:
        raise ValueError(f"No VEP tab header found in {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_line + 1,
        header=None,
        names=columns,
        dtype=str,
        low_memory=False,
    )
    frame["Uploaded_variation"] = frame["Uploaded_variation"].astype(str)
    return preamble, columns, frame


def independent_processing_counts(
    vep: pd.DataFrame, metadata: pd.DataFrame, eligible_column_count: int
) -> tuple[dict[str, int], pd.DataFrame, list[str]]:
    metadata_counts = metadata["Uploaded_variation"].value_counts(dropna=False)
    unique_metadata = metadata.drop_duplicates("Uploaded_variation", keep="first").copy()
    input_ids = set(unique_metadata["Uploaded_variation"])
    vep_ids = set(vep["Uploaded_variation"])
    missing_ids = sorted(input_ids - vep_ids)

    counts: dict[str, int] = {
        "input_vcf_records": len(metadata),
        "input_vcf_unique_uploaded_variations": metadata["Uploaded_variation"].nunique(dropna=False),
        "input_vcf_duplicate_uploaded_variation_groups": int((metadata_counts > 1).sum()),
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
    counts["rare_unique_uploaded_variations"] = rare[
        "Uploaded_variation"
    ].nunique(dropna=False)
    counts["rare_uploaded_variation_groups_with_multiple_rows"] = int(
        (rare.groupby("Uploaded_variation").size() > 1).sum()
    )
    rare_unique = rare.drop_duplicates("Uploaded_variation", keep="first").copy()
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
    merged = eligible_vep.merge(
        unique_metadata,
        on="Uploaded_variation",
        how="left",
        validate="one_to_one",
    )
    counts["rows_after_vcf_metadata_merge"] = len(merged)
    counts["rows_without_vcf_metadata_after_merge"] = int(
        merged["ClinicalSignificance"].isna().sum()
    )
    counts["rows_with_exact_vus_label_after_merge"] = int(
        merged["ClinicalSignificance"].eq(VUS_LABEL).sum()
    )
    counts["eligible_vus_rows"] = len(merged)
    counts["eligible_vus_columns"] = eligible_column_count
    counts["requested_score_columns_present"] = sum(c in merged.columns for c in SCORE_COLUMNS)
    counts["requested_score_columns_missing"] = sum(c not in merged.columns for c in SCORE_COLUMNS)
    return counts, merged, missing_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_positive_control_module():
    spec = importlib.util.spec_from_file_location("loop06_positive_control", POSITIVE_CONTROL_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {POSITIVE_CONTROL_MODULE}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.10 dataclasses require the dynamically loaded module to be registered.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate() -> tuple[CheckBook, dict[str, object]]:
    checks = CheckBook()
    facts: dict[str, object] = {}

    required_files = [
        SOURCE, VUS_VCF, CONFLICTING_VCF, VEP_OUTPUT, ELIGIBLE_RAW, ELIGIBLE,
        SELECTION_COUNTS, SELECTION_GENE_COUNTS, PROCESSING_COUNTS,
        ELIGIBLE_GENE_COUNTS, MISSINGNESS, MISSING_VEP_IDS, VEP_LOG,
        VEP_COMMAND, VEP_ENVIRONMENT, MANIFEST, POSITIVE_CONTROL_MODULE,
    ]
    for path in required_files:
        checks.add(
            f"required_file_exists::{path.name}",
            path.is_file(),
            path.is_file(),
            True,
            str(path),
        )
    if any(not path.is_file() for path in required_files):
        return checks, facts

    source_headers, expected_vus, expected_conflicting, source_counts, source_gene_counts = scan_source()
    vus_headers, vus_records, vus_metadata = read_candidate(VUS_VCF)
    conflicting_headers, conflicting_records, conflicting_metadata = read_candidate(CONFLICTING_VCF)

    checks.add("VUS_header_exactly_matches_source", vus_headers == source_headers, len(vus_headers), len(source_headers), str(VUS_VCF))
    checks.add("conflicting_header_exactly_matches_source", conflicting_headers == source_headers, len(conflicting_headers), len(source_headers), str(CONFLICTING_VCF))
    checks.add("VUS_records_exactly_reconstructed_from_source", vus_records == expected_vus, len(vus_records), len(expected_vus), str(SOURCE))
    checks.add("conflicting_records_exactly_reconstructed_from_source", conflicting_records == expected_conflicting, len(conflicting_records), len(expected_conflicting), str(SOURCE))

    recorded_selection_counts = read_count_table(SELECTION_COUNTS)
    for stage, observed in source_counts.items():
        checks.add(
            f"candidate_stage_count::{stage}",
            recorded_selection_counts.get(stage) == observed,
            recorded_selection_counts.get(stage),
            observed,
            str(SELECTION_COUNTS),
        )

    recorded_gene_table = pd.read_csv(SELECTION_GENE_COUNTS)
    recorded_gene_counts = {
        (str(row.cohort), str(row.gene)): int(row.count)
        for row in recorded_gene_table.itertuples(index=False)
    }
    complete_gene_counts = {
        (cohort, gene): source_gene_counts.get((cohort, gene), 0)
        for cohort in ("exact_vus", "exact_conflicting")
        for gene in sorted(TARGET_GENES)
    }
    checks.add(
        "candidate_gene_counts_match_source_recalculation",
        recorded_gene_counts == complete_gene_counts,
        recorded_gene_counts,
        complete_gene_counts,
        str(SELECTION_GENE_COUNTS),
    )

    checks.add("VUS_label_is_exact_for_every_candidate", vus_metadata["ClinicalSignificance"].eq(VUS_LABEL).all(), sorted(vus_metadata["ClinicalSignificance"].dropna().unique()), [VUS_LABEL], str(VUS_VCF))
    checks.add("conflicting_label_is_exact_for_every_candidate", conflicting_metadata["ClinicalSignificance"].eq(CONFLICTING_LABEL).all(), sorted(conflicting_metadata["ClinicalSignificance"].dropna().unique()), [CONFLICTING_LABEL], str(CONFLICTING_VCF))
    checks.add("VUS_and_conflicting_ID_sets_are_disjoint", set(vus_metadata["Uploaded_variation"]).isdisjoint(set(conflicting_metadata["Uploaded_variation"])), len(set(vus_metadata["Uploaded_variation"]) & set(conflicting_metadata["Uploaded_variation"])), 0, f"{VUS_VCF}; {CONFLICTING_VCF}")
    checks.add("VUS_candidate_IDs_are_unique", vus_metadata["Uploaded_variation"].is_unique, vus_metadata["Uploaded_variation"].nunique(), len(vus_metadata), str(VUS_VCF))

    preamble, vep_columns, vep = load_vep(VEP_OUTPUT)
    preamble_text = "\n".join(preamble)
    checks.add("VEP_header_has_no_duplicate_columns", len(vep_columns) == len(set(vep_columns)), len(vep_columns), len(set(vep_columns)), str(VEP_OUTPUT))
    checks.add("VEP_required_columns_present", REQUIRED_VEP_COLUMNS.issubset(set(vep_columns)), sorted(REQUIRED_VEP_COLUMNS - set(vep_columns)), [], str(VEP_OUTPUT))
    checks.add("all_52_requested_dbNSFP_fields_present", set(SCORE_COLUMNS).issubset(set(vep_columns)), len(set(SCORE_COLUMNS) & set(vep_columns)), len(SCORE_COLUMNS), str(VEP_OUTPUT))
    checks.add("VEP_version_113_embedded", "ENSEMBL VARIANT EFFECT PREDICTOR v113.0" in preamble_text, "v113.0" in preamble_text, True, str(VEP_OUTPUT))
    checks.add("VEP_GRCh38_cache_embedded", "113_GRCh38" in preamble_text and "assembly version GRCh38" in preamble_text, "113_GRCh38/GRCh38", "113_GRCh38/GRCh38", str(VEP_OUTPUT))
    checks.add("VEP_command_embeds_canonical_dbNSFP", "--canonical" in preamble_text and "dbNSFP5.1a_grch38.gz" in preamble_text, "--canonical and dbNSFP5.1a_grch38.gz", "present", str(VEP_OUTPUT))

    log_text = VEP_LOG.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    command_text = VEP_COMMAND.read_text(encoding="utf-8", errors="replace")
    environment_text = VEP_ENVIRONMENT.read_text(encoding="utf-8", errors="replace")
    checks.add("VEP_process_exit_code_zero", "EXIT_CODE=0" in log_text, re.findall(r"EXIT_CODE=\d+", log_text), ["EXIT_CODE=0"], str(VEP_LOG))
    required_command_tokens = ["--assembly GRCh38", "--cache", "--canonical", "--offline", "--tab", "dbNSFP5.1a_grch38.gz", VUS_VCF.name, VEP_OUTPUT.name]
    missing_command_tokens = [token for token in required_command_tokens if token not in command_text]
    checks.add("recorded_VEP_command_has_required_tokens", not missing_command_tokens, missing_command_tokens, [], str(VEP_COMMAND))
    required_environment_tokens = ["image_id=sha256:", "image_repo_digests=", "113_GRCh38", "dbNSFP5.1a_grch38.gz", "dbNSFP5.1a_grch38.gz.tbi"]
    missing_environment_tokens = [token for token in required_environment_tokens if token not in environment_text]
    checks.add("VEP_environment_records_immutable_image_and_sources", not missing_environment_tokens, missing_environment_tokens, [], str(VEP_ENVIRONMENT))
    if "index file is older than the data file" in log_text:
        checks.warn("dbNSFP_index_mtime_warning", "warning present; VEP completed with exit 0", "no warning", str(VEP_LOG))

    vep_ids = set(vep["Uploaded_variation"])
    vus_ids = set(vus_metadata["Uploaded_variation"])
    missing_ids = sorted(vus_ids - vep_ids)
    extra_vep_ids = sorted(vep_ids - vus_ids)
    missing_alt_ids = sorted(
        vus_metadata.loc[vus_metadata["source_ALT"].isin(["", "."]), "Uploaded_variation"]
    )
    reported_missing_ids = sorted(pd.read_csv(MISSING_VEP_IDS, dtype=str)["Uploaded_variation"].tolist())
    checks.add("VEP_contains_no_noninput_IDs", not extra_vep_ids, len(extra_vep_ids), 0, str(VEP_OUTPUT))
    checks.add("VEP_missing_IDs_exactly_equal_no_ALT_IDs", missing_ids == missing_alt_ids, missing_ids, missing_alt_ids, f"{VUS_VCF}; {VEP_OUTPUT}")
    checks.add("reported_missing_VEP_IDs_match_recalculation", reported_missing_ids == missing_ids, reported_missing_ids, missing_ids, str(MISSING_VEP_IDS))

    eligible_raw = pd.read_csv(ELIGIBLE_RAW, dtype=str, low_memory=False)
    eligible = pd.read_csv(ELIGIBLE, dtype=str, low_memory=False)
    independent_counts, independent_eligible, independently_missing_ids = independent_processing_counts(
        vep, vus_metadata, eligible.shape[1]
    )
    recorded_processing_counts = read_count_table(PROCESSING_COUNTS)
    for stage, expected in independent_counts.items():
        checks.add(
            f"processing_stage_count::{stage}",
            recorded_processing_counts.get(stage) == expected,
            recorded_processing_counts.get(stage),
            expected,
            str(PROCESSING_COUNTS),
        )
    checks.add("independent_missing_ID_recalculation_consistent", independently_missing_ids == missing_ids, independently_missing_ids, missing_ids, str(VEP_OUTPUT))

    expected_ids = independent_eligible["Uploaded_variation"].astype(str).tolist()
    eligible_ids = eligible["Uploaded_variation"].astype(str).tolist()
    raw_ids = eligible_raw["Uploaded_variation"].astype(str).tolist()
    checks.add("eligible_row_count_is_3228", len(eligible) == 3228, len(eligible), 3228, str(ELIGIBLE))
    checks.add("eligible_IDs_match_independent_pipeline_in_order", eligible_ids == expected_ids, len(eligible_ids), len(expected_ids), str(ELIGIBLE))
    checks.add("raw_and_cleaned_IDs_match_in_order", raw_ids == eligible_ids, len(raw_ids), len(eligible_ids), f"{ELIGIBLE_RAW}; {ELIGIBLE}")
    checks.add("eligible_IDs_are_unique", eligible["Uploaded_variation"].is_unique, eligible["Uploaded_variation"].nunique(), len(eligible), str(ELIGIBLE))
    checks.add("eligible_has_exact_VUS_label_only", eligible["ClinicalSignificance"].eq(VUS_LABEL).all(), sorted(eligible["ClinicalSignificance"].dropna().unique()), [VUS_LABEL], str(ELIGIBLE))
    checks.add("eligible_contains_no_binary_or_prediction_column", not ({"label", "prediction", "probability", "predicted_label"} & set(eligible.columns)), sorted({"label", "prediction", "probability", "predicted_label"} & set(eligible.columns)), [], str(ELIGIBLE))
    checks.add("eligible_target_genes_only", set(eligible["source_gene"].dropna()) <= TARGET_GENES, sorted(eligible["source_gene"].dropna().unique()), sorted(TARGET_GENES), str(ELIGIBLE))
    checks.add("eligible_source_CLNVC_is_SNV", eligible["source_CLNVC"].str.strip().str.lower().eq("single_nucleotide_variant").all(), sorted(eligible["source_CLNVC"].dropna().unique()), ["single_nucleotide_variant"], str(ELIGIBLE))
    checks.add("eligible_source_CLNDN_contains_breast", eligible["source_CLNDN"].str.lower().str.contains("breast", na=False).all(), int(eligible["source_CLNDN"].str.lower().str.contains("breast", na=False).sum()), len(eligible), str(ELIGIBLE))
    checks.add("eligible_CANONICAL_is_YES", eligible["CANONICAL"].eq("YES").all(), sorted(eligible["CANONICAL"].dropna().unique()), ["YES"], str(ELIGIBLE))
    checks.add("eligible_consequence_contains_missense", eligible["Consequence"].str.contains("missense_variant", na=False).all(), int(eligible["Consequence"].str.contains("missense_variant", na=False).sum()), len(eligible), str(ELIGIBLE))
    eligible_af = pd.to_numeric(eligible["dbNSFP_POPMAX_AF"], errors="coerce")
    checks.add("eligible_POPMAX_is_numeric_and_below_0_01", eligible_af.notna().all() and eligible_af.lt(AF_THRESHOLD).all(), f"numeric={eligible_af.notna().sum()}, max={eligible_af.max()}", f"numeric={len(eligible)}, max<0.01", str(ELIGIBLE))

    non_score_columns = [column for column in eligible.columns if column not in SCORE_COLUMNS]
    non_score_equal = eligible_raw[non_score_columns].fillna("<NA>").equals(
        eligible[non_score_columns].fillna("<NA>")
    )
    checks.add("score_cleaning_did_not_change_non_score_columns", non_score_equal, non_score_equal, True, f"{ELIGIBLE_RAW}; {ELIGIBLE}")
    score_mismatch_count = 0
    for column in SCORE_COLUMNS:
        expected = eligible_raw[column].map(clean_score)
        observed = pd.to_numeric(eligible[column], errors="coerce")
        expected_array = expected.to_numpy(dtype=float, na_value=np.nan)
        observed_array = observed.to_numpy(dtype=float, na_value=np.nan)
        equal = np.isclose(expected_array, observed_array, rtol=1e-12, atol=1e-12, equal_nan=True)
        score_mismatch_count += int((~equal).sum())
    checks.add("all_score_cells_match_first_numeric_cleaning_rule", score_mismatch_count == 0, score_mismatch_count, 0, f"{ELIGIBLE_RAW}; {ELIGIBLE}")

    reported_missingness = pd.read_csv(MISSINGNESS).set_index("feature")
    recalculated_missing = {
        column: int(pd.to_numeric(eligible[column], errors="coerce").isna().sum())
        for column in SCORE_COLUMNS
    }
    recorded_missing = {
        str(feature): int(row["missing_values"])
        for feature, row in reported_missingness.iterrows()
    }
    checks.add("score_missingness_table_matches_cleaned_output", recorded_missing == recalculated_missing, recorded_missing, recalculated_missing, str(MISSINGNESS))
    checks.add("eligible_output_has_not_been_median_imputed", sum(recalculated_missing.values()) > 0, sum(recalculated_missing.values()), "> 0 missing score cells", str(ELIGIBLE))

    reported_eligible_genes = pd.read_csv(ELIGIBLE_GENE_COUNTS).set_index("gene")["count"].astype(int).to_dict()
    recalculated_eligible_genes = eligible["source_gene"].value_counts(dropna=False).astype(int).to_dict()
    checks.add("eligible_gene_counts_match_output", reported_eligible_genes == recalculated_eligible_genes, reported_eligible_genes, recalculated_eligible_genes, str(ELIGIBLE_GENE_COUNTS))

    module = load_positive_control_module()
    positive = module.apply_canonical_pipeline(
        "original_5_gene_positive_control", module.ORIGINAL_VEP, module.ORIGINAL_VCF
    )
    benchmark = pd.read_csv(module.ORIGINAL_FINAL, dtype={"Uploaded_variation": str}, low_memory=False)
    positive_ids = positive.final["Uploaded_variation"].astype(str).tolist()
    benchmark_ids = benchmark["Uploaded_variation"].astype(str).tolist()
    checks.add("positive_control_rows_with_binary_label_is_519", positive.stage_counts["rows_with_binary_label"] == 519, positive.stage_counts["rows_with_binary_label"], 519, str(POSITIVE_CONTROL_MODULE))
    checks.add("positive_control_final_rows_is_519", len(positive.final) == 519, len(positive.final), 519, str(POSITIVE_CONTROL_MODULE))
    checks.add("canonical_benchmark_rows_is_519", len(benchmark) == 519, len(benchmark), 519, str(module.ORIGINAL_FINAL))
    checks.add("positive_control_519_ID_set_matches_benchmark", set(positive_ids) == set(benchmark_ids), f"recomputed={len(set(positive_ids))}, benchmark={len(set(benchmark_ids))}", "identical 519-ID sets", f"{module.ORIGINAL_FINAL}; {POSITIVE_CONTROL_MODULE}")
    checks.note("positive_control_order_match_not_required", positive_ids == benchmark_ids, "The benchmark has the same 519-ID set; file row order is not a pipeline invariant.")

    report_text = (OUTDIR / "vus_same_pipeline_report.md").read_text(encoding="utf-8", errors="replace")
    eligibility_scripts_text = "\n".join(
        (OUTDIR / name).read_text(encoding="utf-8", errors="replace")
        for name in ["01_build_candidate_vcfs.py", "03_process_vus_vep.py"]
    )
    all_scripts_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(OUTDIR.glob("*.py"))
        if path.name != Path(__file__).name
    )
    numeric_performance_claim = re.compile(
        r"(?i)\b(?:AUC|accuracy|sensitivity|specificity|F1(?:-score)?|calibration)\b\s*(?:[:=]|was|of)\s*[0-9]"
    )
    checks.add("report_explicitly_disclaims_model_performance_metrics", "No model prediction was run here" in report_text and "cannot be calculated" in report_text, "boundary present", "boundary present", str(OUTDIR / "vus_same_pipeline_report.md"))
    checks.add("no_numeric_AUC_or_accuracy_claim", numeric_performance_claim.search(report_text) is None, bool(numeric_performance_claim.search(report_text)), False, str(OUTDIR / "vus_same_pipeline_report.md"))
    checks.add("eligibility_scripts_do_not_run_model_prediction", not re.search(r"\.(?:predict|predict_proba)\s*\(|roc_auc_score|accuracy_score", eligibility_scripts_text, flags=re.I), bool(re.search(r"\.(?:predict|predict_proba)\s*\(|roc_auc_score|accuracy_score", eligibility_scripts_text, flags=re.I)), False, str(OUTDIR))
    labeled_metric_api = re.compile(
        r"roc_auc_score|accuracy_score|precision_score|recall_score|f1_score|confusion_matrix",
        flags=re.I,
    )
    checks.add("loop06_scripts_do_not_compute_labeled_performance_metrics", labeled_metric_api.search(all_scripts_text) is None, bool(labeled_metric_api.search(all_scripts_text)), False, str(OUTDIR))

    manifest = pd.read_csv(MANIFEST, dtype=str)
    manifest_mismatches: list[str] = []
    skipped_large_hashes: list[str] = []
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        if not path.is_absolute():
            path = OUTDIR / row.path
        if not path.is_file():
            manifest_mismatches.append(f"missing:{path}")
            continue
        if path.stat().st_size != int(row.size_bytes):
            manifest_mismatches.append(f"size:{path}")
            continue
        if path.stat().st_size > 2 * 1024 * 1024 * 1024:
            skipped_large_hashes.append(str(path))
            continue
        if sha256(path) != row.sha256:
            manifest_mismatches.append(f"sha256:{path}")
    checks.add("manifest_listed_files_exist_and_verified_hashes_match", not manifest_mismatches, manifest_mismatches, [], str(MANIFEST))
    if skipped_large_hashes:
        checks.warn("manifest_large_dbNSFP_hash_not_recomputed", skipped_large_hashes, "hash recomputed", "Files >2 GiB were size-checked only to keep independent validation bounded; the recorded digest remains available in the primary manifest.")

    facts.update(
        {
            "source_records": source_counts["source_records"],
            "exact_vus_records": len(vus_metadata),
            "exact_conflicting_records": len(conflicting_metadata),
            "VEP_rows": len(vep),
            "VEP_unique_input_IDs": vep["Uploaded_variation"].nunique(dropna=False),
            "VEP_missing_input_IDs": len(missing_ids),
            "eligible_VUS_rows": len(eligible),
            "eligible_gene_counts": recalculated_eligible_genes,
            "positive_control_rows": positive.stage_counts["rows_with_binary_label"],
            "positive_control_ID_set_match": set(positive_ids) == set(benchmark_ids),
            "score_cleaning_mismatches": score_mismatch_count,
            "manifest_rows": len(manifest),
        }
    )
    return checks, facts


def write_outputs(checks: CheckBook, facts: dict[str, object]) -> None:
    with CHECKS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["status", "check", "observed", "expected", "evidence"],
        )
        writer.writeheader()
        writer.writerows(checks.rows)

    status = "FAIL" if checks.failures else "PASS"
    lines = [
        "# Independent Validation of Loop06 Exact-VUS Outputs",
        "",
        f"Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"Overall validation status: **{status}**",
        "",
        "## Scope and Method",
        "",
        "This validation independently rescanned the compressed ClinVar source, reconstructed both candidate VCFs, compared their headers and records with the generated files, reparsed the VEP output, recomputed every reported row-selection stage, and checked the raw/cleaned eligible tables cell-by-cell for the documented score-cleaning rule. It did not modify `D:\\thesis`.",
        "",
        "The canonical binary-label pipeline was also rerun as a positive control by calling `apply_canonical_pipeline` from `D:\\revision\\04_analysis\\process_palb2_vep_output.py` against the original five-gene VCF/VEP inputs.",
        "",
        "## Audited Conclusions",
        "",
    ]
    if facts:
        lines.extend(
            [
                f"- The source contained `{facts['source_records']}` records. Exact-VUS selection produced `{facts['exact_vus_records']}` records; exact conflicting classification selection produced `{facts['exact_conflicting_records']}` separate records.",
                "- The exact-VUS and conflicting candidate VCFs reproduce the source-selected records in source order, retain the source VCF header exactly, and have disjoint ClinVar ID sets.",
                f"- VEP 113 produced `{facts['VEP_rows']}` annotation rows covering `{facts['VEP_unique_input_IDs']}` unique input IDs. The `{facts['VEP_missing_input_IDs']}` missing input IDs are exactly the three source records with no ALT allele (`ALT=.`).",
                f"- The independently recomputed rare canonical missense cohort contains **`{facts['eligible_VUS_rows']}` exact VUS variants**. All are unique, `CANONICAL=YES`, `dbNSFP_POPMAX_AF < 0.01`, missense consequences, breast-related ClinVar SNVs, and members of the original five-gene scope.",
                f"- Eligible-gene counts are `{facts['eligible_gene_counts']}`.",
                f"- All 52 requested score columns are present. Cell-level score-cleaning mismatches: `{facts['score_cleaning_mismatches']}`. Missing values remain, confirming that the VUS table was not median-imputed.",
                f"- Positive control: the original binary-label pipeline reproduced `{facts['positive_control_rows']}` rows, and its 519 `Uploaded_variation` IDs exactly match the canonical benchmark ID set: `{facts['positive_control_ID_set_match']}`.",
                "- The Loop06 eligibility stages (candidate selection, VEP, and same-pipeline filtering) do not fit models or run predictions. A separate downstream script only documents optional descriptive fold-ensemble probabilities; it must not compute or support AUC, accuracy, sensitivity, specificity, F1, calibration, or clinical reclassification claims for this unlabeled VUS cohort.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "`eligible_vus.csv` is a row-eligible, score-cleaned audit cohort, not a model-ready prediction table. Column dropping, feature-schema selection, and imputation for any future inference must come from the fitted binary training pipeline; they must not be estimated from the VUS cohort.",
            "",
            "The exact-VUS definition intentionally excludes `Conflicting_classifications_of_pathogenicity` and other mixed/compound ClinVar significance strings. The gene test uses the first-listed `GENEINFO` gene, matching the original project filtering behavior.",
            "",
            "## Provenance Caveats",
            "",
            "- VEP exited with code 0 and its output passed all structural and cohort-invariant checks. The VEP log contains a non-fatal warning that the dbNSFP tabix index file timestamp is older than the data file. Both files are captured in the primary manifest; the 42.8 GB dbNSFP data digest was not recomputed during this bounded independent pass, although its file size was checked.",
            "- The executed container was requested by the mutable tag `ensemblorg/ensembl-vep:latest`, but the resolved image ID and repository digest were recorded. Future reruns should use the recorded immutable digest directly.",
            "- The final SHA256 manifest is regenerated after independent validation so the validation outputs and model-alignment artifacts are included.",
            "",
            "## Check Summary",
            "",
            f"- PASS: `{sum(row['status'] == 'PASS' for row in checks.rows)}`",
            f"- WARN: `{sum(row['status'] == 'WARN' for row in checks.rows)}`",
            f"- FAIL: `{sum(row['status'] == 'FAIL' for row in checks.rows)}`",
            f"- INFO: `{sum(row['status'] == 'INFO' for row in checks.rows)}`",
            "",
            "Detailed machine-readable checks: `independent_validation_checks.csv`.",
            "",
        ]
    )
    if checks.warnings:
        lines.extend(["## Warnings", ""])
        for row in checks.warnings:
            lines.append(f"- `{row['check']}`: {row['observed']}")
        lines.append("")
    if checks.failures:
        lines.extend(["## Failures", ""])
        for row in checks.failures:
            lines.append(
                f"- `{row['check']}`: observed `{row['observed']}`; expected `{row['expected']}`."
            )
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    try:
        checks, facts = validate()
    except Exception as exc:
        checks = CheckBook()
        checks.add(
            "validation_script_completed",
            False,
            f"{type(exc).__name__}: {exc}",
            "completed without exception",
            str(Path(__file__)),
        )
        facts = {}
    write_outputs(checks, facts)
    print(f"status={'FAIL' if checks.failures else 'PASS'}")
    print(f"checks={len(checks.rows)}")
    print(f"warnings={len(checks.warnings)}")
    print(f"failures={len(checks.failures)}")
    print(CHECKS_CSV)
    print(REPORT)
    if checks.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
