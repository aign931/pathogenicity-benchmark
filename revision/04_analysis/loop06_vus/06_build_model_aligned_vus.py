from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import ast
import csv
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path(workspace_path('revision/04_analysis/loop06_vus'))
ELIGIBLE = OUTDIR / "eligible_vus.csv"
VEP_OUTPUT = OUTDIR / "clinvar_5gene_vus_annotated_canonical.txt"
LOOP05_COMMON = Path(workspace_path('revision/04_analysis/loop05_corrected_common.py'))
LOOP05_SCHEMA = Path(
    workspace_path('revision/04_analysis/loop05_corrected_common5fold/inputs/feature_schema_43.csv')
)
PALB2_PROCESSOR = Path(workspace_path('revision/04_analysis/process_palb2_vep_output.py'))

MATRIX_OUTPUT = OUTDIR / "vus_full43_features_unimputed.csv"
METADATA_OUTPUT = OUTDIR / "vus_full43_metadata.csv"
ALIGNED_OUTPUT = OUTDIR / "vus_full43_aligned_unimputed.csv"
MISSINGNESS_OUTPUT = OUTDIR / "vus_full43_missingness.csv"
FIELD_MAPPING_OUTPUT = OUTDIR / "vus_vep52_to_full43_field_mapping.csv"
FIELD_AUDIT_OUTPUT = OUTDIR / "vus_field_count_audit.csv"
REPORT_OUTPUT = OUTDIR / "vus_full43_alignment_report.md"


def parse_literal_list(path: Path, variable_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{variable_name} is not a literal list of strings in {path}")
        return value
    raise ValueError(f"Could not find {variable_name} in {path}")


def read_vep_header(path: Path) -> tuple[list[str], list[str], str]:
    command = ""
    columns: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("## VEP command-line:"):
                command = line.rstrip("\n")
            elif line.startswith("#Uploaded_variation"):
                columns = line.lstrip("#").rstrip("\n").split("\t")
                break
    if not command or not columns:
        raise ValueError("VEP command/header was not found")
    plugin = command.split("--plugin ", 1)[1].split(" --tab", 1)[0].strip()
    plugin_parts = plugin.split(",")
    if len(plugin_parts) < 3 or plugin_parts[0] != "dbNSFP":
        raise ValueError("Could not parse the dbNSFP field list from the VEP header")
    return plugin_parts[2:], columns, command


def write_audit(rows: list[dict[str, object]]) -> None:
    with FIELD_AUDIT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "value", "status", "details"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    full_features = parse_literal_list(LOOP05_COMMON, "FULL_FEATURES")
    original_high_missing = parse_literal_list(PALB2_PROCESSOR, "HIGH_MISSING_DROP_COLS")
    schema_features = pd.read_csv(LOOP05_SCHEMA)["feature"].astype(str).tolist()
    requested_fields, vep_columns, vep_command = read_vep_header(VEP_OUTPUT)
    eligible = pd.read_csv(ELIGIBLE, dtype={"Uploaded_variation": str}, low_memory=False)

    if len(full_features) != 43 or len(set(full_features)) != 43:
        raise AssertionError(f"Loop05 FULL_FEATURES must contain 43 unique fields, got {len(full_features)}")
    if schema_features != full_features:
        raise AssertionError("Loop05 feature_schema_43.csv does not exactly match FULL_FEATURES order")
    if len(requested_fields) != 52 or len(set(requested_fields)) != 52:
        raise AssertionError(f"Expected 52 unique dbNSFP request fields, got {len(requested_fields)}")
    if not set(requested_fields).issubset(set(vep_columns)):
        raise AssertionError("At least one requested dbNSFP field is absent from the VEP tab header")
    if not set(requested_fields).issubset(set(eligible.columns)):
        raise AssertionError("At least one requested dbNSFP field is absent from eligible_vus.csv")
    if not set(full_features).issubset(set(eligible.columns)):
        raise AssertionError("At least one Loop05 FULL_FEATURE is absent from eligible_vus.csv")

    excluded = [field for field in requested_fields if field not in full_features]
    if set(excluded) != set(original_high_missing) or len(excluded) != 9:
        raise AssertionError(
            "The 52-to-43 difference does not match the original nine high-missingness exclusions"
        )

    source_features = eligible[full_features].copy()
    numeric_features = source_features.apply(pd.to_numeric, errors="coerce")
    coercion_failures = int(((source_features.notna()) & (numeric_features.isna())).sum().sum())
    if coercion_failures:
        raise AssertionError(f"Numeric coercion introduced {coercion_failures} new missing values")
    if len(eligible) != 3228:
        raise AssertionError(f"Expected 3,228 eligible VUS rows, got {len(eligible)}")
    if "label" in eligible.columns:
        raise AssertionError("eligible_vus.csv unexpectedly contains a model label column")
    if not eligible["ClinicalSignificance"].eq("Uncertain_significance").all():
        raise AssertionError("The aligned input contains a non-VUS ClinVar significance")

    metadata_columns = [
        "Uploaded_variation", "Location", "Allele", "Gene", "Feature", "Consequence",
        "source_gene", "source_CHROM", "source_POS", "source_REF", "source_ALT",
        "ClinicalSignificance",
    ]
    missing_metadata = [column for column in metadata_columns if column not in eligible.columns]
    if missing_metadata:
        raise AssertionError(f"Missing metadata columns: {missing_metadata}")

    metadata = eligible[metadata_columns].copy()
    metadata.insert(0, "row_id", np.arange(len(eligible), dtype=int))
    aligned = pd.concat(
        [metadata.reset_index(drop=True), numeric_features.reset_index(drop=True)], axis=1
    )

    # Deliberately no fillna/imputation operation occurs here.
    numeric_features.to_csv(MATRIX_OUTPUT, index=False)
    metadata.to_csv(METADATA_OUTPUT, index=False)
    aligned.to_csv(ALIGNED_OUTPUT, index=False)

    reread_matrix = pd.read_csv(MATRIX_OUTPUT)
    if reread_matrix.columns.tolist() != full_features or reread_matrix.shape != (3228, 43):
        raise AssertionError("Written 43-feature matrix failed exact shape/order validation")
    if int(reread_matrix.isna().sum().sum()) != int(numeric_features.isna().sum().sum()):
        raise AssertionError("Missing-value count changed during aligned-table round trip")
    if "label" in aligned.columns:
        raise AssertionError("Aligned VUS table must not contain a model label")

    missingness = pd.DataFrame(
        {
            "feature_index": np.arange(1, len(full_features) + 1),
            "feature": full_features,
            "missing_values": [int(numeric_features[field].isna().sum()) for field in full_features],
            "nonmissing_values": [int(numeric_features[field].notna().sum()) for field in full_features],
            "missing_fraction": [float(numeric_features[field].isna().mean()) for field in full_features],
        }
    )
    missingness.to_csv(MISSINGNESS_OUTPUT, index=False)

    mapping_rows = []
    for index, field in enumerate(requested_fields, start=1):
        included = field in full_features
        mapping_rows.append(
            {
                "vep_request_index": index,
                "field": field,
                "present_in_vep_header": int(field in vep_columns),
                "present_in_eligible_vus": int(field in eligible.columns),
                "included_in_loop05_full43": int(included),
                "loop05_feature_index": full_features.index(field) + 1 if included else "",
                "exclusion_reason": "" if included else "original high-missingness feature exclusion",
            }
        )
    pd.DataFrame(mapping_rows).to_csv(FIELD_MAPPING_OUTPUT, index=False)

    common_41 = [field for field in full_features if field not in {"MutScore_score", "MVP_score"}]
    audit_rows = [
        {"check": "vep_requested_dbnsfp_fields", "value": len(requested_fields), "status": "PASS", "details": "Parsed from embedded VEP command"},
        {"check": "vep_requested_unique_fields", "value": len(set(requested_fields)), "status": "PASS", "details": "No duplicate request fields"},
        {"check": "requested_fields_present_in_vep_header", "value": sum(field in vep_columns for field in requested_fields), "status": "PASS", "details": "All 52 present"},
        {"check": "requested_fields_present_in_eligible_vus", "value": sum(field in eligible.columns for field in requested_fields), "status": "PASS", "details": "All 52 present before model-panel alignment"},
        {"check": "original_high_missingness_exclusions", "value": len(excluded), "status": "PASS", "details": ";".join(excluded)},
        {"check": "loop05_full_features", "value": len(full_features), "status": "PASS", "details": "Exact order parsed from Loop05 source and matched to schema CSV"},
        {"check": "varity_model_features", "value": len(full_features), "status": "PASS", "details": "FULL_FEATURES"},
        {"check": "mutscore_model_features", "value": len(full_features) - 1, "status": "PASS", "details": "FULL_FEATURES excluding MutScore_score"},
        {"check": "mvp_model_features", "value": len(full_features) - 1, "status": "PASS", "details": "FULL_FEATURES excluding MVP_score"},
        {"check": "three_model_common_features", "value": len(common_41), "status": "PASS", "details": "FULL_FEATURES excluding both self-score fields"},
        {"check": "aligned_vus_rows", "value": len(numeric_features), "status": "PASS", "details": "No rows added or removed during feature alignment"},
        {"check": "aligned_feature_columns", "value": numeric_features.shape[1], "status": "PASS", "details": "Exact 43-column model panel"},
        {"check": "numeric_coercion_failures", "value": coercion_failures, "status": "PASS", "details": "No new NaNs created from nonmissing values"},
        {"check": "unimputed_missing_cells", "value": int(numeric_features.isna().sum().sum()), "status": "PASS", "details": "Missing values deliberately retained"},
        {"check": "model_label_columns", "value": int("label" in aligned.columns), "status": "PASS", "details": "No binary or predicted label assigned"},
    ]
    write_audit(audit_rows)

    lines = [
        "# VUS Loop05 Full-Feature Alignment",
        "",
        "- Eligible exact-VUS rows: `3,228`.",
        "- VEP/dbNSFP request fields: `52` (all present).",
        "- Original high-missingness exclusions: `9`.",
        "- Loop05 `FULL_FEATURES`: `43`, in exact source-defined order.",
        "- Missing feature cells retained without imputation: `{}`.".format(int(numeric_features.isna().sum().sum())),
        "- Binary/model label columns assigned: `0`.",
        "",
        "`vus_full43_features_unimputed.csv` contains exactly 43 numeric feature columns. "
        "`vus_full43_metadata.csv` provides row-aligned identifiers and ClinVar VUS metadata. "
        "`vus_full43_aligned_unimputed.csv` combines both for convenience.",
        "",
        "No model inference was performed by this alignment script. Later inference must use each "
        "saved fold's training median and, for MVP, its saved fold-specific MinMaxScaler.",
        "",
        "Embedded VEP command used to audit the 52 requested fields:",
        "",
        "```text", vep_command, "```", "",
    ]
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"matrix_shape={numeric_features.shape}")
    print(f"unimputed_missing_cells={int(numeric_features.isna().sum().sum())}")
    print(MATRIX_OUTPUT)


if __name__ == "__main__":
    main()
