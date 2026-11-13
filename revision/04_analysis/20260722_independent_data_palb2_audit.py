from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


if os.name == "nt":
    THESIS_DATA = Path(workspace_path('thesis/data'))
    REVISION = Path(workspace_path('revision'))
else:
    THESIS_DATA = Path(workspace_path('thesis/data'))
    REVISION = Path(workspace_path('revision'))

SOURCE = THESIS_DATA / "clinvar_20250323.vcf.gz"
ORIGINAL_VCF = THESIS_DATA / "clinvar_bc_variants_for_vep.vcf"
ORIGINAL_VEP = THESIS_DATA / "clinvar_bc_annotated_canonical.txt"
BENCHMARK = THESIS_DATA / "benchmark_dataset.csv"
PALB2_DIR = REVISION / "04_analysis/vep_palb2"
PALB2_VCF = PALB2_DIR / "clinvar_palb2_binary_breast_snv_for_vep.vcf"
SIX_GENE_VCF = PALB2_DIR / "clinvar_6gene_binary_breast_snv_for_vep.vcf"
PALB2_VEP = PALB2_DIR / "clinvar_palb2_annotated_canonical.txt"
OUT = REVISION / "03_audit/20260722_data_palb2_independent_metrics.json"

FIVE_GENES = {"BRCA1", "BRCA2", "TP53", "ATM", "CHEK2"}
SIX_GENES = FIVE_GENES | {"PALB2"}
PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
BINARY = PATHOGENIC | BENIGN


def parse_info(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in text.split(";"):
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
            out[key] = unquote(value)
        else:
            out[field] = "1"
    return out


def genes_from_info(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.split(":", 1)[0] for item in value.split("|") if item]


def first_gene(value: str | None) -> str | None:
    genes = genes_from_info(value)
    return genes[0] if genes else None


def key(fields: list[str]) -> tuple[str, str, str, str, str]:
    return fields[0], fields[1], fields[2], fields[3], fields[4]


def read_vcf(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, object]] = []
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
                    "key": key(fields),
                    "id": fields[2],
                    "alt": fields[4],
                    "first_gene": first_gene(info.get("GENEINFO")),
                    "genes": genes_from_info(info.get("GENEINFO")),
                    "clnvc": info.get("CLNVC", ""),
                    "clndn": info.get("CLNDN", ""),
                    "clnsig": info.get("CLNSIG", ""),
                    "mc": info.get("MC", ""),
                    "clnrevstat": info.get("CLNREVSTAT", ""),
                }
            )
    return records


def basic_candidate(record: dict[str, object], gene_scope: set[str], use_any_gene: bool) -> bool:
    if use_any_gene:
        in_scope = bool(set(record["genes"]) & gene_scope)  # type: ignore[arg-type]
    else:
        in_scope = record["first_gene"] in gene_scope
    return (
        in_scope
        and str(record["clnvc"]).strip().lower() == "single_nucleotide_variant"
        and "breast" in str(record["clndn"]).lower()
    )


def binary_candidate(record: dict[str, object], gene_scope: set[str], use_any_gene: bool) -> bool:
    return basic_candidate(record, gene_scope, use_any_gene) and record["clnsig"] in BINARY


def load_vep(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    header_index = None
    columns = None
    metadata: dict[str, object] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if i < 25 and line.startswith("##"):
                if line.startswith("## ENSEMBL VARIANT EFFECT PREDICTOR"):
                    metadata["vep_version"] = line.strip()
                elif line.startswith("## Using cache"):
                    metadata["cache"] = line.strip()
                elif line.startswith("## assembly version"):
                    metadata["assembly"] = line.strip()
            if line.startswith("## VEP command-line"):
                metadata["command"] = line.strip()
            if line.startswith("#Uploaded_variation"):
                header_index = i
                columns = line.lstrip("#").rstrip("\n").split("\t")
                break
    if header_index is None or columns is None:
        raise RuntimeError("No VEP table header in {}".format(path))
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_index + 1,
        names=columns,
        header=None,
        dtype=str,
        low_memory=False,
    )
    metadata["columns"] = columns
    return df, metadata


def selected_ids(vep: pd.DataFrame) -> tuple[set[str], dict[str, int]]:
    counts: dict[str, int] = {
        "vep_rows": len(vep),
        "vep_unique_ids": vep["Uploaded_variation"].nunique(dropna=False),
    }
    canonical = vep[vep["CANONICAL"] == "YES"].copy()
    counts["canonical_rows"] = len(canonical)
    counts["canonical_unique_ids"] = canonical["Uploaded_variation"].nunique(dropna=False)
    canonical["dbNSFP_POPMAX_AF"] = pd.to_numeric(canonical["dbNSFP_POPMAX_AF"], errors="coerce")
    missense = canonical[canonical["Consequence"].str.contains("missense_variant", na=False)].copy()
    counts["canonical_missense_rows"] = len(missense)
    counts["canonical_missense_unique_ids"] = missense["Uploaded_variation"].nunique(dropna=False)
    counts["canonical_missense_numeric_popmax"] = int(missense["dbNSFP_POPMAX_AF"].notna().sum())
    counts["canonical_missense_popmax_lt_0_01"] = int((missense["dbNSFP_POPMAX_AF"] < 0.01).sum())
    rare = canonical[canonical["dbNSFP_POPMAX_AF"] < 0.01].copy()
    counts["rare_rows"] = len(rare)
    counts["rare_duplicate_id_groups"] = int((rare.groupby("Uploaded_variation").size() > 1).sum())
    rare = rare.drop_duplicates("Uploaded_variation", keep="first")
    final = rare[rare["Consequence"].str.contains("missense_variant", na=False)].copy()
    counts["final_rows_before_label_merge"] = len(final)
    return set(final["Uploaded_variation"].astype(str)), counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = read_vcf(SOURCE)
    original_input = read_vcf(ORIGINAL_VCF)
    palb2_input = read_vcf(PALB2_VCF)
    six_input = read_vcf(SIX_GENE_VCF)

    source_five_first = {r["key"] for r in source if binary_candidate(r, FIVE_GENES, False)}
    source_five_any = {r["key"] for r in source if binary_candidate(r, FIVE_GENES, True)}
    source_palb2_first = {
        r["key"] for r in source if binary_candidate(r, {"PALB2"}, False)
    }
    source_palb2_any = {r["key"] for r in source if binary_candidate(r, {"PALB2"}, True)}
    source_six_first = {r["key"] for r in source if binary_candidate(r, SIX_GENES, False)}

    original_keys = {r["key"] for r in original_input}
    palb2_keys = {r["key"] for r in palb2_input}
    six_keys = {r["key"] for r in six_input}

    original_vep, original_meta = load_vep(ORIGINAL_VEP)
    palb2_vep, palb2_meta = load_vep(PALB2_VEP)
    original_final_ids, original_stage = selected_ids(original_vep)
    palb2_final_ids, palb2_stage = selected_ids(palb2_vep)

    benchmark = pd.read_csv(BENCHMARK, dtype={"Uploaded_variation": str})
    benchmark_ids = set(benchmark["Uploaded_variation"].astype(str))

    original_labels = {str(r["id"]): str(r["clnsig"]) for r in original_input}
    palb2_labels = {str(r["id"]): str(r["clnsig"]) for r in palb2_input}
    original_final_label_counts = Counter(original_labels.get(i, "MISSING") for i in original_final_ids)
    palb2_final_label_counts = Counter(palb2_labels.get(i, "MISSING") for i in palb2_final_ids)
    original_final_review_counts = Counter(
        str(r["clnrevstat"]) for r in original_input if str(r["id"]) in original_final_ids
    )

    missing_palb2_vep_ids = sorted(
        {str(r["id"]) for r in palb2_input}
        - set(palb2_vep["Uploaded_variation"].astype(str))
    )
    missing_palb2_records = [
        {"id": r["id"], "alt": r["alt"], "key": r["key"]}
        for r in palb2_input
        if str(r["id"]) in missing_palb2_vep_ids
    ]

    vus_summary: dict[str, object] = {}
    for name, genes in (("five_gene", FIVE_GENES), ("six_gene", SIX_GENES)):
        basic = [r for r in source if basic_candidate(r, genes, False)]
        vus = [r for r in basic if r["clnsig"] == "Uncertain_significance"]
        conflicting = [
            r for r in basic if "Conflicting_classifications_of_pathogenicity" in str(r["clnsig"])
        ]
        other = [r for r in basic if r["clnsig"] not in BINARY and r not in vus and r not in conflicting]
        vus_summary[name] = {
            "basic_breast_snv": len(basic),
            "binary": sum(r["clnsig"] in BINARY for r in basic),
            "vus_strict": len(vus),
            "conflicting": len(conflicting),
            "vus_strict_mc_missense_proxy": sum("missense_variant" in str(r["mc"]) for r in vus),
            "conflicting_mc_missense_proxy": sum(
                "missense_variant" in str(r["mc"]) for r in conflicting
            ),
            "other_nonbinary": len(other),
            "clnsig": dict(Counter(str(r["clnsig"]) for r in basic)),
        }

    output: dict[str, object] = {
        "file_sha256": {
            str(path): sha256(path)
            for path in (SOURCE, ORIGINAL_VCF, ORIGINAL_VEP, BENCHMARK, PALB2_VCF, PALB2_VEP)
        },
        "source_filter_equivalence": {
            "source_five_first_gene_count": len(source_five_first),
            "source_five_any_gene_count": len(source_five_any),
            "stored_original_input_count": len(original_keys),
            "stored_original_minus_source_first": len(original_keys - source_five_first),
            "source_first_minus_stored_original": len(source_five_first - original_keys),
            "source_palb2_first_gene_count": len(source_palb2_first),
            "source_palb2_any_gene_count": len(source_palb2_any),
            "stored_palb2_input_count": len(palb2_keys),
            "stored_palb2_minus_source_first": len(palb2_keys - source_palb2_first),
            "source_first_minus_stored_palb2": len(source_palb2_first - palb2_keys),
            "source_six_first_gene_count": len(source_six_first),
            "stored_six_input_count": len(six_keys),
            "stored_six_minus_source_first": len(six_keys - source_six_first),
            "source_first_minus_stored_six": len(source_six_first - six_keys),
        },
        "original_519_replication": {
            **original_stage,
            "computed_final_unique_ids": len(original_final_ids),
            "benchmark_rows": len(benchmark),
            "benchmark_unique_ids": len(benchmark_ids),
            "computed_minus_benchmark_ids": len(original_final_ids - benchmark_ids),
            "benchmark_minus_computed_ids": len(benchmark_ids - original_final_ids),
            "final_label_counts": dict(original_final_label_counts),
            "final_review_status_counts": dict(original_final_review_counts),
        },
        "palb2_counterfactual": {
            **palb2_stage,
            "computed_final_unique_ids": len(palb2_final_ids),
            "final_ids": sorted(palb2_final_ids),
            "final_label_counts": dict(palb2_final_label_counts),
            "input_ids_without_vep": missing_palb2_records,
        },
        "vep_comparability": {
            "original": original_meta,
            "palb2": palb2_meta,
            "same_version": original_meta.get("vep_version") == palb2_meta.get("vep_version"),
            "same_cache": original_meta.get("cache") == palb2_meta.get("cache"),
            "same_assembly": original_meta.get("assembly") == palb2_meta.get("assembly"),
            "same_columns": original_meta.get("columns") == palb2_meta.get("columns"),
        },
        "vus_source_only_audit": vus_summary,
        "limitations": [
            "No VUS/conflicting VEP output exists; MC missense counts are proxies, not same-pipeline eligibility counts.",
            "The original notebook globally selected/dropped/imputed features before the train/test split.",
            "The original code did not filter on CLNREVSTAT, so it did not require expert-panel review.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
