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
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


ROOT = Path(workspace_path('thesis'))
DATA = ROOT / "data"
OUT = Path(workspace_path('revision/03_audit'))

TARGET_GENES = ["BRCA1", "BRCA2", "TP53", "ATM", "CHEK2"]
PALB2_PLUS_TARGET = TARGET_GENES + ["PALB2"]

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
BINARY_LABELS = PATHOGENIC_LABELS | BENIGN_LABELS


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


def gene_from_info(geneinfo: str | None) -> str | None:
    if not geneinfo:
        return None
    # ClinVar can encode multiple genes as GENE1:ID|GENE2:ID.
    return geneinfo.split("|", 1)[0].split(":", 1)[0]


def class_bucket(clnsig: str | None) -> str:
    if not clnsig:
        return "missing"
    if clnsig in PATHOGENIC_LABELS:
        return "pathogenic_binary"
    if clnsig in BENIGN_LABELS:
        return "benign_binary"
    if "Uncertain_significance" in clnsig:
        return "vus"
    if "Conflicting" in clnsig:
        return "conflicting"
    return "other_nonbinary"


def count_vcf(path: Path) -> dict[str, object]:
    counts: dict[str, object] = {}
    gene_counts_after_basic: Counter[str] = Counter()
    gene_counts_binary: Counter[str] = Counter()
    gene_counts_binary_missense_proxy: Counter[str] = Counter()
    gene_counts_vus_conflicting: Counter[str] = Counter()
    gene_counts_vus_conflicting_missense_proxy: Counter[str] = Counter()
    clnsig_by_gene: dict[str, Counter[str]] = defaultdict(Counter)
    stage = Counter()

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, vid, ref, alt, _qual, _filter, info_s = fields[:8]
            info = parse_info(info_s)
            gene = gene_from_info(info.get("GENEINFO"))
            clnvc = info.get("CLNVC", "")
            clndn = info.get("CLNDN", "")
            clnsig = info.get("CLNSIG", "")
            molecular_consequence = info.get("MC", "")
            is_missense_proxy = "missense_variant" in molecular_consequence

            if gene not in PALB2_PLUS_TARGET:
                continue
            stage["gene_scope_plus_palb2"] += 1

            if clnvc.strip().lower() != "single_nucleotide_variant":
                continue
            stage["snv"] += 1

            if "breast" not in clndn.lower():
                continue
            stage["breast_relevant_snv"] += 1
            gene_counts_after_basic[gene] += 1
            clnsig_by_gene[gene][clnsig] += 1

            if clnsig in BINARY_LABELS:
                stage["binary_clnsig"] += 1
                gene_counts_binary[gene] += 1
                if is_missense_proxy:
                    stage["binary_clnsig_missense_proxy"] += 1
                    gene_counts_binary_missense_proxy[gene] += 1
            if class_bucket(clnsig) in {"vus", "conflicting"}:
                stage["vus_or_conflicting"] += 1
                gene_counts_vus_conflicting[gene] += 1
                if is_missense_proxy:
                    stage["vus_or_conflicting_missense_proxy"] += 1
                    gene_counts_vus_conflicting_missense_proxy[gene] += 1

    counts["stage"] = dict(stage)
    counts["gene_counts_after_basic"] = dict(gene_counts_after_basic)
    counts["gene_counts_binary"] = dict(gene_counts_binary)
    counts["gene_counts_binary_missense_proxy"] = dict(gene_counts_binary_missense_proxy)
    counts["gene_counts_vus_conflicting"] = dict(gene_counts_vus_conflicting)
    counts["gene_counts_vus_conflicting_missense_proxy"] = dict(gene_counts_vus_conflicting_missense_proxy)
    counts["clnsig_by_gene"] = {g: dict(c) for g, c in clnsig_by_gene.items()}
    return counts


def read_csv_summary(path: Path) -> dict[str, object]:
    sep = "\t" if path.suffix.lower() == ".txt" else ","
    summary: dict[str, object] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "last_write_time": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
    }
    try:
        df = pd.read_csv(path, sep=sep, nrows=0)
        summary["columns"] = list(df.columns)
        with path.open("rb") as handle:
            summary["rows_estimate"] = max(sum(1 for _ in handle) - 1, 0)
    except Exception as exc:
        summary["read_error"] = str(exc)
        return summary

    if path.stat().st_size <= 5_000_000:
        try:
            df_full = pd.read_csv(path, sep=sep)
            summary["shape"] = df_full.shape
            for col in ["Gene", "ClinicalSignificance", "label", "target"]:
                if col in df_full.columns:
                    summary[f"value_counts_{col}"] = df_full[col].value_counts(dropna=False).to_dict()
            if {"Gene", "label"}.issubset(df_full.columns):
                summary["gene_by_label"] = (
                    df_full.groupby(["Gene", "label"], dropna=False)
                    .size()
                    .rename("n")
                    .reset_index()
                    .to_dict(orient="records")
                )
        except Exception as exc:
            summary["full_read_error"] = str(exc)
    return summary


def dataset_role(name: str) -> str:
    roles = {
        "clinvar_20250323.vcf.gz": "ClinVar GRCh38 source VCF named in scripts/README.",
        "clinvar.vcf.gz": "Duplicate/current ClinVar source VCF copy.",
        "clinvar_selected_variants.vcf": "Intermediate VCF after broad target filtering.",
        "clinvar_bc_variants_for_vep.vcf": "VEP input VCF after SNV/gene/breast/binary CLNSIG filtering.",
        "clinvar_bc_annotated.txt": "Older VEP/dbNSFP annotation output used by EDA 556 branch.",
        "clinvar_bc_annotated_canonical.txt": "Canonical VEP/dbNSFP annotation output used by Processing 519 branch.",
        "benchmark_dataset_raw.csv": "Older non-canonical raw benchmark, 556 branch.",
        "benchmark_dataset_cleaned.csv": "Older non-canonical cleaned benchmark, 556 branch.",
        "benchmark_dataset_final.csv": "Older non-canonical final benchmark, 556 branch.",
        "benchmark_dataset_canonical_raw.csv": "Canonical raw benchmark after VEP/canonical filtering, 519 branch.",
        "benchmark_dataset_canonical_cleaned.csv": "Canonical cleaned benchmark before feature pruning/imputation, 519 branch.",
        "benchmark_dataset.csv": "Canonical final benchmark used by manuscript, 519 branch.",
        "training_set.csv": "Canonical 80% training split from benchmark_dataset.csv, 415 variants.",
        "test_set.csv": "Canonical 20% test split from benchmark_dataset.csv, 104 variants.",
        "merged_predictions.csv": "Merged hold-out predictions from VARITY, MVP, and MutScore.",
        "mvp_test_predictions.csv": "MVP hold-out test predictions.",
        "test_predictions.txt": "MutScore hold-out test predictions.",
        "benchmark_varity_test_test_set_prep_MYVARITY_R.csv": "VARITY hold-out test predictions.",
        "appendix_top5_shap_all_models.csv": "Top-5 SHAP summary appendix table.",
        "canonical_transcripts.csv": "Canonical transcript lookup.",
        "mart_export.txt": "BioMart transcript export source.",
    }
    return roles.get(name, "Unclassified auxiliary or intermediate data file.")


def write_markdown(csv_summaries: list[dict[str, object]], vcf_counts: dict[str, object]) -> Path:
    out = OUT / "20260702_data_provenance.md"
    lines: list[str] = []
    lines.append("# Data Provenance Audit")
    lines.append("")
    lines.append("Date: 2026-07-02")
    lines.append("")
    lines.append("Scope: read-only audit of `D:\\thesis\\data`, using canonical 519 variants as the manuscript data line.")
    lines.append("")

    lines.append("## Canonical Decision")
    lines.append("")
    lines.append("- Canonical manuscript line: `benchmark_dataset.csv` with 519 missense SNVs.")
    lines.append("- Historical line: 556-variant outputs from older `EDA.ipynb`; retain only as provenance, not as revision evidence.")
    lines.append("- Current split: `training_set.csv` = 415, `test_set.csv` = 104.")
    lines.append("")

    lines.append("## File Roles")
    lines.append("")
    lines.append("| File | Role | Rows | Notes |")
    lines.append("| --- | --- | ---: | --- |")
    for s in sorted(csv_summaries, key=lambda x: Path(str(x["path"])).name):
        name = Path(str(s["path"])).name
        rows = s.get("rows_estimate", "")
        notes = []
        if "shape" in s:
            notes.append(f"shape={s['shape']}")
        if "read_error" in s:
            notes.append(f"read_error={s['read_error']}")
        lines.append(f"| `{name}` | {dataset_role(name)} | {rows} | {'; '.join(notes)} |")
    lines.append("")

    lines.append("## Core Dataset Counts")
    lines.append("")
    for s in csv_summaries:
        name = Path(str(s["path"])).name
        if name not in {"benchmark_dataset.csv", "training_set.csv", "test_set.csv", "benchmark_dataset_raw.csv", "benchmark_dataset_final.csv"}:
            continue
        lines.append(f"### {name}")
        lines.append("")
        if "shape" in s:
            lines.append(f"- Shape: `{s['shape']}`")
        for key in ["value_counts_Gene", "value_counts_ClinicalSignificance", "value_counts_label", "value_counts_target"]:
            if key in s:
                lines.append(f"- {key.replace('value_counts_', '')}: `{s[key]}`")
        if "gene_by_label" in s:
            lines.append("- Gene by label:")
            for row in s["gene_by_label"]:  # type: ignore[index]
                lines.append(f"  - {row}")
        lines.append("")

    lines.append("## VCF Audit: Target Genes Plus PALB2")
    lines.append("")
    lines.append("Source: `clinvar_20250323.vcf.gz`.")
    lines.append("")
    lines.append("Stage counts among BRCA1, BRCA2, TP53, ATM, CHEK2, and PALB2:")
    lines.append("")
    for k, v in vcf_counts.get("stage", {}).items():  # type: ignore[union-attr]
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Breast-relevant SNVs by gene before binary CLNSIG restriction:")
    lines.append("")
    for gene, n in sorted(vcf_counts.get("gene_counts_after_basic", {}).items()):  # type: ignore[union-attr]
        lines.append(f"- {gene}: {n}")
    lines.append("")
    lines.append("Binary CLNSIG breast-relevant SNVs by gene before VEP/missense/canonical/MAF filtering:")
    lines.append("")
    for gene, n in sorted(vcf_counts.get("gene_counts_binary", {}).items()):  # type: ignore[union-attr]
        lines.append(f"- {gene}: {n}")
    lines.append("")
    lines.append("Binary CLNSIG breast-relevant SNVs with ClinVar MC missense proxy:")
    lines.append("")
    for gene, n in sorted(vcf_counts.get("gene_counts_binary_missense_proxy", {}).items()):  # type: ignore[union-attr]
        lines.append(f"- {gene}: {n}")
    lines.append("")
    lines.append("VUS or conflicting breast-relevant SNVs by gene:")
    lines.append("")
    for gene, n in sorted(vcf_counts.get("gene_counts_vus_conflicting", {}).items()):  # type: ignore[union-attr]
        lines.append(f"- {gene}: {n}")
    lines.append("")
    lines.append("VUS or conflicting breast-relevant SNVs with ClinVar MC missense proxy:")
    lines.append("")
    for gene, n in sorted(vcf_counts.get("gene_counts_vus_conflicting_missense_proxy", {}).items()):  # type: ignore[union-attr]
        lines.append(f"- {gene}: {n}")
    lines.append("")

    lines.append("## CLNSIG Detail by Gene")
    lines.append("")
    for gene, counter in sorted(vcf_counts.get("clnsig_by_gene", {}).items()):  # type: ignore[union-attr]
        lines.append(f"### {gene}")
        for label, n in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{label}`: {n}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_vcf_csv(vcf_counts: dict[str, object]) -> Path:
    out = OUT / "20260702_palb2_vus_clnsig_audit.csv"
    rows: list[dict[str, object]] = []
    for gene, counter in vcf_counts.get("clnsig_by_gene", {}).items():  # type: ignore[union-attr]
        for clnsig, n in counter.items():
            rows.append(
                {
                    "gene": gene,
                    "clnsig": clnsig,
                    "bucket": class_bucket(clnsig),
                    "breast_relevant_snv_count": n,
                }
            )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gene", "clnsig", "bucket", "breast_relevant_snv_count"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (str(r["gene"]), str(r["bucket"]), str(r["clnsig"]))))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    data_files = [
        p
        for p in sorted(DATA.iterdir())
        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".vcf", ".gz", ".tbi", ".html"}
    ]
    csv_like = [p for p in data_files if p.suffix.lower() in {".csv", ".txt"} and p.stat().st_size < 120_000_000]
    csv_summaries = [read_csv_summary(p) for p in csv_like]

    vcf_counts = count_vcf(DATA / "clinvar_20250323.vcf.gz")
    md = write_markdown(csv_summaries, vcf_counts)
    vcf_csv = write_vcf_csv(vcf_counts)
    print(md)
    print(vcf_csv)


if __name__ == "__main__":
    main()
