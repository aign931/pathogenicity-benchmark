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
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


SOURCE = Path(workspace_path('thesis/data/clinvar_20250323.vcf.gz'))
OUTDIR = Path(workspace_path('revision/04_analysis/loop06_vus'))

TARGET_GENES = {"BRCA1", "BRCA2", "TP53", "ATM", "CHEK2"}
VUS_LABEL = "Uncertain_significance"
CONFLICTING_LABEL = "Conflicting_classifications_of_pathogenicity"

VUS_VCF = OUTDIR / "clinvar_5gene_vus_breast_snv_for_vep.vcf"
CONFLICTING_VCF = OUTDIR / "clinvar_5gene_conflicting_breast_snv.vcf"
STAGE_COUNTS = OUTDIR / "candidate_selection_stage_counts.csv"
GENE_COUNTS = OUTDIR / "candidate_selection_gene_counts.csv"
SUMMARY = OUTDIR / "candidate_selection_summary.md"


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
    """Match the project's original first-gene interpretation deterministically."""
    if not geneinfo:
        return None
    return geneinfo.split("|", 1)[0].split(":", 1)[0]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    vus_by_gene = Counter()
    conflicting_by_gene = Counter()

    with gzip.open(SOURCE, "rt", encoding="utf-8", errors="replace") as source, VUS_VCF.open(
        "w", encoding="utf-8", newline=""
    ) as vus_out, CONFLICTING_VCF.open("w", encoding="utf-8", newline="") as conflicting_out:
        for line in source:
            if line.startswith("#"):
                vus_out.write(line)
                conflicting_out.write(line)
                counts["header_lines_copied"] += 1
                continue

            counts["source_records"] += 1
            fields = line.rstrip("\n").split("\t")
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

            clnsig = str(info.get("CLNSIG", "")).strip()
            if clnsig == VUS_LABEL:
                vus_out.write(line)
                counts["exact_vus_records"] += 1
                vus_by_gene[gene] += 1
                if fields[4] in {"", "."}:
                    counts["exact_vus_records_without_alt_allele"] += 1
            elif clnsig == CONFLICTING_LABEL:
                conflicting_out.write(line)
                counts["exact_conflicting_records"] += 1
                conflicting_by_gene[gene] += 1
                if fields[4] in {"", "."}:
                    counts["exact_conflicting_records_without_alt_allele"] += 1

    count_order = [
        "header_lines_copied",
        "source_records",
        "malformed_records",
        "five_gene_records",
        "five_gene_snv_records",
        "five_gene_breast_snv_records",
        "exact_vus_records",
        "exact_vus_records_without_alt_allele",
        "exact_conflicting_records",
        "exact_conflicting_records_without_alt_allele",
    ]
    write_csv(
        STAGE_COUNTS,
        [{"stage": stage, "count": counts[stage]} for stage in count_order],
        ["stage", "count"],
    )

    gene_rows: list[dict[str, object]] = []
    for gene in sorted(TARGET_GENES):
        gene_rows.append({"cohort": "exact_vus", "gene": gene, "count": vus_by_gene[gene]})
        gene_rows.append(
            {"cohort": "exact_conflicting", "gene": gene, "count": conflicting_by_gene[gene]}
        )
    write_csv(GENE_COUNTS, gene_rows, ["cohort", "gene", "count"])

    summary_lines = [
        "# Candidate VCF Selection Summary",
        "",
        f"Source: `{SOURCE}`",
        "",
        "The VUS cohort uses an exact `CLNSIG == Uncertain_significance` match. "
        "Conflicting classifications are counted and written to a separate VCF; they are never merged with VUS.",
        "",
        "Selection order: first-listed target gene in `GENEINFO` -> ClinVar SNV -> `CLNDN` contains `breast` -> exact CLNSIG.",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
    ]
    summary_lines.extend(f"| `{stage}` | {counts[stage]} |" for stage in count_order)
    summary_lines.extend(
        [
            "",
            f"VUS VCF: `{VUS_VCF}`",
            "",
            f"Separate conflicting VCF: `{CONFLICTING_VCF}`",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"vus_records={counts['exact_vus_records']}")
    print(f"conflicting_records={counts['exact_conflicting_records']}")
    print(VUS_VCF)
    print(CONFLICTING_VCF)


if __name__ == "__main__":
    main()
