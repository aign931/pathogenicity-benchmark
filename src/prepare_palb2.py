from __future__ import annotations
from paths import DATA_DIR, SOURCE_DIR, CONFIG_DIR

import gzip
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


SOURCE = DATA_DIR / 'raw/clinvar_20250323.vcf.gz'
OUTDIR = DATA_DIR / 'raw'

ORIGINAL_GENES = {"BRCA1", "BRCA2", "TP53", "ATM", "CHEK2"}
PALB2 = "PALB2"
SIX_GENES = ORIGINAL_GENES | {PALB2}

ALLOWED_LABELS = {
    "Likely_benign",
    "Pathogenic",
    "Benign/Likely_benign",
    "Benign",
    "Pathogenic/Likely_pathogenic",
    "Likely_pathogenic",
}


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


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    palb2_vcf = OUTDIR / "clinvar_palb2_binary_breast_snv_for_vep.vcf"
    six_gene_vcf = OUTDIR / "clinvar_6gene_binary_breast_snv_for_vep.vcf"
    summary = OUTDIR / "candidate_vcf_summary.txt"

    counts = Counter()
    gene_counts = Counter()
    headers: list[str] = []

    with gzip.open(SOURCE, "rt", encoding="utf-8", errors="replace") as src, palb2_vcf.open(
        "w", encoding="utf-8", newline=""
    ) as palb2_out, six_gene_vcf.open("w", encoding="utf-8", newline="") as six_out:
        for line in src:
            if line.startswith("#"):
                headers.append(line)
                palb2_out.write(line)
                six_out.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            info = parse_info(fields[7])
            gene = first_gene(info.get("GENEINFO"))
            if gene not in SIX_GENES:
                continue
            counts["gene_scope"] += 1

            if str(info.get("CLNVC", "")).strip().lower() != "single_nucleotide_variant":
                continue
            counts["snv"] += 1

            clndn = str(info.get("CLNDN", ""))
            if "breast" not in clndn.lower():
                continue
            counts["breast_snv"] += 1

            clnsig = str(info.get("CLNSIG", "")).strip()
            if clnsig not in ALLOWED_LABELS:
                continue
            counts["binary_breast_snv"] += 1
            gene_counts[gene] += 1

            if gene == PALB2:
                palb2_out.write(line)
            six_out.write(line)

    lines = [
        "PALB2 candidate VCF generation",
        f"source={SOURCE}",
        f"palb2_vcf={palb2_vcf}",
        f"six_gene_vcf={six_gene_vcf}",
        "",
        "stage_counts",
    ]
    lines.extend([f"{k}={v}" for k, v in counts.items()])
    lines.append("")
    lines.append("binary_breast_snv_by_gene")
    lines.extend([f"{gene}={count}" for gene, count in sorted(gene_counts.items())])
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary)
    print(palb2_vcf)
    print(six_gene_vcf)


if __name__ == "__main__":
    main()
