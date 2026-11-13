from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import csv
import hashlib
from pathlib import Path


OUTDIR = Path(workspace_path('revision/04_analysis/loop06_vus'))
SOURCE = Path(workspace_path('thesis/data/clinvar_20250323.vcf.gz'))
DBNSFP = Path(workspace_path('vep_data/dbnsfp/dbNSFP5.1a_grch38.gz'))
DBNSFP_INDEX = Path(workspace_path('vep_data/dbnsfp/dbNSFP5.1a_grch38.gz.tbi'))
MANIFEST = OUTDIR / "manifests" / "sha256_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role(path: Path) -> str:
    if path == SOURCE:
        return "source_clinvar"
    if path == DBNSFP:
        return "source_dbnsfp"
    if path == DBNSFP_INDEX:
        return "source_dbnsfp_index"
    if path.suffix in {".py", ".ps1"}:
        return "script"
    if path.suffix == ".vcf":
        return "candidate_vcf"
    if path.name.startswith("clinvar_5gene_vus_annotated"):
        return "vep_output"
    if path.suffix == ".log":
        return "log"
    if path.suffix == ".csv":
        return "table"
    return "audit_metadata"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(OUTDIR))
    except ValueError:
        return str(path)


def main() -> None:
    candidates = [SOURCE, DBNSFP, DBNSFP_INDEX]
    candidates.extend(
        path
        for path in sorted(OUTDIR.rglob("*"))
        if path.is_file()
        and path != MANIFEST
    )

    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        print(f"hashing {path}", flush=True)
        rows.append(
            {
                "role": role(path),
                "path": display_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest_rows={len(rows)}")
    print(MANIFEST)


if __name__ == "__main__":
    main()
