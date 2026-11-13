"""Check a SHA256 CSV manifest without importing model packages."""
import argparse
import csv
import hashlib
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def verify(root, manifest):
    root = Path(root).resolve()
    errors = []
    seen = set()
    with Path(manifest).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ['Empty manifest']
    for row in rows:
        rel = row.get('path') or row.get('relative_path')
        if not rel or rel in seen:
            errors.append('Missing or duplicate path: ' + str(rel)); continue
        seen.add(rel)
        path = (root / rel).resolve()
        if path != root and root not in path.parents:
            errors.append('Unsafe path: ' + rel); continue
        if not path.is_file():
            errors.append('Missing: ' + rel); continue
        size = row.get('size_bytes') or row.get('bytes')
        if size and path.stat().st_size != int(size):
            errors.append('Size mismatch: ' + rel)
        if digest(path) != row['sha256'].lower():
            errors.append('SHA256 mismatch: ' + rel)
    print('Checked {} manifest entries; {} failures'.format(len(rows), len(errors)))
    return errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    errors = verify(args.root, args.manifest or args.root / 'SHA256SUMS.csv')
    for error in errors: print(error)
    raise SystemExit(bool(errors))
