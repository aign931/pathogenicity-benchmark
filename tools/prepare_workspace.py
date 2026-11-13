"""Create a new working copy from repository files and optional local evidence."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--destination', type=Path, required=True)
p.add_argument('--archive-root', type=Path)
p.add_argument('--include-outputs', action='store_true')
args = p.parse_args()
dest = args.destination.resolve()
if dest.exists():
    p.error('Destination must not exist; existing evidence is never overwritten')
with (ROOT / 'SHA256SUMS.csv').open(encoding='utf-8') as f:
    repo_rows = list(csv.DictReader(f))
with (ROOT / 'provenance/processed_inputs.csv').open(encoding='utf-8') as f:
    inputs = list(csv.DictReader(f))
if args.include_outputs and not args.archive_root:
    p.error('--include-outputs requires --archive-root')
if args.archive_root:
    archive = args.archive_root.resolve()
    for row in inputs:
        path = archive / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            p.error('Missing or changed processed input: ' + str(path))
dest.mkdir(parents=True)
if args.archive_root and args.include_outputs:
    for folder in ['revision/04_analysis', 'thesis/data']:
        shutil.copytree(archive / folder, dest / folder, dirs_exist_ok=True)
for row in repo_rows:
    rel = row['path']
    if rel.startswith('legacy/'): continue
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / rel, target)
if args.archive_root:
    for row in inputs:
        target = dest / row['path']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive / row['path'], target)
shutil.copyfile(ROOT / 'SHA256SUMS.csv', dest / 'SHA256SUMS.csv')
print('Workspace created: ' + str(dest))
print('Set PB_WORKSPACE to this directory before running the copied scripts.')
