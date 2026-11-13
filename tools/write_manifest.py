"""Refresh the repository manifest from the Git index after staging intended files."""
import csv,hashlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
with (root/'SHA256SUMS.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['path','size_bytes','sha256'])
    for name in sorted(filter(None,names)):
        if name=='SHA256SUMS.csv':continue
        p=root/name
        if not p.is_file():raise SystemExit('Tracked file missing: '+name)
        w.writerow([name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
