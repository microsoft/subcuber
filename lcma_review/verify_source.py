"""Compare every saved tested source file with the historic final SHA256 manifest."""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify(source_root=ROOT, version='fixed'):
    source_root = source_root.resolve()
    root = Path('\\\\?\\' + str(source_root)) if os.name == 'nt' else source_root
    name = 'source_sha256.json' if version == 'fixed' else 'baseline_source_sha256.json'
    rows = json.loads((ROOT/'lcma_review'/name).read_text())
    mismatches = []
    for row in rows:
        file = root/row['path']
        if not file.is_file():
            mismatches.append({'path': row['path'], 'reason': 'missing'})
        elif hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            mismatches.append({'path': row['path'], 'reason': 'sha256 mismatch'})
    report = {'checked': len(rows), 'matched': len(rows)-len(mismatches), 'mismatches': mismatches}
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--version', choices=['fixed', 'original'], default='fixed')
    a = p.parse_args()
    verify(a.root, a.version)
