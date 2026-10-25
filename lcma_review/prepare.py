"""Prepare pinned public dependencies without resetting local modifications. No GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def git(*args, cwd=None, check=True):
    return subprocess.run(['git', *args], cwd=ROOT if cwd is None else cwd, text=True, capture_output=True, check=check)

def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='No downloads or dependency changes')
    parser.add_argument('--root', type=Path, default=ROOT, help='Also supports a separate baseline worktree')
    args = parser.parse_args()
    meta = json.loads((Path(__file__).resolve().parent/'provenance.json').read_text())
    ROOT = args.root.resolve()
    if not args.check_only:
        git('-c', 'core.autocrlf=false', '-c', 'core.longpaths=true',
            'submodule', 'update', '--init', '--recursive')
    for name, expected in [('cutlass', meta['cutlass_commit']),
                           ('tests/googletest', meta['googletest_commit'])]:
        # An uninitialized submodule must not resolve the parent's HEAD by accident.
        if not (ROOT/name/'.git').exists():
            raise RuntimeError(f'{name}: initialize the pinned submodule first')
        actual = git('rev-parse', 'HEAD', cwd=ROOT/name).stdout.strip()
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}; no reset performed')
    patch = ROOT/'cutlass.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != meta['cutlass_patch_sha256']:
        raise RuntimeError('cutlass.patch fingerprint differs from the recorded dependency patch')
    forward = git('apply', '--check', str(patch), cwd=ROOT/'cutlass', check=False)
    reverse = git('apply', '--reverse', '--check', str(patch), cwd=ROOT/'cutlass', check=False)
    if reverse.returncode == 0:
        print('Original SubCuber CUTLASS patch already present')
    elif forward.returncode == 0 and not args.check_only:
        git('apply', str(patch), cwd=ROOT/'cutlass')
        print('Applied original SubCuber CUTLASS patch')
    elif forward.returncode == 0:
        raise RuntimeError('CUTLASS patch is not applied; rerun without --check-only')
    else:
        raise RuntimeError('CUTLASS differs from both expected patch states; no files changed')
    git('diff', '--check', cwd=ROOT/'cutlass')

if __name__ == '__main__':
    main()
