"""Build the historical Cooperative host bridge; compilation only, no GPU execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nvcc', default=os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc'))
    p.add_argument('--cxx', default=os.environ.get('CXX', 'g++-10'))
    p.add_argument('--curand-include', type=Path, help='Optional pip CUDA curand include directory')
    p.add_argument('--output', type=Path, default=ROOT/'lcma_review/build/cooperative.so')
    p.add_argument('--source-root', type=Path, default=ROOT)
    p.add_argument('--version', choices=['fixed', 'original'], default='fixed')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    source = a.source_root.resolve()
    subprocess.run([sys.executable, str(ROOT/'lcma_review/prepare.py'), '--root', str(source), '--check-only'], check=True)
    subprocess.run([sys.executable, str(ROOT/'lcma_review/verify_source.py'), '--root', str(source), '--version', a.version], check=True)
    includes = [source/'include', source/'cutlass/include', source/'src', source,
                ROOT/'lcma_review/bridge', source/'cutlass/tools/util/include',
                source/'cutlass/examples/common', source/'tests/googletest/googletest/include']
    if a.curand_include:
        includes.append(a.curand_include.resolve())
    command = [a.nvcc, '-std=c++20', '-ccbin='+a.cxx, '-O3', '--expt-relaxed-constexpr',
               '-arch=sm_90a', '--shared', '-Xcompiler=-fPIC', '-DCUTLASS_API_v3',
               '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED',
               '-DORACLE_LCMA_COOPERATIVE']
    command += ['-I'+str(d) for d in includes]
    command += [str(ROOT/'lcma_review/bridge/timing_bridge.cu'), '-o', str(a.output.resolve())]
    print(json.dumps({'command': command, 'gpu_execution': False}, indent=2))
    if a.dry_run:
        return
    # Prevent accidental overwrite of a prior binary or build evidence.
    if a.output.exists() or a.output.with_suffix('.build.json').exists():
        raise RuntimeError('Output already exists; choose a fresh --output path')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    versions = {label: subprocess.check_output(cmd, text=True) for label, cmd in
                [('nvcc', [a.nvcc, '--version']), ('host_compiler', [a.cxx, '--version'])]}
    log = a.output.with_suffix('.log')
    with log.open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    manifest_name = 'source_sha256.json' if a.version == 'fixed' else 'baseline_source_sha256.json'
    data = {'command': command, 'versions': versions, 'source_version': a.version,
            'sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
            'source_manifest_sha256': hashlib.sha256((ROOT/'lcma_review'/manifest_name).read_bytes()).hexdigest(),
            'note': 'Fresh build; not assumed byte-identical to historical release binaries'}
    with a.output.with_suffix('.build.json').open('x') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
