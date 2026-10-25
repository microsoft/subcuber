"""Portable correctness runner for the frozen case list, in an authorized isolated container.

Does not connect to a server or start any keepalive. Run under an external timeout.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--so', required=True, type=Path)
    p.add_argument('--shape', nargs=3, type=int, required=True, metavar=('M', 'N', 'K'))
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--expected-gpu-uuid', required=True)
    p.add_argument('--input-dir', type=Path, help='Frozen files plus input_manifest.json; omit for a NEW draw')
    p.add_argument('--out', required=True, type=Path, help='Fresh directory; never reuse historical output')
    a = p.parse_args()
    catalog = json.loads((HERE/'frozen_inputs.json').read_text())
    matches = [c for c in catalog if c['shape'] == a.shape and c['seed'] == a.seed]
    if len(matches) != 1:
        p.error('Case is outside the documented frozen list; define and review a new test plan first')
    case = matches[0]
    if sys.platform != 'linux' or not Path('/.dockerenv').is_file():
        p.error('Requires the authorized Linux container; host execution is not enabled')
    import fcntl
    lock = open('/tmp/yuelang_kda_gpu_activity.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('GPU activity lock is busy; no GPU work started')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if uuids != a.expected_gpu_uuid:
        p.error('Expected exactly one visible authorized GPU with the requested UUID')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if jobs:
        p.error('Existing GPU processes found; no concurrent experiment is permitted')
    if not a.so.is_file():
        p.error('Shared library does not exist')
    a.out.mkdir(parents=True, exist_ok=False)
    # Import torch only after isolation, UUID, process and lock checks.
    import math
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Requires one visible GPU')
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError('This build targets Hopper sm_90a')
    m, n, k = a.shape
    names = ['a_fp16.npy', 'b_row_fp16.npy', 'reference_fp32.npy']
    if a.input_dir:
        manifest = json.loads((a.input_dir/'input_manifest.json').read_text())
        for name in names:
            if sha(a.input_dir/name) != manifest['files'][name]['sha256']:
                raise RuntimeError('Frozen file fingerprint mismatch: '+name)
        xx, yy, rr = [np.load(a.input_dir/name, allow_pickle=False) for name in names]
        if xx.dtype != np.float16 or yy.dtype != np.float16 or rr.dtype != np.float32:
            raise RuntimeError('Expected FP16 A/B and FP32 reference')
        if (xx.shape, yy.shape, rr.shape) != ((m,k), (k,n), (m,n)):
            raise RuntimeError('Input shapes differ from the declared test case')
        x = torch.from_numpy(xx).cuda(); y = torch.from_numpy(yy).cuda()
        ref = torch.from_numpy(rr).cuda()
        origin = 'saved files; reference provenance remains that of the input manifest'
    else:
        torch.manual_seed(a.seed)
        x = torch.randn((m,k), device='cuda', dtype=torch.float16)/math.sqrt(k)
        ycol = torch.empty_strided((k,n), (1,k), device='cuda', dtype=torch.float16)
        ycol.copy_(torch.randn((k,n), device='cuda', dtype=torch.float16))
        y = ycol.contiguous(); del ycol
        ref = x.float() @ y.float()
        xx, yy, rr = [z.cpu().numpy() for z in (x,y,ref)]
        origin = 'NEW draw; FP32 GEMM with TF32 disabled, not mathematically exact'
    pair = hashlib.sha256(xx.tobytes()+yy.tobytes()).hexdigest()
    files = {}
    for name, array in zip(names, [xx,yy,rr]):
        with (a.out/name).open('xb') as f:
            np.save(f, array, allow_pickle=False)
        files[name] = {'sha256': sha(a.out/name)}
    manifest = {'case': case, 'input_sha256': pair, 'files': files, 'origin': origin,
                'matches_historical_inputs': pair == case['input_sha256'],
                'matches_historical_all_files': files == case['files'],
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'tf32': False}
    (a.out/'input_manifest.json').write_text(json.dumps(manifest, indent=2))
    den = ref.norm().item()
    if den == 0 or not bool(torch.isfinite(ref).all()):
        raise RuntimeError('Invalid reference norm or nonfinite reference')
    c = torch.zeros((m,n), device='cuda', dtype=torch.float16); d = torch.zeros_like(c)
    lib = ctypes.CDLL(str(a.so.resolve()))
    for name, args, result in [
        ('oracle_sub_prepare', [ctypes.c_int]*4+[ctypes.c_void_p]*4, ctypes.c_void_p),
        ('oracle_sub_error', [], ctypes.c_int),
        ('oracle_sub_run', [ctypes.c_void_p]*2, ctypes.c_int),
        ('oracle_sub_destroy', [ctypes.c_void_p], None),
        ('oracle_sub_workspace', [ctypes.c_void_p], ctypes.c_size_t),
        ('oracle_original_gtest_call', [ctypes.c_int]*3+[ctypes.c_void_p]*4, ctypes.c_int)]:
        fn = getattr(lib,name); fn.argtypes = args; fn.restype = result
    ptr = lambda tensor: ctypes.c_void_p(tensor.data_ptr())
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    results = []
    common = {'case_id': case['id'], 'shape': a.shape, 'seed': a.seed, 'gpu_uuid': uuids,
              'binary_sha256': sha(a.so), 'runner_sha256': sha(Path(__file__)),
              'input_sha256': pair, 'historical_replay': manifest['matches_historical_all_files'],
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'criteria': {'atol': .01, 'rtol': .01, 'relative_L2_max': .001}}
    for entry in ['original_helper', 'rebuilt_current']:
        for repeat in range(3):
            c.zero_(); d.zero_(); torch.cuda.synchronize(); ctx = None
            workspace = None
            try:
                if entry == 'original_helper':
                    rc = lib.oracle_original_gtest_call(m,n,k,ptr(x),ptr(y),ptr(c),ptr(d))
                else:
                    ctx = lib.oracle_sub_prepare(m,n,k,props.multi_processor_count,ptr(x),ptr(y),ptr(c),ptr(d))
                    if not ctx:
                        raise RuntimeError('Interface rejected input, native status='+str(lib.oracle_sub_error()))
                    workspace = int(lib.oracle_sub_workspace(ctx))
                    rc = lib.oracle_sub_run(ctx,stream)
                if rc != 0:
                    raise RuntimeError('Native execution failed, status='+str(rc))
                torch.cuda.synchronize()
                error = d.float()-ref
                mask = error.abs() > .01+.01*ref.abs()
                count = int(mask.sum()); rel = error.norm().item()/den
                finite = bool(torch.isfinite(d).all())
                output = d.cpu().numpy(); output_sha = hashlib.sha256(output.tobytes()).hexdigest()
                target = a.out/(output_sha+'.npy')
                if not target.exists():
                    with target.open('xb') as f:
                        np.save(f,output,allow_pickle=False)
                record = dict(common, entry=entry, repeat=repeat, native_rc=rc,
                              passed=finite and count==0 and rel<=.001, finite=finite,
                              bad_count=count, relative_L2=rel, max_abs=float(error.abs().max()),
                              row_bad_counts=mask.sum(1).cpu().tolist(),
                              column_bad_counts=mask.sum(0).cpu().tolist(),
                              workspace_bytes=workspace, output_value_sha256=output_sha)
                with (a.out/'results.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
                results.append(record)
                print(json.dumps({k: record[k] for k in ['entry','repeat','passed','bad_count','relative_L2']}), flush=True)
            finally:
                if ctx:
                    lib.oracle_sub_destroy(ctx)
    after = hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
    if after != pair:
        raise RuntimeError('Input values were modified')
    summary = {'completed': len(results), 'passed': sum(r['passed'] for r in results), 'input_unchanged': True}
    (a.out/'completed.json').write_text(json.dumps(summary, indent=2))
    if summary['passed'] != 6:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
