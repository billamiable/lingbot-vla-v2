import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/experiments/lingbot-ac-mbs-20261108')
PY = '/root/qihank/envs/lingbotvla/bin/python'
OPT = ['--train.teacher_cache_dir', '/raid/teacher_cache/robotwin_clean50_png',
       '--train.fused_joint_qkv_prep', 'true', '--train.fused_swiglu', 'true',
       '--train.fsdp_no_reshard_after_backward', 'true', '--train.fsdp_forward_prefetch_depth', '4',
       '--train.fsdp_prologue_under', 'vit', '--train.fsdp_rs_inflight_depth', '4']

def status(**kw):
    kw['timestamp'] = datetime.datetime.now().isoformat()
    (ROOT / 'status.json').write_text(json.dumps(kw, indent=2))
    print(json.dumps(kw), flush=True)

def run(name, code, steps, extra=(), harness='harness'):
    run_dir = ROOT / 'runs' / name
    if run_dir.exists():
        raise RuntimeError(f'Run already exists: {name}')
    env = os.environ.copy()
    env.update(SRC=str(ROOT / code), RUN=str(run_dir), PY=PY, MODEL_ROOT='/root/models',
               DATA_LIST='/root/keddyj/LingBotVLA-Test/data-preparation/robotwin-clean50-image-training.txt',
               PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0', BENCH_STEPS=str(steps),
               TRITON_CACHE_DIR='/root/qihank/.cache/sync_bench/triton',
               TORCHINDUCTOR_CACHE_DIR='/root/qihank/.cache/sync_bench/inductor',
               CUDA_CACHE_PATH='/root/qihank/.cache/sync_bench/nv')
    env.pop('LINGBOT_COMPILE_RMSNORM', None)
    if code == 'opt':
        env['LINGBOT_COMPILE_RMSNORM'] = '1'
    args = ['bash', str(ROOT / harness / 'bench.sh'), *(OPT if code == 'opt' else []), *extra]
    status(state='running', run=name, steps=steps, command=args)
    started = time.time()
    with (ROOT / f'{name}.launcher.log').open('x') as out:
        result = subprocess.run(args, env=env, stdout=out, stderr=subprocess.STDOUT)
    records = []
    for rank in range(8):
        p = run_dir / f'rank{rank}.jsonl'
        rows = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
        records.append(len(rows))
    passed = result.returncode == 0 and records == [steps] * 8
    receipt = dict(run=name, code=code, steps=steps, returncode=result.returncode,
                   rank_records=records, passed=passed, wall_seconds=time.time()-started,
                   allocator=env.get('PYTORCH_CUDA_ALLOC_CONF', 'default'))
    (ROOT / f'{name}.receipt.json').write_text(json.dumps(receipt, indent=2))
    status(state='finished' if passed else 'failed', **receipt)
    if not passed:
        raise RuntimeError(f'Incomplete training run: {name}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['reproduce', 'smoke', 'study', 'study_alloc'])
    args = parser.parse_args()
    if args.phase == 'reproduce':
        run('repro_main_ac32', 'main', 80)
        run('repro_opt_ac32', 'opt', 80)
    elif args.phase == 'smoke':
        extra = ['--train.micro_batch_size', '16', '--train.gradient_accumulation_steps', '2',
                 '--train.global_batch_size', '256', '--train.enable_gradient_checkpointing', 'false']
        run('smoke_main_noac16', 'main', 5, extra, 'study_harness')
        run('smoke_opt_noac16', 'opt', 5, extra, 'study_harness')
    elif args.phase in ['study', 'study_alloc']:
        # Two mirrored rounds: each code's recipe comparison is ABBA.
        order = [('main_ac32', 'main', True), ('main_noac16', 'main', False),
                 ('opt_ac32', 'opt', True), ('opt_noac16', 'opt', False),
                 ('opt_noac16', 'opt', False), ('opt_ac32', 'opt', True),
                 ('main_noac16', 'main', False), ('main_ac32', 'main', True)]
        prefix = ''
        if args.phase == 'study_alloc':
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            prefix = 'alloc_'
            # Start with the previously failing cell; all cells share the same allocator.
            order = [('main_noac16', 'main', False), ('main_ac32', 'main', True),
                     ('opt_noac16', 'opt', False), ('opt_ac32', 'opt', True),
                     ('opt_ac32', 'opt', True), ('opt_noac16', 'opt', False),
                     ('main_ac32', 'main', True), ('main_noac16', 'main', False)]
        seen = {}
        for cell, code, ac in order:
            seen[cell] = seen.get(cell, 0) + 1
            extra = ['--train.micro_batch_size', '32' if ac else '16',
                     '--train.gradient_accumulation_steps', '1' if ac else '2',
                     '--train.global_batch_size', '256',
                     '--train.enable_gradient_checkpointing', str(ac).lower()]
            run(f'{prefix}{cell}_r{seen[cell]}', code, 200, extra, 'study_harness')
    status(state='phase_complete', phase=args.phase)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status(state='error', error=repr(exc))
        raise
