"""Sequential 2x2 benchmark with guarded smoke stage and mirrored repeats."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/experiments/lingbot-regional-20261109')
PY='/root/qihank/envs/lingbotvla/bin/python'
CASES={'A':(False,False),'B':(True,False),'C':(False,True),'D':(True,True)}
COMMON=['--train.teacher_cache_dir','/raid/teacher_cache/robotwin_clean50_png',
        '--train.fsdp_no_reshard_after_backward','true','--train.fsdp_forward_prefetch_depth','4',
        '--train.fsdp_prologue_under','vit','--train.fsdp_rs_inflight_depth','4',
        '--train.micro_batch_size','32','--train.gradient_accumulation_steps','1',
        '--train.global_batch_size','256','--train.enable_gradient_checkpointing','true']

def state(**kw):
    kw['time']=datetime.datetime.now().isoformat();(ROOT/'status.json').write_text(json.dumps(kw,indent=2));print(json.dumps(kw),flush=True)

def run(name,case,steps,numeric=False):
    region,manual=CASES[case];out=ROOT/'runs'/name
    if out.exists():raise RuntimeError(f'Existing run {name}')
    env=os.environ.copy()
    env.update(SRC=str(ROOT/'src'),RUN=str(out),PY=PY,MODEL_ROOT='/root/models',
        DATA_LIST='/root/keddyj/LingBotVLA-Test/data-preparation/robotwin-clean50-image-training.txt',
        BENCH_STEPS=str(steps),PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',
        LINGBOT_COMPILE_RMSNORM='1' if manual else '0',LINGBOT_REGIONAL_COMPILE='1' if region else '0',
        REGIONAL_CASE=case,REGIONAL_NUMERIC='1' if numeric else '0',
        NUMERIC_REFERENCE=str(ROOT/'numeric_reference'),BPEX_NO_WARN_ON_UNTUNED_CASE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TRITON_CACHE_DIR='/root/qihank/.cache/sync_bench/triton',
        TORCHINDUCTOR_CACHE_DIR='/root/qihank/.cache/sync_bench/inductor',CUDA_CACHE_PATH='/root/qihank/.cache/sync_bench/nv')
    env.pop('TORCH_LOGS',None);env.pop('TORCHDYNAMO_VERBOSE',None)
    command=['bash',str(ROOT/'harness/bench.sh'),*COMMON,
             '--train.fused_joint_qkv_prep',str(manual).lower(),'--train.fused_swiglu',str(manual).lower()]
    state(state='running',run=name,case=case,steps=steps,command=command)
    started=time.time()
    with (ROOT/f'{name}.launcher.log').open('x') as handle:
        p=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
    counts=[len((out/f'rank{i}.jsonl').read_text().splitlines()) if (out/f'rank{i}.jsonl').exists() else 0 for i in range(8)]
    numeric_ok=not numeric or all((out/f'numeric_rank{i}.json').exists() for i in range(8))
    regions=[len((out/f'region_compile_rank{i}.jsonl').read_text().splitlines()) if (out/f'region_compile_rank{i}.jsonl').exists() else 0 for i in range(8)]
    region_ok=not region or all(n>0 for n in regions)
    passed=p.returncode==0 and counts==[steps]*8 and numeric_ok and region_ok
    receipt=dict(run=name,case=case,passed=passed,returncode=p.returncode,records=counts,region_graphs=regions,
                 regional=region,manual=manual,numeric=numeric,seconds=time.time()-started)
    (ROOT/f'{name}.receipt.json').write_text(json.dumps(receipt,indent=2))
    with (ROOT/'summary.log').open('a') as summary_log:
        subprocess.run(['python3',str(ROOT/'summarize.py')],stdout=summary_log,stderr=subprocess.STDOUT)
    state(state='finished' if passed else 'failed',**receipt)
    if not passed:raise RuntimeError(f'Run failed: {name}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','bench','all']);args=parser.parse_args()
    if args.phase in ['smoke','all']:
        for case in ['A','C','B','D']:run('smoke_'+case,case,3,True)
        (ROOT/'smoke_gate.json').write_text(json.dumps(dict(passed=True,cases=list(CASES)),indent=2))
    if args.phase in ['bench','all']:
        assert json.loads((ROOT/'smoke_gate.json').read_text())['passed']
        for case,repeat in [('A',1),('B',1),('D',1),('C',1),('C',2),('D',2),('B',2),('A',2)]:
            run(f'{case}_r{repeat}',case,200)
    state(state='complete',phase=args.phase)

if __name__=='__main__':
    try:main()
    except Exception as error:state(state='error',error=repr(error));raise
