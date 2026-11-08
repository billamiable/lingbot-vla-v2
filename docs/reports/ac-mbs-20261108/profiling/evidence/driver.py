"""Six isolated runs: original endpoint pair plus AC16/GA2 control, for main and opt."""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/root/experiments/lingbot-profile-20261108')
SOURCE=Path('/root/experiments/lingbot-ac-mbs-20261108')
PY='/root/qihank/envs/lingbotvla/bin/python'
OPT=['--train.teacher_cache_dir','/raid/teacher_cache/robotwin_clean50_png',
     '--train.fused_joint_qkv_prep','true','--train.fused_swiglu','true',
     '--train.fsdp_no_reshard_after_backward','true','--train.fsdp_forward_prefetch_depth','4',
     '--train.fsdp_prologue_under','vit','--train.fsdp_rs_inflight_depth','4']


def state(**kw):
    kw['time']=datetime.datetime.now().isoformat()
    (ROOT/'status.json').write_text(json.dumps(kw,indent=2))
    print(json.dumps(kw),flush=True)


def main():
    for code in ['opt','main']:
        for recipe,mb,ga,ac in [('noac16',16,2,False),('ac16',16,2,True),('ac32',32,1,True)]:
            name=f'{code}_{recipe}'
            out=ROOT/'runs'/name
            if out.exists():raise RuntimeError(f'Already exists: {out}')
            env=os.environ.copy()
            env.update(SRC=str(SOURCE/code),RUN=str(out),PY=PY,MODEL_ROOT='/root/models',
                       DATA_LIST=str(SOURCE/'data_list.txt'),BENCH_STEPS='60',PYTHONDONTWRITEBYTECODE='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                       TRITON_CACHE_DIR='/root/qihank/.cache/sync_bench/triton',
                       TORCHINDUCTOR_CACHE_DIR='/root/qihank/.cache/sync_bench/inductor',
                       CUDA_CACHE_PATH='/root/qihank/.cache/sync_bench/nv')
            env.pop('LINGBOT_COMPILE_RMSNORM',None)
            if code=='opt':env['LINGBOT_COMPILE_RMSNORM']='1'
            args=['bash',str(ROOT/'harness/bench.sh'),*(OPT if code=='opt' else []),
                  '--train.micro_batch_size',str(mb),'--train.gradient_accumulation_steps',str(ga),
                  '--train.global_batch_size','256','--train.enable_gradient_checkpointing',str(ac).lower()]
            state(state='running',run=name,command=args)
            with (ROOT/f'{name}.launcher.log').open('x') as handle:
                result=subprocess.run(args,env=env,stdout=handle,stderr=subprocess.STDOUT)
            counts=[len((out/f'rank{r}.jsonl').read_text().splitlines()) if (out/f'rank{r}.jsonl').exists() else 0 for r in range(8)]
            traces=[(out/f'trace_rank{r}.json.gz').exists() for r in range(8)]
            passed=result.returncode==0 and counts==[60]*8 and all(traces)
            receipt=dict(run=name,passed=passed,counts=counts,traces=traces,returncode=result.returncode)
            (ROOT/f'{name}.receipt.json').write_text(json.dumps(receipt,indent=2))
            if not passed:raise RuntimeError(str(receipt))
    state(state='complete')


if __name__=='__main__':
    try:main()
    except Exception as error:
        state(state='error',error=repr(error))
        raise
