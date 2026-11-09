"""Guarded same-distribution accuracy study, followed by independent operator profiles."""
import datetime,itertools,json,os,signal,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/root/experiments/lingbot-compile-accuracy-20261110')
PY='/root/qihank/envs/lingbotvla/bin/python'
CASES={'A':(False,False,False,False,False),'B':(True,False,False,False,False),'B_cast':(True,False,False,False,True),
       'PR1':(False,True,True,True,False),'RMS':(False,True,False,False,False),'SWI':(False,False,True,False,False),'QKV':(False,False,False,True,False)}
COMMON=['--train.teacher_cache_dir','/raid/teacher_cache/robotwin_clean50_png','--train.fsdp_no_reshard_after_backward','true',
 '--train.fsdp_forward_prefetch_depth','4','--train.fsdp_prologue_under','vit','--train.fsdp_rs_inflight_depth','4',
 '--train.micro_batch_size','32','--train.gradient_accumulation_steps','1','--train.global_batch_size','256',
 '--train.enable_gradient_checkpointing','true']

def state(**kw):
 kw['time']=datetime.datetime.now().isoformat();p=ROOT/'status.json';q=ROOT/'status.next';q.write_text(json.dumps(kw,indent=2));q.replace(p);print(json.dumps(kw),flush=True)

def audit_inputs(name,reference,limit=None):
 for rank in range(8):
  a=[json.loads(x) for x in (ROOT/'runs'/reference/f'inputs_rank{rank}.jsonl').read_text().splitlines()]
  b=[json.loads(x) for x in (ROOT/'runs'/name/f'inputs_rank{rank}.jsonl').read_text().splitlines()]
  if limit is not None:a=a[:limit];b=b[:limit]
  assert len(a)==len(b),(name,rank,len(a),len(b))
  for x,y in zip(a,b):assert x['step']==y['step'] and x['tensors']==y['tensors'],('Input mismatch',name,rank,x['step'])
 (ROOT/(name+'.inputs.json')).write_text(json.dumps(dict(passed=True,reference=reference,ranks=8,steps=len(a)),indent=2))

def run(name,case,steps,numeric=False,reference=False,profile=False):
 region,rms,swi,qkv,casts=CASES[case];out=ROOT/'runs'/name
 assert not out.exists(),name
 env=os.environ.copy();env.update(SRC=str(ROOT/'src'),RUN=str(out),PY=PY,MODEL_ROOT='/root/models',
 DATA_LIST='/root/keddyj/LingBotVLA-Test/data-preparation/robotwin-clean50-image-training.txt',BENCH_STEPS=str(steps),
 PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',LINGBOT_COMPILE_RMSNORM=str(int(rms)),LINGBOT_REGIONAL_COMPILE=str(int(region)),
 STUDY_ALIGNED_FLOW='1',STUDY_FULL_INPUT_HASH='1',STUDY_NUMERIC=str(int(numeric)),STUDY_REFERENCE=str(int(reference)),
 STUDY_PROFILE=str(int(profile)),STUDY_CODEGEN=str(int(profile)),STUDY_PRESERVE_CASTS=str(int(casts)),
 NUMERIC_REFERENCE='/raid/experiments/lingbot-compile-accuracy-20261110/references',BPEX_NO_WARN_ON_UNTUNED_CASE='1',
 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TRITON_CACHE_DIR='/root/qihank/.cache/sync_bench/triton',
 TORCHINDUCTOR_CACHE_DIR='/root/qihank/.cache/sync_bench/inductor',CUDA_CACHE_PATH='/root/qihank/.cache/sync_bench/nv')
 for key in ['TORCH_LOGS','TORCHDYNAMO_VERBOSE','TORCHINDUCTOR_EMULATE_PRECISION_CASTS']:env.pop(key,None)
 cmd=['bash',str(ROOT/'harness/bench.sh'),*COMMON,'--train.fused_joint_qkv_prep',str(qkv).lower(),'--train.fused_swiglu',str(swi).lower()]
 state(state='running',run=name,case=case,steps=steps,profile=profile,numeric=numeric)
 started=time.time()
 with (ROOT/(name+'.launcher.log')).open('x') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=3600)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise TimeoutError(name+' exceeded one hour')
 ranks=[]
 for rank in range(8):
  p=out/f'rank{rank}.jsonl';ranks.append([json.loads(x) for x in p.read_text().splitlines()] if p.exists() else [])
 regions=[len((out/f'region_compile_rank{i}.jsonl').read_text().splitlines()) if (out/f'region_compile_rank{i}.jsonl').exists() else 0 for i in range(8)]
 finite=all(__import__('math').isfinite(x[k]) for rows in ranks for x in rows for k in ['loss','vla_loss','grad_norm'])
 passed=code==0 and [len(x) for x in ranks]==[steps]*8 and finite and (not region or min(regions)>0)
 if numeric:passed=passed and all((out/f'{kind}_rank{i}.json').exists() for kind in ['grad','parameter'] for i in range(8))
 if profile:passed=passed and (out/'trace.json.gz').exists() and (out/'operator_profile.json').exists()
 receipt=dict(run=name,case=case,passed=passed,returncode=code,records=[len(x) for x in ranks],finite=finite,region_graphs=regions,
  seconds=time.time()-started,command=cmd,flags=dict(regional=region,rms=rms,swiglu=swi,qkv=qkv,preserve_casts=casts,numeric=numeric,profile=profile))
 (ROOT/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
 if not passed:raise RuntimeError(receipt)
 if name!='num_A':audit_inputs(name,'train_A1' if name.startswith('train_') and name!='train_A1' else 'num_A',None if name.startswith('train_') and name!='train_A1' else 3)
 subprocess.run(['python3',str(ROOT/'summarize.py')],check=True)
 state(state='finished',**receipt)

def main():
 # First update: exact comparison of gradients, updated weights and sampled layer outputs.
 for name,case,ref in [('num_A','A',True),('num_A_repeat','A',False),('num_B','B',False),('num_B_cast','B_cast',False),('num_PR1','PR1',False)]:run(name,case,3,True,ref)
 (ROOT/'numeric_gate.json').write_text(json.dumps(dict(passed=True)))
 # Capture generated code and operator traces separately from the loss trajectories.
 for case in ['A','B','RMS','SWI','QKV','PR1']:run('profile_'+case,case,18,profile=True)
 # Six independent restarts per arm, all same initial weights/data/flow draws; mirrored ABBA blocks.
 order=[('A',1),('B',1),('B',2),('A',2),('B',3),('A',3),('A',4),('B',4),('A',5),('B',5),('B',6),('A',6)]
 for case,rep in order:run(f'train_{case}{rep}',case,300)
 state(state='complete');subprocess.run(['python3',str(ROOT/'summarize.py')],check=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  state(state='error',error=repr(e));subprocess.run(['python3',str(ROOT/'summarize.py')]);raise
