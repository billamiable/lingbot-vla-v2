from pathlib import Path
import datetime,json,os,subprocess,time,hashlib,re
R=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
PY='/root/qihank/envs/lingbotvla/bin/python'
CACHE='/raid/teacher_cache/robotwin_clean50_png'
MINI='/raid/experiments/teacher-cache-pr-20261111/first-episodes'
COMMON=['--train.micro_batch_size','32','--train.global_batch_size','256','--train.gradient_accumulation_steps','1','--train.enable_gradient_checkpointing','true']

def state(**kw):
 kw['time']=datetime.datetime.now().isoformat();p=R/'status.next';p.write_text(json.dumps(kw,indent=2));p.replace(R/'status.json');print(json.dumps(kw),flush=True)

def run(name,kind,steps=31,numeric=False,reference=False,small=False):
 src=R/('baseline' if kind=='base' else 'candidate');out=R/'runs'/name;assert not out.exists(),name
 for p,d in json.loads((R/'source_freeze.json').read_text()).items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==d,p
 flags=[]
 if kind in ['read','verify','build']:flags=['--train.teacher_cache_dir',MINI if small else CACHE,'--train.teacher_cache_mode',kind]
 env=os.environ.copy();env.update(SRC=str(src),RUN=str(out),PY=PY,MODEL_ROOT='/root/models',DATA_LIST='/root/keddyj/LingBotVLA-Test/data-preparation/robotwin-clean50-image-training.txt',BENCH_STEPS=str(steps),
 PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
 STUDY_ALIGNED_FLOW=str(int(numeric)),STUDY_NUMERIC=str(int(numeric)),STUDY_REFERENCE=str(int(reference)),
 NUMERIC_REFERENCE='/raid/experiments/teacher-cache-pr-20261111/reference',CACHE_BUILD_SUBSET=str(int(small)),CACHE_BUILD_ONLY=str(int(kind=='build')),
 TRITON_CACHE_DIR='/root/qihank/.cache/sync_bench/triton',TORCHINDUCTOR_CACHE_DIR='/root/qihank/.cache/sync_bench/inductor',CUDA_CACHE_PATH='/root/qihank/.cache/sync_bench/nv')
 for k in ['LINGBOT_COMPILE_RMSNORM','LINGBOT_REGIONAL_COMPILE','TORCH_LOGS','TORCHDYNAMO_VERBOSE']:env.pop(k,None)
 cmd=['bash',str(R/'harness/bench.sh'),*COMMON,*flags];state(state='running',run=name,kind=kind)
 start=time.time()
 with (R/(name+'.launcher.log')).open('x') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 counts=[len((out/f'rank{i}.jsonl').read_text().splitlines()) if (out/f'rank{i}.jsonl').exists() else 0 for i in range(8)]
 passed=p.returncode==0 and (kind=='build' or counts==[steps]*8)
 if numeric:passed=passed and all((out/f'{k}_rank{i}.json').exists() for k in ['grad','parameter'] for i in range(8))
 verify=[]
 if kind=='verify':
  for line in (out/'train.log').read_text().splitlines():
   if 'TeacherCacheVerify step ' in line:
    values=re.findall(r'max\|diff\| ([\d.e+\-]+) exact ([\d.]+)',line);verify.append(values)
  passed=passed and len(verify)==steps and all(len(x)==4 and all(float(d)==0 and float(e)==1 for d,e in x) for x in verify)
 receipt=dict(run=name,kind=kind,passed=passed,returncode=p.returncode,records=counts,seconds=time.time()-start,verify_steps=len(verify),small=small,numeric=numeric)
 (R/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
 if not passed:raise RuntimeError(receipt)
 if numeric and not reference:
  for rank in range(8):
   a=[json.loads(x) for x in (R/'runs/num_base_1'/f'inputs_rank{rank}.jsonl').read_text().splitlines()]
   b=[json.loads(x) for x in (out/f'inputs_rank{rank}.jsonl').read_text().splitlines()]
   assert len(a)==len(b)==3
   for x,y in zip(a,b):assert x['tensors']==y['tensors'],('Model input mismatch',name,rank,x['step'])
  (R/(name+'.inputs.json')).write_text(json.dumps(dict(passed=True,ranks=8,steps=3)))
 state(state='finished',**receipt)
 subprocess.run(['python3',str(R/'summarize.py')],check=True)

def main():
 run('build_small','build',small=True)
 run('verify_small','verify',3,small=True)
 for name,kind,ref in [('num_base_1','base',True),('num_off','off',False),('num_verify','verify',False),('num_read','read',False),('num_base_2','base',False)]:run(name,kind,3,True,ref)
 for name,kind in [('base_1','base'),('cache_1','read'),('cache_2','read'),('base_2','base')]:run(name,kind)
 run('verify_full','verify')
 state(state='complete');subprocess.run(['python3',str(R/'summarize.py')],check=True)

if __name__=='__main__':
 try:main()
 except Exception as e:state(state='error',error=repr(e));raise
