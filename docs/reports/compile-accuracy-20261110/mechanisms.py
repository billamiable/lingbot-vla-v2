"""Extract trace/code evidence; keyword groups are observations, not a causal time decomposition."""
import ast,collections,gzip,json,re,time
from pathlib import Path
R=Path(__file__).resolve().parent
cases=['A','B','RMS','SWI','QKV','PR1']

def once():
 results={}
 for case in cases:
  p=R/'runs'/('profile_'+case);trace=p/'trace.json.gz'
  if not trace.exists():continue
  with gzip.open(trace,'rt') as f:events=json.load(f)['traceEvents']
  kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
  groups={}
  for label,tags in [('norm',['rsqrt','rmsnorm','rms_norm']),('activation',['silu','swiglu']),('qkv_or_rope',['joint_qkv','qkv_prep','rotary','rope']),('communication',['nccl']),('matmul',['gemm','matmul','cublas']),('attention',['flex_attention','flash_fwd','flash_bwd'])]:
   chosen=[e for e in kernels if any(tag in e.get('name','').lower() for tag in tags)]
   per_name=collections.defaultdict(lambda:dict(count=0,total_us=0.0))
   for e in chosen:
    row=per_name[e['name']];row['count']+=1;row['total_us']+=e.get('dur',0)
   groups[label]=dict(kernels_per_update=len(chosen)/3,gpu_ms_sum_per_update=sum(e.get('dur',0) for e in chosen)/3000,names=dict(sorted(per_name.items(),key=lambda x:-x[1]['total_us'])))
  annotations=[e for e in events if e.get('name','').startswith('decoder.') and e.get('ph')=='X']
  intervals=collections.defaultdict(list)
  for e in annotations:intervals[(e['pid'],e['tid'])].append(e)
  launch_scope={}
  for e in events:
   if e.get('cat') not in ['cuda_runtime','cuda_driver']:continue
   cid=e.get('args',{}).get('correlation');t=e.get('ts',0)
   if cid is None:continue
   matches=[a for a in intervals.get((e.get('pid'),e.get('tid')),[]) if a['ts']<=t<=a['ts']+a['dur']]
   if matches:launch_scope[cid]=min(matches,key=lambda a:a['dur'])['name']
  scopes=collections.defaultdict(lambda:dict(kernel_count=0,gpu_ms=0.0,kernel_names={}))
  for e in kernels:
   name=launch_scope.get(e.get('args',{}).get('correlation'))
   if name:
    phase='.'.join(name.split('.')[1:2]+name.split('.')[-1:]);scopes[phase]['kernel_count']+=1;scopes[phase]['gpu_ms']+=e.get('dur',0)/1000;scopes[phase]['kernel_names'][e['name']]=scopes[phase]['kernel_names'].get(e['name'],0)+1
  code=[]
  for f in p.rglob('output_code.py'):
   text=f.read_text(errors='replace')
   ops=sorted(set(re.findall(r'aten\.[A-Za-z_0-9]+',text)))
   names=sorted(set(re.findall(r'\btriton_[A-Za-z_0-9]+',text)))
   code.append(dict(path=str(f.relative_to(R)),aten_ops=ops,kernel_names=names))
  fx=[]
  for f in sorted((p/'fx').glob('*.txt')):
   text=f.read_text();flags=dict(norm=('rsqrt' in text),silu=('silu' in text),qkv_prep=('joint_qkv_prep' in text))
   fx.append(dict(path=str(f.relative_to(R)),**flags))
  results[case]=dict(profiled_updates=3,rank=0,kernel_count=len(kernels),groups=groups,forward_launch_scopes=dict(scopes),code=code,fx=fx)
 data=dict(cases=results,complete=len(results)==6,limitations=['Keyword groups may omit unnamed/fused operations and may overlap; inspect code and original kernel names.', 'GPU duration sums can overlap and cannot be subtracted from wall time.', 'Decoder scope mapping uses forward CPU launch correlation only; backward/recompute outside that scope is not assigned to a layer.'])
 tmp=R/'mechanisms.next';tmp.write_text(json.dumps(data,indent=2));tmp.replace(R/'mechanisms.json');print('Mechanism evidence extracted',list(results))

if __name__=='__main__':
 if '--wait' in __import__('sys').argv:
  while True:
   state=json.loads((R/'status.json').read_text())
   if state['state']=='error':break
   if all((R/(f'profile_{case}.receipt.json')).exists() for case in cases):once();break
   time.sleep(15)
 else:once()
