from pathlib import Path
import sys,os,json,torch,inspect,py_compile
R=Path('/root/experiments/lingbot-compile-accuracy-20261110');sys.path.insert(0,str(R/'harness'))
from regional_numeric import flow_fixture
print('Inductor',inspect.signature(torch._inductor.compile),'CPU generator',hasattr(torch.random,'default_generator'))
os.environ['STUDY_ALIGNED_FLOW']='1'
x={'actions':torch.zeros(32,32,32),'state':torch.zeros(32,32)};y={k:v.clone() for k,v in x.items()}
before=torch.get_rng_state().clone();flow_fixture([x],1,0);assert torch.equal(before,torch.get_rng_state());flow_fixture([y],1,0)
assert torch.equal(x['noise'],y['noise']) and torch.equal(x['time'],y['time'])
assert 0.001<=x['time'].min()<=x['time'].max()<=1
print('Fixture replay and RNG isolation passed')
for p in [R/'study_runner.py',R/'summarize.py',R/'harness/step_timer.py',R/'src/regional_runtime.py']:py_compile.compile(str(p),doraise=True)
print('Syntax preflight passed')
