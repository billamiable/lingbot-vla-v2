"""Durable overnight continuation: smoke gate -> baseline repeat -> 8 benchmark runs -> final report."""
import importlib.util
import json
import hashlib
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/experiments/lingbot-regional-20261109')
def record(**values):
    (ROOT/'followup_status.json').write_text(json.dumps(values,indent=2));print(json.dumps(values),flush=True)
try:
    record(state='waiting_for_smoke')
    while True:
        state=json.loads((ROOT/'status.json').read_text())
        if state['state'] in ['error','failed']:raise RuntimeError(state)
        if state['state']=='complete' and (ROOT/'smoke_gate.json').exists():break
        time.sleep(10)
    spec=importlib.util.spec_from_file_location('driver',ROOT/'driver.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    record(state='baseline_repeat')
    driver.run('smoke_A_repeat','A',3,True)
    comparisons=[]
    for other in ['B','C','D','A_repeat']:
        matched=0
        for rank in range(8):
            a=list(map(json.loads,(ROOT/'runs/smoke_A'/f'inputs_rank{rank}.jsonl').read_text().splitlines()))
            b=list(map(json.loads,(ROOT/'runs'/('smoke_'+other)/f'inputs_rank{rank}.jsonl').read_text().splitlines()))
            assert len(a)==len(b)==3
            for x,y in zip(a,b):
                assert x['step']==y['step'] and x['tensors']==y['tensors'],(other,rank,x['step'])
                matched+=len(x['tensors'])
        comparisons.append(dict(case=other,matched_tensor_fields=matched,updates_times_ranks=24))
    (ROOT/'input_audit.json').write_text(json.dumps(dict(passed=True,comparisons=comparisons),indent=2))
    for name,digest in json.loads((ROOT/'source_freeze.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('Source changed after freeze',name)
    record(state='benchmarking')
    subprocess.run(['python3',str(ROOT/'driver.py'),'bench'],check=True)
    subprocess.run(['python3',str(ROOT/'summarize.py')],check=True)
    assert json.loads((ROOT/'results.json').read_text())['ready']
    record(state='complete')
except Exception as error:
    record(state='error',error=repr(error))
    subprocess.run(['python3',str(ROOT/'summarize.py')])
    raise
