"""After profiling finishes, validate AC16 controls with the unmodified timing harness."""
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT=Path('/root/experiments/lingbot-profile-20261108')
SOURCE=Path('/root/experiments/lingbot-ac-mbs-20261108')

while True:
    state=json.loads((ROOT/'status.json').read_text())
    if state['state']=='error':raise RuntimeError(state)
    if state['state']=='complete':break
    time.sleep(10)

spec=importlib.util.spec_from_file_location('original_driver',SOURCE/'driver.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
driver.ROOT=ROOT
def status(**kw):
    (ROOT/'validation_status.json').write_text(json.dumps(kw,indent=2))
    print(json.dumps(kw),flush=True)
driver.status=status
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
extra=['--train.micro_batch_size','16','--train.gradient_accumulation_steps','2',
       '--train.global_batch_size','256','--train.enable_gradient_checkpointing','true']
try:
    for code in ['opt','main']:
        driver.run(f'validate_{code}_ac16',code,80,extra,'validation_harness')
    status(state='complete')
except Exception as e:
    status(state='error',error=repr(e));raise
