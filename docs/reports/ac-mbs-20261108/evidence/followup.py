"""Gate the four-cell study on the requested historical reproduction and GA smoke checks."""
import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/experiments/lingbot-ac-mbs-20261108')
sys.path.insert(0, str(ROOT))
import driver

def reproduction_check():
    results = {}
    for name, target in [('repro_main_ac32', 4.527), ('repro_opt_ac32', 2.886)]:
        receipt = json.loads((ROOT/f'{name}.receipt.json').read_text())
        assert receipt['passed'], receipt
        ranks = [{r['step']:r for r in map(json.loads, (ROOT/'runs'/name/f'rank{i}.jsonl').read_text().splitlines())} for i in range(8)]
        times = [max(r[s]['elapsed_s'] for r in ranks) for s in range(31,81)]
        median = statistics.median(times)
        deviation = (median/target-1)*100
        results[name] = dict(target_median_s=target, median_s=median, mean_s=statistics.mean(times), deviation_pct=deviation)
        assert abs(deviation) < 3, results[name]
    (ROOT/'reproduction_gate.json').write_text(json.dumps(results,indent=2))
    print('REPRODUCTION_PASS',json.dumps(results),flush=True)

def smoke_check():
    for name in ['smoke_main_noac16','smoke_opt_noac16']:
        assert json.loads((ROOT/f'{name}.receipt.json').read_text())['passed']
        cfg=json.loads((ROOT/'runs'/name/'effective_args.json').read_text())['train']
        assert cfg['micro_batch_size']==16 and cfg['gradient_accumulation_steps']==2
        assert cfg['global_batch_size']==256 and not cfg['enable_gradient_checkpointing']
        for rank in range(8):
            inputs=[json.loads(s) for s in (ROOT/'runs'/name/f'inputs_rank{rank}.jsonl').read_text().splitlines()]
            assert all(x['micro_batches']==2 for x in inputs)
            rows=[json.loads(s) for s in (ROOT/'runs'/name/f'rank{rank}.jsonl').read_text().splitlines()]
            assert all(math.isfinite(x[k]) for x in rows for k in ['loss','grad_norm'])
    (ROOT/'smoke_gate.json').write_text(json.dumps({'passed':True,'timestamp':datetime.datetime.now().isoformat()},indent=2))
    print('SMOKE_PASS',flush=True)

try:
    while True:
        s=json.loads((ROOT/'status.json').read_text())
        if s.get('state')=='phase_complete' and s.get('phase')=='reproduce':
            break
        if s.get('state') in ['error','failed']:
            raise RuntimeError(s)
        time.sleep(5)
    reproduction_check()
    subprocess.run(['python3',str(ROOT/'driver.py'),'smoke'],check=True)
    smoke_check()
    subprocess.run(['python3',str(ROOT/'driver.py'),'study'],check=True)
except Exception as exc:
    (ROOT/'followup_error.json').write_text(json.dumps({'error':repr(exc)},indent=2))
    print('FOLLOWUP_FAILED',repr(exc),flush=True)
    raise
