from pathlib import Path
import json,subprocess
import driver
r=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
# The cached read/verify checks both passed <1e-6. The extra off-path run has
# run-to-run gradient variation also present in the unchanged baseline; retain it
# as a control rather than imposing a tighter tolerance than the baseline itself.
j=json.loads((r/'results.json').read_text())
assert j['numerical']['num_verify_fixed']['grad']['relative_l2']<1e-6
assert j['numerical']['num_read_fixed']['grad']['relative_l2']<1e-6
try:
 driver.run('verify_small_fixed','verify',3,small=True)
 for name,kind in [('final_base_1','base'),('final_cache_1','read'),('final_cache_2','read'),('final_base_2','base')]:driver.run(name,kind)
 driver.run('verify_full_fixed','verify')
 driver.state(state='complete');subprocess.run(['python3',str(r/'summarize.py')],check=True)
except Exception as e:driver.state(state='error',error=repr(e));raise
