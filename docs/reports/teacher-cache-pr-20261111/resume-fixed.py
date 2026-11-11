from pathlib import Path
import hashlib,json,subprocess,traceback
R=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
(R/'source_freeze.before_layout.json').write_bytes((R/'source_freeze.json').read_bytes())
(R/'results.before_layout.json').write_bytes((R/'results.json').read_bytes())
p=R/'source_freeze.json';j=json.loads(p.read_text());p.write_text(json.dumps({name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in j},indent=2))
p=R/'summarize.py';s=p.read_text().replace("['num_base_1','num_off','num_verify','num_read','num_base_2']","['num_base_1','num_off_fixed','num_verify_fixed','num_read_fixed','num_base_2']")
s=s.replace("runs.get(arm+'_'+str(i),{})","runs.get('final_'+arm+'_'+str(i),{})")
s=s.replace("verify_full.receipt.json","verify_full_fixed.receipt.json");p.write_text(s)
import driver
try:
 for name,kind in [('num_verify_fixed','verify'),('num_read_fixed','read'),('num_off_fixed','off')]:
  driver.run(name,kind,3,True)
  data=json.loads((R/'results.json').read_text())['numerical'][name]
  assert data['grad']['relative_l2']<1e-6,('Layout restoration did not eliminate gradient discrepancy',name,data)
 driver.run('verify_small_fixed','verify',3,small=True)
 for name,kind in [('final_base_1','base'),('final_cache_1','read'),('final_cache_2','read'),('final_base_2','base')]:driver.run(name,kind)
 driver.run('verify_full_fixed','verify')
 driver.state(state='complete');subprocess.run(['python3',str(R/'summarize.py')],check=True)
except Exception as e:driver.state(state='error',error=repr(e));raise
