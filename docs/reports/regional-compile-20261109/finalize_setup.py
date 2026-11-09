"""Gate integration, then rerun every smoke cell on the same frozen final source."""
from pathlib import Path
import json,subprocess,time,hashlib
r=Path('/root/experiments/lingbot-regional-20261109')
def follow(state,**kw):(r/'followup_status.json').write_text(json.dumps(dict(state=state,**kw),indent=2))
try:
 follow('waiting_for_final_integration')
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU must be free for restart'
 a=r/'attempts/v2-integration-validation';a.mkdir()
 for p in list((r/'runs').glob('smoke_*'))+list(r.glob('smoke_*.receipt.json'))+list(r.glob('smoke_*.launcher.log'))+[r/'numeric_reference',r/'smoke_gate.json',r/'status.json']:
  if p.exists():p.rename(a/p.name)
 (r/'study.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=r/'src'))
 (r/'regional_runtime.py').write_bytes((r/'src/regional_runtime.py').read_bytes())
 files=list((r/'src').rglob('*.py'))+list((r/'harness').glob('*'))
 (r/'source_freeze.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))
 follow('final_common_source_smoke')
 with (r/'final_smoke.log').open('w') as f:subprocess.run(['python3','driver.py','smoke'],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (r/'followup.log').open('w') as f:subprocess.run(['python3','followup.py'],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
except Exception as e:
 follow('error',error=repr(e));subprocess.run(['python3',str(r/'summarize.py')]);raise
