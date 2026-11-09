from pathlib import Path
import subprocess,json
r=Path('/root/experiments/lingbot-regional-20261109');a=r/'attempts/v2-swiglu-exp';a.mkdir()
for rel in ['runs/smoke_D','smoke_D.launcher.log','smoke_D.receipt.json','status.json','smoke_queue.log']:
 p=r/rel
 if p.exists():p.rename(a/p.name)
p=r/'src/lingbotvla/ops/swiglu.py';s=p.read_text();assert s.count('_EXP(-g)')==2
p.write_text(s.replace('_EXP(-g)','_libdevice.exp(-g)'))
(r/'status.json').write_text(json.dumps(dict(state='starting')))
with (r/'smoke_queue.log').open('w') as f:p=subprocess.Popen(['python3','resume_smoke.py'],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'smoke_queue.pid').write_text(str(p.pid));print(p.pid)
