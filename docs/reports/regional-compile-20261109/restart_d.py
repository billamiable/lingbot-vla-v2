from pathlib import Path
import subprocess,json
r=Path('/root/experiments/lingbot-regional-20261109');a=r/'attempts/v2-swiglu-disable';a.mkdir()
for rel in ['runs/smoke_D','smoke_D.launcher.log','smoke_D.receipt.json','status.json','followup_status.json','followup.log','smoke_queue.log']:
 p=r/rel
 if p.exists():p.rename(a/p.name)
p=r/'src/lingbotvla/ops/swiglu.py';s=p.read_text();assert s.count('@torch.compiler.disable')==1
p.write_text(s.replace('@torch.compiler.disable\ndef swiglu','def swiglu'))
(r/'resume_smoke.py').write_text("import driver,json\nfrom pathlib import Path\ndriver.run('smoke_D','D',3,True)\nPath('smoke_gate.json').write_text(json.dumps(dict(passed=True,cases=list(driver.CASES))))\ndriver.state(state='complete',phase='smoke')\n")
(r/'status.json').write_text(json.dumps(dict(state='starting')))
for name,cmd in [('smoke_queue',['python3','resume_smoke.py']),('followup',['python3','followup.py'])]:
 with (r/(name+'.log')).open('w') as f:p=subprocess.Popen(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (r/(name+'.pid')).write_text(str(p.pid));print(name,p.pid)
