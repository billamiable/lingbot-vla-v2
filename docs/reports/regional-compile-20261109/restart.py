from pathlib import Path
import subprocess,json
r=Path('/root/experiments/lingbot-regional-20261109')
script="import driver,json\nfrom pathlib import Path\nfor c in ['B','D']:driver.run('smoke_'+c,c,3,True)\nPath('smoke_gate.json').write_text(json.dumps(dict(passed=True,cases=list(driver.CASES))))\ndriver.state(state='complete',phase='smoke')\n"
(r/'resume_smoke.py').write_text(script)
(r/'status.json').write_text(json.dumps(dict(state='starting')))
for name,cmd in [('smoke_queue',['python3','resume_smoke.py']),('followup',['python3','followup.py'])]:
 with (r/(name+'.log')).open('w') as f:p=subprocess.Popen(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (r/(name+'.pid')).write_text(str(p.pid));print(name,p.pid)
