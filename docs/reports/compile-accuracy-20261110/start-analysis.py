from pathlib import Path
import subprocess
r=Path('/root/experiments/lingbot-compile-accuracy-20261110')
with (r/'mechanisms.log').open('x') as f:p=subprocess.Popen(['python3','mechanisms.py','--wait'],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'mechanisms.pid').write_text(str(p.pid));print(p.pid)
