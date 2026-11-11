from pathlib import Path
import subprocess
r=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with (r/'fixed-queue.log').open('x') as f:p=subprocess.Popen(['python3','resume-fixed.py'],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'driver.pid').write_text(str(p.pid));print('Started corrected validation queue',p.pid)
