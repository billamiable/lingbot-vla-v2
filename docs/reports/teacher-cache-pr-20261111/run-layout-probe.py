import os,subprocess,time
from pathlib import Path
r=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
os.environ['CAPTURE_TARGET_LAYOUT']='1'
import driver
driver.run('diag_layout','verify',1)
