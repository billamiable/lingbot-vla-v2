from pathlib import Path
import subprocess,json,hashlib
R=Path('/root/experiments/lingbot-compile-accuracy-20261110')
a=R/'attempts/setup-boundary-dir';a.mkdir(parents=True)
for rel in ['runs/num_A','num_A.receipt.json','num_A.launcher.log','status.json','queue.log']:
 p=R/rel
 if p.exists():p.rename(a/p.name)
ref=Path('/raid/experiments/lingbot-compile-accuracy-20261110/references')
if ref.exists():ref.rename(ref.parent/'references_setup_boundary_dir')
files=list((R/'src').rglob('*.py'))+list((R/'harness').glob('*'))+[R/'study_runner.py',R/'summarize.py']
(R/'source_freeze.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))
with (R/'queue.log').open('x') as f:p=subprocess.Popen(['python3','study_runner.py'],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'queue.pid').write_text(str(p.pid));print('Restarted',p.pid)
