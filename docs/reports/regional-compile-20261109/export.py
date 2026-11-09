"""Export bounded experiment evidence, excluding weights, gradients and source checkout."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,os
r=Path(__file__).resolve().parent
subprocess.run(['python3',str(r/'summarize.py')],check=True)
patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=r/'src')
(r/'study.patch').write_bytes(patch)
(r/'regional_runtime.py').write_bytes((r/'src/regional_runtime.py').read_bytes())
files=[]
for base,dirs,names in os.walk(r):
 dirs[:]=[d for d in dirs if d not in {'src','numeric_reference','output','__pycache__','.git'}]
 for name in names:
  p=Path(base)/name
  if p.suffix in {'.pt','.pth','.pid','.gz','.tar'} or p.is_symlink():continue
  if p.name=='manifest.json':continue
  if p.stat().st_size>90*1024**2:raise RuntimeError(f'Unexpected large artifact {p}')
  files.append(p)
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r/'src',text=True).strip(),
 'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(r/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(r/'evidence.tar.gz','w:gz') as tar:
 for p in files+[r/'manifest.json']:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
print('exported',len(files),'files', (r/'evidence.tar.gz').stat().st_size)
