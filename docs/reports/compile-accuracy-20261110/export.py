"""Bounded evidence export: excludes large tensor references and source checkout."""
from pathlib import Path
import os,hashlib,json,subprocess,tarfile
R=Path(__file__).resolve().parent
subprocess.run(['python3',str(R/'summarize.py')],check=True)
freeze=R/'source_freeze.json';frozen=json.loads(freeze.read_text())
for name,digest in frozen.items():
 actual=hashlib.sha256((R/name).read_bytes()).hexdigest()
 if name.startswith(('src/','harness/')) or name=='study_runner.py':assert actual==digest,('Source changed',name)
 frozen[name]=actual
freeze.write_text(json.dumps(frozen,indent=2))
files=[];omitted=[]
for base,dirs,names in os.walk(R):
 dirs[:]=[d for d in dirs if d not in ['src','references','output','__pycache__','.git']]
 for name in names:
  p=Path(base)/name
  if p.suffix in ['.pt','.pth','.pid','.tar'] or p.name in ['manifest.json','evidence.tar.gz'] or p.is_symlink():continue
  # Profiler traces are compressed independently and can exceed GitHub's per-file limit.
  if p.stat().st_size>90*1024**2:
   omitted.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size));continue
  files.append(p)
(R/'study.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=R/'src'))
m=dict(omitted_large_files=omitted,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'src',text=True).strip(),files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(R/'manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open(R/'evidence.tar.gz','w:gz') as t:
 for p in files+[R/'manifest.json']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
print('Exported',len(files),'files', (R/'evidence.tar.gz').stat().st_size)
