from pathlib import Path
import os,json,hashlib,gzip,shutil,tarfile,subprocess
R=Path(__file__).resolve().parent
subprocess.run(['python3',str(R/'summarize.py')],check=True)
# Check the immutable tested production source before packaging evidence.
frozen=json.loads((R/'source_freeze.json').read_text())
for name,digest in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
(R/'candidate.patch').write_bytes(subprocess.check_output(['git','diff','be969b8','HEAD'],cwd=R/'candidate'))
(R/'environment-gpu.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,pci.device_id,memory.total,driver_version,power.limit','--format=csv,noheader'],text=True))
cache=Path('/raid/teacher_cache/robotwin_clean50_png');meta=json.loads((cache/'meta.json').read_text())
(R/'cache-size.json').write_text(json.dumps(dict(samples=meta['num_samples'],bytes=sum(p.stat().st_size for p in cache.iterdir() if p.is_file()),shapes=meta['shapes']),indent=2))
files=[]
for base,dirs,names in os.walk(R):
 dirs[:]=[x for x in dirs if x not in ['baseline','candidate','__pycache__','output']]
 for name in names:
  p=Path(base)/name
  if p.name in ['manifest.json','evidence.tar.gz'] or p.suffix in ['.pt','.pth','.pid','.bundle'] or p.is_symlink():continue
  if p.suffix=='.log':
   q=p.with_suffix('.log.gz')
   with p.open('rb') as src,gzip.open(q,'wb') as dst:shutil.copyfileobj(src,dst)
   files.append(q);continue
  if p.suffix=='.gz' and p.with_suffix('').exists():continue
  if p.stat().st_size>90*1024**2:raise RuntimeError('Oversized evidence: '+str(p))
  files.append(p)
manifest=dict(candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'candidate',text=True).strip(),files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(R/'evidence.tar.gz','w:gz') as archive:
 for p in files+[R/'manifest.json']:archive.add(p,arcname=str(p.relative_to(R)),recursive=False)
print('Exported',len(files),'files',(R/'evidence.tar.gz').stat().st_size)
