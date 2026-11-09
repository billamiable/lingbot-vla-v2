from pathlib import Path
import subprocess,hashlib,json,shutil
R=Path('/root/experiments/lingbot-compile-accuracy-20261110')
p=R/'src/regional_runtime.py';s=p.read_text()
s=s.replace("        return torch._dynamo.lookup_backend('inductor')(gm,example_inputs)","""        if os.environ.get('STUDY_PRESERVE_CASTS')=='1':
            return torch._inductor.compile(gm,example_inputs,options={'emulate_precision_casts':True})
        return torch._dynamo.lookup_backend('inductor')(gm,example_inputs)""")
s=s.replace("regional=enabled),indent=2)","regional=enabled,preserve_casts=os.environ.get('STUDY_PRESERVE_CASTS')=='1'),indent=2)")
p.write_text(s)
assert s.count("emulate_precision_casts")==1
# Include newly created runtime file in the reproducible patch.
subprocess.run(['git','add','-N','regional_runtime.py'],cwd=R/'src',check=True)
(R/'study.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=R/'src'))
files=list((R/'src').rglob('*.py'))+list((R/'harness').glob('*'))+[R/'study_runner.py',R/'summarize.py']
(R/'source_freeze.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))
subprocess.run(['python3','summarize.py'],cwd=R,check=True)
with (R/'queue.log').open('x') as f:p=subprocess.Popen(['python3','study_runner.py'],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'queue.pid').write_text(str(p.pid));print('Started',p.pid)
