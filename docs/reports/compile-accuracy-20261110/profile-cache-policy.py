from pathlib import Path
import json,hashlib,subprocess
R=Path('/root/experiments/lingbot-compile-accuracy-20261110')
(R/'source_freeze_before_profile_capture.json').write_bytes((R/'source_freeze.json').read_bytes())
p=R/'src/regional_runtime.py';s=p.read_text();old='        inductor_config.trace.enabled=True\n';new=old+'        inductor_config.fx_graph_cache=False  # Profile only: force generated IR/code evidence, retain kernel binary caches.\n';assert old in s;s=s.replace(old,new);p.write_text(s)
(R/'study.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=R/'src'))
j=json.loads((R/'source_freeze.json').read_text())
for name in ['src/regional_runtime.py','summarize.py']:j[name]=hashlib.sha256((R/name).read_bytes()).hexdigest()
(R/'source_freeze.json').write_text(json.dumps(j,indent=2))
(R/'profile_capture_policy.json').write_text(json.dumps(dict(rank=0,profile_only=True,fx_graph_cache=False,triton_binary_cache_preserved=True,reason='Retain generated IR and source on fresh graph compilation; no numerical-mode codepath change'),indent=2))
print('Only profile capture policy changed; numerical and trajectory codepaths unchanged')
