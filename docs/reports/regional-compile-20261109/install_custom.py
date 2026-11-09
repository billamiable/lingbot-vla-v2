from pathlib import Path
import subprocess,json
r=Path('/root/experiments/lingbot-regional-20261109');a=r/'attempts/v2-swiglu-libdevice';a.mkdir()
for rel in ['runs/smoke_D','smoke_D.launcher.log','smoke_D.receipt.json','status.json','smoke_queue.log','followup_status.json']:
 p=r/rel
 if p.exists():p.rename(a/p.name)
p=r/'src/lingbotvla/ops/swiglu.py'
# Restore original exact kernels; register only the compiler-facing boundary.
s=subprocess.check_output(['git','show','HEAD:lingbotvla/ops/swiglu.py'],cwd=r/'src',text=True)
start=s.index('@torch.compiler.disable\ndef swiglu');end=s.index('\ndef _fused_mlp_forward',start)
s=s[:start]+(r/'custom_swiglu.py').read_text()+'\n'+s[end:];p.write_text(s)
(r/'status.json').write_text(json.dumps(dict(state='starting')))
