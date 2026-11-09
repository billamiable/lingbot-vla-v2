from pathlib import Path
import sys,platform,json,subprocess,torch,transformers,triton
r=Path('/root/experiments/lingbot-regional-20261109')
j=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,triton=triton.__version__,platform=platform.platform(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,pci.device_id,memory.total,driver_version','--format=csv,noheader'],text=True).splitlines())
(r/'environment.json').write_text(json.dumps(j,indent=2));print(json.dumps(j,indent=2))
