from pathlib import Path
import json,subprocess,sys,platform,datetime,torch,transformers,triton
r=Path('/root/experiments/lingbot-compile-accuracy-20261110')
j=dict(time=datetime.datetime.now().isoformat(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,triton=triton.__version__,platform=platform.platform(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,pci.device_id,memory.total,driver_version','--format=csv,noheader'],text=True).splitlines())
(r/'environment.json').write_text(json.dumps(j,indent=2))
print({k:v for k,v in j.items() if k in ['torch','transformers','triton','cuda']})
