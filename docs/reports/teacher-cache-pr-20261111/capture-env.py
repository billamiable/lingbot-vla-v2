from pathlib import Path
import json,sys,torch,transformers,triton,platform,datetime
r=Path('/root/experiments/lingbot-teacher-cache-pr-20261111')
j=dict(time=datetime.datetime.now().isoformat(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),transformers=transformers.__version__,triton=triton.__version__,platform=platform.platform())
(r/'environment.json').write_text(json.dumps(j,indent=2));print(j)
