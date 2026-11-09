import importlib.util,os
from pathlib import Path
root=Path("/root/experiments/lingbot-dynamo-20261109")
s=importlib.util.spec_from_file_location("driver","/root/experiments/lingbot-ac-mbs-20261108/driver.py")
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=root
os.environ["TORCH_LOGS"]="graph_breaks,recompiles"
os.environ["TORCHDYNAMO_VERBOSE"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
m.run("real_main_ac32","main",3,harness="harness")
