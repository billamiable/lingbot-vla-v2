"""Region compiler entry outside FSDP/checkpoint control logic, with per-rank compile receipts."""
import json
import os
from pathlib import Path
import torch


def install_regional_compile(model, output_dir):
    from lingbotvla.distributed.torch_parallelize import _resolve
    core=model.model.qwenvl_with_expert
    vlm,_=_resolve(core.qwenvl,[("model","language_model","layers"),("model","layers")])
    expert=core.qwen_expert.model.layers
    path=Path(output_dir).parent/f'region_compile_rank{os.environ.get("RANK","0")}.jsonl'
    # A fullgraph error or cache-limit exception must fail the smoke test, not quietly run eager.
    def backend(gm, example_inputs):
        with path.open('a') as f:
            f.write(json.dumps(dict(nodes=len(list(gm.graph.nodes)),targets=[str(n.target) for n in gm.graph.nodes if n.op.startswith('call')]))+'\n')
        return torch._dynamo.lookup_backend('inductor')(gm,example_inputs)
    for layer in list(vlm)+list(expert):
        layer.forward=torch.compile(layer.forward,backend=backend,fullgraph=True,dynamic=False)
    (Path(output_dir).parent/'region_install.json').write_text(json.dumps(dict(vlm_layers=len(vlm),expert_layers=len(expert),fullgraph=True),indent=2))
