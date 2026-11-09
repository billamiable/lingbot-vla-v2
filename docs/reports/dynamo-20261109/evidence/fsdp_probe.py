"""Two-rank causal test: HF checkpoint __call__ encloses FSDP hooks."""
from datetime import timedelta
import argparse
from functools import partial
import json
import os
from pathlib import Path
import traceback

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLTextDecoderLayer

parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['capture','inductor'],default='capture');options=parser.parse_args()
ROOT=Path('/root/experiments/lingbot-dynamo-20261109')/('fsdp_probe' if options.backend=='capture' else 'fsdp_inductor')
ROOT.mkdir(exist_ok=True)
rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank);torch.set_num_threads(4)
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
mesh=init_device_mesh('cuda',(dist.get_world_size(),))
config=Qwen3VLConfig.from_pretrained('/root/models/Qwen3-VL-4B-Instruct').text_config
results=[];references={}

for operation in ['qkv','output']:
    for mode in ['fsdp_noac','fsdp_ac','fsdp_ac_explicit_forward_compile']:
        torch._dynamo.reset();torch.manual_seed(42)
        model=Qwen3VLTextDecoderLayer(config,0).cuda().float().train()
        model.gradient_checkpointing=mode!='fsdp_noac'
        model._gradient_checkpointing_func=partial(checkpoint,use_reentrant=False)
        fully_shard(model,mesh=mesh,reshard_after_forward=False,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32))
        x=torch.randn(1,8,2560,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        attention=torch.randn(1,12,4096,device='cuda',dtype=torch.float32,requires_grad=True)
        graphs={'outer':[],'region':[]}
        def backend(kind):
            def compile_graph(gm,example_inputs):
                graphs[kind].append(gm.code)
                if options.backend=='inductor':
                    return torch._dynamo.lookup_backend('inductor')(gm,example_inputs)
                return gm.forward
            return compile_graph
        if mode=='fsdp_ac_explicit_forward_compile':
            model.forward=torch.compile(model.forward,backend=backend('region'),fullgraph=True,dynamic=False)
        def call(z,t):
            if operation=='qkv':return model(z,compute_kqv=True)
            return model(z,t,0,8,output_atten=True)
        row=dict(operation=operation,mode=mode,rank=rank,backend=options.backend)
        output=None
        try:
            compiled=torch.compile(call,backend=backend('outer'),fullgraph=False,dynamic=False)
            output=compiled(x,attention)
            tensors=list(output) if isinstance(output,tuple) else [output]
            loss=sum(t.float().square().mean() for t in tensors)
            loss.backward();torch.cuda.synchronize()
            row.update(passed=True,loss=float(loss),outer_graphs=len(graphs['outer']),region_graphs=len(graphs['region']))
            saved=[t.detach().cpu() for t in tensors]+[x.grad.detach().cpu()]
            if mode=='fsdp_noac':references[operation]=saved
            else:row['max_diff_output_and_input_grad']=max(float((p.float()-q.float()).abs().max()) for p,q in zip(saved,references[operation]))
            for k in ['outer','region']:
                if graphs[k]:(ROOT/f'{operation}_{mode}_{k}_rank{rank}.fx.py').write_text('\n\n'.join(graphs[k]))
        except Exception as error:
            row.update(passed=False,error_type=type(error).__name__,error=str(error)[:4000],outer_graphs=len(graphs['outer']),region_graphs=len(graphs['region']))
            (ROOT/f'{operation}_{mode}_rank{rank}.traceback.txt').write_text(traceback.format_exc())
        results.append(row)
        (ROOT/f'results_rank{rank}.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(row),flush=True)
        dist.barrier()
        del model,x,attention,output
        torch.cuda.empty_cache()

dist.destroy_process_group()
