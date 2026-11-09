"""Real decoder classes: capture vs codegen, half layers vs joint, checkpoint off/on."""
import argparse
import ast
from functools import partial
import json
from pathlib import Path
import time
import inspect
import traceback
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLTextDecoderLayer
from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2DecoderLayer, Qwen2TokenMoeBlock
import lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 as joint_module
from lingbotvla.models.vla.lingbot_vla.flex_attention import build_block_mask

parser=argparse.ArgumentParser()
parser.add_argument('--backend',choices=['capture','aot_eager','inductor'],default='capture')
parser.add_argument('--out',required=True,type=Path)
parser.add_argument('--case',default='all')
parser.add_argument('--init-parallel',action='store_true')
parser.add_argument('--strip-noop-assert',action='store_true')
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(42)
device='cuda';torch.cuda.set_device(0)
if args.init_parallel:
    from lingbotvla.distributed.parallel_state import init_parallel_state
    init_parallel_state(dp_mode='fsdp2',device_type='cuda')
if args.strip_noop_assert:
    import lingbotvla.ops.group_gemm.kernel.moe as moe_kernel
    import lingbotvla.ops.fused_moe as fused_moe_module
    tree=ast.parse(inspect.getsource(moe_kernel.moe_scatter))
    class StripNoopAssert(ast.NodeTransformer):
        def visit_Assert(self,node):
            return None if isinstance(node.test,ast.Lambda) else node
    tree=StripNoopAssert().visit(tree);ast.fix_missing_locations(tree)
    exec(compile(tree,moe_kernel.__file__,'exec'),moe_kernel.__dict__)
    fused_moe_module.moe_scatter=moe_kernel.moe_scatter
    (args.out/'diagnostic_patch.txt').write_text('Process-local only: remove assert lambda: index.unique().numel() == M * topk. The original asserts the truthiness of a lambda object and never executes its body. No source file was modified.\n')
tc=Qwen3VLConfig.from_pretrained('/root/models/Qwen3-VL-4B-Instruct').text_config
tc.num_hidden_layers=1
v=Qwen3VLTextDecoderLayer(tc,0).to(device=device,dtype=torch.bfloat16).train()
ac=Qwen2Config(hidden_size=768,intermediate_size=2752,num_hidden_layers=1,num_attention_heads=32,
                num_key_value_heads=8,head_dim=128,hidden_act='silu',use_sliding_window=False)
a=Qwen2DecoderLayer(ac,0).to(device=device,dtype=torch.float32).train()
x=torch.randn(1,8,2560,device=device,dtype=torch.bfloat16,requires_grad=True)
y=torch.randn(1,4,768,device=device,dtype=torch.float32,requires_grad=True)
att=torch.randn(1,12,4096,device=device,dtype=torch.float32,requires_grad=True)
position=torch.arange(12,device=device).view(1,1,12).expand(3,1,12)
mask=torch.ones(1,12,12,device=device,dtype=torch.bool)
prepared=build_block_mask(mask,32,12,12)
original_mask_builder=joint_module.build_block_mask


class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwenvl=nn.Module();self.qwenvl.config=SimpleNamespace(text_config=tc)
        self.qwenvl.model=nn.Module();self.qwenvl.model.language_model=nn.Module()
        language=self.qwenvl.model.language_model
        language.layers=nn.ModuleList([v]);language.norm=nn.Identity()
        language.rotary_emb=Qwen3VLTextRotaryEmbedding(tc,device=device)
        self.qwen_expert=nn.Module();self.qwen_expert.model=nn.Module()
        self.qwen_expert.model.layers=nn.ModuleList([a]);self.qwen_expert.model.norm=nn.Identity()
        self.config=SimpleNamespace(qwen_expert_config=ac,attention_implementation='flex_cached',final_norm_adanorm=False)

    forward=joint_module.QwenvlWithExpertV2Model.forward
    apply_mrope=joint_module.QwenvlWithExpertV2Model.apply_mrope
    handle_kv_cache=joint_module.QwenvlWithExpertV2Model.handle_kv_cache
    _apply_deepstack=joint_module.QwenvlWithExpertV2Model._apply_deepstack


j=Joint().train()
def enable_ac(enabled):
    for layer in [v,a]:
        layer.gradient_checkpointing=enabled
        layer._gradient_checkpointing_func=partial(checkpoint,use_reentrant=False)


def qkv_v(z):return v(z,compute_kqv=True)
def qkv_a(z):return a(z,compute_kqv=True)
def out_v(z,t):return v(z,t,0,8,output_atten=True)
def out_a(z,t):return a(z,t,8,12,output_atten=True)
def joint(z,w):return j(inputs_embeds=[z,w],position_ids=position,attention_mask=mask,use_cache=False)
def prepared_mask(*unused,**kw):return prepared


results=[]
def run(name,fn,inputs,ac_enabled=False,precompute=True):
    if args.case!='all' and args.case not in name:return
    torch._dynamo.reset();enable_ac(ac_enabled)
    joint_module.build_block_mask=prepared_mask if precompute else original_mask_builder
    j.zero_grad(set_to_none=True)
    for tensor in [x,y,att]:tensor.grad=None
    graphs=[]
    def backend(gm,example_inputs):
        graphs.append(gm.code)
        return gm.forward
    result=dict(case=name,backend=args.backend,checkpoint=ac_enabled,precomputed_mask=precompute)
    started=time.time()
    try:
        result['stage']='eager'
        with torch.enable_grad():reference=fn(*inputs)
        result['stage']='capture_forward'
        compiled=torch.compile(fn,backend=backend if args.backend=='capture' else args.backend,fullgraph=True,dynamic=False)
        output=compiled(*inputs)
        flat=torch.utils._pytree.tree_leaves(output)
        tensors=[t for t in flat if isinstance(t,torch.Tensor) and t.is_floating_point() and t.requires_grad]
        loss=sum(t.float().square().mean() for t in tensors)
        result['stage']='backward';loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss)
        result.update(passed=True,stage='complete',loss=float(loss),captured_graphs=len(graphs))
    except Exception as error:
        result.update(passed=False,error_type=type(error).__name__,error=str(error)[:6000])
        (args.out/(name+'.traceback.txt')).write_text(traceback.format_exc())
    result['seconds']=time.time()-started
    if graphs:(args.out/(name+'.fx.py')).write_text('\n\n'.join(graphs))
    results.append(result)
    (args.out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:result[k] for k in ['case','passed','stage','seconds']}),flush=True)


for enabled in [False,True]:
    suffix='ac' if enabled else 'noac'
    run('vlm_qkv_'+suffix,qkv_v,(x,),enabled)
    run('vlm_output_'+suffix,out_v,(x,att),enabled)
    run('action_qkv_'+suffix,qkv_a,(y,),enabled)
    run('action_output_dense_'+suffix,out_a,(y,att),enabled)
run('joint_original_mask_noac',joint,(x,y),False,False)
run('joint_precomputed_noac',joint,(x,y),False,True)
run('joint_precomputed_ac',joint,(x,y),True,True)

if args.case=='all' or 'moe' in args.case:
    cfg=SimpleNamespace(num_experts=32,num_experts_per_tok=4,norm_topk_prob=True,hidden_size=768,
                        hidden_act='silu',moe_intermediate_size=512,shared_expert_intermediate_size=704,
                        _moe_implementation='fused',initializer_range=.02,router_activation='sigmoid',
                        routed_scaling_factor=4.0,use_shared_expert_gate=False)
    a.mlp=Qwen2TokenMoeBlock(cfg).to(device=device,dtype=torch.float32).train()
    run('action_output_moe_noac',out_a,(y,att),False)
    run('action_output_moe_ac',out_a,(y,att),True)
    run('joint_moe_precomputed_noac',joint,(x,y),False,True)
    run('joint_moe_precomputed_ac',joint,(x,y),True,True)
joint_module.build_block_mask=original_mask_builder
if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
