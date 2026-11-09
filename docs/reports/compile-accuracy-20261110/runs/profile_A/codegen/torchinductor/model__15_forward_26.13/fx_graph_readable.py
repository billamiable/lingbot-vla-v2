class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 72, 2560]", primals_2: "bf16[256, 2560]", primals_3: "bf16[256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:654 in torch_dynamo_resume_in_embed_prefix_at_646, code: align_pad_masks = torch.ones(
        full_default: "b8[32, 8]" = torch.ops.aten.full.default([32, 8], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:660 in torch_dynamo_resume_in_embed_prefix_at_646, code: fake_align_ids = torch.full(
        full_default_1: "i64[32, 8]" = torch.ops.aten.full.default([32, 8], 151645, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:650 in _get_align_tokens, code: tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
        view: "bf16[8, 32, 2560]" = torch.ops.aten.view.default(primals_2, [8, 32, 2560]);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:651 in _get_align_tokens, code: tk_weights = tk_weights.mean(dim=1)
        mean: "bf16[8, 2560]" = torch.ops.aten.mean.dim(view, [1]);  view = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:650 in _get_align_tokens, code: tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
        view_1: "bf16[8, 32, 2560]" = torch.ops.aten.view.default(primals_3, [8, 32, 2560]);  primals_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:651 in _get_align_tokens, code: tk_weights = tk_weights.mean(dim=1)
        mean_1: "bf16[8, 2560]" = torch.ops.aten.mean.dim(view_1, [1]);  view_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:675 in torch_dynamo_resume_in_embed_prefix_at_646, code: torch.cat([current_task, current_video_task], dim=-1)
        cat: "bf16[8, 5120]" = torch.ops.aten.cat.default([mean, mean_1], -1);  mean = mean_1 = None
        return (cat, primals_1, full_default, full_default_1)
        