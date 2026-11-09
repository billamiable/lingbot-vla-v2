class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[8, 2560]", primals_2: "bf16[256, 2560]", primals_3: "bf16[256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:677 in torch_dynamo_resume_in_embed_prefix_at_674, code: align_embs = current_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
        repeat: "bf16[32, 8, 2560]" = torch.ops.aten.repeat.default(primals_1, [32, 1, 1]);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:650 in _get_align_tokens, code: tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
        view: "bf16[8, 32, 2560]" = torch.ops.aten.reshape.default(primals_2, [8, 32, 2560]);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:651 in _get_align_tokens, code: tk_weights = tk_weights.mean(dim=1)
        mean: "bf16[8, 2560]" = torch.ops.aten.mean.dim(view, [1]);  view = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:650 in _get_align_tokens, code: tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
        view_1: "bf16[8, 32, 2560]" = torch.ops.aten.reshape.default(primals_3, [8, 32, 2560]);  primals_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:651 in _get_align_tokens, code: tk_weights = tk_weights.mean(dim=1)
        mean_1: "bf16[8, 2560]" = torch.ops.aten.mean.dim(view_1, [1]);  view_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:707 in torch_dynamo_resume_in_embed_prefix_at_674, code: torch.cat([future_task, future_video_task], dim=-1)
        cat: "bf16[8, 5120]" = torch.ops.aten.cat.default([mean, mean_1], -1);  mean = mean_1 = None
        return (cat, repeat)
        