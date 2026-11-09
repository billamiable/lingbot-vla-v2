class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "bf16[8, 5120]", tangents_2: "bf16[32, 8, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:707 in torch_dynamo_resume_in_embed_prefix_at_674, code: torch.cat([future_task, future_video_task], dim=-1)
        slice_1: "bf16[8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, 2560)
        slice_2: "bf16[8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 2560, 5120);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:651 in _get_align_tokens, code: tk_weights = tk_weights.mean(dim=1)
        unsqueeze: "bf16[8, 1, 2560]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
        expand: "bf16[8, 32, 2560]" = torch.ops.aten.expand.default(unsqueeze, [8, 32, 2560]);  unsqueeze = None
        div: "bf16[8, 32, 2560]" = torch.ops.aten.div.Scalar(expand, 32);  expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:650 in _get_align_tokens, code: tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
        view_2: "bf16[256, 2560]" = torch.ops.aten.view.default(div, [256, 2560]);  div = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:651 in _get_align_tokens, code: tk_weights = tk_weights.mean(dim=1)
        unsqueeze_1: "bf16[8, 1, 2560]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
        expand_1: "bf16[8, 32, 2560]" = torch.ops.aten.expand.default(unsqueeze_1, [8, 32, 2560]);  unsqueeze_1 = None
        div_1: "bf16[8, 32, 2560]" = torch.ops.aten.div.Scalar(expand_1, 32);  expand_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:650 in _get_align_tokens, code: tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
        view_3: "bf16[256, 2560]" = torch.ops.aten.view.default(div_1, [256, 2560]);  div_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:677 in torch_dynamo_resume_in_embed_prefix_at_674, code: align_embs = current_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
        sum_1: "bf16[8, 2560]" = torch.ops.aten.sum.dim_IntList(tangents_2, [0]);  tangents_2 = None
        return (sum_1, view_3, view_2)
        