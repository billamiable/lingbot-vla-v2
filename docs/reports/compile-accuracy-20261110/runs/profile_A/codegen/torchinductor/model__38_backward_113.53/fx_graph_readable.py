class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "bf16[32, 72, 2560]", tangents_2: "bf16[32, 256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1498 in video_emb_forward, code: video_preds = query_embs.repeat(video_align_embs.shape[0], 1, 1).to(
        sum_1: "bf16[256, 2560]" = torch.ops.aten.sum.dim_IntList(tangents_2, [0]);  tangents_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1490 in video_emb_forward, code: video_align_embs = torch.cat([image_embs_for_video, video_task_embs], dim=1)
        slice_8: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 64, 72);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:964 in _future_depth_task_tokens, code: return hidden_states[:, -self.num_task_tokens:, :]
        full_default: "bf16[32, 8, 2560]" = torch.ops.aten.full.default([32, 8, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[32, 8, 2560]" = torch.ops.aten.slice_scatter.default(full_default, slice_8, 2, 0, 9223372036854775807);  full_default = slice_8 = None
        full_default_1: "bf16[32, 286, 2560]" = torch.ops.aten.full.default([32, 286, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1: "bf16[32, 286, 2560]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter, 1, -8, 9223372036854775807);  full_default_1 = slice_scatter = None
        return (slice_scatter_1, sum_1)
        