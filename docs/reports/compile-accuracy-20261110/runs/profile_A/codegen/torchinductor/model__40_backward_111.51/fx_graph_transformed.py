class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "bf16[32, 72, 2560]", tangents_2: "bf16[32, 256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1525 in current_video_emb_forward, code: queries = self.current_video_align_embs.repeat(align_embs.shape[0], 1, 1).to(
        sum_1: "bf16[256, 2560]" = torch.ops.aten.sum.dim_IntList(tangents_2, [0]);  tangents_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1524 in current_video_emb_forward, code: align_embs = torch.cat([image_embs, current_task_embs], dim=1)
        slice_7: "bf16[32, 64, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, 64)
        slice_8: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 64, 72);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:831 in _current_depth_task_tokens, code: return hidden_states[:, start:end, :]
        full_default: "bf16[32, 8, 2560]" = torch.ops.aten.full.default([32, 8, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default: "bf16[32, 8, 2560]" = torch.ops.aten.slice_scatter.default(full_default, slice_8, 2, 0, 9223372036854775807);  full_default = slice_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:831 in _current_depth_task_tokens, code: return hidden_states[:, start:end, :]
        full_default_1: "bf16[32, 286, 2560]" = torch.ops.aten.full.default([32, 286, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default_1: "bf16[32, 286, 2560]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default, 1, 270, 278);  slice_scatter_default = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1522 in current_video_emb_forward, code: image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        full_default_3: "bf16[32, 64, 2560]" = torch.ops.aten.full.default([32, 64, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default_2: "bf16[32, 64, 2560]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_7, 2, 0, 9223372036854775807);  full_default_3 = slice_7 = None
        slice_scatter_default_3: "bf16[32, 286, 2560]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default_2, 1, 1, 65);  full_default_1 = slice_scatter_default_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1522 in current_video_emb_forward, code: image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        add: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(slice_scatter_default_1, slice_scatter_default_3);  slice_scatter_default_1 = slice_scatter_default_3 = None
        return (add, sum_1)
        