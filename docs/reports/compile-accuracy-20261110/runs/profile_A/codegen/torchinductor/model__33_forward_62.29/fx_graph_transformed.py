class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "b8[32, 3]", primals_2: "bf16[32, 286, 2560]", primals_3: "bf16[256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1423 in depth_emb_forward, code: image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        slice_2: "bf16[32, 64, 2560]" = torch.ops.aten.slice.Tensor(primals_2, 1, 1, 65)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:831 in _current_depth_task_tokens, code: return hidden_states[:, start:end, :]
        slice_5: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(primals_2, 1, 270, 278);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1425 in depth_emb_forward, code: align_embs = torch.cat([image_embs, align_embs], dim=1)
        cat: "bf16[32, 72, 2560]" = torch.ops.aten.cat.default([slice_2, slice_5], 1);  slice_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1426 in depth_emb_forward, code: depth_preds = self.depth_align_embs.repeat(align_embs.shape[0], 1, 1).to(dtype=align_embs.dtype, device=align_embs.device)
        repeat: "bf16[32, 256, 2560]" = torch.ops.aten.repeat.default(primals_3, [32, 1, 1]);  primals_3 = None
        return (cat, repeat, slice_2)
        