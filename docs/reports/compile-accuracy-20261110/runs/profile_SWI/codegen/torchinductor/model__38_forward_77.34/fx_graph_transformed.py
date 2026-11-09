class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 2560]", primals_2: "bf16[256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1465 in video_emb_forward, code: image_embs = hidden_states[:, chunk_size * 0 + 1 : chunk_size * 1 + 1, :]
        slice_2: "bf16[32, 64, 2560]" = torch.ops.aten.slice.Tensor(primals_1, 1, 1, 65)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:964 in _future_depth_task_tokens, code: return hidden_states[:, -self.num_task_tokens:, :]
        slice_5: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(primals_1, 1, -8, 9223372036854775807);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1490 in video_emb_forward, code: video_align_embs = torch.cat([image_embs_for_video, video_task_embs], dim=1)
        cat: "bf16[32, 72, 2560]" = torch.ops.aten.cat.default([slice_2, slice_5], 1);  slice_2 = slice_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1498 in video_emb_forward, code: video_preds = query_embs.repeat(video_align_embs.shape[0], 1, 1).to(
        repeat: "bf16[32, 256, 2560]" = torch.ops.aten.repeat.default(primals_2, [32, 1, 1]);  primals_2 = None
        return (cat, repeat)
        