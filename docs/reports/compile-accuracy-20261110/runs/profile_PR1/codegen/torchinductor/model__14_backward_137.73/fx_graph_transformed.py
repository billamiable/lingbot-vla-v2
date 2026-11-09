class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "bf16[32, 198, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:641 in torch_dynamo_resume_in_embed_prefix_at_613, code: img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        view_4: "bf16[32, 3, 66, 2560]" = torch.ops.aten.reshape.default(tangents_1, [32, 3, 66, 2560]);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:616 in torch_dynamo_resume_in_embed_prefix_at_613, code: img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
        slice_23: "bf16[32, 3, 1, 2560]" = torch.ops.aten.slice.Tensor(view_4, 2, 0, 1)
        slice_24: "bf16[32, 3, 64, 2560]" = torch.ops.aten.slice.Tensor(view_4, 2, 1, 65)
        slice_25: "bf16[32, 3, 1, 2560]" = torch.ops.aten.slice.Tensor(view_4, 2, 65, 66);  view_4 = None
        return (slice_25, slice_24, slice_23, None)
        