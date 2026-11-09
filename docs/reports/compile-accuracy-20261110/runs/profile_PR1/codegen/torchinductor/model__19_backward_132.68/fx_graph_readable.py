class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "bf16[32, 286, 2560]", tangents_2: "bf16[32, 8, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:771 in torch_dynamo_resume_in_embed_prefix_at_706, code: else:
        slice_1: "bf16[32, 198, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, 198)
        slice_2: "bf16[32, 72, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 198, 270)
        slice_3: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 270, 278)
        slice_4: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 278, 286);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:771 in torch_dynamo_resume_in_embed_prefix_at_706, code: else:
        add: "bf16[32, 8, 2560]" = torch.ops.aten.add.Tensor(tangents_2, slice_4);  tangents_2 = slice_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:709 in torch_dynamo_resume_in_embed_prefix_at_706, code: future_align_embs = future_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
        sum_1: "bf16[8, 2560]" = torch.ops.aten.sum.dim_IntList(add, [0]);  add = None
        return (sum_1, slice_1, None, slice_2, None, slice_3, None, None, None, None, None, None)
        