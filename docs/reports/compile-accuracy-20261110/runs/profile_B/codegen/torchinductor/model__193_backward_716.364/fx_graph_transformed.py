class GraphModule(torch.nn.Module):
    def forward(self, primals_7: "b8[32, 50, 55]", clamp_min_1: "i64[]", tangents_1: "f32[]", tangents_2: "bf16[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1380 in torch_dynamo_resume_in_forward_at_1346, code: loss_vla
        convert_element_type: "bf16[]" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1380 in torch_dynamo_resume_in_forward_at_1346, code: loss_vla
        add_5: "bf16[]" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type);  tangents_2 = convert_element_type = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1372 in torch_dynamo_resume_in_forward_at_1346, code: loss_vla = masked_losses.sum() / joint_mask.sum().clamp(min=1)
        div_2: "bf16[]" = torch.ops.aten.div.Tensor(add_5, clamp_min_1);  add_5 = clamp_min_1 = None
        expand: "bf16[32, 50, 55]" = torch.ops.aten.expand.default(div_2, [32, 50, 55]);  div_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1369 in torch_dynamo_resume_in_forward_at_1346, code: masked_losses = losses * joint_mask
        mul_1: "bf16[32, 50, 55]" = torch.ops.aten.mul.Tensor(expand, primals_7);  expand = primals_7 = None
        return (mul_1, tangents_1, tangents_1, tangents_1, tangents_1, tangents_1, None)
        