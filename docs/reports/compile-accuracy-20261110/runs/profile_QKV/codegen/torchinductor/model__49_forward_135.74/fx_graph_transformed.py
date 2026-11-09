class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 50, 55]", primals_2: "f32[]", primals_3: "f32[]", primals_4: "f32[]", primals_5: "f32[]", primals_6: "f32[]", primals_7: "b8[32, 50, 55]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1369 in torch_dynamo_resume_in_forward_at_1346, code: masked_losses = losses * joint_mask
        mul: "bf16[32, 50, 55]" = torch.ops.aten.mul.Tensor(primals_1, primals_7);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1370 in torch_dynamo_resume_in_forward_at_1346, code: valid_counts = joint_mask.sum(dim=(1, 2)).clamp(min=1)
        sum_1: "i64[32]" = torch.ops.aten.sum.dim_IntList(primals_7, [1, 2])
        clamp_min: "i64[32]" = torch.ops.aten.clamp_min.default(sum_1, 1);  sum_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1371 in torch_dynamo_resume_in_forward_at_1346, code: batch_mean_losses = masked_losses.sum(dim=(1, 2)) / valid_counts
        sum_2: "bf16[32]" = torch.ops.aten.sum.dim_IntList(mul, [1, 2])
        div: "bf16[32]" = torch.ops.aten.div.Tensor(sum_2, clamp_min);  sum_2 = clamp_min = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1372 in torch_dynamo_resume_in_forward_at_1346, code: loss_vla = masked_losses.sum() / joint_mask.sum().clamp(min=1)
        sum_3: "bf16[]" = torch.ops.aten.sum.default(mul);  mul = None
        sum_4: "i64[]" = torch.ops.aten.sum.default(primals_7)
        clamp_min_1: "i64[]" = torch.ops.aten.clamp_min.default(sum_4, 1);  sum_4 = None
        div_1: "bf16[]" = torch.ops.aten.div.Tensor(sum_3, clamp_min_1);  sum_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1380 in torch_dynamo_resume_in_forward_at_1346, code: loss_vla
        add: "f32[]" = torch.ops.aten.add.Tensor(div_1, primals_2);  primals_2 = None
        add_1: "f32[]" = torch.ops.aten.add.Tensor(add, primals_3);  add = primals_3 = None
        add_2: "f32[]" = torch.ops.aten.add.Tensor(add_1, primals_4);  add_1 = primals_4 = None
        add_3: "f32[]" = torch.ops.aten.add.Tensor(add_2, primals_5);  add_2 = primals_5 = None
        add_4: "f32[]" = torch.ops.aten.add.Tensor(add_3, primals_6);  add_3 = None
        return (add_4, div_1, div, primals_6, primals_7, clamp_min_1)
        