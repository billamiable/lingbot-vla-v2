class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "bf16[55, 768]", view: "bf16[1600, 768]", convert_element_type_5: "bf16[32, 50, 55]", tangents_1: "bf16[32, 50, 55]", tangents_2: "f32[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:981 in torch_dynamo_resume_in_forward_at_931, code: losses = F.l1_loss(u_t, v_t, reduction="none")
        mul_3: "bf16[32, 50, 55]" = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        neg: "bf16[32, 50, 55]" = torch.ops.aten.neg.default(mul_3);  mul_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:976 in torch_dynamo_resume_in_forward_at_931, code: v_t = self.action_out_proj(suffix_out)
        view_2: "bf16[1600, 55]" = torch.ops.aten.view.default(neg, [1600, 55]);  neg = None
        permute: "bf16[768, 55]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_1: "bf16[55, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "bf16[1600, 768]" = torch.ops.aten.mm.default(view_2, permute_1);  permute_1 = None
        permute_2: "bf16[55, 1600]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1: "bf16[55, 768]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "bf16[1, 55]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
        view_3: "bf16[55]" = torch.ops.aten.view.default(sum_1, [55]);  sum_1 = None
        view_4: "bf16[32, 50, 768]" = torch.ops.aten.view.default(mm, [32, 50, 768]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:975 in torch_dynamo_resume_in_forward_at_931, code: suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        convert_element_type_10: "f32[32, 50, 768]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:966 in torch_dynamo_resume_in_forward_at_931, code: suffix_out = suffix_out[:, -self.config.n_action_steps :]
        full_default: "f32[32, 51, 768]" = torch.ops.aten.full.default([32, 51, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "f32[32, 51, 768]" = torch.ops.aten.slice_scatter.default(full_default, convert_element_type_10, 1, -50, 9223372036854775807);  full_default = convert_element_type_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:946 in torch_dynamo_resume_in_forward_at_931, code: loss_future_video = video_total_loss * video_weight
        mul_4: "f32[]" = torch.ops.aten.mul.Tensor(tangents_2, 0.004);  tangents_2 = None
        return (mul_4, mul_4, slice_scatter, mm_1, view_3, None)
        