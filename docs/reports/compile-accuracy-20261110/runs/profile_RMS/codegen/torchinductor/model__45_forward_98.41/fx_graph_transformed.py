class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[]", primals_2: "f32[]", primals_3: "f32[32, 51, 768]", primals_4: "bf16[55, 768]", primals_5: "bf16[55]", primals_6: "bf16[32, 50, 55]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:935 in torch_dynamo_resume_in_forward_at_931, code: video_total_loss = video_total_loss + current_video_loss
        add: "f32[]" = torch.ops.aten.add.Tensor(primals_2, primals_1)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:946 in torch_dynamo_resume_in_forward_at_931, code: loss_future_video = video_total_loss * video_weight
        mul: "f32[]" = torch.ops.aten.mul.Tensor(add, 0.004)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:950 in torch_dynamo_resume_in_forward_at_931, code: align_metrics["align/current_video_loss"] * video_weight
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(primals_1, 0.004)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:953 in torch_dynamo_resume_in_forward_at_931, code: align_metrics["align/future_video_loss_weighted"] = (loss_video * video_weight).detach()
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(primals_2, 0.004)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:966 in torch_dynamo_resume_in_forward_at_931, code: suffix_out = suffix_out[:, -self.config.n_action_steps :]
        slice_2: "f32[32, 50, 768]" = torch.ops.aten.slice.Tensor(primals_3, 1, -50, 9223372036854775807);  primals_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:975 in torch_dynamo_resume_in_forward_at_931, code: suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        convert_element_type: "bf16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:976 in torch_dynamo_resume_in_forward_at_931, code: v_t = self.action_out_proj(suffix_out)
        view: "bf16[1600, 768]" = torch.ops.aten.reshape.default(convert_element_type, [1600, 768]);  convert_element_type = None
        permute: "bf16[768, 55]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default: "bf16[1600, 55]" = torch.ops.aten.mm.default(view, permute);  permute = None
        add_tensor: "bf16[1600, 55]" = torch.ops.aten.add.Tensor(mm_default, primals_5);  mm_default = primals_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:976 in torch_dynamo_resume_in_forward_at_931, code: v_t = self.action_out_proj(suffix_out)
        view_1: "bf16[32, 50, 55]" = torch.ops.aten.reshape.default(add_tensor, [32, 50, 55]);  add_tensor = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:981 in torch_dynamo_resume_in_forward_at_931, code: losses = F.l1_loss(u_t, v_t, reduction="none")
        sub: "bf16[32, 50, 55]" = torch.ops.aten.sub.Tensor(primals_6, view_1);  primals_6 = view_1 = None
        abs_1: "bf16[32, 50, 55]" = torch.ops.aten.abs.default(sub)
        convert_element_type_4: "f32[32, 50, 55]" = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        sign: "f32[32, 50, 55]" = torch.ops.aten.sign.default(convert_element_type_4);  convert_element_type_4 = None
        convert_element_type_5: "bf16[32, 50, 55]" = torch.ops.prims.convert_element_type.default(sign, torch.bfloat16);  sign = None
        return (abs_1, mul, primals_1, mul_1, primals_2, mul_2, add, mul, primals_4, view, convert_element_type_5)
        