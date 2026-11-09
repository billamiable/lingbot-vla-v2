class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", tangents_2: "f32[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:915 in torch_dynamo_resume_in_forward_at_913, code: loss_future_depth = loss_future_depth * self.config.align_params.get("future_depth_loss_weight", 1.0)
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(tangents_2, 0.004);  tangents_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:914 in torch_dynamo_resume_in_forward_at_913, code: loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
        mul_3: "f32[]" = torch.ops.aten.mul.Tensor(tangents_1, 0.004);  tangents_1 = None
        return (mul_3, mul_2)
        