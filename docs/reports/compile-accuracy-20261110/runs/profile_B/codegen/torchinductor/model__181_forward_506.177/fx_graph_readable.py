class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[]", primals_2: "f32[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:914 in torch_dynamo_resume_in_forward_at_913, code: loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
        mul: "f32[]" = torch.ops.aten.mul.Tensor(primals_1, 0.004);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:915 in torch_dynamo_resume_in_forward_at_913, code: loss_future_depth = loss_future_depth * self.config.align_params.get("future_depth_loss_weight", 1.0)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(primals_2, 0.004);  primals_2 = None
        return (mul, mul_1)
        