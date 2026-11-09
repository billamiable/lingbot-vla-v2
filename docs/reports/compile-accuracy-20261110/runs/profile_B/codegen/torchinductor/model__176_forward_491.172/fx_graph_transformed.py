class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 51, 768]", primals_2: "f32[768]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:212 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 2)
        mean: "f32[32, 51, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:213 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add: "f32[32, 51, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 51, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_1, rsqrt)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:216 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_1: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_2, mul);  mul = None
        return (mul_1, primals_1, primals_2, rsqrt)
        