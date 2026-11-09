class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 51, 768]", primals_2: "f32[768]", rsqrt: "f32[32, 51, 1]", tangents_1: "f32[32, 51, 768]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:216 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_2: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:213 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_1, rsqrt)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:216 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_3: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
        sum_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True);  mul_3 = None
        view: "f32[768]" = torch.ops.aten.view.default(sum_1, [768]);  sum_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:213 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_4: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_1)
        mul_5: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_2, rsqrt);  mul_2 = None
        sum_2: "f32[32, 51, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        pow_2: "f32[32, 51, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_6: "f32[32, 51, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_7: "f32[32, 51, 1]" = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:212 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand: "f32[32, 51, 768]" = torch.ops.aten.expand.default(mul_7, [32, 51, 768]);  mul_7 = None
        div: "f32[32, 51, 768]" = torch.ops.aten.div.Scalar(expand, 768);  expand = None
        pow_3: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 1.0);  primals_1 = None
        mul_8: "f32[32, 51, 768]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_9: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:212 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_1: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        return (add_1, view)
        