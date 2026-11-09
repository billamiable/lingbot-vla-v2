class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 2560]", primals_2: "bf16[2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:410 in _rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:411 in _rms_norm, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[32, 286, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:412 in _rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + eps)
        add: "f32[32, 286, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 286, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:413 in _rms_norm, code: return weight * hidden_states.to(input_dtype)
        convert_element_type_1: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(primals_2, convert_element_type_1);  convert_element_type_1 = None
        return (mul_1, primals_1, primals_2, rsqrt)
        