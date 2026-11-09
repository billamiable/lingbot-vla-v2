class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 8, 128]", primals_2: "bf16[128]", rsqrt: "f32[32, 286, 8, 1]", tangents_1: "bf16[32, 286, 8, 128]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:413 in _rms_norm, code: return weight * hidden_states.to(input_dtype)
        mul_2: "bf16[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:410 in _rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type: "f32[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:412 in _rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + eps)
        mul: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:413 in _rms_norm, code: return weight * hidden_states.to(input_dtype)
        convert_element_type_1: "bf16[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_3: "bf16[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_1);  tangents_1 = convert_element_type_1 = None
        sum_1: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1, 2], True);  mul_3 = None
        view: "bf16[128]" = torch.ops.aten.view.default(sum_1, [128]);  sum_1 = None
        convert_element_type_2: "f32[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float32);  mul_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:412 in _rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + eps)
        mul_4: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type)
        mul_5: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = None
        sum_2: "f32[32, 286, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [3], True);  mul_4 = None
        pow_2: "f32[32, 286, 8, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_6: "f32[32, 286, 8, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_7: "f32[32, 286, 8, 1]" = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:411 in _rms_norm, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand: "f32[32, 286, 8, 128]" = torch.ops.aten.expand.default(mul_7, [32, 286, 8, 128]);  mul_7 = None
        div: "f32[32, 286, 8, 128]" = torch.ops.aten.div.Scalar(expand, 128);  expand = None
        pow_3: "f32[32, 286, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 1.0);  convert_element_type = None
        mul_8: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_9: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:411 in _rms_norm, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_1: "f32[32, 286, 8, 128]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:410 in _rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "bf16[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        return (convert_element_type_3, view)
        