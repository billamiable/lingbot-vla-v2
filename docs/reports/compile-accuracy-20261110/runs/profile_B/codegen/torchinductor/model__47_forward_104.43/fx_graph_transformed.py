class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 2560]", primals_2: "bf16[4096, 2560]", primals_3: "bf16[2560]", primals_4: "bf16[128]", primals_5: "bf16[1024, 2560]", primals_6: "bf16[128]", primals_7: "bf16[1024, 2560]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[32, 286, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add: "f32[32, 286, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 286, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_1: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(primals_3, convert_element_type_1);  convert_element_type_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:187 in forward, code: query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
        permute: "bf16[2560, 4096]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view: "bf16[9152, 2560]" = torch.ops.aten.reshape.default(mul_1, [9152, 2560]);  mul_1 = None
        mm: "bf16[9152, 4096]" = torch.ops.aten.mm.default(view, permute);  permute = None
        view_1: "bf16[32, 286, 4096]" = torch.ops.aten.reshape.default(mm, [32, 286, 4096]);  mm = None
        view_2: "bf16[32, 286, 32, 128]" = torch.ops.aten.reshape.default(view_1, [32, 286, -1, 128]);  view_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_4: "f32[32, 286, 32, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_2: "f32[32, 286, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
        mean_1: "f32[32, 286, 32, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_1: "f32[32, 286, 32, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[32, 286, 32, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_2: "f32[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt_1);  convert_element_type_4 = rsqrt_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_5: "bf16[32, 286, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        mul_3: "bf16[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(primals_4, convert_element_type_5);  convert_element_type_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:188 in forward, code: key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
        permute_1: "bf16[2560, 1024]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        mm_1: "bf16[9152, 1024]" = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        view_4: "bf16[32, 286, 1024]" = torch.ops.aten.reshape.default(mm_1, [32, 286, 1024]);  mm_1 = None
        view_5: "bf16[32, 286, 8, 128]" = torch.ops.aten.reshape.default(view_4, [32, 286, -1, 128]);  view_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_8: "f32[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_3: "f32[32, 286, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2)
        mean_2: "f32[32, 286, 8, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[32, 286, 8, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[32, 286, 8, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_2);  convert_element_type_8 = rsqrt_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_9: "bf16[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5: "bf16[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(primals_6, convert_element_type_9);  convert_element_type_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:189 in forward, code: value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        permute_2: "bf16[2560, 1024]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        mm_2: "bf16[9152, 1024]" = torch.ops.aten.mm.default(view, permute_2);  view = permute_2 = None
        view_7: "bf16[32, 286, 1024]" = torch.ops.aten.reshape.default(mm_2, [32, 286, 1024]);  mm_2 = None
        view_8: "bf16[32, 286, 8, 128]" = torch.ops.aten.reshape.default(view_7, [32, 286, -1, 128]);  view_7 = None
        return (mul_3, mul_5, view_8, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)
        