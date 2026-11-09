class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 2560]", primals_2: "bf16[4096, 2560]", primals_3: "bf16[2560]", primals_4: "bf16[128]", primals_5: "bf16[1024, 2560]", primals_6: "bf16[128]", primals_7: "bf16[1024, 2560]", tangents_1: "bf16[32, 286, 32, 128]", tangents_2: "bf16[32, 286, 8, 128]", tangents_3: "bf16[32, 286, 8, 128]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:189 in forward, code: value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        view_9: "bf16[32, 286, 1024]" = torch.ops.aten.reshape.default(tangents_3, [32, 286, 1024]);  tangents_3 = None
        view_10: "bf16[9152, 1024]" = torch.ops.aten.reshape.default(view_9, [9152, 1024]);  view_9 = None
        permute_3: "bf16[1024, 9152]" = torch.ops.aten.permute.default(view_10, [1, 0])
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[32, 286, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add: "f32[32, 286, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 286, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_1: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(primals_3, convert_element_type_1)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:187 in forward, code: query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
        view: "bf16[9152, 2560]" = torch.ops.aten.reshape.default(mul_1, [9152, 2560]);  mul_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:189 in forward, code: value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        mm_3: "bf16[1024, 2560]" = torch.ops.aten.mm.default(permute_3, view);  permute_3 = None
        permute_2: "bf16[2560, 1024]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_5: "bf16[1024, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_4: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm_4, [32, 286, 2560]);  mm_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_6: "bf16[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(tangents_2, primals_6);  primals_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:188 in forward, code: key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
        permute_1: "bf16[2560, 1024]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        mm_1: "bf16[9152, 1024]" = torch.ops.aten.mm.default(view, permute_1)
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
        mul_4: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_2)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_9: "bf16[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_7: "bf16[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(tangents_2, convert_element_type_9);  tangents_2 = convert_element_type_9 = None
        sum_1: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1, 2], True);  mul_7 = None
        view_12: "bf16[128]" = torch.ops.aten.reshape.default(sum_1, [128]);  sum_1 = None
        convert_element_type_16: "f32[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float32);  mul_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_8: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_16, convert_element_type_8)
        mul_9: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_16, rsqrt_2);  convert_element_type_16 = None
        sum_2: "f32[32, 286, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [3], True);  mul_8 = None
        pow_4: "f32[32, 286, 8, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_10: "f32[32, 286, 8, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_11: "f32[32, 286, 8, 1]" = torch.ops.aten.mul.Tensor(mul_10, pow_4);  mul_10 = pow_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand: "f32[32, 286, 8, 128]" = torch.ops.aten.expand.default(mul_11, [32, 286, 8, 128]);  mul_11 = None
        div: "f32[32, 286, 8, 128]" = torch.ops.aten.div.Scalar(expand, 128);  expand = None
        pow_5: "f32[32, 286, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 1.0);  convert_element_type_8 = None
        mul_12: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Scalar(pow_5, 2.0);  pow_5 = None
        mul_13: "f32[32, 286, 8, 128]" = torch.ops.aten.mul.Tensor(div, mul_12);  div = mul_12 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_3: "f32[32, 286, 8, 128]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_17: "bf16[32, 286, 8, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:188 in forward, code: key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
        view_13: "bf16[32, 286, 1024]" = torch.ops.aten.reshape.default(convert_element_type_17, [32, 286, 1024]);  convert_element_type_17 = None
        view_14: "bf16[9152, 1024]" = torch.ops.aten.reshape.default(view_13, [9152, 1024]);  view_13 = None
        permute_7: "bf16[1024, 9152]" = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_5: "bf16[1024, 2560]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = None
        permute_9: "bf16[1024, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_6: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view_14, permute_9);  view_14 = permute_9 = None
        view_15: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm_6, [32, 286, 2560]);  mm_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:188 in forward, code: key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
        add_4: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(view_11, view_15);  view_11 = view_15 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_14: "bf16[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(tangents_1, primals_4);  primals_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:187 in forward, code: query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
        permute: "bf16[2560, 4096]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        mm: "bf16[9152, 4096]" = torch.ops.aten.mm.default(view, permute)
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
        mul_2: "f32[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt_1)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_5: "bf16[32, 286, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        mul_15: "bf16[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        sum_3: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_15, [0, 1, 2], True);  mul_15 = None
        view_16: "bf16[128]" = torch.ops.aten.reshape.default(sum_3, [128]);  sum_3 = None
        convert_element_type_22: "f32[32, 286, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_14, torch.float32);  mul_14 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_16: "f32[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_22, convert_element_type_4)
        mul_17: "f32[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_22, rsqrt_1);  convert_element_type_22 = None
        sum_4: "f32[32, 286, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [3], True);  mul_16 = None
        pow_6: "f32[32, 286, 32, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_18: "f32[32, 286, 32, 1]" = torch.ops.aten.mul.Scalar(sum_4, -0.5);  sum_4 = None
        mul_19: "f32[32, 286, 32, 1]" = torch.ops.aten.mul.Tensor(mul_18, pow_6);  mul_18 = pow_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_1: "f32[32, 286, 32, 128]" = torch.ops.aten.expand.default(mul_19, [32, 286, 32, 128]);  mul_19 = None
        div_1: "f32[32, 286, 32, 128]" = torch.ops.aten.div.Scalar(expand_1, 128);  expand_1 = None
        pow_7: "f32[32, 286, 32, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_20: "f32[32, 286, 32, 128]" = torch.ops.aten.mul.Scalar(pow_7, 2.0);  pow_7 = None
        mul_21: "f32[32, 286, 32, 128]" = torch.ops.aten.mul.Tensor(div_1, mul_20);  div_1 = mul_20 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_5: "f32[32, 286, 32, 128]" = torch.ops.aten.add.Tensor(mul_17, mul_21);  mul_17 = mul_21 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_23: "bf16[32, 286, 32, 128]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:187 in forward, code: query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
        view_17: "bf16[32, 286, 4096]" = torch.ops.aten.reshape.default(convert_element_type_23, [32, 286, 4096]);  convert_element_type_23 = None
        view_18: "bf16[9152, 4096]" = torch.ops.aten.reshape.default(view_17, [9152, 4096]);  view_17 = None
        permute_11: "bf16[4096, 9152]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_7: "bf16[4096, 2560]" = torch.ops.aten.mm.default(permute_11, view);  permute_11 = view = None
        permute_13: "bf16[4096, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_8: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view_18, permute_13);  view_18 = permute_13 = None
        view_19: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm_8, [32, 286, 2560]);  mm_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:187 in forward, code: query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
        add_6: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_22: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(add_6, primals_3);  primals_3 = None
        mul_23: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(add_6, convert_element_type_1);  add_6 = convert_element_type_1 = None
        sum_5: "bf16[1, 1, 2560]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1], True);  mul_23 = None
        view_20: "bf16[2560]" = torch.ops.aten.reshape.default(sum_5, [2560]);  sum_5 = None
        convert_element_type_28: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul_22, torch.float32);  mul_22 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_24: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_28, convert_element_type)
        mul_25: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_28, rsqrt);  convert_element_type_28 = None
        sum_6: "f32[32, 286, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        pow_8: "f32[32, 286, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_26: "f32[32, 286, 1]" = torch.ops.aten.mul.Scalar(sum_6, -0.5);  sum_6 = None
        mul_27: "f32[32, 286, 1]" = torch.ops.aten.mul.Tensor(mul_26, pow_8);  mul_26 = pow_8 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_2: "f32[32, 286, 2560]" = torch.ops.aten.expand.default(mul_27, [32, 286, 2560]);  mul_27 = None
        div_2: "f32[32, 286, 2560]" = torch.ops.aten.div.Scalar(expand_2, 2560);  expand_2 = None
        pow_9: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 1.0);  convert_element_type = None
        mul_28: "f32[32, 286, 2560]" = torch.ops.aten.mul.Scalar(pow_9, 2.0);  pow_9 = None
        mul_29: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(div_2, mul_28);  div_2 = mul_28 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_7: "f32[32, 286, 2560]" = torch.ops.aten.add.Tensor(mul_25, mul_29);  mul_25 = mul_29 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_29: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        return (convert_element_type_29, mm_7, view_20, view_16, mm_5, view_12, mm_3)
        