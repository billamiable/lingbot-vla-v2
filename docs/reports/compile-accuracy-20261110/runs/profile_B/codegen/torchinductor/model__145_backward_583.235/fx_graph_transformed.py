class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 2560]", primals_2: "bf16[32, 337, 4096]", primals_3: "bf16[2560, 4096]", primals_4: "bf16[2560]", primals_5: "bf16[9728, 2560]", primals_6: "bf16[9728, 2560]", primals_7: "bf16[2560, 9728]", tangents_1: "bf16[32, 286, 2560]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_12: "bf16[9152, 2560]" = torch.ops.aten.reshape.default(tangents_1, [9152, 2560])
        permute_4: "bf16[2560, 9152]" = torch.ops.aten.permute.default(view_12, [1, 0])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:195 in forward, code: out_emb = self.self_attn.o_proj(att_output[:, start:end])
        slice_2: "bf16[32, 286, 4096]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 286);  primals_2 = None
        permute: "bf16[4096, 2560]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        clone: "bf16[32, 286, 4096]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view: "bf16[9152, 4096]" = torch.ops.aten.reshape.default(clone, [9152, 4096]);  clone = None
        mm: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm, [32, 286, 2560]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:196 in forward, code: out_emb += hidden_states
        add: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_2: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean: "f32[32, 286, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_1: "f32[32, 286, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 286, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_3: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(primals_4, convert_element_type_3)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_1: "bf16[2560, 9728]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        view_4: "bf16[9152, 2560]" = torch.ops.aten.reshape.default(mul_1, [9152, 2560]);  mul_1 = None
        mm_1: "bf16[9152, 9728]" = torch.ops.aten.mm.default(view_4, permute_1)
        view_5: "bf16[32, 286, 9728]" = torch.ops.aten.reshape.default(mm_1, [32, 286, 9728]);  mm_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        convert_element_type_6: "f32[32, 286, 9728]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32)
        sigmoid: "f32[32, 286, 9728]" = torch.ops.aten.sigmoid.default(convert_element_type_6)
        mul_2: "f32[32, 286, 9728]" = torch.ops.aten.mul.Tensor(convert_element_type_6, sigmoid);  convert_element_type_6 = sigmoid = None
        convert_element_type_7: "bf16[32, 286, 9728]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_2: "bf16[2560, 9728]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        mm_2: "bf16[9152, 9728]" = torch.ops.aten.mm.default(view_4, permute_2)
        view_7: "bf16[32, 286, 9728]" = torch.ops.aten.reshape.default(mm_2, [32, 286, 9728]);  mm_2 = None
        mul_3: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(convert_element_type_7, view_7)
        view_8: "bf16[9152, 9728]" = torch.ops.aten.reshape.default(mul_3, [9152, 9728]);  mul_3 = None
        mm_4: "bf16[2560, 9728]" = torch.ops.aten.mm.default(permute_4, view_8);  permute_4 = view_8 = None
        permute_3: "bf16[9728, 2560]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_6: "bf16[2560, 9728]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_5: "bf16[9152, 9728]" = torch.ops.aten.mm.default(view_12, permute_6);  view_12 = permute_6 = None
        view_13: "bf16[32, 286, 9728]" = torch.ops.aten.reshape.default(mm_5, [32, 286, 9728]);  mm_5 = None
        mul_4: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(view_13, convert_element_type_7);  convert_element_type_7 = None
        mul_5: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(view_13, view_7);  view_13 = view_7 = None
        view_14: "bf16[9152, 9728]" = torch.ops.aten.reshape.default(mul_4, [9152, 9728]);  mul_4 = None
        permute_8: "bf16[9728, 9152]" = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_6: "bf16[9728, 2560]" = torch.ops.aten.mm.default(permute_8, view_4);  permute_8 = None
        permute_10: "bf16[9728, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_7: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view_14, permute_10);  view_14 = permute_10 = None
        view_15: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm_7, [32, 286, 2560]);  mm_7 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        sigmoid_1: "bf16[32, 286, 9728]" = torch.ops.aten.sigmoid.default(view_5)
        full_default: "bf16[32, 286, 9728]" = torch.ops.aten.full.default([32, 286, 9728], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub: "bf16[32, 286, 9728]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_1);  full_default = None
        mul_6: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(view_5, sub);  view_5 = sub = None
        add_3: "bf16[32, 286, 9728]" = torch.ops.aten.add.Scalar(mul_6, 1);  mul_6 = None
        mul_7: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(sigmoid_1, add_3);  sigmoid_1 = add_3 = None
        mul_8: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(mul_5, mul_7);  mul_5 = mul_7 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_16: "bf16[9152, 9728]" = torch.ops.aten.reshape.default(mul_8, [9152, 9728]);  mul_8 = None
        permute_13: "bf16[9728, 9152]" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_8: "bf16[9728, 2560]" = torch.ops.aten.mm.default(permute_13, view_4);  permute_13 = view_4 = None
        permute_15: "bf16[9728, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_9: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view_16, permute_15);  view_16 = permute_15 = None
        view_17: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm_9, [32, 286, 2560]);  mm_9 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        add_4: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(view_15, view_17);  view_15 = view_17 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_9: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(add_4, primals_4);  primals_4 = None
        mul_10: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(add_4, convert_element_type_3);  add_4 = convert_element_type_3 = None
        sum_1: "bf16[1, 1, 2560]" = torch.ops.aten.sum.dim_IntList(mul_10, [0, 1], True);  mul_10 = None
        view_18: "bf16[2560]" = torch.ops.aten.reshape.default(sum_1, [2560]);  sum_1 = None
        convert_element_type_24: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul_9, torch.float32);  mul_9 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_11: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_24, convert_element_type_2)
        mul_12: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt);  convert_element_type_24 = None
        sum_2: "f32[32, 286, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [2], True);  mul_11 = None
        pow_2: "f32[32, 286, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_13: "f32[32, 286, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_14: "f32[32, 286, 1]" = torch.ops.aten.mul.Tensor(mul_13, pow_2);  mul_13 = pow_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand: "f32[32, 286, 2560]" = torch.ops.aten.expand.default(mul_14, [32, 286, 2560]);  mul_14 = None
        div: "f32[32, 286, 2560]" = torch.ops.aten.div.Scalar(expand, 2560);  expand = None
        pow_3: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 1.0);  convert_element_type_2 = None
        mul_15: "f32[32, 286, 2560]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_16: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(div, mul_15);  div = mul_15 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_5: "f32[32, 286, 2560]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_25: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:197 in forward, code: after_first_residual = out_emb.clone()
        add_6: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(convert_element_type_25, tangents_1);  convert_element_type_25 = tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:195 in forward, code: out_emb = self.self_attn.o_proj(att_output[:, start:end])
        view_19: "bf16[9152, 2560]" = torch.ops.aten.reshape.default(add_6, [9152, 2560])
        permute_17: "bf16[2560, 9152]" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_10: "bf16[2560, 4096]" = torch.ops.aten.mm.default(permute_17, view);  permute_17 = view = None
        permute_19: "bf16[2560, 4096]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_11: "bf16[9152, 4096]" = torch.ops.aten.mm.default(view_19, permute_19);  view_19 = permute_19 = None
        view_20: "bf16[32, 286, 4096]" = torch.ops.aten.reshape.default(mm_11, [32, 286, 4096]);  mm_11 = None
        full_default_1: "bf16[32, 337, 4096]" = torch.ops.aten.full.default([32, 337, 4096], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default: "bf16[32, 337, 4096]" = torch.ops.aten.slice_scatter.default(full_default_1, view_20, 1, 0, 286);  full_default_1 = view_20 = None
        return (add_6, slice_scatter_default, mm_10, view_18, mm_8, mm_6, mm_4)
        