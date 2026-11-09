class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 286, 2560]", primals_2: "bf16[32, 337, 4096]", primals_3: "bf16[2560, 4096]", primals_4: "bf16[2560]", primals_5: "bf16[9728, 2560]", primals_6: "bf16[9728, 2560]", primals_7: "bf16[2560, 9728]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:195 in forward, code: out_emb = self.self_attn.o_proj(att_output[:, start:end])
        slice_2: "bf16[32, 286, 4096]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 286)
        permute: "bf16[4096, 2560]" = torch.ops.aten.permute.default(primals_3, [1, 0])
        clone: "bf16[32, 286, 4096]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view: "bf16[9152, 4096]" = torch.ops.aten.reshape.default(clone, [9152, 4096]);  clone = None
        mm: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm, [32, 286, 2560]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:196 in forward, code: out_emb += hidden_states
        add: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:349 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_2: "f32[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:350 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 286, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean: "f32[32, 286, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:351 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_1: "f32[32, 286, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 286, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 286, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = rsqrt = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:352 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_3: "bf16[32, 286, 2560]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 286, 2560]" = torch.ops.aten.mul.Tensor(primals_4, convert_element_type_3);  convert_element_type_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_1: "bf16[2560, 9728]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        view_4: "bf16[9152, 2560]" = torch.ops.aten.reshape.default(mul_1, [9152, 2560]);  mul_1 = None
        mm_1: "bf16[9152, 9728]" = torch.ops.aten.mm.default(view_4, permute_1);  permute_1 = None
        view_5: "bf16[32, 286, 9728]" = torch.ops.aten.reshape.default(mm_1, [32, 286, 9728]);  mm_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        convert_element_type_6: "f32[32, 286, 9728]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        sigmoid: "f32[32, 286, 9728]" = torch.ops.aten.sigmoid.default(convert_element_type_6)
        mul_2: "f32[32, 286, 9728]" = torch.ops.aten.mul.Tensor(convert_element_type_6, sigmoid);  convert_element_type_6 = sigmoid = None
        convert_element_type_7: "bf16[32, 286, 9728]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:472 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_2: "bf16[2560, 9728]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_2: "bf16[9152, 9728]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_7: "bf16[32, 286, 9728]" = torch.ops.aten.reshape.default(mm_2, [32, 286, 9728]);  mm_2 = None
        mul_3: "bf16[32, 286, 9728]" = torch.ops.aten.mul.Tensor(convert_element_type_7, view_7);  convert_element_type_7 = view_7 = None
        permute_3: "bf16[9728, 2560]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        view_8: "bf16[9152, 9728]" = torch.ops.aten.reshape.default(mul_3, [9152, 9728]);  mul_3 = None
        mm_3: "bf16[9152, 2560]" = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None
        view_9: "bf16[32, 286, 2560]" = torch.ops.aten.reshape.default(mm_3, [32, 286, 2560]);  mm_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:200 in forward, code: out_emb += after_first_residual
        add_2: "bf16[32, 286, 2560]" = torch.ops.aten.add.Tensor(view_9, add);  view_9 = add = None
        return (add_2, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)
        