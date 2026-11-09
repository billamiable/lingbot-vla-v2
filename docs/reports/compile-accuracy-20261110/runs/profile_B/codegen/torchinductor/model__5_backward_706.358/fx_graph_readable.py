class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024]", primals_3: "bf16[24576, 1024]", primals_4: "bf16[3072, 1024]", primals_6: "f32[24576, 64]", primals_7: "f32[24576, 64]", primals_8: "i32[97]", primals_9: "bf16[1024, 1024]", primals_11: "bf16[1024]", primals_13: "bf16[4096, 1024]", primals_15: "bf16[1024, 4096]", getitem_1: "f32[24576, 1]", rsqrt: "f32[24576, 1]", convert_element_type_1: "bf16[24576, 1024]", view_1: "bf16[24576, 16, 64]", view_2: "bf16[24576, 16, 64]", view_3: "bf16[24576, 16, 64]", getitem_5: "bf16[24576, 16, 64]", getitem_6: "f32[16, 24576]", getitem_8: "i64[2]", addmm_1: "bf16[24576, 1024]", getitem_10: "f32[24576, 1]", rsqrt_1: "f32[24576, 1]", convert_element_type_13: "bf16[24576, 1024]", addmm_2: "bf16[24576, 4096]", convert_element_type_18: "bf16[24576, 4096]", tangents_1: "bf16[24576, 1024]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:56 in forward, code: return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        permute_10: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_11: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm: "bf16[24576, 4096]" = torch.ops.aten.mm.default(tangents_1, permute_11);  permute_11 = None
        permute_12: "bf16[1024, 24576]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_12, convert_element_type_18);  permute_12 = convert_element_type_18 = None
        sum_1: "bf16[1, 1024]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view_6: "bf16[1024]" = torch.ops.aten.view.default(sum_1, [1024]);  sum_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_26: "f32[24576, 4096]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_17: "f32[24576, 4096]" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32);  addmm_2 = None
        mul_8: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_17)
        mul_9: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_8, convert_element_type_17)
        mul_10: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_9, 0.044715);  mul_9 = None
        add_7: "f32[24576, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_17, mul_10);  mul_10 = None
        mul_11: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[24576, 4096]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_8: "f32[24576, 4096]" = torch.ops.aten.add.Tensor(tanh, 1)
        mul_19: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_20: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_2: "f32[24576, 4096]" = torch.ops.aten.sub.Tensor(1, mul_20);  mul_20 = None
        mul_21: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_8, 0.134145);  mul_8 = None
        add_12: "f32[24576, 4096]" = torch.ops.aten.add.Tensor(mul_21, 1);  mul_21 = None
        mul_22: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(add_12, 0.7978845608028654);  add_12 = None
        mul_12: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 0.5);  convert_element_type_17 = None
        mul_23: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_12, sub_2);  mul_12 = sub_2 = None
        mul_24: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_23, mul_22);  mul_23 = mul_22 = None
        add_13: "f32[24576, 4096]" = torch.ops.aten.add.Tensor(mul_19, mul_24);  mul_19 = mul_24 = None
        mul_25: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_26, add_13);  convert_element_type_26 = add_13 = None
        convert_element_type_28: "bf16[24576, 4096]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:56 in forward, code: return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        permute_9: "bf16[1024, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_15: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_2: "bf16[24576, 1024]" = torch.ops.aten.mm.default(convert_element_type_28, permute_15);  permute_15 = None
        permute_16: "bf16[4096, 24576]" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0])
        mm_3: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_16, convert_element_type_13);  permute_16 = convert_element_type_13 = None
        sum_2: "bf16[1, 4096]" = torch.ops.aten.sum.dim_IntList(convert_element_type_28, [0], True);  convert_element_type_28 = None
        view_7: "bf16[4096]" = torch.ops.aten.view.default(sum_2, [4096]);  sum_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:156 in forward, code: hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        convert_element_type_33: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_35: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_11, torch.float32);  primals_11 = None
        mul_27: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_33, convert_element_type_35);  convert_element_type_35 = None
        mul_28: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_27, 1024)
        sum_3: "f32[24576, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [1], True)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:149 in forward, code: hidden_states = hidden_states + self.attn(
        add_4: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(primals_3, addmm_1);  addmm_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:156 in forward, code: hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        convert_element_type_12: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        sub_1: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_10);  convert_element_type_12 = getitem_10 = None
        mul_6: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_29: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_27, mul_6);  mul_27 = None
        sum_4: "f32[24576, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [1], True);  mul_29 = None
        mul_30: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_6, sum_4);  sum_4 = None
        sub_4: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(mul_28, sum_3);  mul_28 = sum_3 = None
        sub_5: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_30);  sub_4 = mul_30 = None
        div: "f32[24576, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_31: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(div, sub_5);  div = sub_5 = None
        mul_32: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_33, mul_6);  mul_6 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_32, [0]);  mul_32 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_33, [0]);  convert_element_type_33 = None
        convert_element_type_37: "bf16[24576, 1024]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        convert_element_type_38: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        convert_element_type_39: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:156 in forward, code: hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        add_14: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(tangents_1, convert_element_type_37);  tangents_1 = convert_element_type_37 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:129 in forward, code: attn_output = self.proj(attn_output)
        permute_8: "bf16[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_19: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_4: "bf16[24576, 1024]" = torch.ops.aten.mm.default(add_14, permute_19);  permute_19 = None
        permute_20: "bf16[1024, 24576]" = torch.ops.aten.permute.default(add_14, [1, 0])
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:660 in _flash_attention_forward, code: out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))
        view_4: "bf16[1, 24576, 16, 64]" = torch.ops.aten.view.default(getitem_5, [1, -1, 16, 64])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:128 in forward, code: attn_output = attn_output.reshape(seq_length, -1).contiguous()
        view_5: "bf16[24576, 1024]" = torch.ops.aten.view.default(view_4, [24576, -1]);  view_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:129 in forward, code: attn_output = self.proj(attn_output)
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_20, view_5);  permute_20 = view_5 = None
        sum_7: "bf16[1, 1024]" = torch.ops.aten.sum.dim_IntList(add_14, [0], True)
        view_8: "bf16[1024]" = torch.ops.aten.view.default(sum_7, [1024]);  sum_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:128 in forward, code: attn_output = attn_output.reshape(seq_length, -1).contiguous()
        view_9: "bf16[1, 24576, 16, 64]" = torch.ops.aten.view.default(mm_4, [1, 24576, 16, 64]);  mm_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:660 in _flash_attention_forward, code: out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))
        view_10: "bf16[24576, 16, 64]" = torch.ops.aten.view.default(view_9, [24576, 16, 64]);  view_9 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py:1443 in flash_attn_varlen_func, code: return FlashAttnVarlenFunc.apply(
        empty: "bf16[24576, 16, 64]" = torch.ops.aten.empty.memory_format([24576, 16, 64], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_23: "bf16[24576, 16, 64]" = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = None
        empty_1: "bf16[24576, 16, 64]" = torch.ops.aten.empty.memory_format([24576, 16, 64], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_24: "bf16[24576, 16, 64]" = torch.ops.aten.permute.default(empty_1, [0, 1, 2]);  empty_1 = None
        empty_2: "bf16[24576, 16, 64]" = torch.ops.aten.empty.memory_format([24576, 16, 64], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25: "bf16[24576, 16, 64]" = torch.ops.aten.permute.default(empty_2, [0, 1, 2]);  empty_2 = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.flash_attn._flash_attn_varlen_backward.default, dout = view_10, q = view_1, k = view_2, v = view_3, out = getitem_5, softmax_lse = getitem_6, cu_seqlens_q = primals_8, cu_seqlens_k = primals_8, max_seqlen_q = 256, max_seqlen_k = 256, dropout_p = 0.0, softmax_scale = 0.125, causal = False, window_size_left = -1, window_size_right = -1, softcap = 0.0, alibi_slopes = None, deterministic = False, rng_state = getitem_8, zero_tensors = False, _dq_base_index = 0, _dk_base_index = 1, _dv_base_index = 2, _all_bases = [permute_23, permute_24, permute_25]);  view_10 = view_1 = view_2 = view_3 = getitem_5 = getitem_6 = primals_8 = getitem_8 = permute_23 = permute_24 = permute_25 = None
        getitem_12: "bf16[24576, 16, 64]" = auto_functionalized_v2[1]
        getitem_13: "bf16[24576, 16, 64]" = auto_functionalized_v2[2]
        getitem_14: "bf16[24576, 16, 64]" = auto_functionalized_v2[3];  auto_functionalized_v2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:126 in apply_rotary_pos_emb_vision, code: k_embed = k_embed.to(orig_k_dtype)
        view_14: "bf16[1, 24576, 16, 64]" = torch.ops.aten.view.default(getitem_13, [1, 24576, 16, 64]);  getitem_13 = None
        permute_32: "bf16[1, 16, 24576, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        squeeze_3: "bf16[16, 24576, 64]" = torch.ops.aten.squeeze.dim(permute_32, 0);  permute_32 = None
        permute_33: "bf16[24576, 16, 64]" = torch.ops.aten.permute.default(squeeze_3, [1, 0, 2]);  squeeze_3 = None
        convert_element_type_44: "f32[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(permute_33, torch.float32);  permute_33 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:125 in apply_rotary_pos_emb_vision, code: q_embed = q_embed.to(orig_q_dtype)
        view_15: "bf16[1, 24576, 16, 64]" = torch.ops.aten.view.default(getitem_12, [1, 24576, 16, 64]);  getitem_12 = None
        permute_34: "bf16[1, 16, 24576, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        squeeze_4: "bf16[16, 24576, 64]" = torch.ops.aten.squeeze.dim(permute_34, 0);  permute_34 = None
        permute_35: "bf16[24576, 16, 64]" = torch.ops.aten.permute.default(squeeze_4, [1, 0, 2]);  squeeze_4 = None
        convert_element_type_45: "f32[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(permute_35, torch.float32);  permute_35 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:122 in apply_rotary_pos_emb_vision, code: cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        unsqueeze_1: "f32[24576, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_7, -2);  primals_7 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:124 in apply_rotary_pos_emb_vision, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_33: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_44, unsqueeze_1)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_5: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(mul_33, 2, 0, 32)
        slice_6: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(mul_33, 2, 32, 64);  mul_33 = None
        neg_2: "f32[24576, 16, 32]" = torch.ops.aten.neg.default(slice_5);  slice_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default: "f32[24576, 16, 64]" = torch.ops.aten.full.default([24576, 16, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "f32[24576, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, neg_2, 2, 32, 9223372036854775807);  neg_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_1: "f32[24576, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, slice_6, 2, 0, 32);  slice_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        add_15: "f32[24576, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:122 in apply_rotary_pos_emb_vision, code: cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        unsqueeze: "f32[24576, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_6, -2);  primals_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:124 in apply_rotary_pos_emb_vision, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_34: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_44, unsqueeze);  convert_element_type_44 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:124 in apply_rotary_pos_emb_vision, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        add_16: "f32[24576, 16, 64]" = torch.ops.aten.add.Tensor(add_15, mul_34);  add_15 = mul_34 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:123 in apply_rotary_pos_emb_vision, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_35: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_45, unsqueeze_1);  unsqueeze_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_7: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(mul_35, 2, 0, 32)
        slice_8: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(mul_35, 2, 32, 64);  mul_35 = None
        neg_3: "f32[24576, 16, 32]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_2: "f32[24576, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, neg_3, 2, 32, 9223372036854775807);  neg_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_3: "f32[24576, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, slice_8, 2, 0, 32);  full_default = slice_8 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        add_17: "f32[24576, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:123 in apply_rotary_pos_emb_vision, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_36: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_45, unsqueeze);  convert_element_type_45 = unsqueeze = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:123 in apply_rotary_pos_emb_vision, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        add_18: "f32[24576, 16, 64]" = torch.ops.aten.add.Tensor(add_17, mul_36);  add_17 = mul_36 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:121 in apply_rotary_pos_emb_vision, code: q, k = q.float(), k.float()
        convert_element_type_46: "bf16[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        convert_element_type_47: "bf16[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:68 in forward, code: self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        view_16: "bf16[1, 24576, 16, 64]" = torch.ops.aten.view.default(getitem_14, [1, 24576, 16, 64]);  getitem_14 = None
        permute_36: "bf16[1, 16, 24576, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        squeeze_5: "bf16[16, 24576, 64]" = torch.ops.aten.squeeze.dim(permute_36, 0);  permute_36 = None
        permute_37: "bf16[24576, 16, 64]" = torch.ops.aten.permute.default(squeeze_5, [1, 0, 2]);  squeeze_5 = None
        cat_2: "bf16[73728, 16, 64]" = torch.ops.aten.cat.default([convert_element_type_47, convert_element_type_46, permute_37]);  convert_element_type_47 = convert_element_type_46 = permute_37 = None
        view_17: "bf16[3, 24576, 16, 64]" = torch.ops.aten.view.default(cat_2, [3, 24576, 16, 64]);  cat_2 = None
        permute_38: "bf16[24576, 3, 16, 64]" = torch.ops.aten.permute.default(view_17, [1, 0, 2, 3]);  view_17 = None
        clone_4: "bf16[24576, 3, 16, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_18: "bf16[24576, 3072]" = torch.ops.aten.view.default(clone_4, [24576, 3072]);  clone_4 = None
        permute: "bf16[1024, 3072]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_39: "bf16[3072, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_6: "bf16[24576, 1024]" = torch.ops.aten.mm.default(view_18, permute_39);  permute_39 = None
        permute_40: "bf16[3072, 24576]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_7: "bf16[3072, 1024]" = torch.ops.aten.mm.default(permute_40, convert_element_type_1);  permute_40 = convert_element_type_1 = None
        sum_8: "bf16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_19: "bf16[3072]" = torch.ops.aten.view.default(sum_8, [3072]);  sum_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:150 in forward, code: self.norm1(hidden_states),
        convert_element_type_52: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_53: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float32);  primals_3 = None
        convert_element_type_54: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        sub_6: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_53, getitem_1);  convert_element_type_53 = getitem_1 = None
        mul_37: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
        mul_38: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, convert_element_type_54);  convert_element_type_54 = None
        mul_39: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_38, 1024)
        sum_9: "f32[24576, 1]" = torch.ops.aten.sum.dim_IntList(mul_38, [1], True)
        mul_40: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_38, mul_37);  mul_38 = None
        sum_10: "f32[24576, 1]" = torch.ops.aten.sum.dim_IntList(mul_40, [1], True);  mul_40 = None
        mul_41: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_37, sum_10);  sum_10 = None
        sub_7: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(mul_39, sum_9);  mul_39 = sum_9 = None
        sub_8: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(sub_7, mul_41);  sub_7 = mul_41 = None
        div_1: "f32[24576, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_42: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_8);  div_1 = sub_8 = None
        mul_43: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, mul_37);  mul_37 = None
        sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_43, [0]);  mul_43 = None
        sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_52, [0]);  convert_element_type_52 = None
        convert_element_type_56: "bf16[24576, 1024]" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        convert_element_type_57: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_11, torch.bfloat16);  sum_11 = None
        convert_element_type_58: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_12, torch.bfloat16);  sum_12 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:150 in forward, code: self.norm1(hidden_states),
        add_19: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_56);  add_14 = convert_element_type_56 = None
        return (convert_element_type_57, convert_element_type_58, add_19, mm_7, view_19, None, None, None, mm_5, view_8, convert_element_type_38, convert_element_type_39, mm_3, view_7, mm_1, view_6)
        