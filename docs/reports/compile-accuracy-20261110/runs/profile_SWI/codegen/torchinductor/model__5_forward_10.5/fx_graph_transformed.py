class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024]", primals_2: "bf16[1024]", primals_3: "bf16[24576, 1024]", primals_4: "bf16[3072, 1024]", primals_5: "bf16[3072]", primals_6: "f32[24576, 64]", primals_7: "f32[24576, 64]", primals_8: "i32[97]", primals_9: "bf16[1024, 1024]", primals_10: "bf16[1024]", primals_11: "bf16[1024]", primals_12: "bf16[1024]", primals_13: "bf16[4096, 1024]", primals_14: "bf16[4096]", primals_15: "bf16[1024, 4096]", primals_16: "bf16[1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:150 in forward, code: self.norm1(hidden_states),
        convert_element_type: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[24576, 1]" = var_mean[0]
        getitem_1: "f32[24576, 1]" = var_mean[1];  var_mean = None
        add: "f32[24576, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[24576, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[24576, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
        convert_element_type_1: "bf16[24576, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:68 in forward, code: self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        permute: "bf16[1024, 3072]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm: "bf16[24576, 3072]" = torch.ops.aten.addmm.default(primals_5, convert_element_type_1, permute);  primals_5 = permute = None
        view: "bf16[24576, 3, 16, 64]" = torch.ops.aten.reshape.default(addmm, [24576, 3, 16, -1]);  addmm = None
        permute_1: "bf16[3, 24576, 16, 64]" = torch.ops.aten.permute.default(view, [1, 0, 2, 3]);  view = None
        unbind = torch.ops.aten.unbind.int(permute_1);  permute_1 = None
        getitem_2: "bf16[24576, 16, 64]" = unbind[0]
        getitem_3: "bf16[24576, 16, 64]" = unbind[1]
        getitem_4: "bf16[24576, 16, 64]" = unbind[2];  unbind = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:121 in apply_rotary_pos_emb_vision, code: q, k = q.float(), k.float()
        convert_element_type_5: "f32[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        convert_element_type_6: "f32[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:122 in apply_rotary_pos_emb_vision, code: cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        unsqueeze: "f32[24576, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_6, -2)
        unsqueeze_1: "f32[24576, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_7, -2)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:123 in apply_rotary_pos_emb_vision, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_2: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_1: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_5, 2, 0, 32)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_2: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_5, 2, 32, 9223372036854775807);  convert_element_type_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "f32[24576, 16, 32]" = torch.ops.aten.neg.default(slice_2);  slice_2 = None
        cat: "f32[24576, 16, 64]" = torch.ops.aten.cat.default([neg, slice_1], -1);  neg = slice_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:123 in apply_rotary_pos_emb_vision, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_3: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = None
        add_2: "f32[24576, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:124 in apply_rotary_pos_emb_vision, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_4: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze);  unsqueeze = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_3: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_6, 2, 0, 32)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_4: "f32[24576, 16, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_6, 2, 32, 9223372036854775807);  convert_element_type_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "f32[24576, 16, 32]" = torch.ops.aten.neg.default(slice_4);  slice_4 = None
        cat_1: "f32[24576, 16, 64]" = torch.ops.aten.cat.default([neg_1, slice_3], -1);  neg_1 = slice_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:124 in apply_rotary_pos_emb_vision, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_5: "f32[24576, 16, 64]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = unsqueeze_1 = None
        add_3: "f32[24576, 16, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:125 in apply_rotary_pos_emb_vision, code: q_embed = q_embed.to(orig_q_dtype)
        convert_element_type_7: "bf16[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:126 in apply_rotary_pos_emb_vision, code: k_embed = k_embed.to(orig_k_dtype)
        convert_element_type_8: "bf16[24576, 16, 64]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:73 in forward, code: query_states = query_states.transpose(0, 1).unsqueeze(0)
        permute_2: "bf16[16, 24576, 64]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0, 2]);  convert_element_type_7 = None
        unsqueeze_2: "bf16[1, 16, 24576, 64]" = torch.ops.aten.unsqueeze.default(permute_2, 0);  permute_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:74 in forward, code: key_states = key_states.transpose(0, 1).unsqueeze(0)
        permute_3: "bf16[16, 24576, 64]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0, 2]);  convert_element_type_8 = None
        unsqueeze_3: "bf16[1, 16, 24576, 64]" = torch.ops.aten.unsqueeze.default(permute_3, 0);  permute_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:75 in forward, code: value_states = value_states.transpose(0, 1).unsqueeze(0)
        permute_4: "bf16[16, 24576, 64]" = torch.ops.aten.permute.default(getitem_4, [1, 0, 2]);  getitem_4 = None
        unsqueeze_4: "bf16[1, 16, 24576, 64]" = torch.ops.aten.unsqueeze.default(permute_4, 0);  permute_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/integrations/flash_attention.py:42 in flash_attention_forward, code: query = query.transpose(1, 2)
        permute_5: "bf16[1, 24576, 16, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 2, 1, 3]);  unsqueeze_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/integrations/flash_attention.py:43 in flash_attention_forward, code: key = key.transpose(1, 2)
        permute_6: "bf16[1, 24576, 16, 64]" = torch.ops.aten.permute.default(unsqueeze_3, [0, 2, 1, 3]);  unsqueeze_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/integrations/flash_attention.py:44 in flash_attention_forward, code: value = value.transpose(1, 2)
        permute_7: "bf16[1, 24576, 16, 64]" = torch.ops.aten.permute.default(unsqueeze_4, [0, 2, 1, 3]);  unsqueeze_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:638 in _flash_attention_forward, code: q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
        view_1: "bf16[24576, 16, 64]" = torch.ops.aten.reshape.default(permute_5, [-1, 16, 64]);  permute_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:639 in _flash_attention_forward, code: k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
        view_2: "bf16[24576, 16, 64]" = torch.ops.aten.reshape.default(permute_6, [-1, 16, 64]);  permute_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:640 in _flash_attention_forward, code: v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))
        view_3: "bf16[24576, 16, 64]" = torch.ops.aten.reshape.default(permute_7, [24576, 16, 64]);  permute_7 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py:1443 in flash_attn_varlen_func, code: return FlashAttnVarlenFunc.apply(
        _flash_attn_varlen_forward = torch.ops.flash_attn._flash_attn_varlen_forward.default(view_1, view_2, view_3, primals_8, primals_8, 256, 256, 0.0, 0.125, False)
        getitem_5: "bf16[24576, 16, 64]" = _flash_attn_varlen_forward[0]
        getitem_6: "f32[16, 24576]" = _flash_attn_varlen_forward[1]
        getitem_8: "i64[2]" = _flash_attn_varlen_forward[3];  _flash_attn_varlen_forward = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py:660 in _flash_attention_forward, code: out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))
        view_4: "bf16[1, 24576, 16, 64]" = torch.ops.aten.reshape.default(getitem_5, [1, -1, 16, 64])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:128 in forward, code: attn_output = attn_output.reshape(seq_length, -1).contiguous()
        view_5: "bf16[24576, 1024]" = torch.ops.aten.reshape.default(view_4, [24576, -1]);  view_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:129 in forward, code: attn_output = self.proj(attn_output)
        permute_8: "bf16[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1: "bf16[24576, 1024]" = torch.ops.aten.addmm.default(primals_10, view_5, permute_8);  primals_10 = view_5 = permute_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:149 in forward, code: hidden_states = hidden_states + self.attn(
        add_4: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(primals_3, addmm_1)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:156 in forward, code: hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        convert_element_type_12: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_12, [1], correction = 0, keepdim = True)
        getitem_9: "f32[24576, 1]" = var_mean_1[0]
        getitem_10: "f32[24576, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[24576, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
        rsqrt_1: "f32[24576, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_10);  convert_element_type_12 = None
        mul_6: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_7: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_6, primals_11);  mul_6 = None
        add_6: "f32[24576, 1024]" = torch.ops.aten.add.Tensor(mul_7, primals_12);  mul_7 = primals_12 = None
        convert_element_type_13: "bf16[24576, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:56 in forward, code: return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        permute_9: "bf16[1024, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_2: "bf16[24576, 4096]" = torch.ops.aten.addmm.default(primals_14, convert_element_type_13, permute_9);  primals_14 = permute_9 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_17: "f32[24576, 4096]" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32)
        mul_8: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_17)
        mul_9: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_8, convert_element_type_17);  mul_8 = None
        mul_10: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_9, 0.044715);  mul_9 = None
        add_7: "f32[24576, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_17, mul_10);  mul_10 = None
        mul_11: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        mul_12: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 0.5);  convert_element_type_17 = None
        tanh: "f32[24576, 4096]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_8: "f32[24576, 4096]" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_13: "f32[24576, 4096]" = torch.ops.aten.mul.Tensor(mul_12, add_8);  mul_12 = add_8 = None
        convert_element_type_18: "bf16[24576, 4096]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:56 in forward, code: return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        permute_10: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default: "bf16[24576, 1024]" = torch.ops.aten.mm.default(convert_element_type_18, permute_10);  permute_10 = None
        add_tensor: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(mm_default, primals_16);  mm_default = primals_16 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:156 in forward, code: hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        add_9: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(add_4, add_tensor);  add_4 = add_tensor = None
        return (add_9, primals_1, primals_3, primals_4, primals_6, primals_7, primals_8, primals_9, primals_11, primals_13, primals_15, getitem_1, rsqrt, convert_element_type_1, view_1, view_2, view_3, getitem_5, getitem_6, getitem_8, addmm_1, getitem_10, rsqrt_1, convert_element_type_13, addmm_2, convert_element_type_18)
        