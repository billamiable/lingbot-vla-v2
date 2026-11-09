class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[3, 32, 337]", primals_2: "f32[64]", primals_3: "f32[32, 337, 32, 128]", primals_4: "f32[32, 337, 8, 128]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:323 in forward, code: inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_2, 0);  primals_2 = None
        unsqueeze_1: "f32[1, 1, 64]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "f32[1, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        expand: "f32[3, 32, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_2, [3, 32, -1, 1]);  unsqueeze_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:324 in forward, code: position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        unsqueeze_3: "i64[3, 32, 1, 337]" = torch.ops.aten.unsqueeze.default(primals_1, 2);  primals_1 = None
        convert_element_type: "f32[3, 32, 1, 337]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:328 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        expand_1: "f32[3, 32, 64, 1]" = torch.ops.aten.expand.default(expand, [3, 32, 64, 1]);  expand = None
        view: "f32[96, 64, 1]" = torch.ops.aten.view.default(expand_1, [96, 64, 1]);  expand_1 = None
        expand_2: "f32[3, 32, 1, 337]" = torch.ops.aten.expand.default(convert_element_type, [3, 32, 1, 337]);  convert_element_type = None
        view_1: "f32[96, 1, 337]" = torch.ops.aten.view.default(expand_2, [96, 1, 337]);  expand_2 = None
        bmm: "f32[96, 64, 337]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[3, 32, 64, 337]" = torch.ops.aten.view.default(bmm, [3, 32, 64, 337]);  bmm = None
        permute: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:313 in apply_interleaved_mrope, code: freqs_t[..., idx] = freqs[dim, ..., idx]
        select_1: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute, 0, 1)
        slice_5: "f32[32, 337, 20]" = torch.ops.aten.slice.Tensor(select_1, 2, 1, 60, 3);  select_1 = None
        select_2: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute, 0, 0)
        slice_scatter: "f32[32, 337, 64]" = torch.ops.aten.slice_scatter.default(select_2, slice_5, 2, 1, 60, 3);  select_2 = slice_5 = None
        select_scatter: "f32[3, 32, 337, 64]" = torch.ops.aten.select_scatter.default(permute, slice_scatter, 0, 0);  permute = slice_scatter = None
        permute_2: "f32[3, 32, 64, 337]" = torch.ops.aten.permute.default(select_scatter, [0, 1, 3, 2]);  select_scatter = None
        view_4: "f32[96, 64, 337]" = torch.ops.aten.view.default(permute_2, [96, 64, 337]);  permute_2 = None
        view_7: "f32[3, 32, 64, 337]" = torch.ops.aten.view.default(view_4, [3, 32, 64, 337]);  view_4 = None
        permute_5: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_7, [0, 1, 3, 2]);  view_7 = None
        select_6: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_5, 0, 2)
        slice_11: "f32[32, 337, 20]" = torch.ops.aten.slice.Tensor(select_6, 2, 2, 60, 3);  select_6 = None
        select_7: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_5, 0, 0)
        slice_scatter_1: "f32[32, 337, 64]" = torch.ops.aten.slice_scatter.default(select_7, slice_11, 2, 2, 60, 3);  select_7 = slice_11 = None
        select_scatter_1: "f32[3, 32, 337, 64]" = torch.ops.aten.select_scatter.default(permute_5, slice_scatter_1, 0, 0);  permute_5 = slice_scatter_1 = None
        permute_7: "f32[3, 32, 64, 337]" = torch.ops.aten.permute.default(select_scatter_1, [0, 1, 3, 2]);  select_scatter_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:330 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        permute_9: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(permute_7, [0, 1, 3, 2]);  permute_7 = None
        select_9: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_9, 0, 0);  permute_9 = None
        unsqueeze_4: "f32[32, 337, 1, 64]" = torch.ops.aten.unsqueeze.default(select_9, 2);  select_9 = None
        expand_3: "f32[32, 337, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_4, [32, 337, 2, 64])
        clone: "f32[32, 337, 2, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "f32[32, 337, 128]" = torch.ops.aten.view.default(clone, [32, 337, 128]);  clone = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:331 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[32, 337, 128]" = torch.ops.aten.cos.default(view_12)
        mul: "f32[32, 337, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:332 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[32, 337, 128]" = torch.ops.aten.sin.default(view_12);  view_12 = None
        mul_1: "f32[32, 337, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:378 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_5: "f32[32, 337, 1, 128]" = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:379 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_6: "f32[32, 337, 1, 128]" = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:380 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_2: "f32[32, 337, 32, 128]" = torch.ops.aten.mul.Tensor(primals_3, unsqueeze_5)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_13: "f32[32, 337, 32, 64]" = torch.ops.aten.slice.Tensor(primals_3, 3, 0, 64)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_14: "f32[32, 337, 32, 64]" = torch.ops.aten.slice.Tensor(primals_3, 3, 64, 9223372036854775807);  primals_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "f32[32, 337, 32, 64]" = torch.ops.aten.neg.default(slice_14);  slice_14 = None
        cat: "f32[32, 337, 32, 128]" = torch.ops.aten.cat.default([neg, slice_13], -1);  neg = slice_13 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:380 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_3: "f32[32, 337, 32, 128]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_6);  cat = None
        add: "f32[32, 337, 32, 128]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:381 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_4: "f32[32, 337, 8, 128]" = torch.ops.aten.mul.Tensor(primals_4, unsqueeze_5);  unsqueeze_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_15: "f32[32, 337, 8, 64]" = torch.ops.aten.slice.Tensor(primals_4, 3, 0, 64)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_16: "f32[32, 337, 8, 64]" = torch.ops.aten.slice.Tensor(primals_4, 3, 64, 9223372036854775807);  primals_4 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "f32[32, 337, 8, 64]" = torch.ops.aten.neg.default(slice_16);  slice_16 = None
        cat_1: "f32[32, 337, 8, 128]" = torch.ops.aten.cat.default([neg_1, slice_15], -1);  neg_1 = slice_15 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:381 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_5: "f32[32, 337, 8, 128]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_6);  cat_1 = unsqueeze_6 = None
        add_1: "f32[32, 337, 8, 128]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        return (add, add_1, unsqueeze_4)
        