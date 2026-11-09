class GraphModule(torch.nn.Module):
    def forward(self, unsqueeze_4: "f32[32, 337, 1, 64]", tangents_1: "f32[32, 337, 32, 128]", tangents_2: "f32[32, 337, 8, 128]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:330 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        expand_3: "f32[32, 337, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_4, [32, 337, 2, 64]);  unsqueeze_4 = None
        clone: "f32[32, 337, 2, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "f32[32, 337, 128]" = torch.ops.aten.reshape.default(clone, [32, 337, 128]);  clone = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:332 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[32, 337, 128]" = torch.ops.aten.sin.default(view_12)
        mul_1: "f32[32, 337, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:379 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_6: "f32[32, 337, 1, 128]" = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:381 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_6: "f32[32, 337, 8, 128]" = torch.ops.aten.mul.Tensor(tangents_2, unsqueeze_6)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_17: "f32[32, 337, 8, 64]" = torch.ops.aten.slice.Tensor(mul_6, 3, 0, 64)
        slice_18: "f32[32, 337, 8, 64]" = torch.ops.aten.slice.Tensor(mul_6, 3, 64, 128);  mul_6 = None
        neg_2: "f32[32, 337, 8, 64]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default: "f32[32, 337, 8, 128]" = torch.ops.aten.full.default([32, 337, 8, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default: "f32[32, 337, 8, 128]" = torch.ops.aten.slice_scatter.default(full_default, neg_2, 3, 64, 9223372036854775807);  neg_2 = None
        slice_scatter_default_1: "f32[32, 337, 8, 128]" = torch.ops.aten.slice_scatter.default(full_default, slice_18, 3, 0, 64);  full_default = slice_18 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        add_2: "f32[32, 337, 8, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:331 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[32, 337, 128]" = torch.ops.aten.cos.default(view_12);  view_12 = None
        mul: "f32[32, 337, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:378 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_5: "f32[32, 337, 1, 128]" = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:381 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7: "f32[32, 337, 8, 128]" = torch.ops.aten.mul.Tensor(tangents_2, unsqueeze_5);  tangents_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:381 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        add_3: "f32[32, 337, 8, 128]" = torch.ops.aten.add.Tensor(add_2, mul_7);  add_2 = mul_7 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:380 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_8: "f32[32, 337, 32, 128]" = torch.ops.aten.mul.Tensor(tangents_1, unsqueeze_6);  unsqueeze_6 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:113 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_19: "f32[32, 337, 32, 64]" = torch.ops.aten.slice.Tensor(mul_8, 3, 0, 64)
        slice_20: "f32[32, 337, 32, 64]" = torch.ops.aten.slice.Tensor(mul_8, 3, 64, 128);  mul_8 = None
        neg_3: "f32[32, 337, 32, 64]" = torch.ops.aten.neg.default(slice_19);  slice_19 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:112 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_2: "f32[32, 337, 32, 128]" = torch.ops.aten.full.default([32, 337, 32, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default_2: "f32[32, 337, 32, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, neg_3, 3, 64, 9223372036854775807);  neg_3 = None
        slice_scatter_default_3: "f32[32, 337, 32, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_20, 3, 0, 64);  full_default_2 = slice_20 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:111 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        add_4: "f32[32, 337, 32, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:380 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_9: "f32[32, 337, 32, 128]" = torch.ops.aten.mul.Tensor(tangents_1, unsqueeze_5);  tangents_1 = unsqueeze_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:380 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        add_5: "f32[32, 337, 32, 128]" = torch.ops.aten.add.Tensor(add_4, mul_9);  add_4 = mul_9 = None
        return (None, None, add_5, add_3)
        