class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[3, 32, 337]", arg1_1: "f32[64]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:323 in forward, code: inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_1: "f32[1, 1, 64]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "f32[1, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        expand: "f32[3, 32, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_2, [3, 32, -1, 1]);  unsqueeze_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:328 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        expand_1: "f32[3, 32, 64, 1]" = torch.ops.aten.expand.default(expand, [3, 32, 64, 1]);  expand = None
        view: "f32[96, 64, 1]" = torch.ops.aten.reshape.default(expand_1, [96, 64, 1]);  expand_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:324 in forward, code: position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        unsqueeze_3: "i64[3, 32, 1, 337]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        convert_element_type: "f32[3, 32, 1, 337]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:328 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        expand_2: "f32[3, 32, 1, 337]" = torch.ops.aten.expand.default(convert_element_type, [3, 32, 1, 337]);  convert_element_type = None
        view_1: "f32[96, 1, 337]" = torch.ops.aten.reshape.default(expand_2, [96, 1, 337]);  expand_2 = None
        bmm: "f32[96, 64, 337]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[3, 32, 64, 337]" = torch.ops.aten.reshape.default(bmm, [3, 32, 64, 337])
        permute: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:309 in apply_interleaved_mrope, code: freqs_t = freqs[0]  # just overwrite the first dimension T
        select: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute, 0, 0)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:313 in apply_interleaved_mrope, code: freqs_t[..., idx] = freqs[dim, ..., idx]
        slice_6: "f32[32, 337, 20]" = torch.ops.aten.slice.Tensor(select, 2, 1, 60, 3);  select = slice_6 = None
        view_3: "f32[3, 32, 64, 337]" = torch.ops.aten.reshape.default(bmm, [3, 32, 64, 337]);  bmm = None
        permute_1: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_3, [0, 1, 3, 2]);  view_3 = None
        select_2: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_1, 0, 0);  select_2 = None
        select_1: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute, 0, 1);  permute = None
        slice_5: "f32[32, 337, 20]" = torch.ops.aten.slice.Tensor(select_1, 2, 1, 60, 3);  select_1 = None
        
        # No stacktrace found for following nodes
        select_int: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_1, 0, 0)
        slice_scatter_default: "f32[32, 337, 64]" = torch.ops.aten.slice_scatter.default(select_int, slice_5, 2, 1, 60, 3);  select_int = slice_5 = None
        select_scatter_default: "f32[3, 32, 337, 64]" = torch.ops.aten.select_scatter.default(permute_1, slice_scatter_default, 0, 0);  permute_1 = slice_scatter_default = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:313 in apply_interleaved_mrope, code: freqs_t[..., idx] = freqs[dim, ..., idx]
        permute_2: "f32[3, 32, 64, 337]" = torch.ops.aten.permute.default(select_scatter_default, [0, 1, 3, 2]);  select_scatter_default = None
        view_4: "f32[96, 64, 337]" = torch.ops.aten.reshape.default(permute_2, [96, 64, 337]);  permute_2 = None
        view_6: "f32[3, 32, 64, 337]" = torch.ops.aten.reshape.default(view_4, [3, 32, 64, 337])
        permute_4: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
        select_5: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_4, 0, 0);  permute_4 = None
        slice_10: "f32[32, 337, 20]" = torch.ops.aten.slice.Tensor(select_5, 2, 2, 60, 3);  select_5 = slice_10 = None
        view_8: "f32[3, 32, 64, 337]" = torch.ops.aten.reshape.default(view_4, [3, 32, 64, 337])
        permute_6: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_8, [0, 1, 3, 2]);  view_8 = None
        select_7: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_6, 0, 0);  select_7 = None
        view_7: "f32[3, 32, 64, 337]" = torch.ops.aten.reshape.default(view_4, [3, 32, 64, 337]);  view_4 = None
        permute_5: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(view_7, [0, 1, 3, 2]);  view_7 = None
        select_6: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_5, 0, 2);  permute_5 = None
        slice_11: "f32[32, 337, 20]" = torch.ops.aten.slice.Tensor(select_6, 2, 2, 60, 3);  select_6 = None
        
        # No stacktrace found for following nodes
        select_int_1: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_6, 0, 0)
        slice_scatter_default_1: "f32[32, 337, 64]" = torch.ops.aten.slice_scatter.default(select_int_1, slice_11, 2, 2, 60, 3);  select_int_1 = slice_11 = None
        select_scatter_default_1: "f32[3, 32, 337, 64]" = torch.ops.aten.select_scatter.default(permute_6, slice_scatter_default_1, 0, 0);  permute_6 = slice_scatter_default_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:313 in apply_interleaved_mrope, code: freqs_t[..., idx] = freqs[dim, ..., idx]
        permute_7: "f32[3, 32, 64, 337]" = torch.ops.aten.permute.default(select_scatter_default_1, [0, 1, 3, 2]);  select_scatter_default_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:330 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        permute_9: "f32[3, 32, 337, 64]" = torch.ops.aten.permute.default(permute_7, [0, 1, 3, 2]);  permute_7 = None
        select_9: "f32[32, 337, 64]" = torch.ops.aten.select.int(permute_9, 0, 0);  permute_9 = None
        unsqueeze_4: "f32[32, 337, 1, 64]" = torch.ops.aten.unsqueeze.default(select_9, 2);  select_9 = None
        expand_3: "f32[32, 337, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_4, [32, 337, 2, 64]);  unsqueeze_4 = None
        clone: "f32[32, 337, 2, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "f32[32, 337, 128]" = torch.ops.aten.reshape.default(clone, [32, 337, 128]);  clone = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:331 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[32, 337, 128]" = torch.ops.aten.cos.default(view_12)
        mul: "f32[32, 337, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:332 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[32, 337, 128]" = torch.ops.aten.sin.default(view_12);  view_12 = None
        mul_1: "f32[32, 337, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        return (mul, mul_1)
        