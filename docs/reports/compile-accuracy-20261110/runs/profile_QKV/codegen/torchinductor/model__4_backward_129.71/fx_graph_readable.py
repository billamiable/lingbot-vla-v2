class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[4, 24576]", primals_3: "bf16[4, 24576]", primals_4: "i64[24576]", tangents_1: "bf16[24576, 1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:346 in fast_pos_embed_interpolate_cached, code: return positions.index_select(0, metadata["permutation"])
        full_default: "bf16[24576, 1024]" = torch.ops.aten.full.default([24576, 1024], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_add: "bf16[24576, 1024]" = torch.ops.aten.index_add.default(full_default, 0, primals_4, tangents_1);  full_default = primals_4 = tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        full_default_1: "bf16[4, 24576, 1024]" = torch.ops.aten.full.default([4, 24576, 1024], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "bf16[4, 24576, 1024]" = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 3)
        select_scatter_1: "bf16[4, 24576, 1024]" = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 2)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        add_3: "bf16[4, 24576, 1024]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        select_scatter_2: "bf16[4, 24576, 1024]" = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 1)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        add_4: "bf16[4, 24576, 1024]" = torch.ops.aten.add.Tensor(add_3, select_scatter_2);  add_3 = select_scatter_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        select_scatter_3: "bf16[4, 24576, 1024]" = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 0);  full_default_1 = index_add = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        add_5: "bf16[4, 24576, 1024]" = torch.ops.aten.add.Tensor(add_4, select_scatter_3);  add_4 = select_scatter_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:344 in fast_pos_embed_interpolate_cached, code: corners = self.pos_embed(metadata["indices"]) * metadata["weights"][:, :, None]
        unsqueeze: "bf16[4, 24576, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 2);  primals_3 = None
        mul_1: "bf16[4, 24576, 1024]" = torch.ops.aten.mul.Tensor(add_5, unsqueeze);  add_5 = unsqueeze = None
        convert_element_type: "f32[4, 24576, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        eq: "b8[4, 24576]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_1: "b8[4, 24576, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[4, 24576, 1024]" = torch.ops.aten.where.self(unsqueeze_1, full_default_5, convert_element_type);  unsqueeze_1 = full_default_5 = convert_element_type = None
        full_default_6: "f32[2304, 1024]" = torch.ops.aten.full.default([2304, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[2304, 1024]" = torch.ops.aten.index_put.default(full_default_6, [primals_2], where, True);  full_default_6 = primals_2 = where = None
        convert_element_type_1: "bf16[2304, 1024]" = torch.ops.prims.convert_element_type.default(index_put, torch.bfloat16);  index_put = None
        return (convert_element_type_1, None, None, None)
        