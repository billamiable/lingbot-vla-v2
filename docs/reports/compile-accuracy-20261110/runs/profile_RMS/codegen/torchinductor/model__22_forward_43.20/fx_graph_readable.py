class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "b8[32, 286]", primals_2: "b8[32, 286]", primals_3: "i64[3, 32, 286]", primals_4: "bf16[32, 55]", primals_5: "bf16[768, 55]", primals_6: "bf16[768]", primals_7: "bf16[32]", primals_8: "bf16[768, 55]", primals_9: "bf16[768]", primals_10: "bf16[32, 50, 55]", primals_11: "bf16[768, 1536]", primals_12: "bf16[768]", primals_13: "bf16[768, 768]", primals_14: "bf16[768]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1132 in embed_suffix, code: state_emb = self._fp32_linear(self.state_proj, state) if _fp32 else self.state_proj(state)
        permute: "bf16[55, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm: "bf16[32, 768]" = torch.ops.aten.addmm.default(primals_6, primals_4, permute);  primals_6 = permute = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:46 in create_sinusoidal_pos_embedding, code: fraction = torch.linspace(
        iota: "i64[384]" = torch.ops.prims.iota.default(384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        lt: "b8[384]" = torch.ops.aten.lt.Scalar(iota, 192.0)
        convert_element_type_3: "f32[384]" = torch.ops.prims.convert_element_type.default(iota, torch.float32)
        mul: "f32[384]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.0026109660574412533);  convert_element_type_3 = None
        add: "f32[384]" = torch.ops.aten.add.Tensor(mul, 0.0);  mul = None
        sub: "i64[384]" = torch.ops.aten.sub.Tensor(383, iota);  iota = None
        convert_element_type_4: "f32[384]" = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_1: "f32[384]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.0026109660574412533);  convert_element_type_4 = None
        sub_1: "f32[384]" = torch.ops.aten.sub.Tensor(1.0, mul_1);  mul_1 = None
        where: "f32[384]" = torch.ops.aten.where.self(lt, add, sub_1);  lt = add = sub_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:49 in create_sinusoidal_pos_embedding, code: period = min_period * (max_period / min_period) ** fraction
        pow_1: "f32[384]" = torch.ops.aten.pow.Scalar(1000.0, where);  where = None
        mul_2: "f32[384]" = torch.ops.aten.mul.Tensor(pow_1, 0.004);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:52 in create_sinusoidal_pos_embedding, code: scaling_factor = 1.0 / period * 2 * math.pi
        reciprocal: "f32[384]" = torch.ops.aten.reciprocal.default(mul_2);  mul_2 = None
        mul_3: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None
        mul_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_3, 2);  mul_3 = None
        mul_5: "f32[384]" = torch.ops.aten.mul.Tensor(mul_4, 3.141592653589793);  mul_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:53 in create_sinusoidal_pos_embedding, code: sin_input = scaling_factor[None, :] * time[:, None]
        unsqueeze: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_1: "bf16[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, 1);  primals_7 = None
        mul_6: "f32[32, 384]" = torch.ops.aten.mul.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:54 in create_sinusoidal_pos_embedding, code: pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        sin: "f32[32, 384]" = torch.ops.aten.sin.default(mul_6)
        cos: "f32[32, 384]" = torch.ops.aten.cos.default(mul_6);  mul_6 = None
        cat: "f32[32, 768]" = torch.ops.aten.cat.default([sin, cos], 1);  sin = cos = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1142 in embed_suffix, code: time_emb = time_emb.type(dtype=dtype)
        convert_element_type_5: "bf16[32, 768]" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1147 in embed_suffix, code: action_emb = self._fp32_linear(self.action_in_proj, noisy_actions) if _fp32 else self.action_in_proj(noisy_actions) # torch.Size([1, state_bs*50, 1024])
        view: "bf16[1600, 55]" = torch.ops.aten.view.default(primals_10, [1600, 55]);  primals_10 = None
        permute_1: "bf16[55, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm_1: "bf16[1600, 768]" = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = permute_1 = None
        view_1: "bf16[32, 50, 768]" = torch.ops.aten.view.default(addmm_1, [32, 50, 768]);  addmm_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1148 in embed_suffix, code: time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1]) # [1, 1024] -> [1, state_bs*50, 1024]
        unsqueeze_2: "bf16[32, 1, 768]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1)
        expand: "bf16[32, 50, 768]" = torch.ops.aten.expand.default(unsqueeze_2, [-1, 50, -1]);  unsqueeze_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1149 in embed_suffix, code: action_time_emb = torch.cat([action_emb, time_emb], dim=-1) # [1, state_bs*50, 2048]
        cat_1: "bf16[32, 50, 1536]" = torch.ops.aten.cat.default([view_1, expand], -1);  view_1 = expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1151 in embed_suffix, code: action_time_emb = self._fp32_linear(self.action_time_mlp_in, action_time_emb) if _fp32 else self.action_time_mlp_in(action_time_emb)
        view_2: "bf16[1600, 1536]" = torch.ops.aten.view.default(cat_1, [1600, 1536]);  cat_1 = None
        permute_2: "bf16[1536, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2: "bf16[1600, 768]" = torch.ops.aten.addmm.default(primals_12, view_2, permute_2);  primals_12 = permute_2 = None
        view_3: "bf16[32, 50, 768]" = torch.ops.aten.view.default(addmm_2, [32, 50, 768])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1152 in embed_suffix, code: action_time_emb = F.silu(action_time_emb)  # swish == silu
        convert_element_type_12: "f32[32, 50, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sigmoid: "f32[32, 50, 768]" = torch.ops.aten.sigmoid.default(convert_element_type_12)
        mul_7: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, sigmoid);  convert_element_type_12 = sigmoid = None
        convert_element_type_13: "bf16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1153 in embed_suffix, code: action_time_emb = self._fp32_linear(self.action_time_mlp_out, action_time_emb) if _fp32 else self.action_time_mlp_out(action_time_emb) # [1, state_bs*50, 1024]
        view_4: "bf16[1600, 768]" = torch.ops.aten.view.default(convert_element_type_13, [1600, 768]);  convert_element_type_13 = None
        permute_3: "bf16[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_3: "bf16[1600, 768]" = torch.ops.aten.addmm.default(primals_14, view_4, permute_3);  primals_14 = permute_3 = None
        view_5: "bf16[32, 50, 768]" = torch.ops.aten.view.default(addmm_3, [32, 50, 768]);  addmm_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1156 in embed_suffix, code: embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        unsqueeze_3: "bf16[32, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        cat_2: "bf16[32, 51, 768]" = torch.ops.aten.cat.default([unsqueeze_3, view_5], 1);  unsqueeze_3 = view_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1157 in embed_suffix, code: pad_masks = torch.ones(
        full_default: "b8[32, 51]" = torch.ops.aten.full.default([32, 51], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1164 in embed_suffix, code: att_masks = torch.zeros(
        full_1: "b8[32, 51]" = torch.ops.aten.full.default([32, 51], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1167 in embed_suffix, code: att_masks[:, :2] = True
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_5: "b8[32, 2]" = torch.ops.aten.slice.Tensor(full_1, 1, 0, 2)
        copy: "b8[32, 2]" = torch.ops.aten.copy.default(slice_5, full_default_1);  slice_5 = full_default_1 = None
        slice_scatter: "b8[32, 51]" = torch.ops.aten.slice_scatter.default(full_1, copy, 1, 0, 2);  copy = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:880 in torch_dynamo_resume_in_forward_at_869, code: pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        cat_3: "b8[32, 337]" = torch.ops.aten.cat.default([primals_1, full_default], 1);  full_default = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:881 in torch_dynamo_resume_in_forward_at_869, code: att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        cat_4: "b8[32, 337]" = torch.ops.aten.cat.default([primals_2, slice_scatter], 1);  primals_2 = slice_scatter = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:90 in make_att_2d_masks, code: cumsum = torch.cumsum(att_masks, dim=1)
        cumsum: "i64[32, 337]" = torch.ops.aten.cumsum.default(cat_4, 1);  cat_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:91 in make_att_2d_masks, code: att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        unsqueeze_4: "i64[32, 1, 337]" = torch.ops.aten.unsqueeze.default(cumsum, 1)
        unsqueeze_5: "i64[32, 337, 1]" = torch.ops.aten.unsqueeze.default(cumsum, 2);  cumsum = None
        le: "b8[32, 337, 337]" = torch.ops.aten.le.Tensor(unsqueeze_4, unsqueeze_5);  unsqueeze_4 = unsqueeze_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:92 in make_att_2d_masks, code: pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        unsqueeze_6: "b8[32, 1, 337]" = torch.ops.aten.unsqueeze.default(cat_3, 1)
        unsqueeze_7: "b8[32, 337, 1]" = torch.ops.aten.unsqueeze.default(cat_3, 2);  cat_3 = None
        mul_8: "b8[32, 337, 337]" = torch.ops.aten.mul.Tensor(unsqueeze_6, unsqueeze_7);  unsqueeze_6 = unsqueeze_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/utils.py:93 in make_att_2d_masks, code: att_2d_masks = att_2d_masks & pad_2d_masks
        bitwise_and: "b8[32, 337, 337]" = torch.ops.aten.bitwise_and.Tensor(le, mul_8);  le = mul_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:808 in _build_full_position_ids, code: valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        unsqueeze_8: "b8[1, 32, 286]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        bitwise_not: "b8[1, 32, 286]" = torch.ops.aten.bitwise_not.default(unsqueeze_8);  unsqueeze_8 = None
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[3, 32, 286]" = torch.ops.aten.where.self(bitwise_not, full_default_2, primals_3);  bitwise_not = full_default_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:809 in _build_full_position_ids, code: prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        amax: "i64[32]" = torch.ops.aten.amax.default(where_1, [0, 2]);  where_1 = None
        add_1: "i64[32]" = torch.ops.aten.add.Tensor(amax, 1);  amax = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:810 in _build_full_position_ids, code: suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        unsqueeze_9: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(add_1, 1);  add_1 = None
        full_default_3: "i64[32, 51]" = torch.ops.aten.full.default([32, 51], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cumsum_1: "i64[32, 51]" = torch.ops.aten.cumsum.default(full_default_3, 1);  full_default_3 = None
        add_2: "i64[32, 51]" = torch.ops.aten.add.Tensor(unsqueeze_9, cumsum_1);  unsqueeze_9 = cumsum_1 = None
        sub_2: "i64[32, 51]" = torch.ops.aten.sub.Tensor(add_2, 1);  add_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:811 in _build_full_position_ids, code: suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        full_default_5: "i64[]" = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[32, 51]" = torch.ops.aten.where.self(full_1, full_default_5, sub_2);  full_1 = full_default_5 = sub_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:812 in _build_full_position_ids, code: suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        unsqueeze_10: "i64[1, 32, 51]" = torch.ops.aten.unsqueeze.default(where_2, 0);  where_2 = None
        expand_1: "i64[3, 32, 51]" = torch.ops.aten.expand.default(unsqueeze_10, [3, -1, -1]);  unsqueeze_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:813 in _build_full_position_ids, code: return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)
        cat_5: "i64[3, 32, 337]" = torch.ops.aten.cat.default([primals_3, expand_1], -1);  primals_3 = expand_1 = None
        return (bitwise_and, cat_5, cat_2, convert_element_type_5, primals_4, primals_11, primals_13, view, view_2, addmm_2, view_4)
        