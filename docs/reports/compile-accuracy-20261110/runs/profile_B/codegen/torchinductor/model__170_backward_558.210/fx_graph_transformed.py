class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 51, 768]", primals_2: "f32[32, 337, 4096]", primals_3: "f32[32, 768]", primals_4: "f32[768, 4096]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[32, 768]", primals_11: "f32[32]", primals_13: "f32[32]", primals_14: "f32[32, 512, 768]", primals_15: "f32[32, 512, 768]", primals_16: "f32[32, 768, 512]", primals_17: "f32[704, 768]", primals_18: "f32[704, 768]", primals_19: "f32[768, 704]", tangents_1: "f32[32, 51, 768]", tangents_2: "f32[1632, 32]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:431 in forward, code: out_emb += after_first_residual
        view_18: "f32[1632, 768]" = torch.ops.aten.reshape.default(tangents_1, [1632, 768]);  tangents_1 = None
        view_20: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(view_18, [32, 51, 768]);  view_18 = None
        view_21: "f32[1632, 768]" = torch.ops.aten.reshape.default(view_20, [1632, 768])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_9: "f32[768, 1632]" = torch.ops.aten.permute.default(view_21, [1, 0])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:416 in forward, code: out_emb = self.self_attn.o_proj(att_output[:, start:end])
        slice_2: "f32[32, 51, 4096]" = torch.ops.aten.slice.Tensor(primals_2, 1, 286, 337);  primals_2 = None
        permute: "f32[4096, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        clone: "f32[32, 51, 4096]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view: "f32[1632, 4096]" = torch.ops.aten.reshape.default(clone, [1632, 4096]);  clone = None
        mm: "f32[1632, 768]" = torch.ops.aten.mm.default(view, permute)
        view_1: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(mm, [32, 51, 768]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:419 in forward, code: out_emb += hidden_states
        add: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean: "f32[32, 51, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:233 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_1: "f32[32, 51, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 51, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add, rsqrt)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:235 in forward, code: hidden_states = self.weight * hidden_states
        mul_1: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_5, mul)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[32, 768]" = torch.ops.aten.mm.default(primals_3, permute_1);  permute_1 = None
        add_tensor_1: "f32[32, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_7);  mm_default_1 = primals_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        unsqueeze: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 1);  add_tensor_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[32, 768]" = torch.ops.aten.mm.default(primals_3, permute_2);  permute_2 = None
        add_tensor: "f32[32, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_9);  mm_default = primals_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        unsqueeze_1: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor, 1);  add_tensor = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:239 in forward, code: hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        add_2: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add_2, mul_1)
        add_3: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:280 in forward, code: hidden_flat = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)
        view_4: "f32[1632, 768]" = torch.ops.aten.reshape.default(add_3, [-1, 768]);  add_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_5: "f32[768, 704]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        mm_2: "f32[1632, 704]" = torch.ops.aten.mm.default(view_4, permute_5)
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        sigmoid_2: "f32[1632, 704]" = torch.ops.aten.sigmoid.default(mm_2)
        mul_7: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mm_2, sigmoid_2)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_6: "f32[768, 704]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        mm_3: "f32[1632, 704]" = torch.ops.aten.mm.default(view_4, permute_6)
        mul_8: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mul_7, mm_3)
        mm_5: "f32[768, 704]" = torch.ops.aten.mm.default(permute_9, mul_8);  permute_9 = mul_8 = None
        permute_7: "f32[704, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_11: "f32[768, 704]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_6: "f32[1632, 704]" = torch.ops.aten.mm.default(view_21, permute_11);  permute_11 = None
        mul_9: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mm_6, mul_7);  mul_7 = None
        mul_10: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mm_6, mm_3);  mm_6 = mm_3 = None
        permute_13: "f32[704, 1632]" = torch.ops.aten.permute.default(mul_9, [1, 0])
        mm_7: "f32[704, 768]" = torch.ops.aten.mm.default(permute_13, view_4);  permute_13 = None
        permute_15: "f32[704, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_8: "f32[1632, 768]" = torch.ops.aten.mm.default(mul_9, permute_15);  mul_9 = permute_15 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        full_default_1: "f32[1632, 704]" = torch.ops.aten.full.default([1632, 704], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub: "f32[1632, 704]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_2);  full_default_1 = None
        mul_11: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mm_2, sub);  mm_2 = sub = None
        add_9: "f32[1632, 704]" = torch.ops.aten.add.Scalar(mul_11, 1);  mul_11 = None
        mul_12: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(sigmoid_2, add_9);  sigmoid_2 = add_9 = None
        mul_13: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mul_10, mul_12);  mul_10 = mul_12 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_18: "f32[704, 1632]" = torch.ops.aten.permute.default(mul_13, [1, 0])
        mm_9: "f32[704, 768]" = torch.ops.aten.mm.default(permute_18, view_4);  permute_18 = None
        permute_20: "f32[704, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_10: "f32[1632, 768]" = torch.ops.aten.mm.default(mul_13, permute_20);  mul_13 = permute_20 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        add_10: "f32[1632, 768]" = torch.ops.aten.add.Tensor(mm_8, mm_10);  mm_8 = mm_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:355 in forward, code: final_hidden_states = final_hidden_states.to(hidden_flat.dtype)
        convert_element_type_11: "bf16[1632, 768]" = torch.ops.prims.convert_element_type.default(view_21, torch.bfloat16);  view_21 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_7: "bf16[6528, 768]" = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:285 in forward, code: router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())  # (B*T, num_experts)
        permute_3: "f32[768, 32]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        mm_1: "f32[1632, 32]" = torch.ops.aten.mm.default(view_4, permute_3)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:288 in forward, code: routing_scores = router_logits.sigmoid()
        sigmoid: "f32[1632, 32]" = torch.ops.aten.sigmoid.default(mm_1);  mm_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:292 in forward, code: scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        unsqueeze_2: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_11, 0);  primals_11 = None
        add_4: "f32[1632, 32]" = torch.ops.aten.add.Tensor(sigmoid, unsqueeze_2);  unsqueeze_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:293 in forward, code: _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk = torch.ops.aten.topk.default(add_4, 4);  add_4 = None
        getitem_1: "i64[1632, 4]" = topk[1];  topk = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        view_7: "i64[6528]" = torch.ops.aten.reshape.default(getitem_1, [6528])
        sort = torch.ops.aten.sort.stable(view_7, stable = True)
        getitem_4: "i64[6528]" = sort[1];  sort = None
        sort_1 = torch.ops.aten.sort.default(getitem_4);  getitem_4 = None
        getitem_6: "i64[6528]" = sort_1[1];  sort_1 = None
        convert_element_type_7: "i32[6528]" = torch.ops.prims.convert_element_type.default(getitem_6, torch.int32);  getitem_6 = None
        view_9: "i32[1632, 4]" = torch.ops.aten.reshape.default(convert_element_type_7, [1632, 4]);  convert_element_type_7 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_11 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 1, constant_args_idx = 482, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': convert_element_type_11, 'O': empty_7, 'index': view_9});  convert_element_type_11 = triton_kernel_wrapper_mutation_11 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_8: "bf16[6528, 512]" = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:293 in fused_moe_forward, code: fc2_weight = fc2_weight.to(compute_dtype)
        convert_element_type_6: "bf16[32, 768, 512]" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        full_default: "i32[32]" = torch.ops.aten.full.default([32], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_10 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 476, grid = [(7, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'out_ptr': full_default, 'x_ptr': view_7});  view_7 = triton_kernel_wrapper_mutation_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        cumsum: "i64[32]" = torch.ops.aten.cumsum.default(full_default, 0);  full_default = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_9 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 483, grid = [(52, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': empty_7, 'b_ptr': convert_element_type_6, 'c_ptr': empty_8, 'cumsum_M': cumsum});  convert_element_type_6 = triton_kernel_wrapper_mutation_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_9: "bf16[32, 768, 512]" = torch.ops.aten.empty.memory_format([32, 768, 512], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_22: "bf16[32, 768, 512]" = torch.ops.aten.permute.default(empty_9, [0, 1, 2]);  empty_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:294 in forward, code: routing_weights = routing_scores.gather(1, selected_experts)
        gather: "f32[1632, 4]" = torch.ops.aten.gather.default(sigmoid, 1, getitem_1)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:298 in forward, code: routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        sum_2: "f32[1632, 1]" = torch.ops.aten.sum.dim_IntList(gather, [-1], True)
        add_6: "f32[1632, 1]" = torch.ops.aten.add.Tensor(sum_2, 1e-20);  sum_2 = None
        div: "f32[1632, 4]" = torch.ops.aten.div.Tensor(gather, add_6);  gather = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:300 in forward, code: routing_weights = routing_weights * self.routed_scaling_factor
        mul_3: "f32[1632, 4]" = torch.ops.aten.mul.Tensor(div, 4.0)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:289 in fused_moe_forward, code: hidden_states = hidden_states.to(compute_dtype)
        convert_element_type_2: "bf16[1632, 768]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:290 in fused_moe_forward, code: routing_weights = routing_weights.to(compute_dtype)
        convert_element_type_3: "bf16[1632, 4]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:291 in fused_moe_forward, code: fc1_1_weight = fc1_1_weight.to(compute_dtype)
        convert_element_type_4: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:292 in fused_moe_forward, code: fc1_2_weight = fc1_2_weight.to(compute_dtype)
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty: "bf16[6528, 768]" = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_8 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 1, constant_args_idx = 477, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': convert_element_type_2, 'O': empty, 'index': view_9});  convert_element_type_2 = triton_kernel_wrapper_mutation_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_1: "bf16[6528, 512]" = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_7 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 478, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': empty, 'b_ptr': convert_element_type_4, 'c_ptr': empty_1, 'cumsum_M': cumsum});  triton_kernel_wrapper_mutation_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_2: "bf16[6528, 512]" = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_6 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 479, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': empty, 'b_ptr': convert_element_type_5, 'c_ptr': empty_2, 'cumsum_M': cumsum});  triton_kernel_wrapper_mutation_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        convert_element_type_8: "f32[6528, 512]" = torch.ops.prims.convert_element_type.default(empty_1, torch.float32)
        sigmoid_1: "f32[6528, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_8)
        mul_4: "f32[6528, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, sigmoid_1);  convert_element_type_8 = sigmoid_1 = None
        convert_element_type_9: "bf16[6528, 512]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, empty_2)
        view_10: "bf16[6528, 1]" = torch.ops.aten.reshape.default(convert_element_type_3, [-1, 1]);  convert_element_type_3 = None
        empty_3: "bf16[6528, 1]" = torch.ops.aten.empty.memory_format([6528, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_4: "bf16[6528, 1]" = torch.ops.aten.permute.default(empty_3, [0, 1]);  empty_3 = None
        view_11: "i32[6528]" = torch.ops.aten.reshape.default(view_9, [6528])
        index_put: "bf16[6528, 1]" = torch.ops.aten.index_put_.default(permute_4, [view_11], view_10);  permute_4 = view_10 = None
        mul_6: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(mul_5, index_put)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_5 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 4, constant_args_idx = 484, grid = [(24, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': empty_7, 'b_ptr': mul_6, 'c_ptr': permute_22, 'cumsum_K': cumsum});  empty_7 = mul_6 = triton_kernel_wrapper_mutation_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        mul_14: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(empty_8, index_put);  index_put = None
        mul_15: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(mul_5, empty_8);  mul_5 = empty_8 = None
        sum_3: "bf16[6528]" = torch.ops.aten.sum.dim_IntList(mul_15, [-1]);  mul_15 = None
        index: "bf16[6528]" = torch.ops.aten.index.Tensor(sum_3, [view_11]);  sum_3 = view_11 = None
        view_25: "bf16[1632, 4]" = torch.ops.aten.reshape.default(index, [1632, 4]);  index = None
        mul_17: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(mul_14, empty_2);  empty_2 = None
        mul_18: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_14);  convert_element_type_9 = mul_14 = None
        empty_10: "bf16[6528, 768]" = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_4 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 485, grid = [(78, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_18, 'b_ptr': convert_element_type_5, 'c_ptr': empty_10, 'cumsum_M': cumsum});  convert_element_type_5 = triton_kernel_wrapper_mutation_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_11: "bf16[32, 512, 768]" = torch.ops.aten.empty.memory_format([32, 512, 768], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_23: "bf16[32, 512, 768]" = torch.ops.aten.permute.default(empty_11, [0, 1, 2]);  empty_11 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_3 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 4, constant_args_idx = 486, grid = [(24, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_18, 'b_ptr': empty, 'c_ptr': permute_23, 'cumsum_K': cumsum});  mul_18 = triton_kernel_wrapper_mutation_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        sigmoid_5: "bf16[6528, 512]" = torch.ops.aten.sigmoid.default(empty_1)
        full_default_2: "bf16[6528, 512]" = torch.ops.aten.full.default([6528, 512], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_1: "bf16[6528, 512]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_5);  full_default_2 = None
        mul_19: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(empty_1, sub_1);  empty_1 = sub_1 = None
        add_11: "bf16[6528, 512]" = torch.ops.aten.add.Scalar(mul_19, 1);  mul_19 = None
        mul_20: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(sigmoid_5, add_11);  sigmoid_5 = add_11 = None
        mul_21: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(mul_17, mul_20);  mul_17 = mul_20 = None
        empty_13: "bf16[6528, 768]" = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 487, grid = [(78, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_21, 'b_ptr': convert_element_type_4, 'c_ptr': empty_13, 'cumsum_M': cumsum});  convert_element_type_4 = triton_kernel_wrapper_mutation_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_14: "bf16[32, 512, 768]" = torch.ops.aten.empty.memory_format([32, 512, 768], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25: "bf16[32, 512, 768]" = torch.ops.aten.permute.default(empty_14, [0, 1, 2]);  empty_14 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 4, constant_args_idx = 488, grid = [(24, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_21, 'b_ptr': empty, 'c_ptr': permute_25, 'cumsum_K': cumsum});  mul_21 = empty = cumsum = triton_kernel_wrapper_mutation_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        add_12: "bf16[6528, 768]" = torch.ops.aten.add.Tensor(empty_13, empty_10);  empty_13 = empty_10 = None
        empty_15: "bf16[1632, 768]" = torch.ops.aten.empty.memory_format([1632, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 3, constant_args_idx = 489, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': add_12, 'Y': empty_15, 'index': view_9});  add_12 = view_9 = triton_kernel_wrapper_mutation = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:293 in fused_moe_forward, code: fc2_weight = fc2_weight.to(compute_dtype)
        convert_element_type_14: "f32[32, 768, 512]" = torch.ops.prims.convert_element_type.default(permute_22, torch.float32);  permute_22 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:292 in fused_moe_forward, code: fc1_2_weight = fc1_2_weight.to(compute_dtype)
        convert_element_type_15: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(permute_23, torch.float32);  permute_23 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:291 in fused_moe_forward, code: fc1_1_weight = fc1_1_weight.to(compute_dtype)
        convert_element_type_16: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:290 in fused_moe_forward, code: routing_weights = routing_weights.to(compute_dtype)
        convert_element_type_17: "f32[1632, 4]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:289 in fused_moe_forward, code: hidden_states = hidden_states.to(compute_dtype)
        convert_element_type_18: "f32[1632, 768]" = torch.ops.prims.convert_element_type.default(empty_15, torch.float32);  empty_15 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:289 in fused_moe_forward, code: hidden_states = hidden_states.to(compute_dtype)
        add_13: "f32[1632, 768]" = torch.ops.aten.add.Tensor(add_10, convert_element_type_18);  add_10 = convert_element_type_18 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:300 in forward, code: routing_weights = routing_weights * self.routed_scaling_factor
        mul_22: "f32[1632, 4]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 4.0);  convert_element_type_17 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:298 in forward, code: routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        div_2: "f32[1632, 4]" = torch.ops.aten.div.Tensor(div, add_6);  div = None
        neg: "f32[1632, 4]" = torch.ops.aten.neg.default(mul_22)
        mul_23: "f32[1632, 4]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3: "f32[1632, 4]" = torch.ops.aten.div.Tensor(mul_22, add_6);  mul_22 = add_6 = None
        sum_4: "f32[1632, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [1], True);  mul_23 = None
        expand: "f32[1632, 4]" = torch.ops.aten.expand.default(sum_4, [1632, 4]);  sum_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:298 in forward, code: routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        add_14: "f32[1632, 4]" = torch.ops.aten.add.Tensor(div_3, expand);  div_3 = expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:294 in forward, code: routing_weights = routing_scores.gather(1, selected_experts)
        full_default_3: "f32[1632, 32]" = torch.ops.aten.full.default([1632, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add: "f32[1632, 32]" = torch.ops.aten.scatter_add.default(full_default_3, 1, getitem_1, add_14);  full_default_3 = add_14 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:288 in forward, code: routing_scores = router_logits.sigmoid()
        sub_2: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_24: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(sigmoid, sub_2);  sigmoid = sub_2 = None
        mul_25: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(scatter_add, mul_24);  scatter_add = mul_24 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:288 in forward, code: routing_scores = router_logits.sigmoid()
        add_15: "f32[1632, 32]" = torch.ops.aten.add.Tensor(tangents_2, mul_25);  tangents_2 = mul_25 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:285 in forward, code: router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())  # (B*T, num_experts)
        permute_26: "f32[32, 1632]" = torch.ops.aten.permute.default(add_15, [1, 0])
        mm_11: "f32[32, 768]" = torch.ops.aten.mm.default(permute_26, view_4);  permute_26 = view_4 = None
        permute_28: "f32[32, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        
        # No stacktrace found for following nodes
        addmm_default: "f32[1632, 768]" = torch.ops.aten.addmm.default(add_13, add_15, permute_28);  add_13 = add_15 = permute_28 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:280 in forward, code: hidden_flat = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)
        view_28: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(addmm_default, [32, 51, 768]);  addmm_default = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:239 in forward, code: hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        sum_5: "f32[32, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_28, [1], True)
        mul_26: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(view_28, add_2);  add_2 = None
        mul_27: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(view_28, mul_1);  view_28 = mul_1 = None
        sum_6: "f32[32, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_27, [1], True);  mul_27 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        squeeze: "f32[32, 768]" = torch.ops.aten.squeeze.dim(sum_5, 1);  sum_5 = None
        permute_30: "f32[768, 32]" = torch.ops.aten.permute.default(squeeze, [1, 0])
        mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_30, primals_3);  permute_30 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze, [0], True);  squeeze = None
        view_29: "f32[768]" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        squeeze_1: "f32[32, 768]" = torch.ops.aten.squeeze.dim(sum_6, 1);  sum_6 = None
        permute_33: "f32[768, 32]" = torch.ops.aten.permute.default(squeeze_1, [1, 0])
        mm_14: "f32[768, 768]" = torch.ops.aten.mm.default(permute_33, primals_3);  permute_33 = primals_3 = None
        sum_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze_1, [0], True);  squeeze_1 = None
        view_30: "f32[768]" = torch.ops.aten.reshape.default(sum_8, [768]);  sum_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:235 in forward, code: hidden_states = self.weight * hidden_states
        mul_28: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_5);  primals_5 = None
        mul_29: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_26, mul);  mul_26 = mul = None
        sum_9: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_29, [0, 1], True);  mul_29 = None
        view_31: "f32[768]" = torch.ops.aten.reshape.default(sum_9, [768]);  sum_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:233 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_30: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_28, add)
        mul_31: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_28, rsqrt);  mul_28 = None
        sum_10: "f32[32, 51, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True);  mul_30 = None
        pow_2: "f32[32, 51, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_32: "f32[32, 51, 1]" = torch.ops.aten.mul.Scalar(sum_10, -0.5);  sum_10 = None
        mul_33: "f32[32, 51, 1]" = torch.ops.aten.mul.Tensor(mul_32, pow_2);  mul_32 = pow_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_1: "f32[32, 51, 768]" = torch.ops.aten.expand.default(mul_33, [32, 51, 768]);  mul_33 = None
        div_4: "f32[32, 51, 768]" = torch.ops.aten.div.Scalar(expand_1, 768);  expand_1 = None
        pow_3: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(add, 1.0);  add = None
        mul_34: "f32[32, 51, 768]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_35: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(div_4, mul_34);  div_4 = mul_34 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_17: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_31, mul_35);  mul_31 = mul_35 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:420 in forward, code: after_first_residual = out_emb.clone()
        add_18: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(add_17, view_20);  add_17 = view_20 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:416 in forward, code: out_emb = self.self_attn.o_proj(att_output[:, start:end])
        view_32: "f32[1632, 768]" = torch.ops.aten.reshape.default(add_18, [1632, 768])
        permute_36: "f32[768, 1632]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_15: "f32[768, 4096]" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = view = None
        permute_38: "f32[768, 4096]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_16: "f32[1632, 4096]" = torch.ops.aten.mm.default(view_32, permute_38);  view_32 = permute_38 = None
        view_33: "f32[32, 51, 4096]" = torch.ops.aten.reshape.default(mm_16, [32, 51, 4096]);  mm_16 = None
        full_default_4: "f32[32, 337, 4096]" = torch.ops.aten.full.default([32, 337, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default: "f32[32, 337, 4096]" = torch.ops.aten.slice_scatter.default(full_default_4, view_33, 1, 286, 337);  full_default_4 = view_33 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:51 in _update_moe_runtime_stats, code: selected_experts.detach().reshape(-1),
        view_6: "i64[6528]" = torch.ops.aten.reshape.default(getitem_1, [-1]);  getitem_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:50 in _update_moe_runtime_stats, code: counts = F.one_hot(
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[6528, 1]" = torch.ops.aten.unsqueeze.default(view_6, -1);  view_6 = None
        eq: "b8[6528, 32]" = torch.ops.aten.eq.Tensor(unsqueeze_3, iota);  unsqueeze_3 = iota = None
        convert_element_type: "i64[6528, 32]" = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:53 in _update_moe_runtime_stats, code: ).sum(dim=0)
        sum_1: "i64[32]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0]);  convert_element_type = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:55 in _update_moe_runtime_stats, code: counts.to(
        convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:54 in _update_moe_runtime_stats, code: block.tokens_per_expert.add_(
        add_5: "f32[32]" = torch.ops.aten.add.Tensor(primals_13, convert_element_type_1);  convert_element_type_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/__init__.py:2425 in __call__, code: return self.compiler_fn(model_, inputs_, **self.kwargs)
        copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_13, add_5);  primals_13 = add_5 = copy__1 = None
        return (add_18, slice_scatter_default, None, mm_15, view_31, mm_14, view_30, mm_13, view_29, mm_11, None, None, None, convert_element_type_16, convert_element_type_15, convert_element_type_14, mm_9, mm_7, mm_5)
        