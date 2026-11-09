class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 51, 768]", primals_2: "f32[32, 337, 4096]", primals_3: "f32[32, 768]", primals_4: "f32[768, 4096]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[32, 768]", primals_11: "f32[32]", primals_12: "f32[1]", primals_13: "f32[32]", primals_14: "f32[32, 512, 768]", primals_15: "f32[32, 512, 768]", primals_16: "f32[32, 768, 512]", primals_17: "f32[704, 768]", primals_18: "f32[704, 768]", primals_19: "f32[768, 704]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:416 in forward, code: out_emb = self.self_attn.o_proj(att_output[:, start:end])
        slice_2: "f32[32, 51, 4096]" = torch.ops.aten.slice.Tensor(primals_2, 1, 286, 337)
        permute: "f32[4096, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        clone: "f32[32, 51, 4096]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view: "f32[1632, 4096]" = torch.ops.aten.reshape.default(clone, [1632, 4096]);  clone = None
        mm: "f32[1632, 768]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(mm, [32, 51, 768]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:419 in forward, code: out_emb += hidden_states
        add: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean: "f32[32, 51, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:233 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_1: "f32[32, 51, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 51, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add, rsqrt);  rsqrt = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:235 in forward, code: hidden_states = self.weight * hidden_states
        mul_1: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_5, mul);  mul = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[32, 768]" = torch.ops.aten.mm.default(primals_3, permute_1);  permute_1 = None
        add_tensor_1: "f32[32, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_7);  mm_default_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        unsqueeze: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 1);  add_tensor_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default: "f32[32, 768]" = torch.ops.aten.mm.default(primals_3, permute_2);  permute_2 = None
        add_tensor: "f32[32, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_9);  mm_default = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        unsqueeze_1: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor, 1);  add_tensor = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:239 in forward, code: hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        add_2: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add_2, mul_1);  add_2 = mul_1 = None
        add_3: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:280 in forward, code: hidden_flat = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)
        view_4: "f32[1632, 768]" = torch.ops.aten.reshape.default(add_3, [-1, 768]);  add_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:285 in forward, code: router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())  # (B*T, num_experts)
        permute_3: "f32[768, 32]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        mm_1: "f32[1632, 32]" = torch.ops.aten.mm.default(view_4, permute_3);  permute_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:288 in forward, code: routing_scores = router_logits.sigmoid()
        sigmoid: "f32[1632, 32]" = torch.ops.aten.sigmoid.default(mm_1)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:292 in forward, code: scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        unsqueeze_2: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_11, 0)
        add_4: "f32[1632, 32]" = torch.ops.aten.add.Tensor(sigmoid, unsqueeze_2);  unsqueeze_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:293 in forward, code: _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk = torch.ops.aten.topk.default(add_4, 4);  add_4 = None
        getitem_1: "i64[1632, 4]" = topk[1];  topk = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:294 in forward, code: routing_weights = routing_scores.gather(1, selected_experts)
        gather: "f32[1632, 4]" = torch.ops.aten.gather.default(sigmoid, 1, getitem_1);  sigmoid = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:41 in _update_moe_runtime_stats, code: avg_score = routing_weights.detach().float().mean()
        mean_1: "f32[]" = torch.ops.aten.mean.default(gather)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:43 in _update_moe_runtime_stats, code: avg_score.reshape_as(block.avg_topk_sigmoid_score).to(
        view_5: "f32[1]" = torch.ops.aten.reshape.default(mean_1, [1]);  mean_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:51 in _update_moe_runtime_stats, code: selected_experts.detach().reshape(-1),
        view_6: "i64[6528]" = torch.ops.aten.reshape.default(getitem_1, [-1])
        
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
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:298 in forward, code: routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        sum_2: "f32[1632, 1]" = torch.ops.aten.sum.dim_IntList(gather, [-1], True)
        add_6: "f32[1632, 1]" = torch.ops.aten.add.Tensor(sum_2, 1e-20);  sum_2 = None
        div: "f32[1632, 4]" = torch.ops.aten.div.Tensor(gather, add_6);  gather = add_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:300 in forward, code: routing_weights = routing_weights * self.routed_scaling_factor
        mul_3: "f32[1632, 4]" = torch.ops.aten.mul.Tensor(div, 4.0);  div = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:289 in fused_moe_forward, code: hidden_states = hidden_states.to(compute_dtype)
        convert_element_type_2: "bf16[1632, 768]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:290 in fused_moe_forward, code: routing_weights = routing_weights.to(compute_dtype)
        convert_element_type_3: "bf16[1632, 4]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:291 in fused_moe_forward, code: fc1_1_weight = fc1_1_weight.to(compute_dtype)
        convert_element_type_4: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:292 in fused_moe_forward, code: fc1_2_weight = fc1_2_weight.to(compute_dtype)
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:293 in fused_moe_forward, code: fc2_weight = fc2_weight.to(compute_dtype)
        convert_element_type_6: "bf16[32, 768, 512]" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        view_7: "i64[6528]" = torch.ops.aten.reshape.default(getitem_1, [6528]);  getitem_1 = None
        full_default: "i32[32]" = torch.ops.aten.full.default([32], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_5 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 322, grid = [(7, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'out_ptr': full_default, 'x_ptr': view_7});  triton_kernel_wrapper_mutation_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        sort = torch.ops.aten.sort.stable(view_7, stable = True);  view_7 = None
        getitem_4: "i64[6528]" = sort[1];  sort = None
        sort_1 = torch.ops.aten.sort.default(getitem_4);  getitem_4 = None
        getitem_6: "i64[6528]" = sort_1[1];  sort_1 = None
        convert_element_type_7: "i32[6528]" = torch.ops.prims.convert_element_type.default(getitem_6, torch.int32);  getitem_6 = None
        view_9: "i32[1632, 4]" = torch.ops.aten.reshape.default(convert_element_type_7, [1632, 4]);  convert_element_type_7 = None
        empty: "bf16[6528, 768]" = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_4 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 1, constant_args_idx = 323, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': convert_element_type_2, 'O': empty, 'index': view_9});  convert_element_type_2 = triton_kernel_wrapper_mutation_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        cumsum: "i64[32]" = torch.ops.aten.cumsum.default(full_default, 0);  full_default = None
        empty_1: "bf16[6528, 512]" = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_3 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 324, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': empty, 'b_ptr': convert_element_type_4, 'c_ptr': empty_1, 'cumsum_M': cumsum});  convert_element_type_4 = triton_kernel_wrapper_mutation_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_2: "bf16[6528, 512]" = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 325, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': empty, 'b_ptr': convert_element_type_5, 'c_ptr': empty_2, 'cumsum_M': cumsum});  empty = convert_element_type_5 = triton_kernel_wrapper_mutation_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        convert_element_type_8: "f32[6528, 512]" = torch.ops.prims.convert_element_type.default(empty_1, torch.float32);  empty_1 = None
        sigmoid_1: "f32[6528, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_8)
        mul_4: "f32[6528, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, sigmoid_1);  convert_element_type_8 = sigmoid_1 = None
        convert_element_type_9: "bf16[6528, 512]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, empty_2);  convert_element_type_9 = empty_2 = None
        view_10: "bf16[6528, 1]" = torch.ops.aten.reshape.default(convert_element_type_3, [-1, 1]);  convert_element_type_3 = None
        empty_3: "bf16[6528, 1]" = torch.ops.aten.empty.memory_format([6528, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_4: "bf16[6528, 1]" = torch.ops.aten.permute.default(empty_3, [0, 1]);  empty_3 = None
        view_11: "i32[6528]" = torch.ops.aten.reshape.default(view_9, [6528])
        index_put: "bf16[6528, 1]" = torch.ops.aten.index_put_.default(permute_4, [view_11], view_10);  permute_4 = view_11 = view_10 = None
        mul_6: "bf16[6528, 512]" = torch.ops.aten.mul.Tensor(mul_5, index_put);  mul_5 = index_put = None
        empty_4: "bf16[6528, 768]" = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 326, grid = [(306, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_6, 'b_ptr': convert_element_type_6, 'c_ptr': empty_4, 'cumsum_M': cumsum});  mul_6 = convert_element_type_6 = cumsum = triton_kernel_wrapper_mutation_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/fused_moe.py:363 in fused_moe_forward, code: final_hidden_states = FusedMoeExpertFunction.apply(
        empty_5: "bf16[1632, 768]" = torch.ops.aten.empty.memory_format([1632, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 3, constant_args_idx = 327, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': empty_4, 'Y': empty_5, 'index': view_9});  empty_4 = view_9 = triton_kernel_wrapper_mutation = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:355 in forward, code: final_hidden_states = final_hidden_states.to(hidden_flat.dtype)
        convert_element_type_10: "f32[1632, 768]" = torch.ops.prims.convert_element_type.default(empty_5, torch.float32);  empty_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_5: "f32[768, 704]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        mm_2: "f32[1632, 704]" = torch.ops.aten.mm.default(view_4, permute_5);  permute_5 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        sigmoid_2: "f32[1632, 704]" = torch.ops.aten.sigmoid.default(mm_2)
        mul_7: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mm_2, sigmoid_2);  mm_2 = sigmoid_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:110 in forward, code: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_6: "f32[768, 704]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        mm_3: "f32[1632, 704]" = torch.ops.aten.mm.default(view_4, permute_6);  view_4 = permute_6 = None
        mul_8: "f32[1632, 704]" = torch.ops.aten.mul.Tensor(mul_7, mm_3);  mul_7 = mm_3 = None
        permute_7: "f32[704, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        mm_4: "f32[1632, 768]" = torch.ops.aten.mm.default(mul_8, permute_7);  mul_8 = permute_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:359 in forward, code: final_hidden_states = final_hidden_states + shared_expert_output
        add_7: "f32[1632, 768]" = torch.ops.aten.add.Tensor(convert_element_type_10, mm_4);  convert_element_type_10 = mm_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:361 in forward, code: final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        view_14: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(add_7, [32, 51, 768]);  add_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:431 in forward, code: out_emb += after_first_residual
        add_8: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(view_14, add);  view_14 = add = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/__init__.py:2425 in __call__, code: return self.compiler_fn(model_, inputs_, **self.kwargs)
        copy_: "f32[1]" = torch.ops.aten.copy_.default(primals_12, view_5);  primals_12 = view_5 = copy_ = None
        copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_13, add_5);  add_5 = copy__1 = None
        return (add_8, mm_1, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19)
        