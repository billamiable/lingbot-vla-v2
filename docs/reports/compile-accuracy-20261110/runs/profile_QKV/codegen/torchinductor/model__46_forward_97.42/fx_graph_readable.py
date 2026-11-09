class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[58752]", primals_2: "i64[1152]", primals_3: "i64[1152]", primals_4: "f32[58752, 32]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:403 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: all_scores = all_logits.sigmoid()
        sigmoid: "f32[58752, 32]" = torch.ops.aten.sigmoid.default(primals_4)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:404 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: all_probs = all_scores / all_scores.sum(dim=-1, keepdim=True)
        sum_1: "f32[58752, 1]" = torch.ops.aten.sum.dim_IntList(sigmoid, [-1], True)
        div: "f32[58752, 32]" = torch.ops.aten.div.Tensor(sigmoid, sum_1);  sigmoid = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:407 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)  # [L * N_valid, E]
        unsqueeze: "i64[58752, 1]" = torch.ops.aten.unsqueeze.default(primals_1, 1)
        expand: "i64[58752, 32]" = torch.ops.aten.expand.default(unsqueeze, [-1, 32]);  unsqueeze = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:409 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: P_sum = torch.zeros(total_segments, E, device=device, dtype=torch.float32)
        full_default: "f32[1152, 32]" = torch.ops.aten.full.default([1152, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:410 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: P_sum.scatter_add_(0, seg_ids_exp, all_probs)
        scatter_add: "f32[1152, 32]" = torch.ops.aten.scatter_add.default(full_default, 0, expand, div);  expand = div = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:411 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: P_i = P_sum / seg_lengths_t.unsqueeze(1).float().clamp(min=1)  # [total_segments, E]
        unsqueeze_1: "i64[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 1)
        convert_element_type: "f32[1152, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        clamp_min: "f32[1152, 1]" = torch.ops.aten.clamp_min.default(convert_element_type, 1);  convert_element_type = None
        div_1: "f32[1152, 32]" = torch.ops.aten.div.Tensor(scatter_add, clamp_min);  scatter_add = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:280 in _triton_segment_f_i, code: _topk_segment_count_kernel[(S_total,)](
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(1152, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'logits_ptr': primals_4, 'seg_starts_ptr': primals_2, 'seg_lengths_ptr': primals_3, 'f_out_ptr': full_default}, tensors_to_clone = ['f_out_ptr']);  primals_2 = primals_3 = full_default = None
        getitem: "f32[1152, 32]" = triton_kernel_wrapper_functional_proxy['f_out_ptr'];  triton_kernel_wrapper_functional_proxy = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:292 in _triton_segment_f_i, code: inv_lens = (float(E) / top_k) / seg_lengths.unsqueeze(1).float().clamp(min=1)
        reciprocal: "f32[1152, 1]" = torch.ops.aten.reciprocal.default(clamp_min)
        mul: "f32[1152, 1]" = torch.ops.aten.mul.Tensor(reciprocal, 8.0);  reciprocal = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:293 in _triton_segment_f_i, code: return f_counts * inv_lens
        mul_1: "f32[1152, 32]" = torch.ops.aten.mul.Tensor(getitem, mul);  mul = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:421 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: loss_per_seg = (f_i * P_i).sum(dim=-1)  # [total_segments]
        mul_2: "f32[1152, 32]" = torch.ops.aten.mul.Tensor(mul_1, div_1);  mul_1 = div_1 = None
        sum_2: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1]);  mul_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:422 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: loss_per_seg = loss_per_seg.reshape(num_layers, S)
        view: "f32[36, 32]" = torch.ops.aten.view.default(sum_2, [36, 32]);  sum_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:423 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: layer_losses = loss_per_seg.mean(dim=1)  # [L]
        mean: "f32[36]" = torch.ops.aten.mean.dim(view, [1]);  view = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:425 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: return list(layer_losses.unbind(0))
        unbind = torch.ops.aten.unbind.int(mean);  mean = None
        getitem_1: "f32[]" = unbind[0]
        getitem_2: "f32[]" = unbind[1]
        getitem_3: "f32[]" = unbind[2]
        getitem_4: "f32[]" = unbind[3]
        getitem_5: "f32[]" = unbind[4]
        getitem_6: "f32[]" = unbind[5]
        getitem_7: "f32[]" = unbind[6]
        getitem_8: "f32[]" = unbind[7]
        getitem_9: "f32[]" = unbind[8]
        getitem_10: "f32[]" = unbind[9]
        getitem_11: "f32[]" = unbind[10]
        getitem_12: "f32[]" = unbind[11]
        getitem_13: "f32[]" = unbind[12]
        getitem_14: "f32[]" = unbind[13]
        getitem_15: "f32[]" = unbind[14]
        getitem_16: "f32[]" = unbind[15]
        getitem_17: "f32[]" = unbind[16]
        getitem_18: "f32[]" = unbind[17]
        getitem_19: "f32[]" = unbind[18]
        getitem_20: "f32[]" = unbind[19]
        getitem_21: "f32[]" = unbind[20]
        getitem_22: "f32[]" = unbind[21]
        getitem_23: "f32[]" = unbind[22]
        getitem_24: "f32[]" = unbind[23]
        getitem_25: "f32[]" = unbind[24]
        getitem_26: "f32[]" = unbind[25]
        getitem_27: "f32[]" = unbind[26]
        getitem_28: "f32[]" = unbind[27]
        getitem_29: "f32[]" = unbind[28]
        getitem_30: "f32[]" = unbind[29]
        getitem_31: "f32[]" = unbind[30]
        getitem_32: "f32[]" = unbind[31]
        getitem_33: "f32[]" = unbind[32]
        getitem_34: "f32[]" = unbind[33]
        getitem_35: "f32[]" = unbind[34]
        getitem_36: "f32[]" = unbind[35];  unbind = None
        return (getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, primals_1, primals_4, sum_1, clamp_min, getitem)
        