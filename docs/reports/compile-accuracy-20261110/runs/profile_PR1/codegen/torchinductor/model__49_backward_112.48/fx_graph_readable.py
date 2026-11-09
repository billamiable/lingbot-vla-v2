class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[58752]", primals_4: "f32[58752, 32]", sum_1: "f32[58752, 1]", clamp_min: "f32[1152, 1]", getitem: "f32[1152, 32]", tangents_1: "f32[]", tangents_2: "f32[]", tangents_3: "f32[]", tangents_4: "f32[]", tangents_5: "f32[]", tangents_6: "f32[]", tangents_7: "f32[]", tangents_8: "f32[]", tangents_9: "f32[]", tangents_10: "f32[]", tangents_11: "f32[]", tangents_12: "f32[]", tangents_13: "f32[]", tangents_14: "f32[]", tangents_15: "f32[]", tangents_16: "f32[]", tangents_17: "f32[]", tangents_18: "f32[]", tangents_19: "f32[]", tangents_20: "f32[]", tangents_21: "f32[]", tangents_22: "f32[]", tangents_23: "f32[]", tangents_24: "f32[]", tangents_25: "f32[]", tangents_26: "f32[]", tangents_27: "f32[]", tangents_28: "f32[]", tangents_29: "f32[]", tangents_30: "f32[]", tangents_31: "f32[]", tangents_32: "f32[]", tangents_33: "f32[]", tangents_34: "f32[]", tangents_35: "f32[]", tangents_36: "f32[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:425 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: return list(layer_losses.unbind(0))
        unsqueeze_3: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_1, 0);  tangents_1 = None
        unsqueeze_4: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_2, 0);  tangents_2 = None
        unsqueeze_5: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_3, 0);  tangents_3 = None
        unsqueeze_6: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_4, 0);  tangents_4 = None
        unsqueeze_7: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_5, 0);  tangents_5 = None
        unsqueeze_8: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_6, 0);  tangents_6 = None
        unsqueeze_9: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_7, 0);  tangents_7 = None
        unsqueeze_10: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_8, 0);  tangents_8 = None
        unsqueeze_11: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_9, 0);  tangents_9 = None
        unsqueeze_12: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_10, 0);  tangents_10 = None
        unsqueeze_13: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_11, 0);  tangents_11 = None
        unsqueeze_14: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_12, 0);  tangents_12 = None
        unsqueeze_15: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_13, 0);  tangents_13 = None
        unsqueeze_16: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_14, 0);  tangents_14 = None
        unsqueeze_17: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_15, 0);  tangents_15 = None
        unsqueeze_18: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_16, 0);  tangents_16 = None
        unsqueeze_19: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_17, 0);  tangents_17 = None
        unsqueeze_20: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_18, 0);  tangents_18 = None
        unsqueeze_21: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_19, 0);  tangents_19 = None
        unsqueeze_22: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_20, 0);  tangents_20 = None
        unsqueeze_23: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_21, 0);  tangents_21 = None
        unsqueeze_24: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_22, 0);  tangents_22 = None
        unsqueeze_25: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_23, 0);  tangents_23 = None
        unsqueeze_26: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_24, 0);  tangents_24 = None
        unsqueeze_27: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_25, 0);  tangents_25 = None
        unsqueeze_28: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_26, 0);  tangents_26 = None
        unsqueeze_29: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_27, 0);  tangents_27 = None
        unsqueeze_30: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_28, 0);  tangents_28 = None
        unsqueeze_31: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_29, 0);  tangents_29 = None
        unsqueeze_32: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_30, 0);  tangents_30 = None
        unsqueeze_33: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_31, 0);  tangents_31 = None
        unsqueeze_34: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_32, 0);  tangents_32 = None
        unsqueeze_35: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_33, 0);  tangents_33 = None
        unsqueeze_36: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_34, 0);  tangents_34 = None
        unsqueeze_37: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_35, 0);  tangents_35 = None
        unsqueeze_38: "f32[1]" = torch.ops.aten.unsqueeze.default(tangents_36, 0);  tangents_36 = None
        cat: "f32[36]" = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, unsqueeze_8, unsqueeze_9, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38]);  unsqueeze_3 = unsqueeze_4 = unsqueeze_5 = unsqueeze_6 = unsqueeze_7 = unsqueeze_8 = unsqueeze_9 = unsqueeze_10 = unsqueeze_11 = unsqueeze_12 = unsqueeze_13 = unsqueeze_14 = unsqueeze_15 = unsqueeze_16 = unsqueeze_17 = unsqueeze_18 = unsqueeze_19 = unsqueeze_20 = unsqueeze_21 = unsqueeze_22 = unsqueeze_23 = unsqueeze_24 = unsqueeze_25 = unsqueeze_26 = unsqueeze_27 = unsqueeze_28 = unsqueeze_29 = unsqueeze_30 = unsqueeze_31 = unsqueeze_32 = unsqueeze_33 = unsqueeze_34 = unsqueeze_35 = unsqueeze_36 = unsqueeze_37 = unsqueeze_38 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:423 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: layer_losses = loss_per_seg.mean(dim=1)  # [L]
        unsqueeze_39: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand_1: "f32[36, 32]" = torch.ops.aten.expand.default(unsqueeze_39, [36, 32]);  unsqueeze_39 = None
        div_2: "f32[36, 32]" = torch.ops.aten.div.Scalar(expand_1, 32);  expand_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:422 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: loss_per_seg = loss_per_seg.reshape(num_layers, S)
        view_1: "f32[1152]" = torch.ops.aten.view.default(div_2, [1152]);  div_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:421 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: loss_per_seg = (f_i * P_i).sum(dim=-1)  # [total_segments]
        unsqueeze_40: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        expand_2: "f32[1152, 32]" = torch.ops.aten.expand.default(unsqueeze_40, [1152, 32]);  unsqueeze_40 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:292 in _triton_segment_f_i, code: inv_lens = (float(E) / top_k) / seg_lengths.unsqueeze(1).float().clamp(min=1)
        reciprocal: "f32[1152, 1]" = torch.ops.aten.reciprocal.default(clamp_min)
        mul: "f32[1152, 1]" = torch.ops.aten.mul.Tensor(reciprocal, 8.0);  reciprocal = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:293 in _triton_segment_f_i, code: return f_counts * inv_lens
        mul_1: "f32[1152, 32]" = torch.ops.aten.mul.Tensor(getitem, mul);  getitem = mul = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:421 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: loss_per_seg = (f_i * P_i).sum(dim=-1)  # [total_segments]
        mul_3: "f32[1152, 32]" = torch.ops.aten.mul.Tensor(expand_2, mul_1);  expand_2 = mul_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:411 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: P_i = P_sum / seg_lengths_t.unsqueeze(1).float().clamp(min=1)  # [total_segments, E]
        div_3: "f32[1152, 32]" = torch.ops.aten.div.Tensor(mul_3, clamp_min);  mul_3 = clamp_min = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:407 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)  # [L * N_valid, E]
        unsqueeze: "i64[58752, 1]" = torch.ops.aten.unsqueeze.default(primals_1, 1);  primals_1 = None
        expand: "i64[58752, 32]" = torch.ops.aten.expand.default(unsqueeze, [-1, 32]);  unsqueeze = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:410 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: P_sum.scatter_add_(0, seg_ids_exp, all_probs)
        gather: "f32[58752, 32]" = torch.ops.aten.gather.default(div_3, 0, expand);  div_3 = expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:403 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: all_scores = all_logits.sigmoid()
        sigmoid: "f32[58752, 32]" = torch.ops.aten.sigmoid.default(primals_4);  primals_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:404 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: all_probs = all_scores / all_scores.sum(dim=-1, keepdim=True)
        div: "f32[58752, 32]" = torch.ops.aten.div.Tensor(sigmoid, sum_1)
        div_5: "f32[58752, 32]" = torch.ops.aten.div.Tensor(div, sum_1);  div = None
        neg: "f32[58752, 32]" = torch.ops.aten.neg.default(gather)
        mul_4: "f32[58752, 32]" = torch.ops.aten.mul.Tensor(neg, div_5);  neg = div_5 = None
        div_6: "f32[58752, 32]" = torch.ops.aten.div.Tensor(gather, sum_1);  gather = sum_1 = None
        sum_3: "f32[58752, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        expand_3: "f32[58752, 32]" = torch.ops.aten.expand.default(sum_3, [58752, 32]);  sum_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:404 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: all_probs = all_scores / all_scores.sum(dim=-1, keepdim=True)
        add: "f32[58752, 32]" = torch.ops.aten.add.Tensor(div_6, expand_3);  div_6 = expand_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:403 in torch_dynamo_resume_in_triton_sequence_wise_balance_loss_at_397, code: all_scores = all_logits.sigmoid()
        sub: "f32[58752, 32]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_5: "f32[58752, 32]" = torch.ops.aten.mul.Tensor(sigmoid, sub);  sigmoid = sub = None
        mul_6: "f32[58752, 32]" = torch.ops.aten.mul.Tensor(add, mul_5);  add = mul_5 = None
        return (None, None, None, mul_6)
        