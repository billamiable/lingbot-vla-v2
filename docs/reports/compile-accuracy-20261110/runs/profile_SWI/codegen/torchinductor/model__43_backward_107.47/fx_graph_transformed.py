class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[1632, 32]", primals_3: "f32[1632, 32]", primals_4: "f32[1632, 32]", primals_5: "f32[1632, 32]", primals_6: "f32[1632, 32]", primals_7: "f32[1632, 32]", primals_8: "f32[1632, 32]", primals_9: "f32[1632, 32]", primals_10: "f32[1632, 32]", primals_11: "f32[1632, 32]", primals_12: "f32[1632, 32]", primals_13: "f32[1632, 32]", primals_14: "f32[1632, 32]", primals_15: "f32[1632, 32]", primals_16: "f32[1632, 32]", primals_17: "f32[1632, 32]", primals_18: "f32[1632, 32]", primals_19: "f32[1632, 32]", primals_20: "f32[1632, 32]", primals_21: "f32[1632, 32]", primals_22: "f32[1632, 32]", primals_23: "f32[1632, 32]", primals_24: "f32[1632, 32]", primals_25: "f32[1632, 32]", primals_26: "f32[1632, 32]", primals_27: "f32[1632, 32]", primals_28: "f32[1632, 32]", primals_29: "f32[1632, 32]", primals_30: "f32[1632, 32]", primals_31: "f32[1632, 32]", primals_32: "f32[1632, 32]", primals_33: "f32[1632, 32]", primals_34: "f32[1632, 32]", primals_35: "f32[1632, 32]", primals_36: "f32[1632, 32]", primals_37: "f32[1632, 32]", add: "f32[1632]", add_1: "f32[1632]", add_2: "f32[1632]", add_3: "f32[1632]", add_4: "f32[1632]", add_5: "f32[1632]", add_6: "f32[1632]", add_7: "f32[1632]", add_8: "f32[1632]", add_9: "f32[1632]", add_10: "f32[1632]", add_11: "f32[1632]", add_12: "f32[1632]", add_13: "f32[1632]", add_14: "f32[1632]", add_15: "f32[1632]", add_16: "f32[1632]", add_17: "f32[1632]", add_18: "f32[1632]", add_19: "f32[1632]", add_20: "f32[1632]", add_21: "f32[1632]", add_22: "f32[1632]", add_23: "f32[1632]", add_24: "f32[1632]", add_25: "f32[1632]", add_26: "f32[1632]", add_27: "f32[1632]", add_28: "f32[1632]", add_29: "f32[1632]", add_30: "f32[1632]", add_31: "f32[1632]", add_32: "f32[1632]", add_33: "f32[1632]", add_34: "f32[1632]", add_35: "f32[1632]", tangents_1: "f32[]", tangents_2: "f32[]", tangents_3: "f32[]", tangents_4: "f32[]", tangents_5: "f32[]", tangents_6: "f32[]", tangents_7: "f32[]", tangents_8: "f32[]", tangents_9: "f32[]", tangents_10: "f32[]", tangents_11: "f32[]", tangents_12: "f32[]", tangents_13: "f32[]", tangents_14: "f32[]", tangents_15: "f32[]", tangents_16: "f32[]", tangents_17: "f32[]", tangents_18: "f32[]", tangents_19: "f32[]", tangents_20: "f32[]", tangents_21: "f32[]", tangents_22: "f32[]", tangents_23: "f32[]", tangents_24: "f32[]", tangents_25: "f32[]", tangents_26: "f32[]", tangents_27: "f32[]", tangents_28: "f32[]", tangents_29: "f32[]", tangents_30: "f32[]", tangents_31: "f32[]", tangents_32: "f32[]", tangents_33: "f32[]", tangents_34: "f32[]", tangents_35: "f32[]", tangents_36: "f32[]", tangents_37: "f32[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1156 in _moe_losses_and_metrics, code: router_z_loss = router_z_loss_coeff * torch.stack(router_z_layer_losses).mean()
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(tangents_1, 0.0001);  tangents_1 = None
        expand: "f32[36]" = torch.ops.aten.expand.default(mul_1, [36]);  mul_1 = None
        div: "f32[36]" = torch.ops.aten.div.Scalar(expand, 36);  expand = None
        select: "f32[]" = torch.ops.aten.select.int(div, 0, 0)
        select_1: "f32[]" = torch.ops.aten.select.int(div, 0, 1)
        select_2: "f32[]" = torch.ops.aten.select.int(div, 0, 2)
        select_3: "f32[]" = torch.ops.aten.select.int(div, 0, 3)
        select_4: "f32[]" = torch.ops.aten.select.int(div, 0, 4)
        select_5: "f32[]" = torch.ops.aten.select.int(div, 0, 5)
        select_6: "f32[]" = torch.ops.aten.select.int(div, 0, 6)
        select_7: "f32[]" = torch.ops.aten.select.int(div, 0, 7)
        select_8: "f32[]" = torch.ops.aten.select.int(div, 0, 8)
        select_9: "f32[]" = torch.ops.aten.select.int(div, 0, 9)
        select_10: "f32[]" = torch.ops.aten.select.int(div, 0, 10)
        select_11: "f32[]" = torch.ops.aten.select.int(div, 0, 11)
        select_12: "f32[]" = torch.ops.aten.select.int(div, 0, 12)
        select_13: "f32[]" = torch.ops.aten.select.int(div, 0, 13)
        select_14: "f32[]" = torch.ops.aten.select.int(div, 0, 14)
        select_15: "f32[]" = torch.ops.aten.select.int(div, 0, 15)
        select_16: "f32[]" = torch.ops.aten.select.int(div, 0, 16)
        select_17: "f32[]" = torch.ops.aten.select.int(div, 0, 17)
        select_18: "f32[]" = torch.ops.aten.select.int(div, 0, 18)
        select_19: "f32[]" = torch.ops.aten.select.int(div, 0, 19)
        select_20: "f32[]" = torch.ops.aten.select.int(div, 0, 20)
        select_21: "f32[]" = torch.ops.aten.select.int(div, 0, 21)
        select_22: "f32[]" = torch.ops.aten.select.int(div, 0, 22)
        select_23: "f32[]" = torch.ops.aten.select.int(div, 0, 23)
        select_24: "f32[]" = torch.ops.aten.select.int(div, 0, 24)
        select_25: "f32[]" = torch.ops.aten.select.int(div, 0, 25)
        select_26: "f32[]" = torch.ops.aten.select.int(div, 0, 26)
        select_27: "f32[]" = torch.ops.aten.select.int(div, 0, 27)
        select_28: "f32[]" = torch.ops.aten.select.int(div, 0, 28)
        select_29: "f32[]" = torch.ops.aten.select.int(div, 0, 29)
        select_30: "f32[]" = torch.ops.aten.select.int(div, 0, 30)
        select_31: "f32[]" = torch.ops.aten.select.int(div, 0, 31)
        select_32: "f32[]" = torch.ops.aten.select.int(div, 0, 32)
        select_33: "f32[]" = torch.ops.aten.select.int(div, 0, 33)
        select_34: "f32[]" = torch.ops.aten.select.int(div, 0, 34)
        select_35: "f32[]" = torch.ops.aten.select.int(div, 0, 35);  div = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1156 in _moe_losses_and_metrics, code: router_z_loss = router_z_loss_coeff * torch.stack(router_z_layer_losses).mean()
        add_36: "f32[]" = torch.ops.aten.add.Tensor(tangents_2, select);  tangents_2 = select = None
        add_37: "f32[]" = torch.ops.aten.add.Tensor(tangents_3, select_1);  tangents_3 = select_1 = None
        add_38: "f32[]" = torch.ops.aten.add.Tensor(tangents_4, select_2);  tangents_4 = select_2 = None
        add_39: "f32[]" = torch.ops.aten.add.Tensor(tangents_5, select_3);  tangents_5 = select_3 = None
        add_40: "f32[]" = torch.ops.aten.add.Tensor(tangents_6, select_4);  tangents_6 = select_4 = None
        add_41: "f32[]" = torch.ops.aten.add.Tensor(tangents_7, select_5);  tangents_7 = select_5 = None
        add_42: "f32[]" = torch.ops.aten.add.Tensor(tangents_8, select_6);  tangents_8 = select_6 = None
        add_43: "f32[]" = torch.ops.aten.add.Tensor(tangents_9, select_7);  tangents_9 = select_7 = None
        add_44: "f32[]" = torch.ops.aten.add.Tensor(tangents_10, select_8);  tangents_10 = select_8 = None
        add_45: "f32[]" = torch.ops.aten.add.Tensor(tangents_11, select_9);  tangents_11 = select_9 = None
        add_46: "f32[]" = torch.ops.aten.add.Tensor(tangents_12, select_10);  tangents_12 = select_10 = None
        add_47: "f32[]" = torch.ops.aten.add.Tensor(tangents_13, select_11);  tangents_13 = select_11 = None
        add_48: "f32[]" = torch.ops.aten.add.Tensor(tangents_14, select_12);  tangents_14 = select_12 = None
        add_49: "f32[]" = torch.ops.aten.add.Tensor(tangents_15, select_13);  tangents_15 = select_13 = None
        add_50: "f32[]" = torch.ops.aten.add.Tensor(tangents_16, select_14);  tangents_16 = select_14 = None
        add_51: "f32[]" = torch.ops.aten.add.Tensor(tangents_17, select_15);  tangents_17 = select_15 = None
        add_52: "f32[]" = torch.ops.aten.add.Tensor(tangents_18, select_16);  tangents_18 = select_16 = None
        add_53: "f32[]" = torch.ops.aten.add.Tensor(tangents_19, select_17);  tangents_19 = select_17 = None
        add_54: "f32[]" = torch.ops.aten.add.Tensor(tangents_20, select_18);  tangents_20 = select_18 = None
        add_55: "f32[]" = torch.ops.aten.add.Tensor(tangents_21, select_19);  tangents_21 = select_19 = None
        add_56: "f32[]" = torch.ops.aten.add.Tensor(tangents_22, select_20);  tangents_22 = select_20 = None
        add_57: "f32[]" = torch.ops.aten.add.Tensor(tangents_23, select_21);  tangents_23 = select_21 = None
        add_58: "f32[]" = torch.ops.aten.add.Tensor(tangents_24, select_22);  tangents_24 = select_22 = None
        add_59: "f32[]" = torch.ops.aten.add.Tensor(tangents_25, select_23);  tangents_25 = select_23 = None
        add_60: "f32[]" = torch.ops.aten.add.Tensor(tangents_26, select_24);  tangents_26 = select_24 = None
        add_61: "f32[]" = torch.ops.aten.add.Tensor(tangents_27, select_25);  tangents_27 = select_25 = None
        add_62: "f32[]" = torch.ops.aten.add.Tensor(tangents_28, select_26);  tangents_28 = select_26 = None
        add_63: "f32[]" = torch.ops.aten.add.Tensor(tangents_29, select_27);  tangents_29 = select_27 = None
        add_64: "f32[]" = torch.ops.aten.add.Tensor(tangents_30, select_28);  tangents_30 = select_28 = None
        add_65: "f32[]" = torch.ops.aten.add.Tensor(tangents_31, select_29);  tangents_31 = select_29 = None
        add_66: "f32[]" = torch.ops.aten.add.Tensor(tangents_32, select_30);  tangents_32 = select_30 = None
        add_67: "f32[]" = torch.ops.aten.add.Tensor(tangents_33, select_31);  tangents_33 = select_31 = None
        add_68: "f32[]" = torch.ops.aten.add.Tensor(tangents_34, select_32);  tangents_34 = select_32 = None
        add_69: "f32[]" = torch.ops.aten.add.Tensor(tangents_35, select_33);  tangents_35 = select_33 = None
        add_70: "f32[]" = torch.ops.aten.add.Tensor(tangents_36, select_34);  tangents_36 = select_34 = None
        add_71: "f32[]" = torch.ops.aten.add.Tensor(tangents_37, select_35);  tangents_37 = select_35 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:1153 in _moe_losses_and_metrics, code: torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        expand_1: "f32[1632]" = torch.ops.aten.expand.default(add_71, [1632]);  add_71 = None
        div_1: "f32[1632]" = torch.ops.aten.div.Scalar(expand_1, 1632);  expand_1 = None
        pow_37: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_35, 1.0)
        mul_2: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
        mul_3: "f32[1632]" = torch.ops.aten.mul.Tensor(div_1, mul_2);  div_1 = mul_2 = None
        unsqueeze_36: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_37: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_35, -1);  add_35 = None
        sub_36: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_37, unsqueeze_37);  primals_37 = unsqueeze_37 = None
        exp_36: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        mul_4: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_36, exp_36);  unsqueeze_36 = exp_36 = None
        expand_2: "f32[1632]" = torch.ops.aten.expand.default(add_70, [1632]);  add_70 = None
        div_2: "f32[1632]" = torch.ops.aten.div.Scalar(expand_2, 1632);  expand_2 = None
        pow_38: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 1.0)
        mul_5: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
        mul_6: "f32[1632]" = torch.ops.aten.mul.Tensor(div_2, mul_5);  div_2 = mul_5 = None
        unsqueeze_38: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_39: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_34, -1);  add_34 = None
        sub_37: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_36, unsqueeze_39);  primals_36 = unsqueeze_39 = None
        exp_37: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        mul_7: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_38, exp_37);  unsqueeze_38 = exp_37 = None
        expand_3: "f32[1632]" = torch.ops.aten.expand.default(add_69, [1632]);  add_69 = None
        div_3: "f32[1632]" = torch.ops.aten.div.Scalar(expand_3, 1632);  expand_3 = None
        pow_39: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 1.0)
        mul_8: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
        mul_9: "f32[1632]" = torch.ops.aten.mul.Tensor(div_3, mul_8);  div_3 = mul_8 = None
        unsqueeze_40: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_41: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_33, -1);  add_33 = None
        sub_38: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_35, unsqueeze_41);  primals_35 = unsqueeze_41 = None
        exp_38: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        mul_10: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_40, exp_38);  unsqueeze_40 = exp_38 = None
        expand_4: "f32[1632]" = torch.ops.aten.expand.default(add_68, [1632]);  add_68 = None
        div_4: "f32[1632]" = torch.ops.aten.div.Scalar(expand_4, 1632);  expand_4 = None
        pow_40: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_32, 1.0)
        mul_11: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
        mul_12: "f32[1632]" = torch.ops.aten.mul.Tensor(div_4, mul_11);  div_4 = mul_11 = None
        unsqueeze_42: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_43: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_32, -1);  add_32 = None
        sub_39: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_34, unsqueeze_43);  primals_34 = unsqueeze_43 = None
        exp_39: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        mul_13: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_42, exp_39);  unsqueeze_42 = exp_39 = None
        expand_5: "f32[1632]" = torch.ops.aten.expand.default(add_67, [1632]);  add_67 = None
        div_5: "f32[1632]" = torch.ops.aten.div.Scalar(expand_5, 1632);  expand_5 = None
        pow_41: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0)
        mul_14: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
        mul_15: "f32[1632]" = torch.ops.aten.mul.Tensor(div_5, mul_14);  div_5 = mul_14 = None
        unsqueeze_44: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_45: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_31, -1);  add_31 = None
        sub_40: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_33, unsqueeze_45);  primals_33 = unsqueeze_45 = None
        exp_40: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        mul_16: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_44, exp_40);  unsqueeze_44 = exp_40 = None
        expand_6: "f32[1632]" = torch.ops.aten.expand.default(add_66, [1632]);  add_66 = None
        div_6: "f32[1632]" = torch.ops.aten.div.Scalar(expand_6, 1632);  expand_6 = None
        pow_42: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_30, 1.0)
        mul_17: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
        mul_18: "f32[1632]" = torch.ops.aten.mul.Tensor(div_6, mul_17);  div_6 = mul_17 = None
        unsqueeze_46: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_47: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_30, -1);  add_30 = None
        sub_41: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_32, unsqueeze_47);  primals_32 = unsqueeze_47 = None
        exp_41: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        mul_19: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_46, exp_41);  unsqueeze_46 = exp_41 = None
        expand_7: "f32[1632]" = torch.ops.aten.expand.default(add_65, [1632]);  add_65 = None
        div_7: "f32[1632]" = torch.ops.aten.div.Scalar(expand_7, 1632);  expand_7 = None
        pow_43: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 1.0)
        mul_20: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
        mul_21: "f32[1632]" = torch.ops.aten.mul.Tensor(div_7, mul_20);  div_7 = mul_20 = None
        unsqueeze_48: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_49: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_29, -1);  add_29 = None
        sub_42: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_31, unsqueeze_49);  primals_31 = unsqueeze_49 = None
        exp_42: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        mul_22: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_48, exp_42);  unsqueeze_48 = exp_42 = None
        expand_8: "f32[1632]" = torch.ops.aten.expand.default(add_64, [1632]);  add_64 = None
        div_8: "f32[1632]" = torch.ops.aten.div.Scalar(expand_8, 1632);  expand_8 = None
        pow_44: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 1.0)
        mul_23: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
        mul_24: "f32[1632]" = torch.ops.aten.mul.Tensor(div_8, mul_23);  div_8 = mul_23 = None
        unsqueeze_50: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_51: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_28, -1);  add_28 = None
        sub_43: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_30, unsqueeze_51);  primals_30 = unsqueeze_51 = None
        exp_43: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        mul_25: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_50, exp_43);  unsqueeze_50 = exp_43 = None
        expand_9: "f32[1632]" = torch.ops.aten.expand.default(add_63, [1632]);  add_63 = None
        div_9: "f32[1632]" = torch.ops.aten.div.Scalar(expand_9, 1632);  expand_9 = None
        pow_45: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 1.0)
        mul_26: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
        mul_27: "f32[1632]" = torch.ops.aten.mul.Tensor(div_9, mul_26);  div_9 = mul_26 = None
        unsqueeze_52: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_53: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_27, -1);  add_27 = None
        sub_44: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_29, unsqueeze_53);  primals_29 = unsqueeze_53 = None
        exp_44: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        mul_28: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_52, exp_44);  unsqueeze_52 = exp_44 = None
        expand_10: "f32[1632]" = torch.ops.aten.expand.default(add_62, [1632]);  add_62 = None
        div_10: "f32[1632]" = torch.ops.aten.div.Scalar(expand_10, 1632);  expand_10 = None
        pow_46: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0)
        mul_29: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
        mul_30: "f32[1632]" = torch.ops.aten.mul.Tensor(div_10, mul_29);  div_10 = mul_29 = None
        unsqueeze_54: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_55: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_26, -1);  add_26 = None
        sub_45: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_28, unsqueeze_55);  primals_28 = unsqueeze_55 = None
        exp_45: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        mul_31: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_54, exp_45);  unsqueeze_54 = exp_45 = None
        expand_11: "f32[1632]" = torch.ops.aten.expand.default(add_61, [1632]);  add_61 = None
        div_11: "f32[1632]" = torch.ops.aten.div.Scalar(expand_11, 1632);  expand_11 = None
        pow_47: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_25, 1.0)
        mul_32: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
        mul_33: "f32[1632]" = torch.ops.aten.mul.Tensor(div_11, mul_32);  div_11 = mul_32 = None
        unsqueeze_56: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_57: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_25, -1);  add_25 = None
        sub_46: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_27, unsqueeze_57);  primals_27 = unsqueeze_57 = None
        exp_46: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        mul_34: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_56, exp_46);  unsqueeze_56 = exp_46 = None
        expand_12: "f32[1632]" = torch.ops.aten.expand.default(add_60, [1632]);  add_60 = None
        div_12: "f32[1632]" = torch.ops.aten.div.Scalar(expand_12, 1632);  expand_12 = None
        pow_48: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 1.0)
        mul_35: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
        mul_36: "f32[1632]" = torch.ops.aten.mul.Tensor(div_12, mul_35);  div_12 = mul_35 = None
        unsqueeze_58: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_59: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_24, -1);  add_24 = None
        sub_47: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_26, unsqueeze_59);  primals_26 = unsqueeze_59 = None
        exp_47: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        mul_37: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_58, exp_47);  unsqueeze_58 = exp_47 = None
        expand_13: "f32[1632]" = torch.ops.aten.expand.default(add_59, [1632]);  add_59 = None
        div_13: "f32[1632]" = torch.ops.aten.div.Scalar(expand_13, 1632);  expand_13 = None
        pow_49: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 1.0)
        mul_38: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
        mul_39: "f32[1632]" = torch.ops.aten.mul.Tensor(div_13, mul_38);  div_13 = mul_38 = None
        unsqueeze_60: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_61: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_23, -1);  add_23 = None
        sub_48: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_25, unsqueeze_61);  primals_25 = unsqueeze_61 = None
        exp_48: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        mul_40: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_60, exp_48);  unsqueeze_60 = exp_48 = None
        expand_14: "f32[1632]" = torch.ops.aten.expand.default(add_58, [1632]);  add_58 = None
        div_14: "f32[1632]" = torch.ops.aten.div.Scalar(expand_14, 1632);  expand_14 = None
        pow_50: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_22, 1.0)
        mul_41: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
        mul_42: "f32[1632]" = torch.ops.aten.mul.Tensor(div_14, mul_41);  div_14 = mul_41 = None
        unsqueeze_62: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_63: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_22, -1);  add_22 = None
        sub_49: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_24, unsqueeze_63);  primals_24 = unsqueeze_63 = None
        exp_49: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        mul_43: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_62, exp_49);  unsqueeze_62 = exp_49 = None
        expand_15: "f32[1632]" = torch.ops.aten.expand.default(add_57, [1632]);  add_57 = None
        div_15: "f32[1632]" = torch.ops.aten.div.Scalar(expand_15, 1632);  expand_15 = None
        pow_51: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0)
        mul_44: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_51, 2.0);  pow_51 = None
        mul_45: "f32[1632]" = torch.ops.aten.mul.Tensor(div_15, mul_44);  div_15 = mul_44 = None
        unsqueeze_64: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_65: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_21, -1);  add_21 = None
        sub_50: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_23, unsqueeze_65);  primals_23 = unsqueeze_65 = None
        exp_50: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
        mul_46: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_64, exp_50);  unsqueeze_64 = exp_50 = None
        expand_16: "f32[1632]" = torch.ops.aten.expand.default(add_56, [1632]);  add_56 = None
        div_16: "f32[1632]" = torch.ops.aten.div.Scalar(expand_16, 1632);  expand_16 = None
        pow_52: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_20, 1.0)
        mul_47: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
        mul_48: "f32[1632]" = torch.ops.aten.mul.Tensor(div_16, mul_47);  div_16 = mul_47 = None
        unsqueeze_66: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_67: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_20, -1);  add_20 = None
        sub_51: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_22, unsqueeze_67);  primals_22 = unsqueeze_67 = None
        exp_51: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        mul_49: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_66, exp_51);  unsqueeze_66 = exp_51 = None
        expand_17: "f32[1632]" = torch.ops.aten.expand.default(add_55, [1632]);  add_55 = None
        div_17: "f32[1632]" = torch.ops.aten.div.Scalar(expand_17, 1632);  expand_17 = None
        pow_53: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 1.0)
        mul_50: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_53, 2.0);  pow_53 = None
        mul_51: "f32[1632]" = torch.ops.aten.mul.Tensor(div_17, mul_50);  div_17 = mul_50 = None
        unsqueeze_68: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
        unsqueeze_69: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_19, -1);  add_19 = None
        sub_52: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_21, unsqueeze_69);  primals_21 = unsqueeze_69 = None
        exp_52: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        mul_52: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_68, exp_52);  unsqueeze_68 = exp_52 = None
        expand_18: "f32[1632]" = torch.ops.aten.expand.default(add_54, [1632]);  add_54 = None
        div_18: "f32[1632]" = torch.ops.aten.div.Scalar(expand_18, 1632);  expand_18 = None
        pow_54: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 1.0)
        mul_53: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
        mul_54: "f32[1632]" = torch.ops.aten.mul.Tensor(div_18, mul_53);  div_18 = mul_53 = None
        unsqueeze_70: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
        unsqueeze_71: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_18, -1);  add_18 = None
        sub_53: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_20, unsqueeze_71);  primals_20 = unsqueeze_71 = None
        exp_53: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        mul_55: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_70, exp_53);  unsqueeze_70 = exp_53 = None
        expand_19: "f32[1632]" = torch.ops.aten.expand.default(add_53, [1632]);  add_53 = None
        div_19: "f32[1632]" = torch.ops.aten.div.Scalar(expand_19, 1632);  expand_19 = None
        pow_55: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_17, 1.0)
        mul_56: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_55, 2.0);  pow_55 = None
        mul_57: "f32[1632]" = torch.ops.aten.mul.Tensor(div_19, mul_56);  div_19 = mul_56 = None
        unsqueeze_72: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_73: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_17, -1);  add_17 = None
        sub_54: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_19, unsqueeze_73);  primals_19 = unsqueeze_73 = None
        exp_54: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
        mul_58: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_72, exp_54);  unsqueeze_72 = exp_54 = None
        expand_20: "f32[1632]" = torch.ops.aten.expand.default(add_52, [1632]);  add_52 = None
        div_20: "f32[1632]" = torch.ops.aten.div.Scalar(expand_20, 1632);  expand_20 = None
        pow_56: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0)
        mul_59: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
        mul_60: "f32[1632]" = torch.ops.aten.mul.Tensor(div_20, mul_59);  div_20 = mul_59 = None
        unsqueeze_74: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
        unsqueeze_75: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_16, -1);  add_16 = None
        sub_55: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_18, unsqueeze_75);  primals_18 = unsqueeze_75 = None
        exp_55: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        mul_61: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_74, exp_55);  unsqueeze_74 = exp_55 = None
        expand_21: "f32[1632]" = torch.ops.aten.expand.default(add_51, [1632]);  add_51 = None
        div_21: "f32[1632]" = torch.ops.aten.div.Scalar(expand_21, 1632);  expand_21 = None
        pow_57: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_15, 1.0)
        mul_62: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_57, 2.0);  pow_57 = None
        mul_63: "f32[1632]" = torch.ops.aten.mul.Tensor(div_21, mul_62);  div_21 = mul_62 = None
        unsqueeze_76: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
        unsqueeze_77: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_15, -1);  add_15 = None
        sub_56: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_17, unsqueeze_77);  primals_17 = unsqueeze_77 = None
        exp_56: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
        mul_64: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_76, exp_56);  unsqueeze_76 = exp_56 = None
        expand_22: "f32[1632]" = torch.ops.aten.expand.default(add_50, [1632]);  add_50 = None
        div_22: "f32[1632]" = torch.ops.aten.div.Scalar(expand_22, 1632);  expand_22 = None
        pow_58: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 1.0)
        mul_65: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
        mul_66: "f32[1632]" = torch.ops.aten.mul.Tensor(div_22, mul_65);  div_22 = mul_65 = None
        unsqueeze_78: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
        unsqueeze_79: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_14, -1);  add_14 = None
        sub_57: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_16, unsqueeze_79);  primals_16 = unsqueeze_79 = None
        exp_57: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        mul_67: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_78, exp_57);  unsqueeze_78 = exp_57 = None
        expand_23: "f32[1632]" = torch.ops.aten.expand.default(add_49, [1632]);  add_49 = None
        div_23: "f32[1632]" = torch.ops.aten.div.Scalar(expand_23, 1632);  expand_23 = None
        pow_59: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 1.0)
        mul_68: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_59, 2.0);  pow_59 = None
        mul_69: "f32[1632]" = torch.ops.aten.mul.Tensor(div_23, mul_68);  div_23 = mul_68 = None
        unsqueeze_80: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
        unsqueeze_81: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_13, -1);  add_13 = None
        sub_58: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_15, unsqueeze_81);  primals_15 = unsqueeze_81 = None
        exp_58: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        mul_70: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_80, exp_58);  unsqueeze_80 = exp_58 = None
        expand_24: "f32[1632]" = torch.ops.aten.expand.default(add_48, [1632]);  add_48 = None
        div_24: "f32[1632]" = torch.ops.aten.div.Scalar(expand_24, 1632);  expand_24 = None
        pow_60: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_12, 1.0)
        mul_71: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_72: "f32[1632]" = torch.ops.aten.mul.Tensor(div_24, mul_71);  div_24 = mul_71 = None
        unsqueeze_82: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
        unsqueeze_83: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_12, -1);  add_12 = None
        sub_59: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_14, unsqueeze_83);  primals_14 = unsqueeze_83 = None
        exp_59: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
        mul_73: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_82, exp_59);  unsqueeze_82 = exp_59 = None
        expand_25: "f32[1632]" = torch.ops.aten.expand.default(add_47, [1632]);  add_47 = None
        div_25: "f32[1632]" = torch.ops.aten.div.Scalar(expand_25, 1632);  expand_25 = None
        pow_61: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0)
        mul_74: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_61, 2.0);  pow_61 = None
        mul_75: "f32[1632]" = torch.ops.aten.mul.Tensor(div_25, mul_74);  div_25 = mul_74 = None
        unsqueeze_84: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
        unsqueeze_85: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_11, -1);  add_11 = None
        sub_60: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_13, unsqueeze_85);  primals_13 = unsqueeze_85 = None
        exp_60: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        mul_76: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_84, exp_60);  unsqueeze_84 = exp_60 = None
        expand_26: "f32[1632]" = torch.ops.aten.expand.default(add_46, [1632]);  add_46 = None
        div_26: "f32[1632]" = torch.ops.aten.div.Scalar(expand_26, 1632);  expand_26 = None
        pow_62: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_10, 1.0)
        mul_77: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
        mul_78: "f32[1632]" = torch.ops.aten.mul.Tensor(div_26, mul_77);  div_26 = mul_77 = None
        unsqueeze_86: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
        unsqueeze_87: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_10, -1);  add_10 = None
        sub_61: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_12, unsqueeze_87);  primals_12 = unsqueeze_87 = None
        exp_61: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        mul_79: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_86, exp_61);  unsqueeze_86 = exp_61 = None
        expand_27: "f32[1632]" = torch.ops.aten.expand.default(add_45, [1632]);  add_45 = None
        div_27: "f32[1632]" = torch.ops.aten.div.Scalar(expand_27, 1632);  expand_27 = None
        pow_63: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0)
        mul_80: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
        mul_81: "f32[1632]" = torch.ops.aten.mul.Tensor(div_27, mul_80);  div_27 = mul_80 = None
        unsqueeze_88: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
        unsqueeze_89: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_9, -1);  add_9 = None
        sub_62: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_11, unsqueeze_89);  primals_11 = unsqueeze_89 = None
        exp_62: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_62);  sub_62 = None
        mul_82: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_88, exp_62);  unsqueeze_88 = exp_62 = None
        expand_28: "f32[1632]" = torch.ops.aten.expand.default(add_44, [1632]);  add_44 = None
        div_28: "f32[1632]" = torch.ops.aten.div.Scalar(expand_28, 1632);  expand_28 = None
        pow_64: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 1.0)
        mul_83: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_84: "f32[1632]" = torch.ops.aten.mul.Tensor(div_28, mul_83);  div_28 = mul_83 = None
        unsqueeze_90: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
        unsqueeze_91: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_8, -1);  add_8 = None
        sub_63: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_10, unsqueeze_91);  primals_10 = unsqueeze_91 = None
        exp_63: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        mul_85: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_90, exp_63);  unsqueeze_90 = exp_63 = None
        expand_29: "f32[1632]" = torch.ops.aten.expand.default(add_43, [1632]);  add_43 = None
        div_29: "f32[1632]" = torch.ops.aten.div.Scalar(expand_29, 1632);  expand_29 = None
        pow_65: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_7, 1.0)
        mul_86: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
        mul_87: "f32[1632]" = torch.ops.aten.mul.Tensor(div_29, mul_86);  div_29 = mul_86 = None
        unsqueeze_92: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
        unsqueeze_93: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_7, -1);  add_7 = None
        sub_64: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_9, unsqueeze_93);  primals_9 = unsqueeze_93 = None
        exp_64: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        mul_88: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_92, exp_64);  unsqueeze_92 = exp_64 = None
        expand_30: "f32[1632]" = torch.ops.aten.expand.default(add_42, [1632]);  add_42 = None
        div_30: "f32[1632]" = torch.ops.aten.div.Scalar(expand_30, 1632);  expand_30 = None
        pow_66: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 1.0)
        mul_89: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_90: "f32[1632]" = torch.ops.aten.mul.Tensor(div_30, mul_89);  div_30 = mul_89 = None
        unsqueeze_94: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
        unsqueeze_95: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
        sub_65: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_8, unsqueeze_95);  primals_8 = unsqueeze_95 = None
        exp_65: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        mul_91: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_94, exp_65);  unsqueeze_94 = exp_65 = None
        expand_31: "f32[1632]" = torch.ops.aten.expand.default(add_41, [1632]);  add_41 = None
        div_31: "f32[1632]" = torch.ops.aten.div.Scalar(expand_31, 1632);  expand_31 = None
        pow_67: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_5, 1.0)
        mul_92: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
        mul_93: "f32[1632]" = torch.ops.aten.mul.Tensor(div_31, mul_92);  div_31 = mul_92 = None
        unsqueeze_96: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
        unsqueeze_97: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_5, -1);  add_5 = None
        sub_66: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_7, unsqueeze_97);  primals_7 = unsqueeze_97 = None
        exp_66: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
        mul_94: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_96, exp_66);  unsqueeze_96 = exp_66 = None
        expand_32: "f32[1632]" = torch.ops.aten.expand.default(add_40, [1632]);  add_40 = None
        div_32: "f32[1632]" = torch.ops.aten.div.Scalar(expand_32, 1632);  expand_32 = None
        pow_68: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_4, 1.0)
        mul_95: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_96: "f32[1632]" = torch.ops.aten.mul.Tensor(div_32, mul_95);  div_32 = mul_95 = None
        unsqueeze_98: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
        unsqueeze_99: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_4, -1);  add_4 = None
        sub_67: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_6, unsqueeze_99);  primals_6 = unsqueeze_99 = None
        exp_67: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        mul_97: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_98, exp_67);  unsqueeze_98 = exp_67 = None
        expand_33: "f32[1632]" = torch.ops.aten.expand.default(add_39, [1632]);  add_39 = None
        div_33: "f32[1632]" = torch.ops.aten.div.Scalar(expand_33, 1632);  expand_33 = None
        pow_69: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_3, 1.0)
        mul_98: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_69, 2.0);  pow_69 = None
        mul_99: "f32[1632]" = torch.ops.aten.mul.Tensor(div_33, mul_98);  div_33 = mul_98 = None
        unsqueeze_100: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
        unsqueeze_101: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_3, -1);  add_3 = None
        sub_68: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_5, unsqueeze_101);  primals_5 = unsqueeze_101 = None
        exp_68: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        mul_100: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_100, exp_68);  unsqueeze_100 = exp_68 = None
        expand_34: "f32[1632]" = torch.ops.aten.expand.default(add_38, [1632]);  add_38 = None
        div_34: "f32[1632]" = torch.ops.aten.div.Scalar(expand_34, 1632);  expand_34 = None
        pow_70: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_2, 1.0)
        mul_101: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_102: "f32[1632]" = torch.ops.aten.mul.Tensor(div_34, mul_101);  div_34 = mul_101 = None
        unsqueeze_102: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
        unsqueeze_103: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_2, -1);  add_2 = None
        sub_69: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_4, unsqueeze_103);  primals_4 = unsqueeze_103 = None
        exp_69: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
        mul_103: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_102, exp_69);  unsqueeze_102 = exp_69 = None
        expand_35: "f32[1632]" = torch.ops.aten.expand.default(add_37, [1632]);  add_37 = None
        div_35: "f32[1632]" = torch.ops.aten.div.Scalar(expand_35, 1632);  expand_35 = None
        pow_71: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add_1, 1.0)
        mul_104: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_71, 2.0);  pow_71 = None
        mul_105: "f32[1632]" = torch.ops.aten.mul.Tensor(div_35, mul_104);  div_35 = mul_104 = None
        unsqueeze_104: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
        unsqueeze_105: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add_1, -1);  add_1 = None
        sub_70: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_3, unsqueeze_105);  primals_3 = unsqueeze_105 = None
        exp_70: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        mul_106: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_104, exp_70);  unsqueeze_104 = exp_70 = None
        expand_36: "f32[1632]" = torch.ops.aten.expand.default(add_36, [1632]);  add_36 = None
        div_36: "f32[1632]" = torch.ops.aten.div.Scalar(expand_36, 1632);  expand_36 = None
        pow_72: "f32[1632]" = torch.ops.aten.pow.Tensor_Scalar(add, 1.0)
        mul_107: "f32[1632]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_108: "f32[1632]" = torch.ops.aten.mul.Tensor(div_36, mul_107);  div_36 = mul_107 = None
        unsqueeze_106: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
        unsqueeze_107: "f32[1632, 1]" = torch.ops.aten.unsqueeze.default(add, -1);  add = None
        sub_71: "f32[1632, 32]" = torch.ops.aten.sub.Tensor(primals_2, unsqueeze_107);  primals_2 = unsqueeze_107 = None
        exp_71: "f32[1632, 32]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        mul_109: "f32[1632, 32]" = torch.ops.aten.mul.Tensor(unsqueeze_106, exp_71);  unsqueeze_106 = exp_71 = None
        return (None, mul_109, mul_106, mul_103, mul_100, mul_97, mul_94, mul_91, mul_88, mul_85, mul_82, mul_79, mul_76, mul_73, mul_70, mul_67, mul_64, mul_61, mul_58, mul_55, mul_52, mul_49, mul_46, mul_43, mul_40, mul_37, mul_34, mul_31, mul_28, mul_25, mul_22, mul_19, mul_16, mul_13, mul_10, mul_7, mul_4)
        