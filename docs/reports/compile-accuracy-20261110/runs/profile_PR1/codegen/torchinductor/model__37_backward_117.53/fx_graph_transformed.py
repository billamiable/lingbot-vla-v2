class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2560, 2560]", primals_4: "bf16[2560, 2560]", primals_7: "bf16[2560]", primals_9: "bf16[2560]", primals_11: "bf16[128, 2560]", primals_12: "bf16[256, 2560]", primals_13: "bf16[2560, 128]", primals_14: "bf16[2560]", primals_16: "bf16[2560, 2560]", primals_17: "bf16[2560, 2560]", primals_18: "bf16[1024, 2560]", primals_20: "bf16[1024]", view: "bf16[8192, 2560]", addmm: "bf16[8192, 2560]", view_2: "bf16[2304, 2560]", addmm_1: "bf16[2304, 2560]", getitem_1: "f32[32, 72, 1]", rsqrt: "f32[32, 72, 1]", getitem_3: "f32[32, 256, 1]", rsqrt_1: "f32[32, 256, 1]", view_4: "bf16[8192, 2560]", view_6: "bf16[10496, 2560]", bmm: "bf16[128, 256, 328]", amax: "f32[32, 4, 256, 1]", sum_1: "f32[32, 4, 256, 1]", view_21: "bf16[8192, 128]", add_4: "bf16[32, 256, 2560]", getitem_7: "f32[32, 256, 1]", rsqrt_2: "f32[32, 256, 1]", view_23: "bf16[8192, 2560]", mm_3: "bf16[8192, 2560]", view_25: "bf16[8192, 2560]", view_27: "bf16[8192, 2560]", addmm_2: "bf16[8192, 1024]", getitem_9: "f32[32, 256, 1]", rsqrt_3: "f32[32, 256, 1]", permute_30: "bf16[128, 328, 256]", permute_31: "bf16[128, 32, 328]", permute_32: "bf16[128, 32, 256]", permute_33: "bf16[128, 328, 32]", tangents_1: "bf16[32, 256, 1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:201 in forward, code: queries = self.norm_out(queries)
        convert_element_type_35: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        convert_element_type_37: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_20, torch.float32);  primals_20 = None
        mul_14: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_37);  convert_element_type_37 = None
        mul_15: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_14, 1024)
        sum_2: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:200 in forward, code: queries = self.proj_out(queries)
        view_28: "bf16[32, 256, 1024]" = torch.ops.aten.reshape.default(addmm_2, [32, 256, 1024]);  addmm_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:201 in forward, code: queries = self.norm_out(queries)
        convert_element_type_33: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        sub_4: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_9);  convert_element_type_33 = getitem_9 = None
        mul_11: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_16: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_14, mul_11);  mul_14 = None
        sum_3: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [2], True);  mul_16 = None
        mul_17: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_11, sum_3);  sum_3 = None
        sub_6: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(mul_15, sum_2);  mul_15 = sum_2 = None
        sub_7: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(sub_6, mul_17);  sub_6 = mul_17 = None
        div_1: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
        mul_18: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_19: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, mul_11);  mul_11 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_19, [0, 1]);  mul_19 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_35, [0, 1]);  convert_element_type_35 = None
        convert_element_type_39: "bf16[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        convert_element_type_40: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_4, torch.bfloat16);  sum_4 = None
        convert_element_type_41: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:200 in forward, code: queries = self.proj_out(queries)
        view_29: "bf16[8192, 1024]" = torch.ops.aten.reshape.default(convert_element_type_39, [8192, 1024]);  convert_element_type_39 = None
        permute_12: "bf16[2560, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_13: "bf16[1024, 2560]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_5: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_29, permute_13);  permute_13 = None
        permute_14: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_6: "bf16[1024, 2560]" = torch.ops.aten.mm.default(permute_14, view_27);  permute_14 = view_27 = None
        sum_6: "bf16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_29, [0], True);  view_29 = None
        view_30: "bf16[1024]" = torch.ops.aten.reshape.default(sum_6, [1024]);  sum_6 = None
        view_31: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(mm_5, [32, 256, 2560]);  mm_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:198 in forward, code: queries = ff(queries) + queries
        view_32: "bf16[8192, 2560]" = torch.ops.aten.reshape.default(view_31, [8192, 2560])
        permute_17: "bf16[2560, 8192]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_7: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_17, view_25);  permute_17 = view_25 = None
        permute_11: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_19: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_8: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_32, permute_19);  view_32 = permute_19 = None
        view_33: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(mm_8, [32, 256, 2560]);  mm_8 = None
        convert_element_type_50: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        view_24: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(mm_3, [32, 256, 2560]);  mm_3 = None
        convert_element_type_26: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        mul_9: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf: "f32[32, 256, 2560]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[32, 256, 2560]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_21: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_22: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_23: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_22, -0.5);  mul_22 = None
        exp_1: "f32[32, 256, 2560]" = torch.ops.aten.exp.default(mul_23);  mul_23 = None
        mul_24: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_25: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_24);  convert_element_type_26 = mul_24 = None
        add_12: "f32[32, 256, 2560]" = torch.ops.aten.add.Tensor(mul_21, mul_25);  mul_21 = mul_25 = None
        mul_26: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_50, add_12);  convert_element_type_50 = add_12 = None
        convert_element_type_52: "bf16[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        view_34: "bf16[8192, 2560]" = torch.ops.aten.reshape.default(convert_element_type_52, [8192, 2560]);  convert_element_type_52 = None
        permute_21: "bf16[2560, 8192]" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_9: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_21, view_23);  permute_21 = view_23 = None
        permute_10: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_23: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_10: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_34, permute_23);  view_34 = permute_23 = None
        view_35: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(mm_10, [32, 256, 2560]);  mm_10 = None
        convert_element_type_57: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        convert_element_type_59: "f32[2560]" = torch.ops.prims.convert_element_type.default(primals_14, torch.float32);  primals_14 = None
        mul_28: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_57, convert_element_type_59);  convert_element_type_59 = None
        mul_29: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_28, 2560)
        sum_7: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_28, [2], True)
        convert_element_type_22: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        sub_3: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_7);  convert_element_type_22 = getitem_7 = None
        mul_6: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_30: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_28, mul_6);  mul_28 = None
        sum_8: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True);  mul_30 = None
        mul_31: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_6, sum_8);  sum_8 = None
        sub_9: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(mul_29, sum_7);  mul_29 = sum_7 = None
        sub_10: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(sub_9, mul_31);  sub_9 = mul_31 = None
        div_2: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2560);  rsqrt_2 = None
        mul_32: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(div_2, sub_10);  div_2 = sub_10 = None
        mul_33: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_57, mul_6);  mul_6 = None
        sum_9: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_33, [0, 1]);  mul_33 = None
        sum_10: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_57, [0, 1]);  convert_element_type_57 = None
        convert_element_type_61: "bf16[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        convert_element_type_62: "bf16[2560]" = torch.ops.prims.convert_element_type.default(sum_9, torch.bfloat16);  sum_9 = None
        convert_element_type_63: "bf16[2560]" = torch.ops.prims.convert_element_type.default(sum_10, torch.bfloat16);  sum_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:198 in forward, code: queries = ff(queries) + queries
        add_13: "bf16[32, 256, 2560]" = torch.ops.aten.add.Tensor(view_31, convert_element_type_61);  view_31 = convert_element_type_61 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:75 in forward, code: return self.to_out(out)
        view_36: "bf16[8192, 2560]" = torch.ops.aten.reshape.default(add_13, [8192, 2560])
        permute_25: "bf16[2560, 8192]" = torch.ops.aten.permute.default(view_36, [1, 0])
        mm_11: "bf16[2560, 128]" = torch.ops.aten.mm.default(permute_25, view_21);  permute_25 = view_21 = None
        permute_9: "bf16[128, 2560]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_27: "bf16[2560, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_12: "bf16[8192, 128]" = torch.ops.aten.mm.default(view_36, permute_27);  view_36 = permute_27 = None
        view_37: "bf16[32, 256, 128]" = torch.ops.aten.reshape.default(mm_12, [32, 256, 128]);  mm_12 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:73 in forward, code: out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        view_38: "bf16[32, 256, 4, 32]" = torch.ops.aten.reshape.default(view_37, [32, 256, 4, 32]);  view_37 = None
        permute_29: "bf16[32, 4, 256, 32]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:71 in forward, code: out = weight @ v
        clone_12: "bf16[32, 4, 256, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_39: "bf16[128, 256, 32]" = torch.ops.aten.reshape.default(clone_12, [128, 256, 32]);  clone_12 = None
        bmm_2: "bf16[128, 328, 32]" = torch.ops.aten.bmm.default(permute_30, view_39);  permute_30 = None
        bmm_3: "bf16[128, 256, 328]" = torch.ops.aten.bmm.default(view_39, permute_31);  view_39 = permute_31 = None
        view_40: "bf16[32, 4, 328, 32]" = torch.ops.aten.reshape.default(bmm_2, [32, 4, 328, 32]);  bmm_2 = None
        view_41: "bf16[32, 4, 256, 328]" = torch.ops.aten.reshape.default(bmm_3, [32, 4, 256, 328]);  bmm_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:70 in forward, code: weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        convert_element_type_72: "f32[32, 4, 256, 328]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:69 in forward, code: weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        view_16: "bf16[32, 4, 256, 328]" = torch.ops.aten.reshape.default(bmm, [32, 4, 256, 328]);  bmm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:70 in forward, code: weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        convert_element_type_16: "f32[32, 4, 256, 328]" = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        sub_2: "f32[32, 4, 256, 328]" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = amax = None
        exp: "f32[32, 4, 256, 328]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div: "f32[32, 4, 256, 328]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_34: "f32[32, 4, 256, 328]" = torch.ops.aten.mul.Tensor(convert_element_type_72, div);  convert_element_type_72 = None
        sum_11: "f32[32, 4, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_34, [-1], True)
        neg: "f32[32, 4, 256, 328]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[32, 4, 256, 328]" = torch.ops.prims.fma.default(neg, sum_11, mul_34);  neg = sum_11 = mul_34 = None
        convert_element_type_73: "bf16[32, 4, 256, 328]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:69 in forward, code: weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        view_42: "bf16[128, 256, 328]" = torch.ops.aten.reshape.default(convert_element_type_73, [128, 256, 328]);  convert_element_type_73 = None
        bmm_4: "bf16[128, 32, 328]" = torch.ops.aten.bmm.default(permute_32, view_42);  permute_32 = None
        bmm_5: "bf16[128, 256, 32]" = torch.ops.aten.bmm.default(view_42, permute_33);  view_42 = permute_33 = None
        view_43: "bf16[32, 4, 32, 328]" = torch.ops.aten.reshape.default(bmm_4, [32, 4, 32, 328]);  bmm_4 = None
        view_44: "bf16[32, 4, 256, 32]" = torch.ops.aten.reshape.default(bmm_5, [32, 4, 256, 32]);  bmm_5 = None
        permute_34: "bf16[32, 4, 328, 32]" = torch.ops.aten.permute.default(view_43, [0, 1, 3, 2]);  view_43 = None
        mul_35: "bf16[32, 4, 328, 32]" = torch.ops.aten.mul.Tensor(permute_34, 0.4204482076268573);  permute_34 = None
        mul_36: "bf16[32, 4, 256, 32]" = torch.ops.aten.mul.Tensor(view_44, 0.4204482076268573);  view_44 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:24 in reshape_tensor, code: x = x.transpose(1, 2)
        permute_35: "bf16[32, 328, 4, 32]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:22 in reshape_tensor, code: x = x.view(bs, length, heads, -1)
        clone_13: "bf16[32, 328, 4, 32]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_46: "bf16[32, 328, 128]" = torch.ops.aten.reshape.default(clone_13, [32, 328, 128]);  clone_13 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:24 in reshape_tensor, code: x = x.transpose(1, 2)
        permute_36: "bf16[32, 328, 4, 32]" = torch.ops.aten.permute.default(mul_35, [0, 2, 1, 3]);  mul_35 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:22 in reshape_tensor, code: x = x.view(bs, length, heads, -1)
        view_48: "bf16[32, 328, 128]" = torch.ops.aten.reshape.default(permute_36, [32, 328, 128]);  permute_36 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:24 in reshape_tensor, code: x = x.transpose(1, 2)
        permute_37: "bf16[32, 256, 4, 32]" = torch.ops.aten.permute.default(mul_36, [0, 2, 1, 3]);  mul_36 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:22 in reshape_tensor, code: x = x.view(bs, length, heads, -1)
        clone_14: "bf16[32, 256, 4, 32]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_50: "bf16[32, 256, 128]" = torch.ops.aten.reshape.default(clone_14, [32, 256, 128]);  clone_14 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:61 in forward, code: k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        cat_1: "bf16[32, 328, 256]" = torch.ops.aten.cat.default([view_48, view_46], 2);  view_48 = view_46 = None
        view_51: "bf16[10496, 256]" = torch.ops.aten.reshape.default(cat_1, [10496, 256]);  cat_1 = None
        permute_38: "bf16[256, 10496]" = torch.ops.aten.permute.default(view_51, [1, 0])
        mm_13: "bf16[256, 2560]" = torch.ops.aten.mm.default(permute_38, view_6);  permute_38 = view_6 = None
        permute_3: "bf16[2560, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_40: "bf16[256, 2560]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_14: "bf16[10496, 2560]" = torch.ops.aten.mm.default(view_51, permute_40);  view_51 = permute_40 = None
        view_52: "bf16[32, 328, 2560]" = torch.ops.aten.reshape.default(mm_14, [32, 328, 2560]);  mm_14 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:60 in forward, code: kv_input = torch.cat((x, latents), dim=-2)
        slice_1: "bf16[32, 72, 2560]" = torch.ops.aten.slice.Tensor(view_52, 1, 0, 72)
        slice_2: "bf16[32, 256, 2560]" = torch.ops.aten.slice.Tensor(view_52, 1, 72, 328);  view_52 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:59 in forward, code: q = self.to_q(latents)
        view_53: "bf16[8192, 128]" = torch.ops.aten.reshape.default(view_50, [8192, 128]);  view_50 = None
        permute_42: "bf16[128, 8192]" = torch.ops.aten.permute.default(view_53, [1, 0])
        mm_15: "bf16[128, 2560]" = torch.ops.aten.mm.default(permute_42, view_4);  permute_42 = view_4 = None
        permute_2: "bf16[2560, 128]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_44: "bf16[128, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_16: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_53, permute_44);  view_53 = permute_44 = None
        view_54: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(mm_16, [32, 256, 2560]);  mm_16 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:59 in forward, code: q = self.to_q(latents)
        add_14: "bf16[32, 256, 2560]" = torch.ops.aten.add.Tensor(slice_2, view_54);  slice_2 = view_54 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:55 in forward, code: latents = self.norm2(latents)
        convert_element_type_86: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        convert_element_type_88: "f32[2560]" = torch.ops.prims.convert_element_type.default(primals_9, torch.float32);  primals_9 = None
        mul_38: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_86, convert_element_type_88);  convert_element_type_88 = None
        mul_39: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_38, 2560)
        sum_12: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_38, [2], True)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:193 in forward, code: queries = self.proj_in1(queries)
        view_1: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(addmm, [32, 256, 2560]);  addmm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:55 in forward, code: latents = self.norm2(latents)
        convert_element_type_8: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub_1: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_3);  convert_element_type_8 = getitem_3 = None
        mul_2: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_40: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_38, mul_2);  mul_38 = None
        sum_13: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_40, [2], True);  mul_40 = None
        mul_41: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_2, sum_13);  sum_13 = None
        sub_12: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(mul_39, sum_12);  mul_39 = sum_12 = None
        sub_13: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(sub_12, mul_41);  sub_12 = mul_41 = None
        div_3: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_42: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(div_3, sub_13);  div_3 = sub_13 = None
        mul_43: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_86, mul_2);  mul_2 = None
        sum_14: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_43, [0, 1]);  mul_43 = None
        sum_15: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_86, [0, 1]);  convert_element_type_86 = None
        convert_element_type_90: "bf16[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        convert_element_type_91: "bf16[2560]" = torch.ops.prims.convert_element_type.default(sum_14, torch.bfloat16);  sum_14 = None
        convert_element_type_92: "bf16[2560]" = torch.ops.prims.convert_element_type.default(sum_15, torch.bfloat16);  sum_15 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:55 in forward, code: latents = self.norm2(latents)
        add_15: "bf16[32, 256, 2560]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_90);  add_13 = convert_element_type_90 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:54 in forward, code: x = self.norm1(x)
        convert_element_type_93: "f32[32, 72, 2560]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_95: "f32[2560]" = torch.ops.prims.convert_element_type.default(primals_7, torch.float32);  primals_7 = None
        mul_45: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_93, convert_element_type_95);  convert_element_type_95 = None
        mul_46: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(mul_45, 2560)
        sum_16: "f32[32, 72, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [2], True)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:194 in forward, code: x = self.proj_in2(x)
        view_3: "bf16[32, 72, 2560]" = torch.ops.aten.reshape.default(addmm_1, [32, 72, 2560]);  addmm_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:54 in forward, code: x = self.norm1(x)
        convert_element_type_6: "f32[32, 72, 2560]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sub: "f32[32, 72, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_1);  convert_element_type_6 = getitem_1 = None
        mul: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_47: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(mul_45, mul);  mul_45 = None
        sum_17: "f32[32, 72, 1]" = torch.ops.aten.sum.dim_IntList(mul_47, [2], True);  mul_47 = None
        mul_48: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(mul, sum_17);  sum_17 = None
        sub_15: "f32[32, 72, 2560]" = torch.ops.aten.sub.Tensor(mul_46, sum_16);  mul_46 = sum_16 = None
        sub_16: "f32[32, 72, 2560]" = torch.ops.aten.sub.Tensor(sub_15, mul_48);  sub_15 = mul_48 = None
        div_4: "f32[32, 72, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_49: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(div_4, sub_16);  div_4 = sub_16 = None
        mul_50: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_93, mul);  mul = None
        sum_18: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_50, [0, 1]);  mul_50 = None
        sum_19: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_93, [0, 1]);  convert_element_type_93 = None
        convert_element_type_97: "bf16[32, 72, 2560]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        convert_element_type_98: "bf16[2560]" = torch.ops.prims.convert_element_type.default(sum_18, torch.bfloat16);  sum_18 = None
        convert_element_type_99: "bf16[2560]" = torch.ops.prims.convert_element_type.default(sum_19, torch.bfloat16);  sum_19 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:194 in forward, code: x = self.proj_in2(x)
        view_55: "bf16[2304, 2560]" = torch.ops.aten.reshape.default(convert_element_type_97, [2304, 2560]);  convert_element_type_97 = None
        permute_1: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_46: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_17: "bf16[2304, 2560]" = torch.ops.aten.mm.default(view_55, permute_46);  permute_46 = None
        permute_47: "bf16[2560, 2304]" = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_18: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_47, view_2);  permute_47 = view_2 = None
        sum_20: "bf16[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        view_56: "bf16[2560]" = torch.ops.aten.reshape.default(sum_20, [2560]);  sum_20 = None
        view_57: "bf16[32, 72, 2560]" = torch.ops.aten.reshape.default(mm_17, [32, 72, 2560]);  mm_17 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:193 in forward, code: queries = self.proj_in1(queries)
        view_58: "bf16[8192, 2560]" = torch.ops.aten.reshape.default(add_15, [8192, 2560]);  add_15 = None
        permute: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_50: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_19: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_58, permute_50);  permute_50 = None
        permute_51: "bf16[2560, 8192]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_20: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_51, view);  permute_51 = view = None
        sum_21: "bf16[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True);  view_58 = None
        view_59: "bf16[2560]" = torch.ops.aten.reshape.default(sum_21, [2560]);  sum_21 = None
        view_60: "bf16[32, 256, 2560]" = torch.ops.aten.reshape.default(mm_19, [32, 256, 2560]);  mm_19 = None
        return (mm_20, view_59, view_60, mm_18, view_56, view_57, convert_element_type_98, convert_element_type_99, convert_element_type_91, convert_element_type_92, mm_15, mm_13, mm_11, convert_element_type_62, convert_element_type_63, mm_9, mm_7, mm_6, view_30, convert_element_type_40, convert_element_type_41)
        