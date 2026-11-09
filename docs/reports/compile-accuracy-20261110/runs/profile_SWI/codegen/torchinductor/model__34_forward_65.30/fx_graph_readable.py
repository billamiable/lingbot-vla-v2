class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2560, 2560]", primals_2: "bf16[2560]", primals_3: "bf16[32, 256, 2560]", primals_4: "bf16[2560, 2560]", primals_5: "bf16[2560]", primals_6: "bf16[32, 72, 2560]", primals_7: "bf16[2560]", primals_8: "bf16[2560]", primals_9: "bf16[2560]", primals_10: "bf16[2560]", primals_11: "bf16[128, 2560]", primals_12: "bf16[256, 2560]", primals_13: "bf16[2560, 128]", primals_14: "bf16[2560]", primals_15: "bf16[2560]", primals_16: "bf16[2560, 2560]", primals_17: "bf16[2560, 2560]", primals_18: "bf16[1024, 2560]", primals_19: "bf16[1024]", primals_20: "bf16[1024]", primals_21: "bf16[1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:193 in forward, code: queries = self.proj_in1(queries)
        view: "bf16[8192, 2560]" = torch.ops.aten.view.default(primals_3, [8192, 2560]);  primals_3 = None
        permute: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_1, [1, 0])
        addmm: "bf16[8192, 2560]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = permute = None
        view_1: "bf16[32, 256, 2560]" = torch.ops.aten.view.default(addmm, [32, 256, 2560])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:194 in forward, code: x = self.proj_in2(x)
        view_2: "bf16[2304, 2560]" = torch.ops.aten.view.default(primals_6, [2304, 2560]);  primals_6 = None
        permute_1: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "bf16[2304, 2560]" = torch.ops.aten.addmm.default(primals_5, view_2, permute_1);  primals_5 = permute_1 = None
        view_3: "bf16[32, 72, 2560]" = torch.ops.aten.view.default(addmm_1, [32, 72, 2560])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:54 in forward, code: x = self.norm1(x)
        convert_element_type_6: "f32[32, 72, 2560]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_6, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 72, 1]" = var_mean[0]
        getitem_1: "f32[32, 72, 1]" = var_mean[1];  var_mean = None
        add: "f32[32, 72, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 72, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 72, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_1);  convert_element_type_6 = None
        mul: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 72, 2560]" = torch.ops.aten.mul.Tensor(mul, primals_7);  mul = None
        add_1: "f32[32, 72, 2560]" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None
        convert_element_type_7: "bf16[32, 72, 2560]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:55 in forward, code: latents = self.norm2(latents)
        convert_element_type_8: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 256, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 256, 1]" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_3);  convert_element_type_8 = None
        mul_2: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_2, primals_9);  mul_2 = None
        add_3: "f32[32, 256, 2560]" = torch.ops.aten.add.Tensor(mul_3, primals_10);  mul_3 = primals_10 = None
        convert_element_type_9: "bf16[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:59 in forward, code: q = self.to_q(latents)
        permute_2: "bf16[2560, 128]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        view_4: "bf16[8192, 2560]" = torch.ops.aten.view.default(convert_element_type_9, [8192, 2560])
        mm: "bf16[8192, 128]" = torch.ops.aten.mm.default(view_4, permute_2);  permute_2 = None
        view_5: "bf16[32, 256, 128]" = torch.ops.aten.view.default(mm, [32, 256, 128]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:60 in forward, code: kv_input = torch.cat((x, latents), dim=-2)
        cat: "bf16[32, 328, 2560]" = torch.ops.aten.cat.default([convert_element_type_7, convert_element_type_9], -2);  convert_element_type_7 = convert_element_type_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:61 in forward, code: k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        permute_3: "bf16[2560, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        view_6: "bf16[10496, 2560]" = torch.ops.aten.view.default(cat, [10496, 2560]);  cat = None
        mm_1: "bf16[10496, 256]" = torch.ops.aten.mm.default(view_6, permute_3);  permute_3 = None
        view_7: "bf16[32, 328, 256]" = torch.ops.aten.view.default(mm_1, [32, 328, 256]);  mm_1 = None
        split = torch.ops.aten.split.Tensor(view_7, 128, -1);  view_7 = None
        getitem_4: "bf16[32, 328, 128]" = split[0]
        getitem_5: "bf16[32, 328, 128]" = split[1];  split = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:22 in reshape_tensor, code: x = x.view(bs, length, heads, -1)
        view_8: "bf16[32, 256, 4, 32]" = torch.ops.aten.view.default(view_5, [32, 256, 4, -1]);  view_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:24 in reshape_tensor, code: x = x.transpose(1, 2)
        permute_4: "bf16[32, 4, 256, 32]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:22 in reshape_tensor, code: x = x.view(bs, length, heads, -1)
        view_10: "bf16[32, 328, 4, 32]" = torch.ops.aten.view.default(getitem_4, [32, 328, 4, -1]);  getitem_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:24 in reshape_tensor, code: x = x.transpose(1, 2)
        permute_5: "bf16[32, 4, 328, 32]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:22 in reshape_tensor, code: x = x.view(bs, length, heads, -1)
        view_12: "bf16[32, 328, 4, 32]" = torch.ops.aten.view.default(getitem_5, [32, 328, 4, -1]);  getitem_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:24 in reshape_tensor, code: x = x.transpose(1, 2)
        permute_6: "bf16[32, 4, 328, 32]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:69 in forward, code: weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        mul_4: "bf16[32, 4, 256, 32]" = torch.ops.aten.mul.Tensor(permute_4, 0.4204482076268573);  permute_4 = None
        mul_5: "bf16[32, 4, 328, 32]" = torch.ops.aten.mul.Tensor(permute_5, 0.4204482076268573);  permute_5 = None
        permute_7: "bf16[32, 4, 32, 328]" = torch.ops.aten.permute.default(mul_5, [0, 1, 3, 2]);  mul_5 = None
        expand: "bf16[32, 4, 256, 32]" = torch.ops.aten.expand.default(mul_4, [32, 4, 256, 32]);  mul_4 = None
        clone: "bf16[32, 4, 256, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_14: "bf16[128, 256, 32]" = torch.ops.aten.view.default(clone, [128, 256, 32]);  clone = None
        expand_1: "bf16[32, 4, 32, 328]" = torch.ops.aten.expand.default(permute_7, [32, 4, 32, 328]);  permute_7 = None
        clone_1: "bf16[32, 4, 32, 328]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_15: "bf16[128, 32, 328]" = torch.ops.aten.view.default(clone_1, [128, 32, 328]);  clone_1 = None
        bmm: "bf16[128, 256, 328]" = torch.ops.aten.bmm.default(view_14, view_15)
        view_16: "bf16[32, 4, 256, 328]" = torch.ops.aten.view.default(bmm, [32, 4, 256, 328])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:70 in forward, code: weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        convert_element_type_16: "f32[32, 4, 256, 328]" = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        amax: "f32[32, 4, 256, 1]" = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_2: "f32[32, 4, 256, 328]" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = None
        exp: "f32[32, 4, 256, 328]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[32, 4, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 4, 256, 328]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_17: "bf16[32, 4, 256, 328]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:71 in forward, code: out = weight @ v
        expand_2: "bf16[32, 4, 256, 328]" = torch.ops.aten.expand.default(convert_element_type_17, [32, 4, 256, 328]);  convert_element_type_17 = None
        view_17: "bf16[128, 256, 328]" = torch.ops.aten.view.default(expand_2, [128, 256, 328]);  expand_2 = None
        expand_3: "bf16[32, 4, 328, 32]" = torch.ops.aten.expand.default(permute_6, [32, 4, 328, 32]);  permute_6 = None
        clone_2: "bf16[32, 4, 328, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_18: "bf16[128, 328, 32]" = torch.ops.aten.view.default(clone_2, [128, 328, 32]);  clone_2 = None
        bmm_1: "bf16[128, 256, 32]" = torch.ops.aten.bmm.default(view_17, view_18)
        view_19: "bf16[32, 4, 256, 32]" = torch.ops.aten.view.default(bmm_1, [32, 4, 256, 32]);  bmm_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:73 in forward, code: out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        permute_8: "bf16[32, 256, 4, 32]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        clone_3: "bf16[32, 256, 4, 32]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_20: "bf16[32, 256, 128]" = torch.ops.aten.view.default(clone_3, [32, 256, 128]);  clone_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:75 in forward, code: return self.to_out(out)
        permute_9: "bf16[128, 2560]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        view_21: "bf16[8192, 128]" = torch.ops.aten.view.default(view_20, [8192, 128]);  view_20 = None
        mm_2: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_21, permute_9);  permute_9 = None
        view_22: "bf16[32, 256, 2560]" = torch.ops.aten.view.default(mm_2, [32, 256, 2560]);  mm_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:197 in forward, code: queries = attn(x, queries) + queries
        add_4: "bf16[32, 256, 2560]" = torch.ops.aten.add.Tensor(view_22, view_1);  view_22 = view_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:198 in forward, code: queries = ff(queries) + queries
        convert_element_type_22: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_22, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 256, 1]" = var_mean_2[0]
        getitem_7: "f32[32, 256, 1]" = var_mean_2[1];  var_mean_2 = None
        add_5: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_2: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_3: "f32[32, 256, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_7);  convert_element_type_22 = None
        mul_6: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_7: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_6, primals_14);  mul_6 = None
        add_6: "f32[32, 256, 2560]" = torch.ops.aten.add.Tensor(mul_7, primals_15);  mul_7 = primals_15 = None
        convert_element_type_23: "bf16[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        permute_10: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        view_23: "bf16[8192, 2560]" = torch.ops.aten.view.default(convert_element_type_23, [8192, 2560]);  convert_element_type_23 = None
        mm_3: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_23, permute_10);  permute_10 = None
        view_24: "bf16[32, 256, 2560]" = torch.ops.aten.view.default(mm_3, [32, 256, 2560])
        convert_element_type_26: "f32[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        mul_8: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_9: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf: "f32[32, 256, 2560]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[32, 256, 2560]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_10: "f32[32, 256, 2560]" = torch.ops.aten.mul.Tensor(mul_8, add_7);  mul_8 = add_7 = None
        convert_element_type_27: "bf16[32, 256, 2560]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        permute_11: "bf16[2560, 2560]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        view_25: "bf16[8192, 2560]" = torch.ops.aten.view.default(convert_element_type_27, [8192, 2560]);  convert_element_type_27 = None
        mm_4: "bf16[8192, 2560]" = torch.ops.aten.mm.default(view_25, permute_11);  permute_11 = None
        view_26: "bf16[32, 256, 2560]" = torch.ops.aten.view.default(mm_4, [32, 256, 2560]);  mm_4 = None
        add_8: "bf16[32, 256, 2560]" = torch.ops.aten.add.Tensor(view_26, add_4);  view_26 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:200 in forward, code: queries = self.proj_out(queries)
        view_27: "bf16[8192, 2560]" = torch.ops.aten.view.default(add_8, [8192, 2560]);  add_8 = None
        permute_12: "bf16[2560, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(primals_19, view_27, permute_12);  primals_19 = permute_12 = None
        view_28: "bf16[32, 256, 1024]" = torch.ops.aten.view.default(addmm_2, [32, 256, 1024])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:201 in forward, code: queries = self.norm_out(queries)
        convert_element_type_33: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 256, 1]" = var_mean_3[0]
        getitem_9: "f32[32, 256, 1]" = var_mean_3[1];  var_mean_3 = None
        add_9: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_3: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_4: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_9);  convert_element_type_33 = None
        mul_11: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_12: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_11, primals_20);  mul_11 = None
        add_10: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(mul_12, primals_21);  mul_12 = primals_21 = None
        convert_element_type_34: "bf16[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:71 in forward, code: out = weight @ v
        permute_30: "bf16[128, 328, 256]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        permute_31: "bf16[128, 32, 328]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/vision_models/align_heads/resampler.py:69 in forward, code: weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        permute_32: "bf16[128, 32, 256]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_33: "bf16[128, 328, 32]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        return (convert_element_type_34, primals_1, primals_4, primals_7, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, view, addmm, view_2, addmm_1, getitem_1, rsqrt, getitem_3, rsqrt_1, view_4, view_6, bmm, amax, sum_1, view_21, add_4, getitem_7, rsqrt_2, view_23, mm_3, view_25, view_27, addmm_2, getitem_9, rsqrt_3, permute_30, permute_31, permute_32, permute_33)
        