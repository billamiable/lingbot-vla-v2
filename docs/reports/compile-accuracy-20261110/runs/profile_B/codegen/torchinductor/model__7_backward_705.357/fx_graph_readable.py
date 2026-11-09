class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024]", primals_3: "bf16[24576, 1024]", primals_5: "bf16[4096, 4096]", primals_7: "bf16[2560, 4096]", getitem_1: "f32[24576, 1]", rsqrt: "f32[24576, 1]", view: "bf16[6144, 4096]", addmm: "bf16[6144, 4096]", convert_element_type_6: "bf16[6144, 4096]", tangents_1: "bf16[6144, 2560]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:105 in forward, code: x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        permute_1: "bf16[4096, 2560]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_2: "bf16[2560, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm: "bf16[6144, 4096]" = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3: "bf16[2560, 6144]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[2560, 4096]" = torch.ops.aten.mm.default(permute_3, convert_element_type_6);  permute_3 = convert_element_type_6 = None
        sum_1: "bf16[1, 2560]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "bf16[2560]" = torch.ops.aten.view.default(sum_1, [2560]);  sum_1 = None
        convert_element_type_14: "f32[6144, 4096]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_5: "f32[6144, 4096]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        mul_3: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476)
        erf: "f32[6144, 4096]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2: "f32[6144, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(add_2, 0.5);  add_2 = None
        mul_7: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, convert_element_type_5)
        mul_8: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(mul_7, -0.5);  mul_7 = None
        exp: "f32[6144, 4096]" = torch.ops.aten.exp.default(mul_8);  mul_8 = None
        mul_9: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_10: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mul_9);  convert_element_type_5 = mul_9 = None
        add_4: "f32[6144, 4096]" = torch.ops.aten.add.Tensor(mul_6, mul_10);  mul_6 = mul_10 = None
        mul_11: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_14, add_4);  convert_element_type_14 = add_4 = None
        convert_element_type_16: "bf16[6144, 4096]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_6: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_2: "bf16[6144, 4096]" = torch.ops.aten.mm.default(convert_element_type_16, permute_6);  permute_6 = None
        permute_7: "bf16[4096, 6144]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0])
        mm_3: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        sum_2: "bf16[1, 4096]" = torch.ops.aten.sum.dim_IntList(convert_element_type_16, [0], True);  convert_element_type_16 = None
        view_2: "bf16[4096]" = torch.ops.aten.view.default(sum_2, [4096]);  sum_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:104 in forward, code: x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        view_3: "bf16[24576, 1024]" = torch.ops.aten.view.default(mm_2, [24576, 1024]);  mm_2 = None
        convert_element_type_21: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_23: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        mul_13: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, convert_element_type_23);  convert_element_type_23 = None
        mul_14: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_13, 1024)
        sum_3: "f32[24576, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [1], True)
        convert_element_type: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float32);  primals_3 = None
        sub: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_15: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul_13, mul);  mul_13 = None
        sum_4: "f32[24576, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [1], True);  mul_15 = None
        mul_16: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_4);  sum_4 = None
        sub_2: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(mul_14, sum_3);  mul_14 = sum_3 = None
        sub_3: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(sub_2, mul_16);  sub_2 = mul_16 = None
        div: "f32[24576, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_17: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_18: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, mul);  mul = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_18, [0]);  mul_18 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_21, [0]);  convert_element_type_21 = None
        convert_element_type_25: "bf16[24576, 1024]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        convert_element_type_26: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        convert_element_type_27: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        return (convert_element_type_26, convert_element_type_27, convert_element_type_25, None, mm_3, view_2, mm_1, view_1)
        