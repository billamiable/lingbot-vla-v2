class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024]", primals_2: "bf16[1024]", primals_3: "bf16[24576, 1024]", primals_4: "Sym(1024)", primals_5: "bf16[4096, 4096]", primals_6: "bf16[4096]", primals_7: "bf16[2560, 4096]", primals_8: "bf16[2560]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:104 in forward, code: x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        convert_element_type: "f32[24576, 1024]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[24576, 1]" = var_mean[0]
        getitem_1: "f32[24576, 1]" = var_mean[1];  var_mean = None
        add: "f32[24576, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[24576, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[24576, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[24576, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[24576, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
        convert_element_type_1: "bf16[24576, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[6144, 4096]" = torch.ops.aten.view.default(convert_element_type_1, [-1, 4096]);  convert_element_type_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:105 in forward, code: x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        addmm: "bf16[6144, 4096]" = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = permute = None
        convert_element_type_5: "f32[6144, 4096]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        mul_2: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_3: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476);  convert_element_type_5 = None
        erf: "f32[6144, 4096]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2: "f32[6144, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_6: "bf16[6144, 4096]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        permute_1: "bf16[4096, 2560]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm_1: "bf16[6144, 2560]" = torch.ops.aten.addmm.default(primals_8, convert_element_type_6, permute_1);  primals_8 = permute_1 = None
        return (addmm_1, primals_1, primals_3, primals_5, primals_7, getitem_1, rsqrt, view, addmm, convert_element_type_6)
        