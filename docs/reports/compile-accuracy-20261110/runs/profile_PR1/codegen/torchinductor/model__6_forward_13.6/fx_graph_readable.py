class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[24576, 1024]", primals_2: "bf16[4096]", primals_3: "bf16[4096]", primals_4: "bf16[4096, 4096]", primals_5: "bf16[4096]", primals_6: "bf16[2560, 4096]", primals_7: "bf16[2560]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:104 in forward, code: x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        view: "bf16[6144, 4096]" = torch.ops.aten.view.default(primals_1, [-1, 4096])
        convert_element_type: "f32[6144, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[6144, 1]" = var_mean[0]
        getitem_1: "f32[6144, 1]" = var_mean[1];  var_mean = None
        add: "f32[6144, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[6144, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[6144, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None
        add_1: "f32[6144, 4096]" = torch.ops.aten.add.Tensor(mul_1, primals_3);  mul_1 = primals_3 = None
        convert_element_type_1: "bf16[6144, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:105 in forward, code: x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm: "bf16[6144, 4096]" = torch.ops.aten.addmm.default(primals_5, convert_element_type_1, permute);  primals_5 = permute = None
        convert_element_type_5: "f32[6144, 4096]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        mul_2: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_3: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476);  convert_element_type_5 = None
        erf: "f32[6144, 4096]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2: "f32[6144, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[6144, 4096]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_6: "bf16[6144, 4096]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        permute_1: "bf16[4096, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_1: "bf16[6144, 2560]" = torch.ops.aten.addmm.default(primals_7, convert_element_type_6, permute_1);  primals_7 = permute_1 = None
        return (addmm_1, primals_1, primals_2, primals_4, primals_6, getitem_1, rsqrt, convert_element_type_1, addmm, convert_element_type_6)
        