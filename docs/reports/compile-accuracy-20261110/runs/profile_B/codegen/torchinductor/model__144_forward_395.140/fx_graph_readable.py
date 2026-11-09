class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 51, 768]", primals_2: "f32[32, 768]", primals_3: "f32[4096, 768]", primals_4: "f32[768]", primals_5: "f32[768, 768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[4096]", primals_10: "f32[1024, 768]", primals_11: "f32[1024]", primals_12: "f32[1024, 768]", primals_13: "f32[1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 2)
        mean: "f32[32, 51, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:233 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add: "f32[32, 51, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 51, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_1, rsqrt);  rsqrt = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:235 in forward, code: hidden_states = self.weight * hidden_states
        mul_1: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_4, mul);  mul = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        addmm: "f32[32, 768]" = torch.ops.aten.addmm.default(primals_6, primals_2, permute);  permute = None
        unsqueeze: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm_1: "f32[32, 768]" = torch.ops.aten.addmm.default(primals_8, primals_2, permute_1);  permute_1 = None
        unsqueeze_1: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:239 in forward, code: hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        add_1: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        add_2: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:407 in forward, code: query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
        view: "f32[1632, 768]" = torch.ops.aten.view.default(add_2, [1632, 768]);  add_2 = None
        permute_2: "f32[768, 4096]" = torch.ops.aten.permute.default(primals_3, [1, 0])
        addmm_2: "f32[1632, 4096]" = torch.ops.aten.addmm.default(primals_9, view, permute_2);  primals_9 = permute_2 = None
        view_1: "f32[32, 51, 4096]" = torch.ops.aten.view.default(addmm_2, [32, 51, 4096]);  addmm_2 = None
        view_2: "f32[32, 51, 32, 128]" = torch.ops.aten.view.default(view_1, [32, 51, -1, 128]);  view_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:408 in forward, code: key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
        permute_3: "f32[768, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_3: "f32[1632, 1024]" = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = permute_3 = None
        view_4: "f32[32, 51, 1024]" = torch.ops.aten.view.default(addmm_3, [32, 51, 1024]);  addmm_3 = None
        view_5: "f32[32, 51, 8, 128]" = torch.ops.aten.view.default(view_4, [32, 51, -1, 128]);  view_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:409 in forward, code: value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        permute_4: "f32[768, 1024]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_4: "f32[1632, 1024]" = torch.ops.aten.addmm.default(primals_13, view, permute_4);  primals_13 = view = permute_4 = None
        view_7: "f32[32, 51, 1024]" = torch.ops.aten.view.default(addmm_4, [32, 51, 1024]);  addmm_4 = None
        view_8: "f32[32, 51, 8, 128]" = torch.ops.aten.view.default(view_7, [32, 51, -1, 128]);  view_7 = None
        return (view_2, view_5, view_8, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_12)
        