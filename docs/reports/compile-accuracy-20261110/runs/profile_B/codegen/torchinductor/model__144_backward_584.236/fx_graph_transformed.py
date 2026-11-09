class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 51, 768]", primals_2: "f32[32, 768]", primals_3: "f32[4096, 768]", primals_4: "f32[768]", primals_5: "f32[768, 768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_10: "f32[1024, 768]", primals_12: "f32[1024, 768]", tangents_1: "f32[32, 51, 32, 128]", tangents_2: "f32[32, 51, 8, 128]", tangents_3: "f32[32, 51, 8, 128]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:409 in forward, code: value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        view_9: "f32[32, 51, 1024]" = torch.ops.aten.reshape.default(tangents_3, [32, 51, 1024]);  tangents_3 = None
        view_10: "f32[1632, 1024]" = torch.ops.aten.reshape.default(view_9, [1632, 1024]);  view_9 = None
        permute_4: "f32[768, 1024]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_5: "f32[1024, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm: "f32[1632, 768]" = torch.ops.aten.mm.default(view_10, permute_5);  permute_5 = None
        permute_6: "f32[1024, 1632]" = torch.ops.aten.permute.default(view_10, [1, 0])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 2)
        mean: "f32[32, 51, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:233 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add: "f32[32, 51, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 51, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_1, rsqrt)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:235 in forward, code: hidden_states = self.weight * hidden_states
        mul_1: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(primals_4, mul)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[32, 768]" = torch.ops.aten.mm.default(primals_2, permute);  permute = None
        add_tensor_1: "f32[32, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_6);  mm_default_1 = primals_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        unsqueeze: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 1);  add_tensor_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[32, 768]" = torch.ops.aten.mm.default(primals_2, permute_1);  permute_1 = None
        add_tensor: "f32[32, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_8);  mm_default = primals_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        unsqueeze_1: "f32[32, 1, 768]" = torch.ops.aten.unsqueeze.default(add_tensor, 1);  add_tensor = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:239 in forward, code: hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        add_1: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add_1, mul_1)
        add_2: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:407 in forward, code: query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
        view: "f32[1632, 768]" = torch.ops.aten.reshape.default(add_2, [1632, 768]);  add_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:409 in forward, code: value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        mm_1: "f32[1024, 768]" = torch.ops.aten.mm.default(permute_6, view);  permute_6 = None
        sum_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True);  view_10 = None
        view_11: "f32[1024]" = torch.ops.aten.reshape.default(sum_1, [1024]);  sum_1 = None
        view_12: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(mm, [32, 51, 768]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:408 in forward, code: key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
        view_13: "f32[32, 51, 1024]" = torch.ops.aten.reshape.default(tangents_2, [32, 51, 1024]);  tangents_2 = None
        view_14: "f32[1632, 1024]" = torch.ops.aten.reshape.default(view_13, [1632, 1024]);  view_13 = None
        permute_3: "f32[768, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_9: "f32[1024, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_2: "f32[1632, 768]" = torch.ops.aten.mm.default(view_14, permute_9);  permute_9 = None
        permute_10: "f32[1024, 1632]" = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_3: "f32[1024, 768]" = torch.ops.aten.mm.default(permute_10, view);  permute_10 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_14, [0], True);  view_14 = None
        view_15: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        view_16: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(mm_2, [32, 51, 768]);  mm_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:408 in forward, code: key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
        add_3: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(view_12, view_16);  view_12 = view_16 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:407 in forward, code: query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
        view_17: "f32[32, 51, 4096]" = torch.ops.aten.reshape.default(tangents_1, [32, 51, 4096]);  tangents_1 = None
        view_18: "f32[1632, 4096]" = torch.ops.aten.reshape.default(view_17, [1632, 4096]);  view_17 = None
        permute_2: "f32[768, 4096]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        permute_13: "f32[4096, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_4: "f32[1632, 768]" = torch.ops.aten.mm.default(view_18, permute_13);  permute_13 = None
        permute_14: "f32[4096, 1632]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_5: "f32[4096, 768]" = torch.ops.aten.mm.default(permute_14, view);  permute_14 = view = None
        sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_19: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        view_20: "f32[32, 51, 768]" = torch.ops.aten.reshape.default(mm_4, [32, 51, 768]);  mm_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:407 in forward, code: query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
        add_4: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(add_3, view_20);  add_3 = view_20 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:239 in forward, code: hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        sum_4: "f32[32, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_4, [1], True)
        mul_3: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add_4, add_1);  add_1 = None
        mul_4: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(add_4, mul_1);  add_4 = mul_1 = None
        sum_5: "f32[32, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:238 in forward, code: beta  = self.beta(cond).unsqueeze(1)   # [B, 1, H]
        squeeze: "f32[32, 768]" = torch.ops.aten.squeeze.dim(sum_4, 1);  sum_4 = None
        permute_17: "f32[768, 32]" = torch.ops.aten.permute.default(squeeze, [1, 0])
        mm_6: "f32[768, 768]" = torch.ops.aten.mm.default(permute_17, primals_2);  permute_17 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze, [0], True);  squeeze = None
        view_21: "f32[768]" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:237 in forward, code: gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        squeeze_1: "f32[32, 768]" = torch.ops.aten.squeeze.dim(sum_5, 1);  sum_5 = None
        permute_20: "f32[768, 32]" = torch.ops.aten.permute.default(squeeze_1, [1, 0])
        mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_20, primals_2);  permute_20 = primals_2 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze_1, [0], True);  squeeze_1 = None
        view_22: "f32[768]" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:235 in forward, code: hidden_states = self.weight * hidden_states
        mul_5: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_4);  primals_4 = None
        mul_6: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_3, mul);  mul_3 = mul = None
        sum_8: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1], True);  mul_6 = None
        view_23: "f32[768]" = torch.ops.aten.reshape.default(sum_8, [768]);  sum_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:233 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_7: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_1)
        mul_8: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(mul_5, rsqrt);  mul_5 = None
        sum_9: "f32[32, 51, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        pow_2: "f32[32, 51, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_9: "f32[32, 51, 1]" = torch.ops.aten.mul.Scalar(sum_9, -0.5);  sum_9 = None
        mul_10: "f32[32, 51, 1]" = torch.ops.aten.mul.Tensor(mul_9, pow_2);  mul_9 = pow_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand: "f32[32, 51, 768]" = torch.ops.aten.expand.default(mul_10, [32, 51, 768]);  mul_10 = None
        div: "f32[32, 51, 768]" = torch.ops.aten.div.Scalar(expand, 768);  expand = None
        pow_3: "f32[32, 51, 768]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 1.0);  primals_1 = None
        mul_11: "f32[32, 51, 768]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_12: "f32[32, 51, 768]" = torch.ops.aten.mul.Tensor(div, mul_11);  div = mul_11 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:232 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        add_5: "f32[32, 51, 768]" = torch.ops.aten.add.Tensor(mul_8, mul_12);  mul_8 = mul_12 = None
        return (add_5, None, mm_5, view_23, mm_7, view_22, mm_6, view_21, view_19, mm_3, view_15, mm_1, view_11)
        