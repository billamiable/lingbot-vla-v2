class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "bf16[32, 55]", primals_11: "bf16[768, 1536]", primals_13: "bf16[768, 768]", view: "bf16[1600, 55]", view_2: "bf16[1600, 1536]", addmm_2: "bf16[1600, 768]", view_4: "bf16[1600, 768]", tangents_1: "bf16[32, 51, 768]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1156 in embed_suffix, code: embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        slice_18: "bf16[32, 1, 768]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, 1)
        slice_19: "bf16[32, 50, 768]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 1, 51);  tangents_1 = None
        squeeze: "bf16[32, 768]" = torch.ops.aten.squeeze.dim(slice_18, 1);  slice_18 = None
        full_default_6: "bf16[32, 768]" = torch.ops.aten.full.default([32, 768], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default: "bf16[32, 768]" = torch.ops.aten.slice_scatter.default(full_default_6, squeeze, 0, 0, 9223372036854775807);  full_default_6 = squeeze = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1153 in embed_suffix, code: action_time_emb = self._fp32_linear(self.action_time_mlp_out, action_time_emb) if _fp32 else self.action_time_mlp_out(action_time_emb) # [1, state_bs*50, 1024]
        clone: "bf16[32, 50, 768]" = torch.ops.aten.clone.default(slice_19, memory_format = torch.contiguous_format);  slice_19 = None
        view_6: "bf16[1600, 768]" = torch.ops.aten.reshape.default(clone, [1600, 768]);  clone = None
        permute_3: "bf16[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_4: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm: "bf16[1600, 768]" = torch.ops.aten.mm.default(view_6, permute_4);  permute_4 = None
        permute_5: "bf16[768, 1600]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm_1: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_5, view_4);  permute_5 = view_4 = None
        sum_1: "bf16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True);  view_6 = None
        view_7: "bf16[768]" = torch.ops.aten.reshape.default(sum_1, [768]);  sum_1 = None
        view_8: "bf16[32, 50, 768]" = torch.ops.aten.reshape.default(mm, [32, 50, 768]);  mm = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1151 in embed_suffix, code: action_time_emb = self._fp32_linear(self.action_time_mlp_in, action_time_emb) if _fp32 else self.action_time_mlp_in(action_time_emb)
        view_3: "bf16[32, 50, 768]" = torch.ops.aten.reshape.default(addmm_2, [32, 50, 768]);  addmm_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1152 in embed_suffix, code: action_time_emb = F.silu(action_time_emb)  # swish == silu
        sigmoid_1: "bf16[32, 50, 768]" = torch.ops.aten.sigmoid.default(view_3)
        full_default_7: "bf16[32, 50, 768]" = torch.ops.aten.full.default([32, 50, 768], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_3: "bf16[32, 50, 768]" = torch.ops.aten.sub.Tensor(full_default_7, sigmoid_1);  full_default_7 = None
        mul_9: "bf16[32, 50, 768]" = torch.ops.aten.mul.Tensor(view_3, sub_3);  view_3 = sub_3 = None
        add_3: "bf16[32, 50, 768]" = torch.ops.aten.add.Scalar(mul_9, 1);  mul_9 = None
        mul_10: "bf16[32, 50, 768]" = torch.ops.aten.mul.Tensor(sigmoid_1, add_3);  sigmoid_1 = add_3 = None
        mul_11: "bf16[32, 50, 768]" = torch.ops.aten.mul.Tensor(view_8, mul_10);  view_8 = mul_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1151 in embed_suffix, code: action_time_emb = self._fp32_linear(self.action_time_mlp_in, action_time_emb) if _fp32 else self.action_time_mlp_in(action_time_emb)
        view_9: "bf16[1600, 768]" = torch.ops.aten.reshape.default(mul_11, [1600, 768]);  mul_11 = None
        permute_2: "bf16[1536, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_9: "bf16[768, 1536]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_2: "bf16[1600, 1536]" = torch.ops.aten.mm.default(view_9, permute_9);  permute_9 = None
        permute_10: "bf16[768, 1600]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_3: "bf16[768, 1536]" = torch.ops.aten.mm.default(permute_10, view_2);  permute_10 = view_2 = None
        sum_2: "bf16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True);  view_9 = None
        view_10: "bf16[768]" = torch.ops.aten.reshape.default(sum_2, [768]);  sum_2 = None
        view_11: "bf16[32, 50, 1536]" = torch.ops.aten.reshape.default(mm_2, [32, 50, 1536]);  mm_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1149 in embed_suffix, code: action_time_emb = torch.cat([action_emb, time_emb], dim=-1) # [1, state_bs*50, 2048]
        slice_20: "bf16[32, 50, 768]" = torch.ops.aten.slice.Tensor(view_11, 2, 0, 768);  view_11 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1147 in embed_suffix, code: action_emb = self._fp32_linear(self.action_in_proj, noisy_actions) if _fp32 else self.action_in_proj(noisy_actions) # torch.Size([1, state_bs*50, 1024])
        view_12: "bf16[1600, 768]" = torch.ops.aten.reshape.default(slice_20, [1600, 768]);  slice_20 = None
        permute_13: "bf16[768, 1600]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_4: "bf16[768, 55]" = torch.ops.aten.mm.default(permute_13, view);  permute_13 = view = None
        sum_3: "bf16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True);  view_12 = None
        view_13: "bf16[768]" = torch.ops.aten.reshape.default(sum_3, [768]);  sum_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1132 in embed_suffix, code: state_emb = self._fp32_linear(self.state_proj, state) if _fp32 else self.state_proj(state)
        permute_16: "bf16[768, 32]" = torch.ops.aten.permute.default(slice_scatter_default, [1, 0])
        mm_5: "bf16[768, 55]" = torch.ops.aten.mm.default(permute_16, primals_4);  permute_16 = primals_4 = None
        sum_4: "bf16[1, 768]" = torch.ops.aten.sum.dim_IntList(slice_scatter_default, [0], True);  slice_scatter_default = None
        view_14: "bf16[768]" = torch.ops.aten.reshape.default(sum_4, [768]);  sum_4 = None
        return (None, None, None, None, mm_5, view_14, None, mm_4, view_13, None, mm_3, view_10, mm_1, view_7)
        