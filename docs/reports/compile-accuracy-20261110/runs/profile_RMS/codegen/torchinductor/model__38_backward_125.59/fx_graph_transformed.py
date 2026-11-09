class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "bf16[32, 256, 1024]", convert_element_type: "f32[32, 256, 1024]", tangents_1: "bf16[32, 72, 2560]", tangents_2: "bf16[32, 256, 2560]", tangents_3: "f32[32, 256, 1024]", tangents_4: "f32[]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1438 in torch_dynamo_resume_in_depth_emb_forward_at_1427, code: future_depth_preds = self.future_depth_align_embs.repeat(future_align_embs.shape[0], 1, 1).to(dtype=future_align_embs.dtype, device=future_align_embs.device)
        sum_1: "bf16[256, 2560]" = torch.ops.aten.sum.dim_IntList(tangents_2, [0]);  tangents_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1437 in torch_dynamo_resume_in_depth_emb_forward_at_1427, code: future_align_embs = torch.cat([future_image_embs, future_align_embs], dim=1)
        slice_5: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 64, 72);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:964 in _future_depth_task_tokens, code: return hidden_states[:, -self.num_task_tokens:, :]
        full_default: "bf16[32, 8, 2560]" = torch.ops.aten.full.default([32, 8, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default: "bf16[32, 8, 2560]" = torch.ops.aten.slice_scatter.default(full_default, slice_5, 2, 0, 9223372036854775807);  full_default = slice_5 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:964 in _future_depth_task_tokens, code: return hidden_states[:, -self.num_task_tokens:, :]
        full_default_1: "bf16[32, 286, 2560]" = torch.ops.aten.full.default([32, 286, 2560], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        slice_scatter_default_1: "bf16[32, 286, 2560]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default, 1, -8, 9223372036854775807);  full_default_1 = slice_scatter_default = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1606 in _emb_loss, code: return l1_loss.mean()
        expand: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(tangents_4, [32, 256, 1024]);  tangents_4 = None
        div_1: "f32[32, 256, 1024]" = torch.ops.aten.div.Scalar(expand, 8388608);  expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1605 in _emb_loss, code: l1_loss = F.smooth_l1_loss(emb_preds.float(), emb_targets.float().detach(), reduction="none")
        mul_1: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(div_1, 1.0);  div_1 = None
        convert_element_type_1: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(primals_2, torch.float32);  primals_2 = None
        sub: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mul_2: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_1, sub)
        div_2: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(mul_2, 1.0);  mul_2 = None
        sign: "f32[32, 256, 1024]" = torch.ops.aten.sign.default(sub)
        mul_3: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_1, sign);  mul_1 = sign = None
        abs_1: "f32[32, 256, 1024]" = torch.ops.aten.abs.default(sub);  sub = None
        lt: "b8[32, 256, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 1.0);  abs_1 = None
        where_1: "f32[32, 256, 1024]" = torch.ops.aten.where.self(lt, div_2, mul_3);  lt = div_2 = mul_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1605 in _emb_loss, code: l1_loss = F.smooth_l1_loss(emb_preds.float(), emb_targets.float().detach(), reduction="none")
        add: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(tangents_3, where_1);  tangents_3 = where_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1427 in torch_dynamo_resume_in_depth_emb_forward_at_1427, code: depth_preds = self.depth_align_head(align_embs, depth_preds).contiguous().float()
        convert_element_type_2: "bf16[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        return (convert_element_type_2, None, slice_scatter_default_1, None, sum_1)
        