class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 256, 1024]", primals_2: "bf16[32, 256, 1024]", primals_3: "bf16[32, 286, 2560]", primals_4: "bf16[32, 64, 2560]", primals_5: "bf16[256, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1427 in torch_dynamo_resume_in_depth_emb_forward_at_1427, code: depth_preds = self.depth_align_head(align_embs, depth_preds).contiguous().float()
        convert_element_type: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1605 in _emb_loss, code: l1_loss = F.smooth_l1_loss(emb_preds.float(), emb_targets.float().detach(), reduction="none")
        convert_element_type_1: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(primals_2, torch.float32)
        sub: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1);  convert_element_type_1 = None
        abs_1: "f32[32, 256, 1024]" = torch.ops.aten.abs.default(sub);  sub = None
        lt: "b8[32, 256, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 1.0)
        pow_1: "f32[32, 256, 1024]" = torch.ops.aten.pow.Tensor_Scalar(abs_1, 2)
        mul: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(pow_1, 0.5);  pow_1 = None
        div: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(mul, 1.0);  mul = None
        sub_1: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(abs_1, 0.5);  abs_1 = None
        where: "f32[32, 256, 1024]" = torch.ops.aten.where.self(lt, div, sub_1);  lt = div = sub_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1606 in _emb_loss, code: return l1_loss.mean()
        mean: "f32[]" = torch.ops.aten.mean.default(where);  where = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:964 in _future_depth_task_tokens, code: return hidden_states[:, -self.num_task_tokens:, :]
        slice_2: "bf16[32, 8, 2560]" = torch.ops.aten.slice.Tensor(primals_3, 1, -8, 9223372036854775807);  primals_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1437 in torch_dynamo_resume_in_depth_emb_forward_at_1427, code: future_align_embs = torch.cat([future_image_embs, future_align_embs], dim=1)
        cat: "bf16[32, 72, 2560]" = torch.ops.aten.cat.default([primals_4, slice_2], 1);  primals_4 = slice_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1438 in torch_dynamo_resume_in_depth_emb_forward_at_1427, code: future_depth_preds = self.future_depth_align_embs.repeat(future_align_embs.shape[0], 1, 1).to(dtype=future_align_embs.dtype, device=future_align_embs.device)
        repeat: "bf16[32, 256, 2560]" = torch.ops.aten.repeat.default(primals_5, [32, 1, 1]);  primals_5 = None
        return (cat, repeat, convert_element_type, mean, primals_2, convert_element_type)
        