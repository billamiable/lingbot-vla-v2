class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 256, 1024]", primals_2: "bf16[32, 256, 1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1439 in torch_dynamo_resume_in_depth_emb_forward_at_1439, code: future_depth_preds = self.future_depth_align_head(future_align_embs, future_depth_preds).contiguous().float()
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
        return (mean, convert_element_type, primals_2, convert_element_type)
        