class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "bf16[32, 256, 1024]", convert_element_type: "f32[32, 256, 1024]", tangents_1: "f32[]", tangents_2: "f32[32, 256, 1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1606 in _emb_loss, code: return l1_loss.mean()
        expand: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(tangents_1, [32, 256, 1024]);  tangents_1 = None
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
        add: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(tangents_2, where_1);  tangents_2 = where_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1439 in torch_dynamo_resume_in_depth_emb_forward_at_1439, code: future_depth_preds = self.future_depth_align_head(future_align_embs, future_depth_preds).contiguous().float()
        convert_element_type_2: "bf16[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        return (convert_element_type_2, None)
        