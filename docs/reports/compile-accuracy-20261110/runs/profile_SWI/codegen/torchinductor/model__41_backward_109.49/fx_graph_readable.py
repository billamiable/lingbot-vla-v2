class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "bf16[32, 256, 1024]", convert_element_type: "f32[32, 256, 1024]", pow_3: "f32[32, 256, 1]", clamp_min_1: "f32[32, 256, 1]", pow_7: "f32[32, 256, 1]", clamp_min_3: "f32[32, 256, 1]", tangents_1: "f32[]", tangents_2: "f32[32, 256, 1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1568 in _video_emb_loss, code: weighted_cosine_loss = cosine_loss * cosine_weight
        mul_5: "f32[]" = torch.ops.aten.mul.Tensor(tangents_1, 0.2)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1564 in _video_emb_loss, code: cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
        neg: "f32[]" = torch.ops.aten.neg.default(mul_5);  mul_5 = None
        expand_2: "f32[32, 256]" = torch.ops.aten.expand.default(neg, [32, 256]);  neg = None
        div_4: "f32[32, 256]" = torch.ops.aten.div.Scalar(expand_2, 8192);  expand_2 = None
        unsqueeze: "f32[32, 256, 1]" = torch.ops.aten.unsqueeze.default(div_4, -1);  div_4 = None
        expand_3: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(unsqueeze, [32, 256, 1024]);  unsqueeze = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1553 in _video_emb_loss, code: target = future_video_targets.to(dtype=video_preds.dtype, device=video_preds.device)
        convert_element_type_1: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(primals_2, torch.float32);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1563 in _video_emb_loss, code: target_norm = F.normalize(target.float().detach(), dim=-1, eps=1e-6)
        expand_1: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(clamp_min_1, [32, 256, 1024]);  clamp_min_1 = None
        div_1: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, expand_1);  expand_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1564 in _video_emb_loss, code: cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
        div_2: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div_1, clamp_min_3);  div_1 = clamp_min_3 = None
        mul_6: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(expand_3, div_2);  expand_3 = div_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1562 in _video_emb_loss, code: pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
        clamp_min: "f32[32, 256, 1]" = torch.ops.aten.clamp_min.default(pow_3, 1e-06)
        expand: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(clamp_min, [32, 256, 1024]);  clamp_min = None
        div: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(convert_element_type, expand)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1564 in _video_emb_loss, code: cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
        clamp_min_2: "f32[32, 256, 1]" = torch.ops.aten.clamp_min.default(pow_7, 1e-06)
        div_3: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div, clamp_min_2)
        div_6: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div_3, clamp_min_2);  div_3 = None
        neg_1: "f32[32, 256, 1024]" = torch.ops.aten.neg.default(mul_6)
        mul_7: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(neg_1, div_6);  neg_1 = div_6 = None
        div_7: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(mul_6, clamp_min_2);  mul_6 = clamp_min_2 = None
        sum_6: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        div_8: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div, pow_7)
        eq: "b8[32, 256, 1]" = torch.ops.aten.eq.Scalar(pow_7, 0);  pow_7 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 256, 1024]" = torch.ops.aten.where.self(eq, full_default, div_8);  eq = div_8 = None
        mul_8: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(sum_6, where);  sum_6 = where = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1564 in _video_emb_loss, code: cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
        add_1: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(div_7, mul_8);  div_7 = mul_8 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1562 in _video_emb_loss, code: pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
        div_10: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div, expand);  div = None
        neg_2: "f32[32, 256, 1024]" = torch.ops.aten.neg.default(add_1)
        mul_9: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(neg_2, div_10);  neg_2 = div_10 = None
        div_11: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(add_1, expand);  add_1 = expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1562 in _video_emb_loss, code: pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
        add_2: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(tangents_2, div_11);  tangents_2 = div_11 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1562 in _video_emb_loss, code: pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
        sum_7: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
        ge: "b8[32, 256, 1]" = torch.ops.aten.ge.Scalar(pow_3, 1e-06)
        where_1: "f32[32, 256, 1]" = torch.ops.aten.where.self(ge, sum_7, full_default);  ge = sum_7 = None
        div_12: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(convert_element_type, pow_3)
        eq_1: "b8[32, 256, 1]" = torch.ops.aten.eq.Scalar(pow_3, 0);  pow_3 = None
        where_2: "f32[32, 256, 1024]" = torch.ops.aten.where.self(eq_1, full_default, div_12);  eq_1 = full_default = div_12 = None
        mul_10: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1562 in _video_emb_loss, code: pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
        add_3: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_10);  add_2 = mul_10 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1558 in _video_emb_loss, code: weighted_mse_loss = mse_loss * mse_weight
        mul_11: "f32[]" = torch.ops.aten.mul.Tensor(tangents_1, 1.0);  tangents_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1554 in _video_emb_loss, code: mse_loss = F.mse_loss(video_preds.float(), target.float().detach())
        sub: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mul_12: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(sub, 2.384185791015625e-07);  sub = None
        mul_13: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_12, mul_11);  mul_12 = mul_11 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1554 in _video_emb_loss, code: mse_loss = F.mse_loss(video_preds.float(), target.float().detach())
        add_4: "f32[32, 256, 1024]" = torch.ops.aten.add.Tensor(add_3, mul_13);  add_3 = mul_13 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1529 in torch_dynamo_resume_in_current_video_emb_forward_at_1529, code: preds = self.current_video_align_head(align_embs, queries).contiguous().float()
        convert_element_type_3: "bf16[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        return (convert_element_type_3, None)
        