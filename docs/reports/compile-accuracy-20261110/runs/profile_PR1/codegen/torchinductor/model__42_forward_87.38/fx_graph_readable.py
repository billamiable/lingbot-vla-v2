class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 256, 1024]", primals_2: "bf16[32, 256, 1024]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1501 in torch_dynamo_resume_in_video_emb_forward_at_1501, code: video_preds = self.future_video_align_head(video_align_embs, video_preds).contiguous().float()
        convert_element_type: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1553 in _video_emb_loss, code: target = future_video_targets.to(dtype=video_preds.dtype, device=video_preds.device)
        convert_element_type_1: "f32[32, 256, 1024]" = torch.ops.prims.convert_element_type.default(primals_2, torch.float32)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1554 in _video_emb_loss, code: mse_loss = F.mse_loss(video_preds.float(), target.float().detach())
        sub: "f32[32, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1)
        pow_1: "f32[32, 256, 1024]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        mean: "f32[]" = torch.ops.aten.mean.default(pow_1);  pow_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1557 in _video_emb_loss, code: metrics[f"align/{metric_prefix}_mse_loss_weighted"] = (mse_loss * mse_weight).detach()
        mul: "f32[]" = torch.ops.aten.mul.Tensor(mean, 1.0)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1558 in _video_emb_loss, code: weighted_mse_loss = mse_loss * mse_weight
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(mean, 1.0)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1562 in _video_emb_loss, code: pred_norm = F.normalize(video_preds.float(), dim=-1, eps=1e-6)
        pow_2: "f32[32, 256, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2.0)
        sum_1: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(pow_2, [-1], True);  pow_2 = None
        pow_3: "f32[32, 256, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min: "f32[32, 256, 1]" = torch.ops.aten.clamp_min.default(pow_3, 1e-06)
        expand: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(clamp_min, [32, 256, 1024]);  clamp_min = None
        div: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(convert_element_type, expand);  expand = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1563 in _video_emb_loss, code: target_norm = F.normalize(target.float().detach(), dim=-1, eps=1e-6)
        pow_4: "f32[32, 256, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2.0)
        sum_2: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(pow_4, [-1], True);  pow_4 = None
        pow_5: "f32[32, 256, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        clamp_min_1: "f32[32, 256, 1]" = torch.ops.aten.clamp_min.default(pow_5, 1e-06);  pow_5 = None
        expand_1: "f32[32, 256, 1024]" = torch.ops.aten.expand.default(clamp_min_1, [32, 256, 1024])
        div_1: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, expand_1);  convert_element_type_1 = expand_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1564 in _video_emb_loss, code: cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
        pow_6: "f32[32, 256, 1024]" = torch.ops.aten.pow.Tensor_Scalar(div, 2)
        sum_3: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(pow_6, [-1], True);  pow_6 = None
        pow_7: "f32[32, 256, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None
        pow_8: "f32[32, 256, 1024]" = torch.ops.aten.pow.Tensor_Scalar(div_1, 2)
        sum_4: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(pow_8, [-1], True);  pow_8 = None
        pow_9: "f32[32, 256, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        clamp_min_2: "f32[32, 256, 1]" = torch.ops.aten.clamp_min.default(pow_7, 1e-06)
        clamp_min_3: "f32[32, 256, 1]" = torch.ops.aten.clamp_min.default(pow_9, 1e-06);  pow_9 = None
        div_2: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div_1, clamp_min_3);  div_1 = None
        div_3: "f32[32, 256, 1024]" = torch.ops.aten.div.Tensor(div, clamp_min_2);  div = clamp_min_2 = None
        mul_2: "f32[32, 256, 1024]" = torch.ops.aten.mul.Tensor(div_3, div_2);  div_3 = div_2 = None
        sum_5: "f32[32, 256]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1]);  mul_2 = None
        mean_1: "f32[]" = torch.ops.aten.mean.default(sum_5);  sum_5 = None
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(1.0, mean_1);  mean_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1567 in _video_emb_loss, code: metrics[f"align/{metric_prefix}_cosine_loss_weighted"] = (cosine_loss * cosine_weight).detach()
        mul_3: "f32[]" = torch.ops.aten.mul.Tensor(sub_1, 0.2)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1568 in _video_emb_loss, code: weighted_cosine_loss = cosine_loss * cosine_weight
        mul_4: "f32[]" = torch.ops.aten.mul.Tensor(sub_1, 0.2)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py:1569 in _video_emb_loss, code: loss = weighted_cosine_loss if loss is None else loss + weighted_cosine_loss
        add: "f32[]" = torch.ops.aten.add.Tensor(mul_1, mul_4);  mul_1 = mul_4 = None
        return (add, convert_element_type, mean, mul, sub_1, mul_3, primals_2, convert_element_type, pow_3, clamp_min_1, pow_7, clamp_min_3)
        