class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[8, 2560]", primals_2: "bf16[32, 198, 2560]", primals_3: "i64[32, 72]", primals_4: "bf16[32, 72, 2560]", primals_5: "b8[32, 72]", primals_6: "bf16[32, 8, 2560]", primals_7: "b8[32, 8]", primals_8: "b8[32, 198]", primals_9: "i64[32, 8]", primals_10: "i64[32, 198]", primals_11: "b8[32, 198]", primals_12: "b8[32, 3]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:709 in torch_dynamo_resume_in_embed_prefix_at_706, code: future_align_embs = future_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
        repeat: "bf16[32, 8, 2560]" = torch.ops.aten.repeat.default(primals_1, [32, 1, 1]);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:693 in _append, code: token_visual_masks = torch.zeros_like(token_masks)
        full_default: "b8[32, 72]" = torch.ops.aten.full.default([32, 72], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:693 in _append, code: token_visual_masks = torch.zeros_like(token_masks)
        full_default_1: "b8[32, 8]" = torch.ops.aten.full.default([32, 8], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:693 in _append, code: token_visual_masks = torch.zeros_like(token_masks)
        full_default_2: "b8[32, 8]" = torch.ops.aten.full.default([32, 8], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:771 in torch_dynamo_resume_in_embed_prefix_at_706, code: else:
        cat: "bf16[32, 286, 2560]" = torch.ops.aten.cat.default([primals_2, primals_4, primals_6, repeat], 1);  primals_2 = primals_4 = primals_6 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:772 in torch_dynamo_resume_in_embed_prefix_at_706, code: embs = torch.cat([img_emb, lang_emb], dim=1)
        cat_1: "b8[32, 286]" = torch.ops.aten.cat.default([primals_8, primals_5, primals_7, primals_7], 1);  primals_8 = primals_5 = primals_7 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:773 in torch_dynamo_resume_in_embed_prefix_at_706, code: pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
        cat_2: "i64[32, 286]" = torch.ops.aten.cat.default([primals_10, primals_3, primals_9, primals_9], 1);  primals_10 = primals_9 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:774 in torch_dynamo_resume_in_embed_prefix_at_706, code: prefix_input_ids = torch.cat([fake_image_ids, lang_tokens.to(device)], dim=1)
        cat_3: "b8[32, 286]" = torch.ops.aten.cat.default([primals_11, full_default, full_default_1, full_default_2], 1);  primals_11 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:782 in torch_dynamo_resume_in_embed_prefix_at_706, code: flat_img_masks = einops.rearrange(img_masks, "b n -> (b n)")
        full_default_3: "b8[32, 286]" = torch.ops.aten.full.default([32, 286], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:786 in torch_dynamo_resume_in_embed_prefix_at_706, code: prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
        view: "b8[96]" = torch.ops.aten.reshape.default(primals_12, [96]);  primals_12 = None
        return (view, cat, cat_1, cat_2, cat_3, full_default_3, repeat, primals_3, full_default, full_default_1, full_default_2)
        