class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[32, 3, 1, 2560]", primals_2: "bf16[32, 3, 64, 2560]", primals_3: "bf16[32, 3, 1, 2560]", primals_4: "b8[32, 3]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:616 in torch_dynamo_resume_in_embed_prefix_at_613, code: img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
        cat: "bf16[32, 3, 66, 2560]" = torch.ops.aten.cat.default([primals_3, primals_2, primals_1], 2);  primals_3 = primals_2 = primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:618 in torch_dynamo_resume_in_embed_prefix_at_613, code: image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
        unsqueeze: "b8[32, 3, 1]" = torch.ops.aten.unsqueeze.default(primals_4, 2);  primals_4 = None
        expand: "b8[32, 3, 66]" = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 66])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:619 in torch_dynamo_resume_in_embed_prefix_at_613, code: image_visual_masks = torch.zeros_like(image_pad_masks)
        full: "b8[32, 3, 66]" = torch.ops.aten.full.default([32, 3, 66], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:620 in torch_dynamo_resume_in_embed_prefix_at_613, code: image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        expand_1: "b8[32, 3, 64]" = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 64]);  unsqueeze = None
        slice_3: "b8[32, 3, 64]" = torch.ops.aten.slice.Tensor(full, 2, 1, 65)
        copy: "b8[32, 3, 64]" = torch.ops.aten.copy.default(slice_3, expand_1);  slice_3 = expand_1 = None
        slice_scatter: "b8[32, 3, 66]" = torch.ops.aten.slice_scatter.default(full, copy, 2, 1, 65);  full = copy = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:621 in torch_dynamo_resume_in_embed_prefix_at_613, code: fake_image_ids = torch.full(
        full_default: "i64[32, 3, 66]" = torch.ops.aten.full.default([32, 3, 66], 151655, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:627 in torch_dynamo_resume_in_embed_prefix_at_613, code: fake_image_ids[:, :, 0] = cfg.vision_start_token_id
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 151652, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select: "i64[32, 3]" = torch.ops.aten.select.int(full_default, 2, 0)
        copy_1: "i64[32, 3]" = torch.ops.aten.copy.default(select, full_default_1);  select = full_default_1 = None
        select_scatter: "i64[32, 3, 66]" = torch.ops.aten.select_scatter.default(full_default, copy_1, 2, 0);  full_default = copy_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:628 in torch_dynamo_resume_in_embed_prefix_at_613, code: fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 151653, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_3: "i64[32, 3]" = torch.ops.aten.select.int(select_scatter, 2, -1)
        copy_2: "i64[32, 3]" = torch.ops.aten.copy.default(select_3, full_default_2);  select_3 = full_default_2 = None
        select_scatter_1: "i64[32, 3, 66]" = torch.ops.aten.select_scatter.default(select_scatter, copy_2, 2, -1);  select_scatter = copy_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:641 in torch_dynamo_resume_in_embed_prefix_at_613, code: img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        view: "bf16[32, 198, 2560]" = torch.ops.aten.view.default(cat, [32, 198, 2560]);  cat = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:642 in torch_dynamo_resume_in_embed_prefix_at_613, code: image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        clone: "b8[32, 3, 66]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_1: "b8[32, 198]" = torch.ops.aten.view.default(clone, [32, 198]);  clone = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/__init__.py:2380 in __call__, code: return compile_fx(model_, inputs_, config_patches=self.config)
        view_5: "i64[32, 198]" = torch.ops.aten.view.default(select_scatter_1, [32, 198]);  select_scatter_1 = None
        view_6: "b8[32, 198]" = torch.ops.aten.view.default(slice_scatter, [32, 198]);  slice_scatter = None
        return (view, view_1, view_5, view_6)
        