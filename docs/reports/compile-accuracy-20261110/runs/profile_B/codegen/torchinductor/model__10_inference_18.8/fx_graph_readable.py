class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[96, 64, 2560]", arg1_1: "bf16[96, 64, 2560]", arg2_1: "bf16[96, 64, 2560]", arg3_1: "bf16[96, 64, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:598 in torch_dynamo_resume_in_embed_prefix_at_592, code: img_emb = einops.rearrange(img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images)
        view: "bf16[32, 3, 64, 2560]" = torch.ops.aten.view.default(arg0_1, [32, 3, 64, 2560]);  arg0_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:600 in torch_dynamo_resume_in_embed_prefix_at_592, code: einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images)
        view_1: "bf16[32, 3, 64, 2560]" = torch.ops.aten.view.default(arg1_1, [32, 3, 64, 2560]);  arg1_1 = None
        view_2: "bf16[32, 3, 64, 2560]" = torch.ops.aten.view.default(arg2_1, [32, 3, 64, 2560]);  arg2_1 = None
        view_3: "bf16[32, 3, 64, 2560]" = torch.ops.aten.view.default(arg3_1, [32, 3, 64, 2560]);  arg3_1 = None
        return (view, view_1, view_2, view_3)
        