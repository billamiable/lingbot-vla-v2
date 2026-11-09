class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 3, 3]", arg1_1: "bf16[32, 3, 256, 1536]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:584 in embed_prefix, code: images = einops.rearrange(images, "b n l d -> (b n) l d")
        view: "bf16[96, 256, 1536]" = torch.ops.aten.reshape.default(arg1_1, [96, 256, 1536]);  arg1_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:588 in embed_prefix, code: flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        view_1: "i64[96, 3]" = torch.ops.aten.reshape.default(arg0_1, [96, 3]);  arg0_1 = None
        return (view, view_1)
        