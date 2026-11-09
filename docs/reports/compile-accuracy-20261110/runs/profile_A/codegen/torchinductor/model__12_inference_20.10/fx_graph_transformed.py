class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 2560]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:258 in torch_dynamo_resume_in_embed_special_token_at_257, code: return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)
        view: "bf16[1, 1, 1, 2560]" = torch.ops.aten.reshape.default(arg0_1, [1, 1, 1, -1]);  arg0_1 = None
        expand: "bf16[32, 3, 1, 2560]" = torch.ops.aten.expand.default(view, [32, 3, 1, -1]);  view = None
        return (expand,)
        