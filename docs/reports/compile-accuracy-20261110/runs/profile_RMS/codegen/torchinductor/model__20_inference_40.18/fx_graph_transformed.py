class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "b8[32, 286]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:788 in torch_dynamo_resume_in_embed_prefix_at_787, code: pad_masks.long(),
        convert_element_type: "i64[32, 286]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int64);  arg0_1 = None
        return (convert_element_type,)
        