class <lambda>(torch.nn.Module):
    def forward(self):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:256 in embed_special_token, code: token = torch.tensor([token_id], device=device, dtype=torch.long)
        _tensor_constant0: "i64[1]" = self._tensor_constant0;  _tensor_constant0 = None
        full_default: "i64[1]" = torch.ops.aten.full.default([1], 151652, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (full_default,)
        