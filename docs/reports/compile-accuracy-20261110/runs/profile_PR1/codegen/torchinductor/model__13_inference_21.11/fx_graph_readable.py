class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s73)"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:256 in embed_special_token, code: token = torch.tensor([token_id], device=device, dtype=torch.long)
        scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(arg0_1, dtype = torch.int64, device = device(type='cpu'), pin_memory = False);  arg0_1 = None
        unsqueeze: "i64[1]" = torch.ops.aten.unsqueeze.default(scalar_tensor, 0);  scalar_tensor = None
        device_put: "i64[1]" = torch.ops.prims.device_put.default(unsqueeze, device(type='cuda', index=0));  unsqueeze = None
        return (device_put,)
        