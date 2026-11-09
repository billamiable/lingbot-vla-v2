class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "b8[32, 3]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/einops/einops.py:650 in repeat, code: return reduce(tensor, pattern, reduction="repeat", **axes_lengths)
        unsqueeze: "b8[32, 3, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand: "b8[32, 3, 64]" = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 64]);  unsqueeze = None
        return (expand,)
        