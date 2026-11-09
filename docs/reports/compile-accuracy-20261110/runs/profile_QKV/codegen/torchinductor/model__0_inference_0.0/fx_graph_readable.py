class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 50, 55]", arg1_1: "bf16[32]", arg2_1: "bf16[32, 50, 55]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:858 in forward, code: time_expanded = time[:, None, None]
        unsqueeze: "bf16[32, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        unsqueeze_1: "bf16[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:859 in forward, code: x_t = time_expanded * noise + (1 - time_expanded) * actions
        mul: "bf16[32, 50, 55]" = torch.ops.aten.mul.Tensor(unsqueeze_1, arg0_1)
        sub: "bf16[32, 1, 1]" = torch.ops.aten.sub.Tensor(1, unsqueeze_1);  unsqueeze_1 = None
        mul_1: "bf16[32, 50, 55]" = torch.ops.aten.mul.Tensor(sub, arg2_1);  sub = None
        add: "bf16[32, 50, 55]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:860 in forward, code: u_t = noise - actions
        sub_1: "bf16[32, 50, 55]" = torch.ops.aten.sub.Tensor(arg0_1, arg2_1);  arg0_1 = arg2_1 = None
        return (add, sub_1)
        