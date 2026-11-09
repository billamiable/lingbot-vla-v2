class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2560, 5120]", primals_2: "bf16[2560]", primals_3: "bf16[8, 5120]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/_dynamo/external_utils.py:70 in inner, code: return fn(*args, **kwargs)
        permute: "bf16[5120, 2560]" = torch.ops.aten.permute.default(primals_1, [1, 0])
        addmm: "bf16[8, 2560]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        return (addmm, primals_1, primals_3)
        