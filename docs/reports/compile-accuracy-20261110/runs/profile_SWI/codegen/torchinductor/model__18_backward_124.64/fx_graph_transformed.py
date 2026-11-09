class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2560, 5120]", primals_3: "bf16[8, 5120]", tangents_1: "bf16[8, 2560]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/_dynamo/external_utils.py:70 in inner, code: return fn(*args, **kwargs)
        permute: "bf16[5120, 2560]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_1: "bf16[2560, 5120]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "bf16[8, 5120]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[2560, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[2560, 5120]" = torch.ops.aten.mm.default(permute_2, primals_3);  permute_2 = primals_3 = None
        sum_1: "bf16[1, 2560]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "bf16[2560]" = torch.ops.aten.reshape.default(sum_1, [2560]);  sum_1 = None
        return (mm_1, view, mm)
        