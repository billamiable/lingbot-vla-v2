class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2304, 1024]", primals_2: "i64[4, 24576]", primals_3: "bf16[4, 24576]", primals_4: "i64[24576]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:344 in fast_pos_embed_interpolate_cached, code: corners = self.pos_embed(metadata["indices"]) * metadata["weights"][:, :, None]
        embedding: "bf16[4, 24576, 1024]" = torch.ops.aten.embedding.default(primals_1, primals_2);  primals_1 = None
        unsqueeze: "bf16[4, 24576, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 2)
        mul: "bf16[4, 24576, 1024]" = torch.ops.aten.mul.Tensor(embedding, unsqueeze);  embedding = unsqueeze = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:345 in fast_pos_embed_interpolate_cached, code: positions = corners[0] + corners[1] + corners[2] + corners[3]
        select: "bf16[24576, 1024]" = torch.ops.aten.select.int(mul, 0, 0)
        select_1: "bf16[24576, 1024]" = torch.ops.aten.select.int(mul, 0, 1)
        add: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(select, select_1);  select = select_1 = None
        select_2: "bf16[24576, 1024]" = torch.ops.aten.select.int(mul, 0, 2)
        add_1: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(add, select_2);  add = select_2 = None
        select_3: "bf16[24576, 1024]" = torch.ops.aten.select.int(mul, 0, 3);  mul = None
        add_2: "bf16[24576, 1024]" = torch.ops.aten.add.Tensor(add_1, select_3);  add_1 = select_3 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py:346 in fast_pos_embed_interpolate_cached, code: return positions.index_select(0, metadata["permutation"])
        index: "bf16[24576, 1024]" = torch.ops.aten.index.Tensor(add_2, [primals_4]);  add_2 = None
        return (index, primals_2, primals_3, primals_4)
        