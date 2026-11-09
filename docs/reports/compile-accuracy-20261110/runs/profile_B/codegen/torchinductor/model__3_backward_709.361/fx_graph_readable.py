class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024, 3, 2, 16, 16]", view: "bf16[24576, 3, 2, 16, 16]", tangents_1: "bf16[24576, 1024]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:75 in forward, code: hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        view_2: "bf16[24576, 1024, 1, 1, 1]" = torch.ops.aten.view.default(tangents_1, [24576, 1024, 1, 1, 1]);  tangents_1 = None
        sum_1: "bf16[1024]" = torch.ops.aten.sum.dim_IntList(view_2, [0, 2, 3, 4])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_2, view, primals_1, [1024], [2, 16, 16], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1, [False, True, False]);  view_2 = view = primals_1 = None
        getitem_1: "bf16[1024, 3, 2, 16, 16]" = convolution_backward[1];  convolution_backward = None
        return (getitem_1, None, sum_1)
        