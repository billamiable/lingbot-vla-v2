class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024, 3, 2, 16, 16]", primals_2: "bf16[96, 256, 1536]", primals_3: "bf16[1024]"):
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:72 in forward, code: hidden_states = hidden_states.view(
        view: "bf16[24576, 3, 2, 16, 16]" = torch.ops.aten.reshape.default(primals_2, [-1, 3, 2, 16, 16]);  primals_2 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:75 in forward, code: hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        convolution: "bf16[24576, 1024, 1, 1, 1]" = torch.ops.aten.convolution.default(view, primals_1, primals_3, [2, 16, 16], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1);  primals_3 = None
        view_1: "bf16[24576, 1024]" = torch.ops.aten.reshape.default(convolution, [-1, 1024]);  convolution = None
        return (view_1, primals_1, view)
        