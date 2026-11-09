class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 337, 32, 128]", primals_2: "f32[32, 337, 8, 128]", primals_3: "f32[32, 337, 8, 128]", primals_4: "i32[32, 32, 3, 3]", primals_5: "i32[32, 32, 3]", primals_6: "b8[32, 32, 337, 337]", primals_7: "i32[32, 32, 3]", primals_8: "i32[32, 32, 3, 3]", primals_9: "i32[32, 32, 3]", primals_10: "i32[32, 32, 3, 3]", primals_11: "i32[32, 32, 3]", primals_12: "i32[32, 32, 3, 3]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:214 in flex_attention_with_block_mask, code: query_states = query_states.transpose(1, 2).to(torch.float32)
        permute: "f32[32, 32, 337, 128]" = torch.ops.aten.permute.default(primals_1, [0, 2, 1, 3]);  primals_1 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:215 in flex_attention_with_block_mask, code: key_states = key_states.transpose(1, 2).to(torch.float32)
        permute_1: "f32[32, 8, 337, 128]" = torch.ops.aten.permute.default(primals_2, [0, 2, 1, 3]);  primals_2 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:216 in flex_attention_with_block_mask, code: value_states = value_states.transpose(1, 2).to(torch.float32)
        permute_2: "f32[32, 8, 337, 128]" = torch.ops.aten.permute.default(primals_3, [0, 2, 1, 3]);  primals_3 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1401 in flex_attention, code: out, lse = flex_attention_hop(
        sdpa_score0 = self.sdpa_score0
        sdpa_mask0 = self.sdpa_mask0
        flex_attention = torch.ops.higher_order.flex_attention(permute, permute_1, permute_2, sdpa_score0, (337, 337, primals_5, primals_4, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, 128, 128, sdpa_mask0), 0.08838834764831845, {'BLOCK_M': 32, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2, 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True}, (), (primals_6,));  sdpa_score0 = sdpa_mask0 = None
        getitem: "f32[32, 32, 337, 128]" = flex_attention[0]
        getitem_1: "f32[32, 32, 337]" = flex_attention[1];  flex_attention = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:242 in flex_attention_with_block_mask, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[32, 337, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:243 in flex_attention_with_block_mask, code: attn_output = attn_output.reshape(batch_size, -1, attn_output.shape[2] * attn_output.shape[3])
        view: "f32[32, 337, 4096]" = torch.ops.aten.reshape.default(permute_3, [32, -1, 4096]);  permute_3 = None
        return (view, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, permute, permute_1, permute_2, getitem, getitem_1)
        
    class sdpa_score0(torch.nn.Module):
        def forward(self, arg0_1: "f32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1
            
    class sdpa_mask0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "b8[32, 32, 337, 337]"):
             # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:177 in mask_mod, code: return precomputed_mask[b][h][q_idx][kv_idx]
            index: "b8[32, 337, 337]" = torch.ops.aten.index.Tensor(arg4_1, [arg0_1]);  arg4_1 = arg0_1 = None
            index_1: "b8[337, 337]" = torch.ops.aten.index.Tensor(index, [arg1_1]);  index = arg1_1 = None
            index_2: "b8[337]" = torch.ops.aten.index.Tensor(index_1, [arg2_1]);  index_1 = arg2_1 = None
            index_3: "b8[]" = torch.ops.aten.index.Tensor(index_2, [arg3_1]);  index_2 = arg3_1 = None
            return index_3
            