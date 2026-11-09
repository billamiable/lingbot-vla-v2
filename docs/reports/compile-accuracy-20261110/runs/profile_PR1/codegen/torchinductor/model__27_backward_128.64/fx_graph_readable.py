class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 32, 337, 128]", primals_2: "f32[32, 8, 337, 128]", primals_3: "f32[32, 8, 337, 128]", primals_4: "i32[32, 32, 3, 3]", primals_5: "i32[32, 32, 3]", primals_6: "b8[32, 32, 337, 337]", primals_7: "i32[32, 32, 3]", primals_8: "i32[32, 32, 3, 3]", primals_9: "i32[32, 32, 3]", primals_10: "i32[32, 32, 3, 3]", primals_11: "i32[32, 32, 3]", primals_12: "i32[32, 32, 3, 3]", getitem: "f32[32, 32, 337, 128]", getitem_1: "f32[32, 32, 337]", tangents_1: "f32[32, 337, 4096]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:262 in flex_attention_prepared, code: return out.transpose(1, 2).reshape(B, L, Hq * D)
        view_1: "f32[32, 337, 32, 128]" = torch.ops.aten.view.default(tangents_1, [32, 337, 32, 128]);  tangents_1 = None
        permute_1: "f32[32, 32, 337, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        
         # File: /root/qihank/envs/lingbotvla/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1401 in flex_attention, code: out, lse = flex_attention_hop(
        full_default: "f32[32, 32, 337]" = torch.ops.aten.full.default([32, 32, 337], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_1, primals_2, primals_3, getitem, getitem_1, permute_1, full_default, fw_graph0, joint_graph0, (337, 337, primals_5, primals_4, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, 128, 128, mask_graph0), 0.08838834764831845, {'BLOCK_M': 32, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2, 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True}, (), (primals_6,));  primals_1 = primals_2 = primals_3 = getitem = getitem_1 = permute_1 = full_default = fw_graph0 = joint_graph0 = primals_5 = primals_4 = primals_7 = primals_8 = primals_9 = primals_10 = primals_11 = primals_12 = mask_graph0 = primals_6 = None
        getitem_2: "f32[32, 32, 337, 128]" = flex_attention_backward[0]
        getitem_3: "f32[32, 8, 337, 128]" = flex_attention_backward[1]
        getitem_4: "f32[32, 8, 337, 128]" = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_2, getitem_3, getitem_4, None, None, None, None, None, None, None, None, None)
        
    class fw_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            return arg0_1
            
    class joint_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f32[]"):
            return [arg5_1, None, None, None, None]
            
    class mask_graph0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "b8[32, 32, 337, 337]"):
             # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/flex_attention.py:177 in mask_mod, code: return precomputed_mask[b][h][q_idx][kv_idx]
            index: "b8[32, 337, 337]" = torch.ops.aten.index.Tensor(arg4_1, [arg0_1]);  arg4_1 = arg0_1 = None
            index_1: "b8[337, 337]" = torch.ops.aten.index.Tensor(index, [arg1_1]);  index = arg1_1 = None
            index_2: "b8[337]" = torch.ops.aten.index.Tensor(index_1, [arg2_1]);  index_1 = arg2_1 = None
            index_3: "b8[]" = torch.ops.aten.index.Tensor(index_2, [arg3_1]);  index_2 = arg3_1 = None
            return index_3
            