class <lambda>(torch.nn.Module):
    def forward(self):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:188 in _build_segment_info, code: seg_lengths_t = torch.tensor(seq_lengths_per_layer, dtype=torch.int64, device=device)
        _tensor_constant0: "i64[32]" = self._tensor_constant0
        lift_fresh_copy: "i64[32]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:191 in _build_segment_info, code: all_seg_lengths = seg_lengths_t.repeat(num_layers)  # [total_segments]
        repeat: "i64[1152]" = torch.ops.aten.repeat.default(lift_fresh_copy, [36])
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:194 in _build_segment_info, code: per_layer_starts = torch.cumsum(seg_lengths_t, 0) - seg_lengths_t  # [S]
        cumsum: "i64[32]" = torch.ops.aten.cumsum.default(lift_fresh_copy, 0)
        sub: "i64[32]" = torch.ops.aten.sub.Tensor(cumsum, lift_fresh_copy);  cumsum = lift_fresh_copy = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:196 in _build_segment_info, code: layer_offsets = torch.arange(num_layers, device=device, dtype=torch.int64) * N_per_layer
        iota: "i64[36]" = torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i64[36]" = torch.ops.aten.mul.Tensor(iota, 1632);  iota = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:198 in _build_segment_info, code: all_seg_starts = (per_layer_starts.unsqueeze(0) + layer_offsets.unsqueeze(1)).reshape(-1)
        unsqueeze: "i64[1, 32]" = torch.ops.aten.unsqueeze.default(sub, 0);  sub = None
        unsqueeze_1: "i64[36, 1]" = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        add: "i64[36, 32]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        view: "i64[1152]" = torch.ops.aten.view.default(add, [-1]);  add = None
        
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:202 in _build_segment_info, code: torch.arange(total_segments, device=device), all_seg_lengths,
        iota_1: "i64[1152]" = torch.ops.prims.iota.default(1152, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        return (iota_1, repeat, view)
        