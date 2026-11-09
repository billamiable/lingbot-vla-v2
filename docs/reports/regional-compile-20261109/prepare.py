"""Prepare an isolated opt snapshot with shared compile-compatibility fixes."""
from pathlib import Path
ROOT=Path('/root/experiments/lingbot-regional-20261109')
src=ROOT/'src'
def edit(relative,old,new,count=1):
    p=src/relative;s=p.read_text();assert s.count(old)==count,(relative,s.count(old));p.write_text(s.replace(old,new))

# All four cases use identical, unpadded FP32 attention. Handwritten prep already uses this policy.
edit('lingbotvla/models/vla/lingbot_vla/flex_attention.py',
     "    pad_q = q_len_rounded - query_states.shape[2]\n    pad_k = kv_len_rounded - key_states.shape[2]",
     "    # Study-wide policy: match the handwritten prep path, no sequence padding.\n    pad_q = 0\n    pad_k = 0")
edit('lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
     "                            _full_len,\n                            _full_len,\n                        )",
     "                            _full_len,\n                            _full_len,\n                            pad_to_block=False,\n                        )")
# Identical no-op-assert removal in all four groups; warning is suppressed through the shared environment.
edit('lingbotvla/ops/group_gemm/kernel/moe.py',
     '    assert lambda: index.unique().numel() == M * topk, "Holes in output?"',
     '    # Removed vacuous assert-lambda: its body was never executed in eager.')
# Deliberately leaves FSDP hooks/checkpoint policy unchanged; only the tensor forward is compiled.
edit('tasks/vla/train_lingbotvla.py',
     '    if args.train.use_compile:\n        model = torch.compile(model)',
     '''    from regional_runtime import install_regional_compile
    install_regional_compile(model, args.train.output_dir, enabled=os.environ.get("LINGBOT_REGIONAL_COMPILE", "0") == "1")
    if args.train.use_compile:
        model = torch.compile(model)''')
print(src)
