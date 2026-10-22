"""Check the future-depth dedup: is the future frame decoded for sample i pixel-identical to the
current frame decoded for sample future_index[i]?

    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 python experiment/teacher_cache/check_future_frames.py CONFIG_YAML CACHE_DIR [N]

CONFIG_YAML is the config.yaml a training run wrote to its output directory.
"""

import sys

import numpy as np
import torch

from lingbotvla.data import build_vla_dataset
from lingbotvla.models import build_processor
from lingbotvla.utils.arguments import parse_args

sys.path.insert(0, "tasks/vla")
from train_lingbotvla import Arguments  # noqa: E402


def main():
    config, cache_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    # One process: shrink the global batch so the batch-size check passes.
    sys.argv = [sys.argv[0], config, "--train.global_batch_size", "32", "--train.micro_batch_size", "32"]
    args = parse_args(Arguments)
    args.data.chunk_size = args.train.chunk_size
    processor = build_processor(args.model.tokenizer_path)

    from lingbotvla.models.config_registry import get_config_registry

    cfg_cls = get_config_registry().get_config_cls_from_config_key(args.model.config_key)
    cfg = cfg_cls(**{**vars(args.model), **vars(args.train)})
    ds = build_vla_dataset(dataset_config=args.data, model_config=args.model, config=cfg, processor=processor, use_depth_align=True)
    future_index = np.load(f"{cache_dir}/future_index.npy")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), n, replace=False)
    same = clamped = 0
    bad = []
    for i in idx:
        j = int(future_index[i])
        fut = ds[int(i)]["future_pil_images"][0]
        cur = ds[j]["pil_images"][0]
        eq = torch.equal(fut, cur)
        same += eq
        clamped += j != i + 49
        if not eq:
            d = (fut.float() - cur.float()).abs()
            bad.append((int(i), j, j != i + 49, float(d.max()), float((d > 0).float().mean())))
    print(f"{same}/{n} future frames identical to the current frame of future_index ({clamped} clamped at episode end)")
    for b in bad[:20]:
        print(f"  sample {b[0]} -> {b[1]} clamped={b[2]} max|pixel diff| {b[3]:.0f} differing {b[4] * 100:.2f}%")


if __name__ == "__main__":
    main()
