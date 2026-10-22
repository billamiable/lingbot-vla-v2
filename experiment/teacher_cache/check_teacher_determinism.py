"""Is the online depth teacher itself deterministic? Run it twice on the same batches and compare
the two results with each other and with the cache.

    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 python experiment/teacher_cache/check_teacher_determinism.py \
        CONFIG_YAML CACHE_DIR [BATCHES] [SEED] [INDICES_NPY]

CONFIG_YAML is the config.yaml a training run wrote to its output directory. Uses one GPU; run
several copies with different SEED on different GPUs to cover more samples. With INDICES_NPY,
the batches are drawn from those sample indices only (padded with random samples to 32).
"""

import sys

import numpy as np
import torch

from lingbotvla.data import VLADataCollatorWithPacking, build_vla_dataset
from lingbotvla.data.vla_data.teacher_cache import TeacherCache
from lingbotvla.models import build_processor
from lingbotvla.models.config_registry import get_config_registry
from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, get_depth_target
from lingbotvla.utils.arguments import parse_args

sys.path.insert(0, "tasks/vla")
from train_lingbotvla import Arguments  # noqa: E402


def main():
    config, cache_dir = sys.argv[1], sys.argv[2]
    batches = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    only = np.load(sys.argv[5]) if len(sys.argv) > 5 else None
    sys.argv = [sys.argv[0], config, "--train.global_batch_size", "32", "--train.micro_batch_size", "32"]
    args = parse_args(Arguments)
    torch.cuda.set_device(0)
    args.data.chunk_size = args.train.chunk_size
    cfg = get_config_registry().get_config_cls_from_config_key(args.model.config_key)(**{**vars(args.model), **vars(args.train)})
    ds = build_vla_dataset(
        dataset_config=args.data, model_config=args.model, config=cfg,
        processor=build_processor(args.model.tokenizer_path), use_depth_align=True,
    )
    moge, morgbd = build_depth_model(args.train.align_params)
    if args.train.use_compile:
        moge, morgbd = torch.compile(moge), torch.compile(morgbd)
    cache = TeacherCache(cache_dir)
    collate = VLADataCollatorWithPacking()
    rng = np.random.default_rng(seed)
    run_vs_run = run_vs_cache = total = 0
    for b in range(batches):
        if only is not None:
            chosen = only[b * 16 : (b + 1) * 16]
            if len(chosen) == 0:
                break
            idx = np.concatenate([chosen, rng.choice(len(ds), 32 - len(chosen), replace=False)])
        else:
            idx = rng.choice(len(ds), 32, replace=False)
        batch = collate([ds[int(i)] for i in idx])
        pil = batch["pil_images"].cuda()
        outs = []
        for _ in range(2):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(get_depth_target("MoRGBD", (moge, morgbd), pil)[0])
        cached = torch.stack([cache.get(int(i))["teacher_depth"] for i in idx]).cuda()
        a = (outs[0] != outs[1]).flatten(1).any(1)
        c = (outs[0] != cached).flatten(1).any(1)
        run_vs_run += int(a.sum())
        run_vs_cache += int(c.sum())
        total += len(idx)
        if a.any() or c.any():
            d = (outs[0].float() - outs[1].float()).abs().max()
            print(
                f"batch {b}: {int(a.sum())} samples differ run-to-run (max {d:.3f}), {int(c.sum())} differ "
                f"from the cache: run-to-run {idx[a.cpu().numpy()].tolist()}, vs cache {idx[c.cpu().numpy()].tolist()}",
                flush=True,
            )
    print(f"{total} samples: online run 1 vs run 2 differ in {run_vs_run}, online vs cache differ in {run_vs_cache}")


if __name__ == "__main__":
    main()
