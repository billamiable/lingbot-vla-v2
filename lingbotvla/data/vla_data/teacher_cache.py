"""Offline cache of the frozen teacher targets used for depth / video alignment.

The three teachers (MoGe2 + MoRGBD for depth, DINO video) are frozen and their inputs only
depend on the sample (image augmentation off, future frame at a fixed offset), so their
targets can be computed once and read back by the data loader instead of running the teachers
every step.

Layout of a cache directory, one row per sample index of the training dataset:

    depth.npy           [N, T, D]  current-frame depth target (bf16 bits stored as int16)
    video_current.npy   [N, T, D]  DINO patch target of the current frame
    video_future.npy    [N, T, D]  DINO patch target of the future frame
    future_index.npy    [N]        sample index whose current frame is this sample's future
                                   frame; the future depth target is depth[future_index]
    done.npy            [N]        1 once a row has been written
    meta.json                      dataset / teacher fingerprint, checked when reading

The future frame is clamped to the last frame of the episode exactly as LeRobot does for
delta timestamps, so the future depth target needs no row of its own.
"""

import json
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lingbotvla.utils import helper

logger = helper.create_logger(__name__)

TARGET_KEYS = ("depth", "video_current", "video_future")
# Keys added to every sample by TeacherCacheDataset; the training loop pops them.
BATCH_KEYS = {
    "depth": "teacher_depth",
    "future_depth": "teacher_future_depth",
    "video_current": "teacher_video_current",
    "video_future": "teacher_video_future",
}


def future_frame_offset(multi_dataset) -> int:
    offsets = {round(d.chunk_size - 1) for d in multi_dataset._datasets}
    assert len(offsets) == 1, f"datasets disagree on the future frame offset: {offsets}"
    return offsets.pop()


def compute_future_index(multi_dataset) -> np.ndarray:
    """Sample index of each sample's future frame, clamped to its episode like LeRobot does."""
    offset = future_frame_offset(multi_dataset)
    out = np.empty(len(multi_dataset), dtype=np.int64)
    for start, sub in zip(multi_dataset.dataset_start_index, multi_dataset._datasets):
        episodes = sub.dataset.meta.episodes
        for ep_from, ep_to in zip(episodes["dataset_from_index"], episodes["dataset_to_index"]):
            local = np.arange(ep_from, ep_to)
            out[start + local] = start + np.minimum(local + offset, ep_to - 1)
    return out


def fingerprint(multi_dataset, align_params, image_augment: bool, cameras) -> Dict:
    return {
        "datasets": [[sub.dataset.repo_id, int(sub.dataset.num_frames)] for sub in multi_dataset._datasets],
        "future_frame_offset": future_frame_offset(multi_dataset),
        "image_augment": bool(image_augment),
        # The teachers only see the first camera, so the order matters.
        "cameras": list(cameras or []),
        "moge_path": align_params["depth"]["moge_path"],
        "morgbd_path": align_params["depth"]["morgbd_path"],
        "video_ckpt_path": align_params["video"]["ckpt_path"],
        "video_input_size": align_params["video"].get("input_size"),
        "video_use_warmup_frame": align_params["video"].get("use_warmup_frame"),
    }


def _to_int16(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.int16).cpu().numpy()


class TeacherCache:
    """Read-only view of a cache directory. Memory maps are opened lazily, once per process,
    so that each data loader worker gets its own file handles after fork."""

    def __init__(self, cache_dir: str, expected_meta: Optional[Dict] = None):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        if expected_meta is not None and self.meta["fingerprint"] != expected_meta:
            raise ValueError(
                f"teacher cache {cache_dir} was built for different data or teachers:\n"
                f"  cache:    {self.meta['fingerprint']}\n  expected: {expected_meta}"
            )
        done = np.load(os.path.join(cache_dir, "done.npy"))
        if not done.all():
            raise ValueError(f"teacher cache {cache_dir} is incomplete: {int((done == 0).sum())} rows missing")
        self._arrays = None

    def __len__(self):
        return self.meta["num_samples"]

    def _open(self):
        if self._arrays is None:
            self._arrays = {
                key: np.load(os.path.join(self.cache_dir, f"{key}.npy"), mmap_mode="r")
                for key in (*TARGET_KEYS, "future_index")
            }
        return self._arrays

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        arrays = self._open()
        rows = {
            "depth": arrays["depth"][index],
            "future_depth": arrays["depth"][int(arrays["future_index"][index])],
            "video_current": arrays["video_current"][index],
            "video_future": arrays["video_future"][index],
        }
        return {
            BATCH_KEYS[key]: torch.from_numpy(np.array(row)).view(torch.bfloat16) for key, row in rows.items()
        }


class TeacherCacheDataset(Dataset):
    """Adds the cached teacher targets to every sample of a MultiVLADataset and drops the
    teacher input images, which are then no longer needed."""

    def __init__(self, dataset, cache: TeacherCache, keep_teacher_images: bool = False):
        assert len(cache) == len(dataset), f"cache has {len(cache)} rows, dataset {len(dataset)}"
        self.dataset = dataset
        self.cache = cache
        self.keep_teacher_images = keep_teacher_images
        dataset.return_sample_index = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The dataset may substitute another sample when one fails to load, so use the index it
        # reports rather than idx.
        item.update(self.cache.get(item.pop("sample_index")))
        if not self.keep_teacher_images:
            item.pop("pil_images", None)
            item.pop("future_pil_images", None)
        return item


class _IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[int(self.indices[i])]


def build_teacher_cache(
    dataset,
    compute_targets: Callable[[Dict], Dict[str, torch.Tensor]],
    cache_dir: str,
    meta: Dict,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    collate_fn,
    barrier: Callable[[], None],
):
    """Run the teachers over every sample once and write the targets to cache_dir.

    Each rank takes a contiguous slice of sample indices. compute_targets maps a collated batch
    (on GPU) to {"depth", "video_current", "video_future"} bf16 tensors of shape [B, T, D].
    Rows already marked done are skipped, so an interrupted build can be resumed.

    Every batch has exactly batch_size samples: the compiled depth teacher picks kernels by batch
    size, so a smaller last batch gives bf16 results a little off from training, which always runs
    the teachers on micro batches of batch_size. The last batch is padded by repeating its final
    sample; which samples share a batch does not change the result.
    """
    n = len(dataset)
    dataset.return_sample_index = True
    os.makedirs(cache_dir, exist_ok=True)
    done_path = os.path.join(cache_dir, "done.npy")

    # Rank 0 sizes the files from one batch, then every rank opens them for writing.
    if rank == 0 and not os.path.exists(done_path):
        probe = collate_fn([dataset[i] for i in range(batch_size)])
        probe = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in probe.items()}
        shapes = {k: tuple(v.shape[1:]) for k, v in compute_targets(probe).items()}
        for key in TARGET_KEYS:
            np.lib.format.open_memmap(
                os.path.join(cache_dir, f"{key}.npy"), mode="w+", dtype=np.int16, shape=(n, *shapes[key])
            )
        np.save(os.path.join(cache_dir, "future_index.npy"), compute_future_index(dataset))
        np.save(done_path, np.zeros(n, dtype=np.uint8))
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump({"num_samples": n, "shapes": shapes, "fingerprint": meta}, f, indent=2)
    barrier()
    with open(os.path.join(cache_dir, "meta.json")) as f:
        if json.load(f)["fingerprint"] != meta:
            raise ValueError(f"{cache_dir} holds a cache of different data or teachers")

    arrays = {key: np.load(os.path.join(cache_dir, f"{key}.npy"), mmap_mode="r+") for key in TARGET_KEYS}
    done = np.load(done_path)
    lo, hi = n * rank // world_size, n * (rank + 1) // world_size
    todo = np.arange(lo, hi)[done[lo:hi] == 0]
    if len(todo) % batch_size:
        todo = np.concatenate([todo, np.full(batch_size - len(todo) % batch_size, todo[-1])])
    my_done = np.zeros(n, dtype=np.uint8)
    my_done[lo:hi] = done[lo:hi]
    logger.info(f"rank {rank}: samples [{lo}, {hi}), {len(todo)} to compute")

    loader = DataLoader(
        _IndexedSubset(dataset, todo),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    start, written = time.time(), 0
    for batch in loader:
        index = batch.pop("sample_index").numpy()
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets = compute_targets(batch)
        for key in TARGET_KEYS:
            arrays[key][index] = _to_int16(targets[key])
        my_done[index] = 1
        written += len(np.unique(index))
        if rank == 0 and written % (50 * batch_size) < len(index):
            rate = written / (time.time() - start)
            logger.info(f"rank 0: {written}/{len(todo)} samples, {rate:.1f} samples/s per rank")
    for arr in arrays.values():
        arr.flush()
    np.save(os.path.join(cache_dir, f"done.rank{rank}.npy"), my_done[lo:hi])
    barrier()
    if rank == 0:
        for r in range(world_size):
            r_lo, r_hi = n * r // world_size, n * (r + 1) // world_size
            path = os.path.join(cache_dir, f"done.rank{r}.npy")
            done[r_lo:r_hi] = np.load(path)
            os.remove(path)
        np.save(done_path, done)
        logger.info(f"teacher cache {cache_dir}: {int(done.sum())}/{n} rows written in {time.time() - start:.0f}s")
    barrier()
