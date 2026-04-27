"""FC2 dataset loading and batch sampling."""

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np


def _resolve_path(base_path: Optional[str], raw_path: str) -> Path:
    """Resolve a possibly relative dataset path."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if base_path:
        return Path(base_path) / candidate
    return candidate


def _read_flow_file(path_like: str) -> np.ndarray:
    """Read a .flo optical flow file."""
    with open(path_like, "rb") as handle:
        magic = handle.read(4)
        if magic.decode("utf-8") != "PIEH":
            raise ValueError(f"invalid flow header: {path_like}")
        width = np.fromfile(handle, np.int32, 1).squeeze()
        height = np.fromfile(handle, np.int32, 1).squeeze()
        data = np.fromfile(handle, np.float32, width * height * 2)
        flow = data.reshape((height, width, 2)).astype(np.float32)
    return flow


def _build_fc2_triplet(img0_path: str) -> Tuple[str, str, str]:
    """Infer img_1 and flow_01 paths from img_0 path."""
    suffix = "-img_0.png"
    if not img0_path.endswith(suffix):
        raise ValueError(f"unexpected FC2 sample name: {img0_path}")
    base = img0_path[: -len(suffix)]
    img1_path = base + "-img_1.png"
    flow_path = base + "-flow_01.flo"
    return img0_path, img1_path, flow_path


def resolve_fc2_samples_from_folder(base_path: Optional[str], split_dir: str) -> List[str]:
    """Collect valid FC2 img_0 samples from a split folder."""
    split_root = _resolve_path(base_path=base_path, raw_path=split_dir)
    if not split_root.exists() or not split_root.is_dir():
        return []

    samples: List[str] = []
    for img0_path in sorted(split_root.rglob("*-img_0.png")):
        img0, img1, flow = _build_fc2_triplet(str(img0_path))
        if Path(img1).exists() and Path(flow).exists():
            samples.append(img0)
    return samples


def _random_crop_triplet(
    img0: np.ndarray,
    img1: np.ndarray,
    flow: np.ndarray,
    crop_h: int,
    crop_w: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop for paired frames and flow."""
    h, w = img0.shape[0], img0.shape[1]
    if h < crop_h or w < crop_w:
        raise ValueError(f"input too small for crop: {h}x{w} vs {crop_h}x{crop_w}")
    top = rng.randint(0, h - crop_h)
    left = rng.randint(0, w - crop_w)
    img0_c = img0[top : top + crop_h, left : left + crop_w, :]
    img1_c = img1[top : top + crop_h, left : left + crop_w, :]
    flow_c = flow[top : top + crop_h, left : left + crop_w, :]
    return img0_c, img1_c, flow_c


def _center_crop_triplet(
    img0: np.ndarray,
    img1: np.ndarray,
    flow: np.ndarray,
    crop_h: int,
    crop_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center crop for paired frames and flow."""
    h, w = img0.shape[0], img0.shape[1]
    if h < crop_h or w < crop_w:
        raise ValueError(f"input too small for crop: {h}x{w} vs {crop_h}x{crop_w}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    img0_c = img0[top : top + crop_h, left : left + crop_w, :]
    img1_c = img1[top : top + crop_h, left : left + crop_w, :]
    flow_c = flow[top : top + crop_h, left : left + crop_w, :]
    return img0_c, img1_c, flow_c


class FC2BatchProvider:
    """Provide FC2 training/validation batches."""

    def __init__(
        self,
        samples: List[str],
        crop_h: int,
        crop_w: int,
        seed: int = 42,
        source_dir: str = "",
        sampling_mode: str = "random",
        crop_mode: str = "random",
        num_workers: int = 1,
    ):
        self.samples = list(samples)
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)
        self.source_dir = str(source_dir)
        self.rng = random.Random(int(seed))
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.crop_mode = str(crop_mode).strip().lower()
        self.num_workers = max(1, int(num_workers))
        self._executor: Optional[ThreadPoolExecutor] = None
        if self.sampling_mode not in ("random", "sequential", "shuffle_no_replacement"):
            raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
        if self.crop_mode not in ("random", "center"):
            raise ValueError(f"unsupported crop_mode: {crop_mode}")
        self._cursor = 0
        self._order = list(range(len(self.samples)))
        if self.sampling_mode == "shuffle_no_replacement":
            self.start_epoch(shuffle=True)

    def __len__(self) -> int:
        return len(self.samples)

    def reset_cursor(self, index: int = 0) -> None:
        """Reset sequential cursor for deterministic eval batching."""
        if not self.samples:
            self._cursor = 0
            return
        self._cursor = int(index) % len(self.samples)

    def start_epoch(self, shuffle: bool = True) -> None:
        """Start one train epoch for no-replacement sampling."""
        if self.sampling_mode != "shuffle_no_replacement":
            return
        self._order = list(range(len(self.samples)))
        if shuffle:
            self.rng.shuffle(self._order)
        self._cursor = 0

    def _next_sample_path(self) -> str:
        """Pick one sample path according to sampling mode."""
        if self.sampling_mode == "random":
            return self.samples[self.rng.randint(0, len(self.samples) - 1)]
        if self.sampling_mode == "shuffle_no_replacement":
            if not self._order:
                raise RuntimeError("shuffle_no_replacement order is empty")
            if self._cursor >= len(self._order):
                # 同一 epoch 内溢出时循环取头部，保证批次维度稳定。
                self._cursor = 0
            sample_path = self.samples[self._order[self._cursor]]
            self._cursor += 1
            return sample_path
        sample_path = self.samples[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.samples)
        return sample_path

    def _load_one_from_sample(self, img0_path: str, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one valid sample triplet and crop from a claimed path."""
        img0_path, img1_path, flow_path = _build_fc2_triplet(img0_path=img0_path)
        if not os.path.exists(img0_path) or not os.path.exists(img1_path) or not os.path.exists(flow_path):
            raise FileNotFoundError(img0_path)

        img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            raise ValueError(f"failed to read FC2 images: {img0_path}")

        flow = _read_flow_file(flow_path)
        img0 = img0.astype(np.float32)
        img1 = img1.astype(np.float32)
        flow = np.clip(flow, a_min=-50.0, a_max=50.0).astype(np.float32)

        if self.crop_mode == "random":
            return _random_crop_triplet(
                img0=img0,
                img1=img1,
                flow=flow,
                crop_h=self.crop_h,
                crop_w=self.crop_w,
                rng=rng,
            )
        return _center_crop_triplet(
            img0=img0,
            img1=img1,
            flow=flow,
            crop_h=self.crop_h,
            crop_w=self.crop_w,
        )

    def _load_job(self, job: Tuple[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_path, seed = job
        return self._load_one_from_sample(sample_path, random.Random(int(seed)))

    def _executor_instance(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="fc2_loader")
        return self._executor

    def _load_one(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one valid sample triplet and crop."""
        if not self.samples:
            raise RuntimeError(
                f"FC2 sample list is empty. source_dir={self.source_dir}. "
                "Please check train_dir/val_dir and dataset extraction."
            )
        if cv2 is None:
            raise RuntimeError("OpenCV not available")

        retry_limit = max(64, len(self.samples))
        for _ in range(retry_limit):
            img0_path = self._next_sample_path()
            seed = self.rng.randint(0, 2**31 - 1)
            try:
                return self._load_job((img0_path, seed))
            except Exception:
                continue

        raise RuntimeError("failed to load valid FC2 sample after retries")

    def next_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load one mini-batch."""
        if self.num_workers > 1:
            jobs = [(self._next_sample_path(), self.rng.randint(0, 2**31 - 1)) for _ in range(int(batch_size))]
            loaded = list(self._executor_instance().map(self._load_job, jobs))
        else:
            loaded = [self._load_one() for _ in range(int(batch_size))]

        p1_batch = []
        p2_batch = []
        flow_batch = []
        for img0, img1, flow in loaded:
            p1_batch.append(img0)
            p2_batch.append(img1)
            flow_batch.append(flow)

        p1 = np.asarray(p1_batch, dtype=np.float32)
        p2 = np.asarray(p2_batch, dtype=np.float32)
        label = np.asarray(flow_batch, dtype=np.float32)
        input_pair = np.concatenate([p1, p2], axis=3).astype(np.float32)
        return input_pair, p1, p2, label
