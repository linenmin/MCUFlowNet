"""FT3D dataset loading and batch sampling with folder scanning."""

import os
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np


def _resolve_path(base_path: Optional[str], raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if base_path:
        return Path(base_path) / candidate
    return candidate


def _read_pfm(path_like: str) -> np.ndarray:
    with open(path_like, "rb") as handle:
        header = handle.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"invalid PFM header: {path_like}")
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", handle.readline().decode("ascii"))
        if not dim_match:
            raise ValueError(f"malformed PFM shape: {path_like}")
        width, height = [int(v) for v in dim_match.groups()]
        scale = float(handle.readline().decode("ascii").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, endian + "f")
        shape = (height, width, 3) if header == "PF" else (height, width)
        return np.flipud(np.reshape(data, shape))


def _read_flow(path_like: str) -> np.ndarray:
    if path_like.lower().endswith(".pfm"):
        flow = _read_pfm(path_like)
        if flow.ndim == 3:
            return flow[:, :, :2].astype(np.float32)
        raise ValueError(f"PFM flow must be HxWx3: {path_like}")
    raise ValueError(f"unsupported FT3D flow format: {path_like}")


def _build_ft3d_triplet(img0_path: str, frames_root: Optional[str] = None, flow_root: Optional[str] = None) -> Tuple[str, str, str]:
    img0 = Path(img0_path).resolve()
    next_name = f"{int(img0.stem) + 1:04d}.png"
    img1 = str(img0.with_name(next_name).resolve())
    if frames_root and flow_root:
        frames_root_path = Path(frames_root).resolve()
        flow_root_path = Path(flow_root).resolve()
        relative = img0.relative_to(frames_root_path)
        if len(relative.parts) < 3 or relative.parts[-2] != "left":
            raise ValueError(f"unexpected FT3D frame layout: {img0}")
        flow_relative = Path(*relative.parts[:-2]) / "into_future" / "left" / f"OpticalFlowIntoFuture_{img0.stem}_L.pfm"
        flow_path = str((flow_root_path / flow_relative).resolve())
        return str(img0), img1, flow_path
    raise ValueError("frames_root and flow_root are required")


def resolve_ft3d_samples_from_folder(
    frames_base_path: Optional[str],
    flow_base_path: Optional[str],
    split_dir: str,
    frames_subdir: str = "",
    flow_subdir: str = "",
) -> List[str]:
    frames_split_root = _resolve_path(base_path=frames_base_path, raw_path=str(Path(frames_subdir) / split_dir) if frames_subdir else split_dir)
    flow_split_root = _resolve_path(base_path=flow_base_path, raw_path=str(Path(flow_subdir) / split_dir) if flow_subdir else split_dir)
    if not frames_split_root.exists() or not frames_split_root.is_dir():
        return []
    if not flow_split_root.exists() or not flow_split_root.is_dir():
        return []

    samples: List[str] = []
    for img0_path in sorted(frames_split_root.rglob("left/*.png")):
        try:
            img0, img1, flow = _build_ft3d_triplet(str(img0_path), frames_root=str(frames_split_root), flow_root=str(flow_split_root))
        except Exception:
            continue
        if Path(img1).exists() and Path(flow).exists():
            samples.append(img0)
    return samples


def _random_crop_triplet(img0, img1, flow, crop_h: int, crop_w: int, rng: random.Random):
    h, w = img0.shape[0], img0.shape[1]
    if h < crop_h or w < crop_w:
        raise ValueError(f"input too small for crop: {h}x{w} vs {crop_h}x{crop_w}")
    top = rng.randint(0, h - crop_h)
    left = rng.randint(0, w - crop_w)
    return (
        img0[top : top + crop_h, left : left + crop_w, :],
        img1[top : top + crop_h, left : left + crop_w, :],
        flow[top : top + crop_h, left : left + crop_w, :],
    )


def _center_crop_triplet(img0, img1, flow, crop_h: int, crop_w: int):
    h, w = img0.shape[0], img0.shape[1]
    if h < crop_h or w < crop_w:
        raise ValueError(f"input too small for crop: {h}x{w} vs {crop_h}x{crop_w}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return (
        img0[top : top + crop_h, left : left + crop_w, :],
        img1[top : top + crop_h, left : left + crop_w, :],
        flow[top : top + crop_h, left : left + crop_w, :],
    )


class FT3DBatchProvider:
    """Provide FT3D train/validation batches from scanned folders."""

    def __init__(
        self,
        samples: List[str],
        crop_h: int,
        crop_w: int,
        frames_root: str,
        flow_root: str,
        seed: int = 42,
        source_dir: str = "",
        sampling_mode: str = "random",
        crop_mode: str = "random",
        flow_divisor: float = 12.5,
    ):
        self.samples = list(samples)
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)
        self.frames_root = str(frames_root)
        self.flow_root = str(flow_root)
        self.source_dir = str(source_dir)
        self.flow_divisor = float(flow_divisor)
        self.rng = random.Random(int(seed))
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.crop_mode = str(crop_mode).strip().lower()
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
        if not self.samples:
            self._cursor = 0
            return
        self._cursor = int(index) % len(self.samples)

    def start_epoch(self, shuffle: bool = True) -> None:
        if self.sampling_mode != "shuffle_no_replacement":
            return
        self._order = list(range(len(self.samples)))
        if shuffle:
            self.rng.shuffle(self._order)
        self._cursor = 0

    def _next_sample_path(self) -> str:
        if self.sampling_mode == "random":
            return self.samples[self.rng.randint(0, len(self.samples) - 1)]
        if self.sampling_mode == "shuffle_no_replacement":
            if self._cursor >= len(self._order):
                self._cursor = 0
            sample_path = self.samples[self._order[self._cursor]]
            self._cursor += 1
            return sample_path
        sample_path = self.samples[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.samples)
        return sample_path

    def _load_one(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.samples:
            raise RuntimeError(
                f"FT3D sample list is empty. source_dir={self.source_dir}. "
                "Please check TRAIN/TEST roots and dataset extraction."
            )
        if cv2 is None:
            raise RuntimeError("OpenCV not available")

        retry_limit = max(64, len(self.samples))
        for _ in range(retry_limit):
            img0_path = self._next_sample_path()
            img0_path, img1_path, flow_path = _build_ft3d_triplet(
                img0_path=img0_path,
                frames_root=self.frames_root,
                flow_root=self.flow_root,
            )
            if not os.path.exists(img0_path) or not os.path.exists(img1_path) or not os.path.exists(flow_path):
                continue
            img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            if img0 is None or img1 is None:
                continue
            try:
                flow = _read_flow(flow_path)
            except Exception:
                continue
            img0 = img0.astype(np.float32)
            img1 = img1.astype(np.float32)
            flow = np.clip(flow / self.flow_divisor, a_min=-50.0, a_max=50.0).astype(np.float32)
            try:
                if self.crop_mode == "random":
                    return _random_crop_triplet(img0, img1, flow, self.crop_h, self.crop_w, self.rng)
                return _center_crop_triplet(img0, img1, flow, self.crop_h, self.crop_w)
            except Exception:
                continue

        raise RuntimeError("failed to load valid FT3D sample after retries")

    def next_batch(self, batch_size: int):
        p1_batch = []
        p2_batch = []
        flow_batch = []
        for _ in range(int(batch_size)):
            img0, img1, flow = self._load_one()
            p1_batch.append(img0)
            p2_batch.append(img1)
            flow_batch.append(flow)
        p1 = np.asarray(p1_batch, dtype=np.float32)
        p2 = np.asarray(p2_batch, dtype=np.float32)
        label = np.asarray(flow_batch, dtype=np.float32)
        input_pair = np.concatenate([p1, p2], axis=3).astype(np.float32)
        return input_pair, p1, p2, label
