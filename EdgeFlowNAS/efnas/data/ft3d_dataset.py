"""FT3D dataset loading and batch sampling with folder scanning."""

from concurrent.futures import ThreadPoolExecutor
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np


FT3DSample = Tuple[str, str, str]


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


def _build_ft3d_triplet(
    img0_path: str,
    frames_root: Optional[str] = None,
    flow_root: Optional[str] = None,
    direction: str = "into_future",
) -> FT3DSample:
    img0 = Path(img0_path).resolve()
    direction_text = str(direction).strip().lower()
    if direction_text not in ("into_future", "into_past"):
        raise ValueError(f"unsupported FT3D direction: {direction}")
    if frames_root and flow_root:
        frames_root_path = Path(frames_root).resolve()
        flow_root_path = Path(flow_root).resolve()
        relative = img0.relative_to(frames_root_path)
        if len(relative.parts) < 3 or relative.parts[-2] != "left":
            raise ValueError(f"unexpected FT3D frame layout: {img0}")
        frame_index = int(img0.stem)
        if direction_text == "into_future":
            img1_name = f"{frame_index + 1:04d}.png"
            flow_name = f"OpticalFlowIntoFuture_{img0.stem}_L.pfm"
        else:
            img1_name = f"{frame_index - 1:04d}.png"
            flow_name = f"OpticalFlowIntoPast_{img0.stem}_L.pfm"
        img1 = str(img0.with_name(img1_name).resolve())
        flow_relative = Path(*relative.parts[:-2]) / direction_text / "left" / flow_name
        flow_path = str((flow_root_path / flow_relative).resolve())
        return str(img0), img1, flow_path
    raise ValueError("frames_root and flow_root are required")


def _rand_uniform(rng: Any, low: float, high: float) -> float:
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return float(np.random.uniform(low, high))


def _rand_int(rng: Any, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"invalid randint bounds: low={low} high={high}")
    if hasattr(rng, "randint"):
        try:
            return int(rng.randint(low, high + 1))
        except TypeError:
            return int(rng.randint(low, high))
    return int(np.random.randint(low, high + 1))


def _maybe_resize_triplet(img0, img1, flow, scale_x: float, scale_y: float):
    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    img0 = cv2.resize(img0, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    flow = flow * np.asarray([scale_x, scale_y], dtype=np.float32)
    return img0, img1, flow


def _apply_photometric_augment(img0, img1, rng: Any, aug_cfg: Optional[Dict[str, Any]]):
    cfg = aug_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return img0, img1

    brightness = float(cfg.get("brightness", 0.4))
    contrast = float(cfg.get("contrast", 0.4))
    saturation = float(cfg.get("saturation", 0.4))
    asym_prob = float(cfg.get("asymmetric_color_aug_prob", 0.2))
    apply_prob = float(cfg.get("photometric_aug_prob", 1.0))
    if _rand_uniform(rng, 0.0, 1.0) >= apply_prob:
        return img0, img1

    def _augment_one(image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32).copy()
        if brightness > 0:
            out *= _rand_uniform(rng, 1.0 - brightness, 1.0 + brightness)
        if contrast > 0:
            mean = np.mean(out, axis=(0, 1), keepdims=True)
            out = (out - mean) * _rand_uniform(rng, 1.0 - contrast, 1.0 + contrast) + mean
        if saturation > 0:
            gray = np.mean(out, axis=2, keepdims=True)
            out = gray + (out - gray) * _rand_uniform(rng, 1.0 - saturation, 1.0 + saturation)
        return np.clip(out, 0.0, 255.0).astype(np.float32)

    if _rand_uniform(rng, 0.0, 1.0) < asym_prob:
        return _augment_one(img0), _augment_one(img1)

    bright = _rand_uniform(rng, 1.0 - brightness, 1.0 + brightness) if brightness > 0 else 1.0
    cont = _rand_uniform(rng, 1.0 - contrast, 1.0 + contrast) if contrast > 0 else 1.0
    sat = _rand_uniform(rng, 1.0 - saturation, 1.0 + saturation) if saturation > 0 else 1.0

    def _apply_shared(image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32).copy() * bright
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        out = (out - mean) * cont + mean
        gray = np.mean(out, axis=2, keepdims=True)
        out = gray + (out - gray) * sat
        return np.clip(out, 0.0, 255.0).astype(np.float32)

    return _apply_shared(img0), _apply_shared(img1)


def _apply_eraser_augment(img0, img1, rng: Any, aug_cfg: Optional[Dict[str, Any]]):
    cfg = aug_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return img0, img1
    eraser_prob = float(cfg.get("eraser_aug_prob", 0.5))
    if _rand_uniform(rng, 0.0, 1.0) >= eraser_prob:
        return img0, img1

    min_size = int(cfg.get("eraser_min_size", 50))
    max_size = int(cfg.get("eraser_max_size", 100))
    h, w = img1.shape[:2]
    mean_color = np.mean(img1.reshape(-1, 3), axis=0)
    count = _rand_int(rng, 1, 2)
    out = img1.copy()
    for _ in range(count):
        x0 = _rand_int(rng, 0, max(0, w - 1))
        y0 = _rand_int(rng, 0, max(0, h - 1))
        dx = _rand_int(rng, min_size, max_size)
        dy = _rand_int(rng, min_size, max_size)
        out[y0 : min(h, y0 + dy), x0 : min(w, x0 + dx), :] = mean_color
    return img0, out.astype(np.float32)


def _apply_spatial_augment(
    img0,
    img1,
    flow,
    crop_h: int,
    crop_w: int,
    rng: Any,
    aug_cfg: Optional[Dict[str, Any]],
):
    cfg = aug_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return _random_crop_triplet(img0, img1, flow, crop_h, crop_w, rng)

    h, w = img0.shape[:2]
    min_scale = max((crop_h + 8) / float(h), (crop_w + 8) / float(w))
    scale = 2 ** _rand_uniform(rng, float(cfg.get("min_scale", -0.4)), float(cfg.get("max_scale", 0.8)))
    scale_x = scale
    scale_y = scale
    stretch_prob = float(cfg.get("stretch_prob", 0.8))
    max_stretch = float(cfg.get("max_stretch", 0.2))
    if _rand_uniform(rng, 0.0, 1.0) < stretch_prob:
        scale_x *= 2 ** _rand_uniform(rng, -max_stretch, max_stretch)
        scale_y *= 2 ** _rand_uniform(rng, -max_stretch, max_stretch)

    scale_x = max(scale_x, min_scale)
    scale_y = max(scale_y, min_scale)
    spatial_aug_prob = float(cfg.get("spatial_aug_prob", 0.8))
    if _rand_uniform(rng, 0.0, 1.0) < spatial_aug_prob:
        img0, img1, flow = _maybe_resize_triplet(img0, img1, flow, scale_x=scale_x, scale_y=scale_y)

    if bool(cfg.get("do_flip", True)):
        if _rand_uniform(rng, 0.0, 1.0) < float(cfg.get("h_flip_prob", 0.5)):
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            flow = flow[:, ::-1] * np.asarray([-1.0, 1.0], dtype=np.float32)
        if _rand_uniform(rng, 0.0, 1.0) < float(cfg.get("v_flip_prob", 0.1)):
            img0 = img0[::-1, :]
            img1 = img1[::-1, :]
            flow = flow[::-1, :] * np.asarray([1.0, -1.0], dtype=np.float32)

    return _random_crop_triplet(img0, img1, flow, crop_h, crop_w, rng)


def resolve_ft3d_samples_from_folder(
    frames_base_path: Optional[str],
    flow_base_path: Optional[str],
    split_dir: str,
    frames_subdir: str = "",
    flow_subdir: str = "",
    include_directions: Sequence[str] = ("into_future",),
) -> List[FT3DSample]:
    frames_split_root = _resolve_path(base_path=frames_base_path, raw_path=str(Path(frames_subdir) / split_dir) if frames_subdir else split_dir)
    flow_split_root = _resolve_path(base_path=flow_base_path, raw_path=str(Path(flow_subdir) / split_dir) if flow_subdir else split_dir)
    if not frames_split_root.exists() or not frames_split_root.is_dir():
        return []
    if not flow_split_root.exists() or not flow_split_root.is_dir():
        return []

    directions = [str(item).strip().lower() for item in include_directions if str(item).strip()]
    if not directions:
        directions = ["into_future"]

    samples: List[FT3DSample] = []
    for img0_path in sorted(frames_split_root.rglob("left/*.png")):
        for direction in directions:
            try:
                img0, img1, flow = _build_ft3d_triplet(
                    str(img0_path),
                    frames_root=str(frames_split_root),
                    flow_root=str(flow_split_root),
                    direction=direction,
                )
            except Exception:
                continue
            if Path(img1).exists() and Path(flow).exists():
                samples.append((img0, img1, flow))
    return samples


def _random_crop_triplet(img0, img1, flow, crop_h: int, crop_w: int, rng: Any):
    h, w = img0.shape[0], img0.shape[1]
    if h < crop_h or w < crop_w:
        raise ValueError(f"input too small for crop: {h}x{w} vs {crop_h}x{crop_w}")
    top = _rand_int(rng, 0, h - crop_h)
    left = _rand_int(rng, 0, w - crop_w)
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
        samples: List[FT3DSample],
        crop_h: int,
        crop_w: int,
        seed: int = 42,
        source_dir: str = "",
        sampling_mode: str = "random",
        crop_mode: str = "random",
        flow_divisor: float = 12.5,
        augment_cfg: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
    ):
        self.samples = list(samples)
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)
        self.source_dir = str(source_dir)
        self.flow_divisor = float(flow_divisor)
        self.rng = np.random.RandomState(int(seed))
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.crop_mode = str(crop_mode).strip().lower()
        self.augment_cfg = dict(augment_cfg or {})
        self.num_workers = max(1, int(num_workers))
        if self.sampling_mode not in ("random", "sequential", "shuffle_no_replacement"):
            raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
        if self.crop_mode not in ("random", "center"):
            raise ValueError(f"unsupported crop_mode: {crop_mode}")
        self._cursor = 0
        self._order = list(range(len(self.samples)))
        self._sample_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
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
        with self._sample_lock:
            self._order = list(range(len(self.samples)))
            if shuffle:
                self.rng.shuffle(self._order)
            self._cursor = 0

    def _next_sample_path(self) -> FT3DSample:
        if not self.samples:
            raise RuntimeError(
                f"FT3D sample list is empty. source_dir={self.source_dir}. "
                "Please check TRAIN/TEST roots and dataset extraction."
            )
        if self.sampling_mode == "random":
            return self.samples[_rand_int(self.rng, 0, len(self.samples) - 1)]
        if self.sampling_mode == "shuffle_no_replacement":
            if self._cursor >= len(self._order):
                self._cursor = 0
            sample_path = self.samples[self._order[self._cursor]]
            self._cursor += 1
            return sample_path
        sample_path = self.samples[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.samples)
        return sample_path

    def _claim_sample_path(self) -> FT3DSample:
        with self._sample_lock:
            return self._next_sample_path()

    def _next_sample_seed(self) -> int:
        with self._sample_lock:
            return int(self.rng.randint(0, 2**31 - 1))

    def _load_one_from_sample(self, initial_sample_path: FT3DSample, rng: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.samples:
            raise RuntimeError(
                f"FT3D sample list is empty. source_dir={self.source_dir}. "
                "Please check TRAIN/TEST roots and dataset extraction."
            )
        if cv2 is None:
            raise RuntimeError("OpenCV not available")

        retry_limit = max(64, len(self.samples))
        sample_path = initial_sample_path
        for attempt in range(retry_limit):
            img0_path, img1_path, flow_path = sample_path
            if not os.path.exists(img0_path) or not os.path.exists(img1_path) or not os.path.exists(flow_path):
                sample_path = self._claim_sample_path()
                continue
            img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            if img0 is None or img1 is None:
                sample_path = self._claim_sample_path()
                continue
            try:
                flow = _read_flow(flow_path)
            except Exception:
                sample_path = self._claim_sample_path()
                continue
            img0 = img0.astype(np.float32)
            img1 = img1.astype(np.float32)
            flow = np.clip(flow / self.flow_divisor, a_min=-50.0, a_max=50.0).astype(np.float32)
            try:
                if self.crop_mode == "random":
                    img0, img1 = _apply_photometric_augment(img0, img1, self.rng, self.augment_cfg)
                    img0, img1 = _apply_eraser_augment(img0, img1, self.rng, self.augment_cfg)
                    return _apply_spatial_augment(
                        img0=img0,
                        img1=img1,
                        flow=flow,
                        crop_h=self.crop_h,
                        crop_w=self.crop_w,
                        rng=self.rng,
                        aug_cfg=self.augment_cfg,
                    )
                return _center_crop_triplet(img0, img1, flow, self.crop_h, self.crop_w)
            except Exception:
                if attempt + 1 < retry_limit:
                    sample_path = self._claim_sample_path()
                continue

        raise RuntimeError("failed to load valid FT3D sample after retries")

    def _load_one(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_path = self._claim_sample_path()
        rng = np.random.RandomState(self._next_sample_seed())
        return self._load_one_from_sample(sample_path, rng)

    def _executor_instance(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="ft3d_loader")
        return self._executor

    def _load_job(self, job: Tuple[FT3DSample, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_path, seed = job
        rng = np.random.RandomState(int(seed))
        return self._load_one_from_sample(sample_path, rng)

    def next_batch(self, batch_size: int):
        jobs = [(self._claim_sample_path(), self._next_sample_seed()) for _ in range(int(batch_size))]
        if self.num_workers > 1:
            loaded = list(self._executor_instance().map(self._load_job, jobs))
        else:
            loaded = [self._load_job(job) for job in jobs]
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

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
