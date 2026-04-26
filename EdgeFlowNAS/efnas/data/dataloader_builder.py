"""Dataset provider builders for FC2 and FT3D."""

from pathlib import Path
from typing import Dict, List

from efnas.data.fc2_dataset import FC2BatchProvider, resolve_fc2_samples_from_folder
from efnas.data.ft3d_dataset import FT3DBatchProvider, resolve_ft3d_samples_from_folder


def _resolve_source_dir(base_path: str, split_dir: str) -> str:
    split = Path(split_dir)
    if split.is_absolute():
        return str(split)
    if base_path:
        return str((Path(base_path) / split).resolve())
    return str(split.resolve())


def _normalize_path_list(raw_value, fallback=None) -> List[str]:
    if isinstance(raw_value, (list, tuple)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
    elif raw_value is None or str(raw_value).strip() == "":
        values = []
    else:
        values = [str(raw_value).strip()]
    if values:
        return values
    if fallback is None or str(fallback).strip() == "":
        return []
    return [str(fallback).strip()]


def build_fc2_provider(config: Dict, split: str, seed_offset: int = 0, provider_mode: str = "train") -> FC2BatchProvider:
    """Build FC2 batch provider from folder split."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be train or val, got: {split}")
    mode = str(provider_mode).strip().lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"provider_mode must be train or eval, got: {provider_mode}")

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    runtime_cfg = config.get("runtime", {})
    split_key = "train_dir" if split == "train" else "val_dir"
    split_dir = str(data_cfg.get(split_key, "")).strip()
    base_path = data_cfg.get("base_path", None)

    sample_paths = resolve_fc2_samples_from_folder(base_path=base_path, split_dir=split_dir)
    source_dir = _resolve_source_dir(base_path=str(base_path or ""), split_dir=split_dir)

    if mode == "train":
        sampling_mode = str(train_cfg.get("train_sampling_mode", "random")).strip().lower()
        if sampling_mode not in ("random", "sequential", "shuffle_no_replacement"):
            raise ValueError(f"unsupported train_sampling_mode: {sampling_mode}")
        crop_mode = str(train_cfg.get("train_crop_mode", "random")).strip().lower()
        if crop_mode not in ("random", "center"):
            raise ValueError(f"unsupported train_crop_mode: {crop_mode}")
    else:
        sampling_mode = "sequential"
        crop_mode = "center"

    provider = FC2BatchProvider(
        samples=sample_paths,
        crop_h=int(data_cfg.get("input_height", 180)),
        crop_w=int(data_cfg.get("input_width", 240)),
        seed=int(runtime_cfg.get("seed", 42)) + int(seed_offset),
        source_dir=source_dir,
        sampling_mode=sampling_mode,
        crop_mode=crop_mode,
    )
    return provider


def build_ft3d_provider(
    config: Dict, split: str, seed_offset: int = 0, provider_mode: str = "train"
) -> FT3DBatchProvider:
    """Build FT3D batch provider by scanning TRAIN/TEST split roots."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be train or val, got: {split}")
    mode = str(provider_mode).strip().lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"provider_mode must be train or eval, got: {provider_mode}")

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    runtime_cfg = config.get("runtime", {})
    split_key = "train_dir" if split == "train" else "val_dir"
    split_dir = str(data_cfg.get(split_key, "")).strip()
    base_path = data_cfg.get("base_path", None)
    frames_base_paths = _normalize_path_list(
        data_cfg.get("ft3d_frames_base_paths", None),
        fallback=data_cfg.get("ft3d_frames_base_path", base_path),
    )
    flow_base_path = data_cfg.get("ft3d_flow_base_path", base_path)
    frames_subdir = str(data_cfg.get("ft3d_frames_subdir", "")).strip()
    flow_subdir = str(data_cfg.get("ft3d_flow_subdir", "")).strip()
    directions = data_cfg.get("ft3d_directions", ["into_future", "into_past"])

    sample_paths = []
    resolved_source_dirs = []
    for frames_base_path in frames_base_paths:
        sample_paths.extend(
            resolve_ft3d_samples_from_folder(
                frames_base_path=frames_base_path,
                flow_base_path=flow_base_path,
                split_dir=split_dir,
                frames_subdir=frames_subdir,
                flow_subdir=flow_subdir,
                include_directions=directions,
            )
        )
        resolved_source_dirs.append(
            _resolve_source_dir(
                base_path=str(frames_base_path or ""),
                split_dir=str(Path(frames_subdir) / split_dir) if frames_subdir else split_dir,
            )
        )
    source_dir = ";".join(resolved_source_dirs)
    flow_dir = _resolve_source_dir(base_path=str(flow_base_path or ""), split_dir=str(Path(flow_subdir) / split_dir) if flow_subdir else split_dir)

    if mode == "train":
        sampling_mode = str(train_cfg.get("train_sampling_mode", "random")).strip().lower()
        if sampling_mode not in ("random", "sequential", "shuffle_no_replacement"):
            raise ValueError(f"unsupported train_sampling_mode: {sampling_mode}")
        crop_mode = str(train_cfg.get("train_crop_mode", "random")).strip().lower()
        if crop_mode not in ("random", "center"):
            raise ValueError(f"unsupported train_crop_mode: {crop_mode}")
        augment_cfg = dict(
            data_cfg.get(
                "ft3d_train_augment",
                {
                    "enabled": True,
                    "min_scale": -0.4,
                    "max_scale": 0.8,
                    "spatial_aug_prob": 0.8,
                    "stretch_prob": 0.8,
                    "max_stretch": 0.2,
                    "do_flip": True,
                    "h_flip_prob": 0.5,
                    "v_flip_prob": 0.1,
                    "photometric_aug_prob": 1.0,
                    "brightness": 0.4,
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "asymmetric_color_aug_prob": 0.2,
                    "eraser_aug_prob": 0.5,
                    "eraser_min_size": 50,
                    "eraser_max_size": 100,
                },
            )
        )
    else:
        sampling_mode = "sequential"
        crop_mode = "center"
        augment_cfg = {"enabled": False}

    train_num_workers = int(data_cfg.get("ft3d_num_workers", 1))
    eval_num_workers = int(data_cfg.get("ft3d_eval_num_workers", train_num_workers))

    provider = FT3DBatchProvider(
        samples=sample_paths,
        crop_h=int(data_cfg.get("input_height", 480)),
        crop_w=int(data_cfg.get("input_width", 640)),
        seed=int(runtime_cfg.get("seed", 42)) + int(seed_offset),
        source_dir=source_dir,
        sampling_mode=sampling_mode,
        crop_mode=crop_mode,
        flow_divisor=float(data_cfg.get("ft3d_flow_divisor", 12.5)),
        augment_cfg=augment_cfg,
        num_workers=train_num_workers if mode == "train" else eval_num_workers,
    )
    return provider
