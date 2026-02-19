"""FC2 dataloader builder (folder-based)."""

from pathlib import Path
from typing import Dict

from code.data.fc2_dataset import FC2BatchProvider, resolve_fc2_samples_from_folder


def _resolve_source_dir(base_path: str, split_dir: str) -> str:
    """Build absolute display path for logging."""
    split = Path(split_dir)
    if split.is_absolute():
        return str(split)
    if base_path:
        return str((Path(base_path) / split).resolve())
    return str(split.resolve())


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
