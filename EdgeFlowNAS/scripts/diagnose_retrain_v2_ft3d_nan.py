"""Diagnose FT3D NaN sources for retrain_v2 runs.

This script is intentionally read-only. It scans FT3D PFM files for non-finite
or extreme values and can replay the retrain_v2 batch provider order to check
whether pre-TensorFlow inputs/labels already contain NaN/Inf.
"""

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from efnas.data.dataloader_builder import build_ft3d_provider
from efnas.data.ft3d_dataset import _read_flow, resolve_ft3d_samples_from_folder
from efnas.data.transforms_180x240 import standardize_image_tensor


Sample = Tuple[str, str, str]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _normalize_path_list(raw_value: Any, fallback: Optional[str] = None) -> List[str]:
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


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = config.setdefault("data", {})
    train_cfg = config.setdefault("train", {})
    runtime_cfg = config.setdefault("runtime", {})
    if args.base_path is not None:
        data_cfg["base_path"] = args.base_path
    if args.frames_base_path is not None:
        data_cfg["ft3d_frames_base_path"] = args.frames_base_path
        data_cfg["ft3d_frames_base_paths"] = [args.frames_base_path]
    if args.flow_base_path is not None:
        data_cfg["ft3d_flow_base_path"] = args.flow_base_path
    if args.train_dir is not None:
        data_cfg["train_dir"] = args.train_dir
    if args.val_dir is not None:
        data_cfg["val_dir"] = args.val_dir
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        runtime_cfg["seed"] = int(args.seed)
    if args.num_workers is not None:
        data_cfg["ft3d_num_workers"] = int(args.num_workers)
    return config


def _collect_samples(config: Dict[str, Any], split: str) -> List[Sample]:
    data_cfg = config.get("data", {})
    split_key = "train_dir" if split == "train" else "val_dir"
    split_dir = str(data_cfg.get(split_key, "TRAIN" if split == "train" else "TEST")).strip()
    frames_base_paths = _normalize_path_list(
        data_cfg.get("ft3d_frames_base_paths", None),
        fallback=data_cfg.get("ft3d_frames_base_path", data_cfg.get("base_path", None)),
    )
    flow_base_path = data_cfg.get("ft3d_flow_base_path", data_cfg.get("base_path", None))
    frames_subdir = str(data_cfg.get("ft3d_frames_subdir", "")).strip()
    flow_subdir = str(data_cfg.get("ft3d_flow_subdir", "")).strip()
    directions = data_cfg.get("ft3d_directions", ["into_future", "into_past"])

    samples: List[Sample] = []
    for frames_base_path in frames_base_paths:
        samples.extend(
            resolve_ft3d_samples_from_folder(
                frames_base_path=frames_base_path,
                flow_base_path=flow_base_path,
                split_dir=split_dir,
                frames_subdir=frames_subdir,
                flow_subdir=flow_subdir,
                include_directions=directions,
            )
        )
    return samples


def _scan_one_flow(sample: Sample, extreme_threshold: float) -> Dict[str, Any]:
    img0, img1, flow_path = sample
    row: Dict[str, Any] = {
        "img0": img0,
        "img1": img1,
        "flow": flow_path,
        "ok": False,
        "error": "",
        "shape": "",
        "finite": False,
        "nan_count": "",
        "inf_count": "",
        "min": "",
        "max": "",
        "max_abs": "",
        "extreme": "",
    }
    try:
        flow = _read_flow(flow_path)
        finite_mask = np.isfinite(flow)
        finite = bool(np.all(finite_mask))
        nan_count = int(np.isnan(flow).sum())
        inf_count = int(np.isinf(flow).sum())
        finite_values = flow[finite_mask]
        if finite_values.size:
            min_value = float(np.min(finite_values))
            max_value = float(np.max(finite_values))
            max_abs = float(np.max(np.abs(finite_values)))
        else:
            min_value = float("nan")
            max_value = float("nan")
            max_abs = float("nan")
        row.update(
            {
                "ok": finite and (not math.isnan(max_abs)) and max_abs <= extreme_threshold,
                "shape": "x".join(str(v) for v in flow.shape),
                "finite": finite,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "min": min_value,
                "max": max_value,
                "max_abs": max_abs,
                "extreme": bool((not math.isnan(max_abs)) and max_abs > extreme_threshold),
            }
        )
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def _iter_limited(samples: Sequence[Sample], limit: int) -> Sequence[Sample]:
    if limit and limit > 0:
        return samples[: int(limit)]
    return samples


def scan_flows(config: Dict[str, Any], splits: Sequence[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in splits:
        samples = list(_iter_limited(_collect_samples(config, split), int(args.limit or 0)))
        print(f"[scan] split={split} samples={len(samples)}")
        with ThreadPoolExecutor(max_workers=max(1, int(args.scan_workers))) as executor:
            future_to_sample = {
                executor.submit(_scan_one_flow, sample, float(args.extreme_threshold)): sample for sample in samples
            }
            for future in as_completed(future_to_sample):
                row = future.result()
                row["split"] = split
                if args.report_all or not row["ok"]:
                    rows.append(row)
                    status = "OK" if row["ok"] else "BAD"
                    print(f"[scan:{status}] split={split} max_abs={row['max_abs']} flow={row['flow']} error={row['error']}")
    return rows


def _batch_has_bad_values(name: str, array: np.ndarray) -> Optional[str]:
    if not np.all(np.isfinite(array)):
        return f"{name} contains non-finite values"
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    if math.isnan(max_abs) or math.isinf(max_abs):
        return f"{name} max_abs is not finite"
    return None


def probe_batches(config: Dict[str, Any], split: str, args: argparse.Namespace) -> int:
    mode = "train" if split == "train" else "eval"
    provider = build_ft3d_provider(config=config, split=split, seed_offset=0 if split == "train" else 999, provider_mode=mode)
    batch_size = int(config.get("train", {}).get("batch_size", args.batch_size or 32))
    steps_per_epoch = int(math.ceil(float(len(provider)) / float(max(1, batch_size))))
    max_batches = int(args.max_batches_per_epoch or 0)
    if max_batches > 0:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    print(f"[probe] split={split} samples={len(provider)} batch_size={batch_size} steps_per_epoch={steps_per_epoch}")

    for epoch in range(1, int(args.epochs_to_probe) + 1):
        if hasattr(provider, "start_epoch"):
            provider.start_epoch(shuffle=(split == "train"))
        for step in range(1, steps_per_epoch + 1):
            jobs = [(provider._claim_sample_path(), provider._next_sample_seed()) for _ in range(batch_size)]
            loaded = [provider._load_job(job) for job in jobs]
            p1 = np.asarray([item[0] for item in loaded], dtype=np.float32)
            p2 = np.asarray([item[1] for item in loaded], dtype=np.float32)
            label = np.asarray([item[2] for item in loaded], dtype=np.float32)
            input_pair = standardize_image_tensor(np.concatenate([p1, p2], axis=3).astype(np.float32))
            for name, array in (("input", input_pair), ("label", label)):
                reason = _batch_has_bad_values(name, array)
                if reason:
                    print(f"[probe:BAD] epoch={epoch} step={step} reason={reason}")
                    for sample, _seed in jobs:
                        print(f"  flow={sample[2]}")
                    return 1
        print(f"[probe:OK] epoch={epoch}")
    return 0


def _write_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose retrain_v2 FT3D NaN inputs")
    parser.add_argument("--config", default="configs/retrain_v2_ft3d.yaml")
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--frames_base_path", default=None)
    parser.add_argument("--flow_base_path", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--skip_scan", action="store_true")
    parser.add_argument("--report_all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scan_workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--extreme_threshold", type=float, default=1.0e5)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--probe_batches", action="store_true")
    parser.add_argument("--probe_split", choices=["train", "val"], default="train")
    parser.add_argument("--epochs_to_probe", type=int, default=1)
    parser.add_argument("--max_batches_per_epoch", type=int, default=0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = _apply_overrides(_load_yaml(str(config_path)), args)
    splits = ["train", "val"] if args.split == "both" else [args.split]

    status = 0
    rows: List[Dict[str, Any]] = []
    if not args.skip_scan:
        rows = scan_flows(config=config, splits=splits, args=args)
        bad_count = sum(1 for row in rows if not row.get("ok", False))
        print(f"[scan:summary] reported_rows={len(rows)} bad_rows={bad_count}")
        if bad_count:
            status = 1
    if args.output_csv:
        _write_rows(args.output_csv, rows)
        print(f"[scan] wrote {args.output_csv}")
    if args.probe_batches:
        status = max(status, probe_batches(config=config, split=args.probe_split, args=args))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
