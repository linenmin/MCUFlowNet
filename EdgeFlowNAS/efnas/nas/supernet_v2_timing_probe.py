"""Timing probe for comparing FC2-val and Sintel validation cost on one V2 subnet."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from efnas.app.train_supernet_app import _load_yaml, _merge_overrides
from efnas.data.dataloader_builder import build_fc2_provider
from efnas.engine.supernet_v2_evaluator import setup_supernet_v2_eval_model
from efnas.nas.search_space_v2 import V2_REFERENCE_ARCH_CODE
from efnas.nas.supernet_v2_rank_consistency import (
    _evaluate_fc2_one_arch,
    _evaluate_sintel_one_arch,
    _option_or_default,
    _parse_arch_text,
    _prepare_sintel_lists,
    _resolve_path,
    _restore_eval_checkpoint,
    _run_bn_recalibration,
    _to_arch_text,
)
from efnas.utils.path_utils import ensure_directory, project_root


def compute_timing_comparison(
    fc2_seconds: float,
    fc2_samples: int,
    sintel_seconds: float,
    sintel_samples: int,
) -> Dict[str, Any]:
    """Summarize timing comparison metrics with sample-normalized views."""
    fc2_time = float(fc2_seconds)
    sintel_time = float(sintel_seconds)
    fc2_count = max(1, int(fc2_samples))
    sintel_count = max(1, int(sintel_samples))
    fc2_per_sample = fc2_time / float(fc2_count)
    sintel_per_sample = sintel_time / float(sintel_count)
    return {
        "fc2_eval_seconds": fc2_time,
        "fc2_samples": int(fc2_samples),
        "fc2_seconds_per_sample": float(fc2_per_sample),
        "sintel_eval_seconds": sintel_time,
        "sintel_samples": int(sintel_samples),
        "sintel_seconds_per_sample": float(sintel_per_sample),
        "sintel_over_fc2_eval_ratio": float(sintel_time / fc2_time) if fc2_time > 0.0 else None,
        "sintel_over_fc2_per_sample_ratio": float(sintel_per_sample / fc2_per_sample) if fc2_per_sample > 0.0 else None,
    }


def _build_markdown_report(payload: Dict[str, Any]) -> str:
    """Render a short markdown summary for quick inspection."""
    arch_text = str(payload["arch_code"])
    checkpoint_type = str(payload["checkpoint_type"])
    experiment_dir = str(payload["experiment_dir"])
    metrics = payload["timing"]
    bn_seconds = float(payload["bn_recal_seconds"])
    fc2_total = bn_seconds + float(metrics["fc2_eval_seconds"])
    sintel_total = bn_seconds + float(metrics["sintel_eval_seconds"])
    lines = [
        "# V2 Timing Probe",
        "",
        f"- Arch code: `{arch_text}`",
        f"- Checkpoint: `{checkpoint_type}`",
        f"- Experiment dir: `{experiment_dir}`",
        f"- BN recalibration: `{bn_seconds:.3f}s`",
        f"- FC2 eval: `{metrics['fc2_eval_seconds']:.3f}s` over `{metrics['fc2_samples']}` samples",
        f"- Sintel eval: `{metrics['sintel_eval_seconds']:.3f}s` over `{metrics['sintel_samples']}` samples",
        f"- FC2 eval per sample: `{metrics['fc2_seconds_per_sample']:.6f}s`",
        f"- Sintel eval per sample: `{metrics['sintel_seconds_per_sample']:.6f}s`",
        f"- Sintel / FC2 eval ratio: `{metrics['sintel_over_fc2_eval_ratio']:.3f}x`" if metrics["sintel_over_fc2_eval_ratio"] is not None else "- Sintel / FC2 eval ratio: `N/A`",
        f"- Sintel / FC2 per-sample ratio: `{metrics['sintel_over_fc2_per_sample_ratio']:.3f}x`" if metrics["sintel_over_fc2_per_sample_ratio"] is not None else "- Sintel / FC2 per-sample ratio: `N/A`",
        f"- FC2 total incl. shared BN: `{fc2_total:.3f}s`",
        f"- Sintel total incl. shared BN: `{sintel_total:.3f}s`",
        "",
        "## Accuracy Snapshot",
        "",
        f"- FC2 EPE: `{float(payload['fc2_epe']):.6f}`",
        f"- Sintel EPE: `{float(payload['sintel_epe']):.6f}`",
    ]
    return "\n".join(lines) + "\n"


def run_timing_probe(
    config_path: str,
    overrides: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure one fixed V2 subnet on FC2 val and Sintel and compare wall time."""
    config = _merge_overrides(_load_yaml(config_path), overrides)

    arch_opt = options.get("arch_code", "")
    arch_raw = "" if arch_opt is None else str(arch_opt).strip()
    arch_code = list(V2_REFERENCE_ARCH_CODE) if not arch_raw else _parse_arch_text(arch_raw)
    arch_text = _to_arch_text(arch_code)

    output_dir_raw = options.get("output_dir", "")
    output_dir_text = "" if output_dir_raw is None else str(output_dir_raw).strip()
    if output_dir_text:
        output_dir = _resolve_path(output_dir_text)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (project_root() / "outputs" / "search_v2" / f"timing_probe_{timestamp}").resolve()

    if bool(options.get("dry_run", False)):
        return {
            "mode": "dry_run",
            "resolved_output_dir": str(output_dir),
            "arch_code": arch_text,
            "config": config,
        }

    experiment_dir_text = str(options.get("experiment_dir", "")).strip()
    dataset_root_text = str(options.get("dataset_root", "")).strip()
    if not experiment_dir_text:
        raise RuntimeError("--experiment_dir is required unless --dry_run is used")
    if not dataset_root_text:
        raise RuntimeError("--dataset_root is required unless --dry_run is used")

    experiment_dir = _resolve_path(experiment_dir_text)
    dataset_root = _resolve_path(dataset_root_text, base_path=config.get("data", {}).get("base_path", None))
    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment_dir does not exist: {experiment_dir}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    metadata_dir = Path(ensure_directory(str(output_dir / "metadata")))
    report_json = metadata_dir / "timing_probe_summary.json"
    report_md = metadata_dir / "timing_probe_summary.md"

    if "CUDA_VISIBLE_DEVICES" not in os.environ and options.get("gpu_device", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options["gpu_device"])

    fc2_train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    fc2_val_provider = build_fc2_provider(config=config, split="val", seed_offset=1, provider_mode="eval")

    img1_list, img2_list, flo_list, sintel_list_path = _prepare_sintel_lists(
        dataset_root=dataset_root,
        sintel_list_text=str(options.get("sintel_list", "EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")),
    )
    sintel_patch_size = [int(item) for item in str(options.get("sintel_patch_size", "416,1024")).split(",")]
    if len(sintel_patch_size) != 2:
        raise ValueError("sintel_patch_size must be H,W")

    max_fc2_val_samples = _option_or_default(options, "max_fc2_val_samples", None)
    fc2_eval_samples = int(len(fc2_val_provider) if max_fc2_val_samples is None else min(len(fc2_val_provider), int(max_fc2_val_samples)))
    max_sintel_samples = _option_or_default(options, "max_sintel_samples", None)
    sintel_eval_samples = int(len(img1_list) if max_sintel_samples is None else min(len(img1_list), int(max_sintel_samples)))
    fc2_eval_batch_size = int(_option_or_default(options, "eval_batch_size", config.get("train", {}).get("batch_size", 8)))
    bn_recal_batches = int(_option_or_default(options, "bn_recal_batches", 16))
    bn_recal_batch_size = int(_option_or_default(options, "bn_recal_batch_size", config.get("train", {}).get("batch_size", 8)))

    eval_model = setup_supernet_v2_eval_model(
        experiment_dir=experiment_dir,
        checkpoint_type=str(options.get("checkpoint_type", "best")),
        flow_channels=int(config.get("data", {}).get("flow_channels", 2)),
        allow_growth=True,
    )

    try:
        _restore_eval_checkpoint(eval_model)
        bn_t0 = time.perf_counter()
        _run_bn_recalibration(
            eval_model=eval_model,
            train_provider=fc2_train_provider,
            arch_code=arch_code,
            num_batches=bn_recal_batches,
            batch_size=bn_recal_batch_size,
        )
        bn_seconds = float(time.perf_counter() - bn_t0)

        fc2_t0 = time.perf_counter()
        fc2_epe = _evaluate_fc2_one_arch(
            eval_model=eval_model,
            val_provider=fc2_val_provider,
            arch_code=arch_code,
            batch_size=fc2_eval_batch_size,
            max_samples=max_fc2_val_samples,
        )
        fc2_seconds = float(time.perf_counter() - fc2_t0)

        sintel_t0 = time.perf_counter()
        sintel_epe = _evaluate_sintel_one_arch(
            eval_model=eval_model,
            arch_code=arch_code,
            img1_list=img1_list,
            img2_list=img2_list,
            flo_list=flo_list,
            patch_size=sintel_patch_size,
            max_samples=max_sintel_samples,
        )
        sintel_seconds = float(time.perf_counter() - sintel_t0)
    finally:
        eval_model["sess"].close()

    timing = compute_timing_comparison(
        fc2_seconds=fc2_seconds,
        fc2_samples=fc2_eval_samples,
        sintel_seconds=sintel_seconds,
        sintel_samples=sintel_eval_samples,
    )
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "experiment_dir": str(experiment_dir),
        "checkpoint_type": str(options.get("checkpoint_type", "best")),
        "dataset_root": str(dataset_root),
        "sintel_list": sintel_list_path,
        "sintel_patch_size": sintel_patch_size,
        "arch_code": arch_text,
        "bn_recal_batches": int(bn_recal_batches),
        "bn_recal_batch_size": int(bn_recal_batch_size),
        "bn_recal_seconds": float(bn_seconds),
        "fc2_epe": float(fc2_epe),
        "sintel_epe": float(sintel_epe),
        "timing": timing,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_build_markdown_report(payload), encoding="utf-8")

    return {
        "exit_code": 0,
        "output_dir": str(output_dir),
        "summary_json": str(report_json),
        "summary_md": str(report_md),
        "summary": payload,
    }
