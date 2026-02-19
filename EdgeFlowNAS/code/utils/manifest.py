"""Utilities for training manifest and run manifest."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List


def _hash_config(config: Dict[str, Any]) -> str:
    """Compute stable SHA256 hash for full config."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_resume_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build resume-critical signature used by run_manifest validation."""
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    return {
        # 数据与输入口径
        "dataset": data_cfg.get("dataset", ""),
        "input_shape": [int(data_cfg.get("input_height", 0)), int(data_cfg.get("input_width", 0))],
        "flow_channels": int(data_cfg.get("flow_channels", 2)),
        # 超网训练核心模式
        "supernet_mode": train_cfg.get("supernet_mode", "strict_fairness"),
        "uncertainty_type": train_cfg.get("uncertainty_type", "LinearSoftplus"),
        # 架构编码语义：0/1/2 = 3x3/5x5/7x7
        "arch_semantics_version": "kernel_idx_v2_0to3_1to5_2to7",
        # 固定骨架参数（当前实现常量）
        "network": "MultiScaleResNet_supernet",
        "init_neurons": 32,
        "expansion_factor": 2.0,
        "multiscale_outputs": 3,
    }


def compare_run_manifest(current_manifest: Dict[str, Any], resume_manifest: Dict[str, Any]) -> List[str]:
    """Compare two run manifests and return mismatch messages."""
    current_sig = current_manifest.get("resume_signature", {})
    resume_sig = resume_manifest.get("resume_signature", {})
    mismatches: List[str] = []
    for key in sorted(set(current_sig.keys()) | set(resume_sig.keys())):
        if current_sig.get(key) != resume_sig.get(key):
            mismatches.append(
                f"{key}: resume='{resume_sig.get(key)}' current='{current_sig.get(key)}'"
            )
    return mismatches


def build_run_manifest(config: Dict[str, Any], git_commit: str) -> Dict[str, Any]:
    """Build run-level manifest for resume compatibility checks."""
    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    checkpoint_cfg = config.get("checkpoint", {})
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "manifest_type": "run_manifest",
        "experiment_name": runtime_cfg.get("experiment_name", "unknown"),
        "git_commit": git_commit,
        "config_hash": _hash_config(config),
        "resume_signature": _build_resume_signature(config=config),
        "train_snapshot": {
            "num_epochs": int(train_cfg.get("num_epochs", 0)),
            "epoch_mode": str(train_cfg.get("epoch_mode", "fixed_steps")),
            "steps_per_epoch": int(train_cfg.get("steps_per_epoch", 0)),
            "train_sampling_mode": str(train_cfg.get("train_sampling_mode", "random")),
            "batch_size": int(train_cfg.get("batch_size", 0)),
            "micro_batch_size": int(train_cfg.get("micro_batch_size", 0)),
            "lr": float(train_cfg.get("lr", 0.0)),
            "lr_schedule": str(train_cfg.get("lr_schedule", "")),
            "grad_clip_global_norm": float(train_cfg.get("grad_clip_global_norm", 0.0)),
        },
        "eval_snapshot": {
            "eval_every_epoch": int(eval_cfg.get("eval_every_epoch", 1)),
            "eval_pool_size": int(eval_cfg.get("eval_pool_size", 12)),
            "bn_recal_batches": int(eval_cfg.get("bn_recal_batches", 8)),
            "eval_batches_per_arch": int(eval_cfg.get("eval_batches_per_arch", 4)),
        },
        "checkpoint_snapshot": {
            "load_checkpoint": bool(checkpoint_cfg.get("load_checkpoint", False)),
            "resume_experiment_name": str(checkpoint_cfg.get("resume_experiment_name", "")),
        },
        "data_snapshot": {
            "dataset": str(data_cfg.get("dataset", "")),
            "base_path": str(data_cfg.get("base_path", "")),
            "train_dir": str(data_cfg.get("train_dir", "")),
            "val_dir": str(data_cfg.get("val_dir", "")),
        },
    }


def build_manifest(config: Dict[str, Any], git_commit: str) -> Dict[str, Any]:
    """Build legacy train_manifest payload."""
    train_cfg = config.get("train", {})
    runtime_cfg = config.get("runtime", {})
    data_cfg = config.get("data", {})
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "experiment_name": runtime_cfg.get("experiment_name", "unknown"),
        "seed": runtime_cfg.get("seed", 0),
        "git_commit": git_commit,
        "config_hash": _hash_config(config),
        "input_shape": [data_cfg.get("input_height", 0), data_cfg.get("input_width", 0)],
        "flow_channels": data_cfg.get("flow_channels", 2),
        "optimizer": train_cfg.get("optimizer", "adam"),
        "lr_schedule": train_cfg.get("lr_schedule", "cosine"),
        "uncertainty_type": train_cfg.get("uncertainty_type", "LinearSoftplus"),
        "wd": train_cfg.get("weight_decay", 0.0),
        "grad_clip": train_cfg.get("grad_clip_global_norm", 0.0),
    }
