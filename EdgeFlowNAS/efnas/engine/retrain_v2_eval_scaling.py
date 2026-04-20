"""Helpers for scaling retrain_v2 predictions during external evaluation."""

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _extract_processor_mean_epe(processor) -> float | None:
    mean_epe = getattr(processor, "MeanEPE", None)
    if mean_epe is not None:
        return float(mean_epe)
    error_epes = getattr(processor, "errorEPEs", None)
    if not error_epes:
        return None
    return float(np.concatenate(error_epes).mean())


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _safe_positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result <= 0.0:
        return float(default)
    return result


def _resolve_prediction_flow_scale(model_dir: Path, meta_data: Mapping[str, Any] | None) -> float:
    meta = dict(meta_data or {})

    if "flow_divisor" in meta:
        return _safe_positive_float(meta.get("flow_divisor"), 1.0)

    dataset = str(meta.get("dataset", "")).strip().upper()
    if dataset == "FT3D":
        return _safe_positive_float(meta.get("flow_divisor"), 12.5)
    if dataset:
        return 1.0

    experiment_dir = model_dir.parent
    run_manifest = _load_json_if_exists(experiment_dir / "run_manifest.json")
    if run_manifest:
        data_cfg = run_manifest.get("config", {}).get("data", {})
        if isinstance(data_cfg, dict):
            manifest_dataset = str(data_cfg.get("dataset", "")).strip().upper()
            if manifest_dataset == "FT3D":
                return _safe_positive_float(data_cfg.get("ft3d_flow_divisor"), 12.5)
            if manifest_dataset:
                return 1.0

    trainer_state = _load_json_if_exists(experiment_dir / "trainer_state.json")
    if trainer_state:
        state_dataset = str(trainer_state.get("dataset", "")).strip().upper()
        if state_dataset == "FT3D":
            return 12.5
        if state_dataset:
            return 1.0

    experiment_hint = str(experiment_dir).lower()
    if "ft3d" in experiment_hint or "things" in experiment_hint:
        return 12.5
    return 1.0


def _scale_prediction_for_sintel_eval(prediction: np.ndarray, flow_scale: float) -> np.ndarray:
    scaled = np.asarray(prediction, dtype=np.float32).copy()
    if float(flow_scale) != 1.0:
        scaled *= float(flow_scale)
    return scaled
