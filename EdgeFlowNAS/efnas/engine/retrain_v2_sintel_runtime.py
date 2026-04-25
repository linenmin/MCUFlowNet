"""Runtime helpers for in-training Sintel evaluation during retrain_v2."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from efnas.engine.retrain_v2_eval_scaling import _extract_processor_mean_epe, _scale_prediction_for_sintel_eval
from efnas.utils.import_bootstrap import bootstrap_project_paths, resolve_project_paths

bootstrap_project_paths(anchor_file=__file__)
project_root = resolve_project_paths(anchor_file=__file__)["mcu_root"]


def _strip_sintel_prefix(path_str: str) -> str:
    prefix = "Datasets/Sintel/"
    return path_str[len(prefix) :] if path_str.startswith(prefix) else path_str


def _parse_patch_size(raw_value: Any) -> Tuple[int, int]:
    if isinstance(raw_value, (list, tuple)):
        values = [int(item) for item in raw_value]
    else:
        values = [int(item.strip()) for item in str(raw_value).split(",") if str(item).strip()]
    if len(values) != 2:
        raise ValueError("sintel patch_size must be H,W")
    return int(values[0]), int(values[1])


def _resolve_sintel_eval_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eval_cfg = config.get("eval", {})
    sintel_cfg = eval_cfg.get("sintel", {})
    if not isinstance(sintel_cfg, dict):
        return None
    dataset_root = str(sintel_cfg.get("dataset_root", "")).strip()
    if not dataset_root:
        return None
    return {
        "dataset_root": dataset_root,
        "sintel_list": str(
            sintel_cfg.get("sintel_list", "EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")
        ).strip(),
        "patch_size": _parse_patch_size(sintel_cfg.get("patch_size", "416,1024")),
        "max_samples": sintel_cfg.get("max_samples", None),
        "gpu_device": int(config.get("train", {}).get("gpu_device", 0)),
        "ckpt_name": str(sintel_cfg.get("ckpt_name", "sintel_best")).strip(),
    }


def _append_sintel_metrics(row: Dict[str, Any], sintel_epes: Dict[str, float], model_names: Sequence[str]) -> Dict[str, Any]:
    updated = dict(row)
    values: List[float] = []
    for name in model_names:
        metric = float(sintel_epes[name])
        updated[f"sintel_epe_{name}"] = metric
        values.append(metric)
    updated["mean_sintel_epe"] = float(np.mean(values)) if values else float("inf")
    return updated


def _prepare_sintel_lists(dataset_root: Path, sintel_list_text: str) -> Tuple[List[str], List[str], List[str]]:
    from EdgeFlowNet.code.misc.utils import read_sintel_list

    list_path = Path(sintel_list_text)
    if not list_path.is_absolute():
        list_path = (project_root / list_path).resolve()
    if not list_path.exists():
        raise FileNotFoundError(f"Sintel list not found: {list_path}")

    list_args = Namespace(data_list=str(list_path))
    rel_img1_list, rel_img2_list, rel_flo_list = read_sintel_list(list_args)
    img1_list = [str(dataset_root / _strip_sintel_prefix(item)) for item in rel_img1_list]
    img2_list = [str(dataset_root / _strip_sintel_prefix(item)) for item in rel_img2_list]
    flo_list = [str(dataset_root / _strip_sintel_prefix(item)) for item in rel_flo_list]
    return img1_list, img2_list, flo_list


def preprocess_eval_batch(input_batch: np.ndarray) -> np.ndarray:
    return (input_batch / 255.0) * 2.0 - 1.0


def evaluate_model_graph_on_sintel(
    sess,
    input_ph,
    is_training_ph,
    pred_tensor,
    flow_scale: float,
    dataset_root: str,
    sintel_list: str,
    patch_size: Tuple[int, int],
    max_samples: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> float:
    from EdgeFlowNet.code.misc.processor import FlowPostProcessor
    from EdgeFlowNet.code.misc.utils import get_sintel_batch

    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Sintel dataset_root does not exist: {dataset_root_path}")

    img1_list, img2_list, flo_list = _prepare_sintel_lists(dataset_root=dataset_root_path, sintel_list_text=sintel_list)
    total_samples = len(img1_list)
    if max_samples is not None:
        total_samples = min(total_samples, int(max_samples))

    processor = FlowPostProcessor("full", is_multiscale=True)
    args = Namespace(
        Display=False,
        ShiftedFlow=False,
        ResizeToHalf=False,
        ResizeCropStack=False,
        ResizeNearestCropStack=False,
        NumberOfHalves=0,
        ResizeCropStackBlur=False,
        OverlapCropStack=False,
        PatchDelta=0,
        uncertainity=False,
    )
    iterator = range(total_samples)
    if progress_desc:
        iterator = tqdm(iterator, desc=progress_desc, leave=False, unit="sample")

    for idx in iterator:
        input_comb, gt_flow = get_sintel_batch(img1_list[idx], img2_list[idx], flo_list[idx], list(patch_size))
        if input_comb is None or gt_flow is None:
            continue
        input_batch = preprocess_eval_batch(np.expand_dims(input_comb, axis=0))
        preds_results = sess.run(pred_tensor, feed_dict={input_ph: input_batch, is_training_ph: False})
        flow_prediction = _scale_prediction_for_sintel_eval(preds_results[:, :, :, :2], flow_scale)
        processor.update(label=gt_flow, prediction=flow_prediction, Args=args)

    mean_epe = _extract_processor_mean_epe(processor)
    if mean_epe is None:
        return float("inf")
    return float(mean_epe)
