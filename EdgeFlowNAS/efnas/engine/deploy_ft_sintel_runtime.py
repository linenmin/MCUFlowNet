"""Sintel evaluator dispatch for deploy-resolution fine-tune candidates.

Both V3 and mainline paths use **explicit `flow_scale`** rather than the
broken `_resolve_prediction_flow_scale` fallback chain (which only
returns 12.5 if the experiment directory name happens to contain
"ft3d" — our `retrain_v3_deploy_ft_run1` doesn't, so the original V3
eval path silently used flow_scale=1.0 and produced meaningless
Sintel EPE numbers around 10.5). We pass `flow_scale=data.ft3d_flow_divisor`
(12.5 for V3, 1.0 for mainline) directly from `deploy_ft_trainer`.

Both paths produce the same `{"sintel_epe": <float>, ...}` shape so the
trainer can consume them uniformly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.engine.distill_or_not_sintel_runtime import (
    setup_fixed_v3_eval_model,
)
from efnas.engine.eval_step import accumulate_predictions
from efnas.engine.retrain_v2_eval_scaling import (
    _extract_processor_mean_epe,
    _scale_prediction_for_sintel_eval,
)
from efnas.engine.retrain_v2_sintel_runtime import (
    _build_processor_args,
    _prepare_sintel_lists,
    _resolve_sintel_eval_config,
    _strip_sintel_prefix,
    preprocess_eval_batch,
)
from efnas.network.edgeflownet_mainline import (
    EdgeFlowNetMainline,
    MAINLINE_DEFAULT_CONFIG,
)


# ----------------------------------------------------------------------------
# Mainline-specific eval graph + runner
# ----------------------------------------------------------------------------
def setup_mainline_eval_model(
    ckpt_path: str,
    patch_size: Tuple[int, int],
):
    """Build mainline forward graph at `patch_size` and restore weights.

    Mainline ckpt vars live at root scope; we restore everything that's not
    the optimizer scope. BN runs in inference mode (matches training)."""
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tf.compat.v1.placeholder(
            tf.float32,
            shape=[1, int(patch_size[0]), int(patch_size[1]), 6],
            name="input_ph",
        )
        model = EdgeFlowNetMainline(InputPH=input_ph, **MAINLINE_DEFAULT_CONFIG)
        preds = model.build()
        pred_tensor = accumulate_predictions(preds)
        fwd_vars = [
            v for v in tf.compat.v1.global_variables()
            if not v.name.startswith("Optimizer/")
        ]
        saver = tf.compat.v1.train.Saver(var_list=fwd_vars)
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=graph, config=cfg)
    with graph.as_default():
        saver.restore(sess, ckpt_path)
    return sess, input_ph, pred_tensor


def evaluate_mainline_checkpoint_on_sintel(
    model_dir: Path,
    dataset_root: str,
    sintel_list: str,
    patch_size: Tuple[int, int],
    flow_scale: float = 1.0,
    ckpt_name: str = "best",
    max_samples: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one mainline checkpoint on Sintel via the same EPE pipeline
    used for V3 (FlowPostProcessor + optional flow_scale).

    Mainline original training did not divide GT, so flow_scale defaults to
    1.0. The trainer still passes it explicitly so future configs (e.g.
    mainline retrained with FT3D's flow_divisor=12.5) work cleanly."""
    from EdgeFlowNet.code.misc.processor import FlowPostProcessor  # noqa: WPS433
    from EdgeFlowNet.code.misc.utils import get_sintel_batch  # noqa: WPS433

    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Sintel dataset_root missing: {dataset_root_path}")
    ckpt_path = str(Path(model_dir) / "checkpoints" / f"{ckpt_name}.ckpt")
    sess, input_ph, pred_tensor = setup_mainline_eval_model(
        ckpt_path=ckpt_path, patch_size=tuple(patch_size)
    )
    img1_list, img2_list, flo_list = _prepare_sintel_lists(
        dataset_root=dataset_root_path, sintel_list_text=sintel_list
    )
    total = len(img1_list)
    if max_samples is not None:
        total = min(total, int(max_samples))
    processor = FlowPostProcessor("full", is_multiscale=True)
    args = _build_processor_args()
    iterator = (
        tqdm(range(total), desc=progress_desc, leave=False, unit="sample")
        if progress_desc
        else range(total)
    )
    try:
        for idx in iterator:
            input_comb, gt_flow = get_sintel_batch(
                img1_list[idx], img2_list[idx], flo_list[idx], list(patch_size)
            )
            if input_comb is None or gt_flow is None:
                continue
            input_batch = preprocess_eval_batch(input_comb[None, ...])
            preds = sess.run(pred_tensor, feed_dict={input_ph: input_batch})
            flow_prediction = _scale_prediction_for_sintel_eval(
                preds[:, :, :, :2], float(flow_scale)
            )
            processor.update(label=gt_flow, prediction=flow_prediction, Args=args)
    finally:
        sess.close()

    mean_epe = _extract_processor_mean_epe(processor) or float("inf")
    return {
        "model_name": Path(model_dir).name,
        "arch_family": "edgeflownet_mainline",
        "checkpoint_path": ckpt_path,
        "flow_scale": float(flow_scale),
        "sintel_epe": float(mean_epe),
        "patch_size": list(patch_size),
        "num_samples": total,
    }


def evaluate_v3_checkpoint_on_sintel_with_flow_scale(
    model_dir: Path,
    dataset_root: str,
    sintel_list: str,
    patch_size: Tuple[int, int],
    flow_scale: float,
    ckpt_name: str = "best",
    max_samples: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Like `evaluate_v3_checkpoint_dir_on_sintel` but with an EXPLICIT
    flow_scale instead of the broken `_resolve_prediction_flow_scale`
    fallback chain. Pass `flow_scale=12.5` for FT3D-trained V3 candidates."""
    from EdgeFlowNet.code.misc.processor import FlowPostProcessor  # noqa: WPS433
    from EdgeFlowNet.code.misc.utils import get_sintel_batch  # noqa: WPS433

    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Sintel dataset_root missing: {dataset_root_path}")
    sess, input_ph, pred_tensor, meta_data = setup_fixed_v3_eval_model(
        checkpoint_dir=Path(model_dir),
        patch_size=tuple(patch_size),
        ckpt_name=str(ckpt_name),
    )
    img1_list, img2_list, flo_list = _prepare_sintel_lists(
        dataset_root=dataset_root_path, sintel_list_text=sintel_list
    )
    total = len(img1_list)
    if max_samples is not None:
        total = min(total, int(max_samples))
    processor = FlowPostProcessor("full", is_multiscale=True)
    args = _build_processor_args()
    iterator = (
        tqdm(range(total), desc=progress_desc, leave=False, unit="sample")
        if progress_desc
        else range(total)
    )
    try:
        for idx in iterator:
            input_comb, gt_flow = get_sintel_batch(
                img1_list[idx], img2_list[idx], flo_list[idx], list(patch_size)
            )
            if input_comb is None or gt_flow is None:
                continue
            input_batch = preprocess_eval_batch(input_comb[None, ...])
            preds = sess.run(pred_tensor, feed_dict={input_ph: input_batch})
            flow_prediction = _scale_prediction_for_sintel_eval(
                preds[:, :, :, :2], float(flow_scale)
            )
            processor.update(label=gt_flow, prediction=flow_prediction, Args=args)
    finally:
        sess.close()

    mean_epe = _extract_processor_mean_epe(processor) or float("inf")
    return {
        "model_name": Path(model_dir).name,
        "arch_family": "fixed_v3",
        "arch_code": meta_data.get("arch_code"),
        "checkpoint_path": meta_data.get("checkpoint_path"),
        "flow_scale": float(flow_scale),
        "sintel_epe": float(mean_epe),
        "patch_size": list(patch_size),
        "num_samples": total,
    }


# ----------------------------------------------------------------------------
# Family-aware dispatcher used by deploy_ft_trainer
# ----------------------------------------------------------------------------
def run_sintel_for_deploy_ft(
    model_dir: Path,
    config: Dict[str, Any],
    epoch_idx: int,
    ckpt_name: str,
    arch_family: str,
    flow_scale: float,
) -> Optional[Dict[str, Any]]:
    """Dispatch Sintel eval to the right per-family runner with an EXPLICIT
    `flow_scale`. The trainer passes `flow_scale=data.ft3d_flow_divisor`
    (12.5 for V3, 1.0 for mainline) so we don't rely on the broken
    path-name hint in `_resolve_prediction_flow_scale`."""
    sintel_cfg = _resolve_sintel_eval_config(config)
    if sintel_cfg is None:
        return None
    progress = f"sintel/{model_dir.name}/{ckpt_name}/e{epoch_idx}"

    if arch_family == "fixed_v3":
        return evaluate_v3_checkpoint_on_sintel_with_flow_scale(
            model_dir=model_dir,
            dataset_root=str(sintel_cfg["dataset_root"]),
            sintel_list=str(sintel_cfg["sintel_list"]),
            patch_size=tuple(sintel_cfg["patch_size"]),
            flow_scale=float(flow_scale),
            ckpt_name=str(ckpt_name),
            max_samples=sintel_cfg.get("max_samples", None),
            progress_desc=progress,
        )
    if arch_family == "edgeflownet_mainline":
        return evaluate_mainline_checkpoint_on_sintel(
            model_dir=model_dir,
            dataset_root=str(sintel_cfg["dataset_root"]),
            sintel_list=str(sintel_cfg["sintel_list"]),
            patch_size=tuple(sintel_cfg["patch_size"]),
            flow_scale=float(flow_scale),
            ckpt_name=str(ckpt_name),
            max_samples=sintel_cfg.get("max_samples", None),
            progress_desc=progress,
        )
    raise ValueError(f"unsupported arch_family for sintel eval: {arch_family}")
