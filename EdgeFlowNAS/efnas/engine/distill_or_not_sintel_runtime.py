"""Sintel evaluation runtime for fixed V3 scratch-retrain checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from efnas.engine.eval_step import accumulate_predictions
from efnas.engine.retrain_v2_eval_scaling import (
    _extract_processor_mean_epe,
    _resolve_prediction_flow_scale,
    _scale_prediction_for_sintel_eval,
)
from efnas.engine.retrain_v2_sintel_runtime import (
    _build_processor_args,
    _prepare_sintel_lists,
    preprocess_eval_batch,
)
from efnas.network.fixed_arch_models_v3 import FixedArchModelV3


def _load_checkpoint_meta(model_dir: Path, ckpt_name: str) -> Dict[str, Any]:
    meta_path = model_dir / "checkpoints" / f"{ckpt_name}.ckpt.meta.json"
    if not meta_path.exists() and ckpt_name == "best":
        meta_path = model_dir / "checkpoints" / "last.ckpt.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint meta found for {model_dir} ({ckpt_name})")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def setup_fixed_v3_eval_model(checkpoint_dir: Path, patch_size: Tuple[int, int], ckpt_name: str = "best"):
    """Build and restore a fixed V3 checkpoint for one-image Sintel inference."""
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    meta_data = _load_checkpoint_meta(checkpoint_dir, ckpt_name)
    arch_code = meta_data["arch_code"]
    arch_list = [int(x) for x in arch_code] if not isinstance(arch_code, str) else [int(x) for x in arch_code.split(",") if str(x).strip()]
    scope_name = checkpoint_dir.name[len("model_") :] if checkpoint_dir.name.startswith("model_") else checkpoint_dir.name
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, int(patch_size[0]), int(patch_size[1]), 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")
        with tf.compat.v1.variable_scope(scope_name):
            model = FixedArchModelV3(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                arch_code=arch_list,
                num_out=4,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = model.build()
        pred_tensor = accumulate_predictions(preds)
        scope_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_vars)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=graph, config=config)
    ckpt_path = meta_data.get("checkpoint_path") or str(checkpoint_dir / "checkpoints" / f"{ckpt_name}.ckpt")
    with graph.as_default():
        saver.restore(sess, ckpt_path)
    meta = dict(meta_data)
    meta["scope_name"] = scope_name
    meta["checkpoint_dir"] = str(checkpoint_dir)
    meta["checkpoint_path"] = str(ckpt_path)
    return sess, input_ph, pred_tensor, meta


def evaluate_v3_checkpoint_dir_on_sintel(
    model_dir: Path,
    dataset_root: str,
    sintel_list: str,
    patch_size: Tuple[int, int],
    ckpt_name: str = "best",
    max_samples: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one fixed V3 checkpoint on the configured Sintel split."""
    from EdgeFlowNet.code.misc.processor import FlowPostProcessor
    from EdgeFlowNet.code.misc.utils import get_sintel_batch
    from tqdm import tqdm

    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Sintel dataset_root does not exist: {dataset_root_path}")
    sess, input_ph, pred_tensor, meta_data = setup_fixed_v3_eval_model(
        checkpoint_dir=Path(model_dir),
        patch_size=tuple(patch_size),
        ckpt_name=str(ckpt_name),
    )
    flow_scale = _resolve_prediction_flow_scale(Path(model_dir), meta_data)
    img1_list, img2_list, flo_list = _prepare_sintel_lists(dataset_root=dataset_root_path, sintel_list_text=sintel_list)
    total_samples = len(img1_list)
    if max_samples is not None:
        total_samples = min(total_samples, int(max_samples))
    processor = FlowPostProcessor("full", is_multiscale=True)
    args = _build_processor_args()
    iterator = tqdm(range(total_samples), desc=progress_desc, leave=False, unit="sample") if progress_desc else range(total_samples)
    try:
        for idx in iterator:
            input_comb, gt_flow = get_sintel_batch(img1_list[idx], img2_list[idx], flo_list[idx], list(patch_size))
            if input_comb is None or gt_flow is None:
                continue
            input_batch = preprocess_eval_batch(input_comb[None, ...])
            preds = sess.run(pred_tensor, feed_dict={input_ph: input_batch})
            flow_prediction = _scale_prediction_for_sintel_eval(preds[:, :, :, :2], flow_scale)
            processor.update(label=gt_flow, prediction=flow_prediction, Args=args)
    finally:
        sess.close()
    mean_epe = _extract_processor_mean_epe(processor)
    return {
        "model_name": meta_data["scope_name"],
        "arch_code": ",".join(str(v) for v in meta_data["arch_code"]),
        "checkpoint_path": meta_data["checkpoint_path"],
        "fc2_or_stage_metric": meta_data.get("metric", ""),
        "sintel_epe": float(mean_epe) if mean_epe is not None else float("inf"),
    }
