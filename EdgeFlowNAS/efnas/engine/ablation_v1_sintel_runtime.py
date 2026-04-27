"""Sintel runtime helpers for Ablation V1 checkpoints."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from efnas.engine.eval_step import accumulate_predictions
from efnas.engine.retrain_v2_sintel_runtime import evaluate_checkpoint_dir_on_sintel as _eval_retrain_sintel
from efnas.network.ablation_edgeflownet_v1 import ABlationEdgeFlowNetV1
from efnas.utils.json_io import read_json


def _load_checkpoint_meta(model_dir: Path, ckpt_name: str) -> Dict[str, Any]:
    meta_path = model_dir / "checkpoints" / f"{ckpt_name}.ckpt.meta.json"
    if not meta_path.exists() and ckpt_name == "best":
        meta_path = model_dir / "checkpoints" / "last.ckpt.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint meta found for {model_dir} ({ckpt_name})")
    payload = read_json(str(meta_path))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint meta is not an object: {meta_path}")
    return payload


def _load_variant_config(model_dir: Path, meta_data: Dict[str, Any]) -> Dict[str, Any]:
    variant = meta_data.get("variant_config", None)
    if isinstance(variant, dict):
        return variant
    variant_path = model_dir / "variant_config.json"
    if variant_path.exists():
        payload = read_json(str(variant_path))
        if isinstance(payload, dict):
            return payload
    name = model_dir.name[len("model_") :] if model_dir.name.startswith("model_") else model_dir.name
    return {"name": name, "upsample_mode": "bilinear", "bottleneck_eca": False, "gate_4x": False}


def setup_ablation_v1_eval_model(checkpoint_dir: Path, patch_size: Tuple[int, int], ckpt_name: str = "best"):
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    meta_data = _load_checkpoint_meta(checkpoint_dir, ckpt_name)
    variant_config = _load_variant_config(checkpoint_dir, meta_data)
    scope_name = checkpoint_dir.name[len("model_") :] if checkpoint_dir.name.startswith("model_") else checkpoint_dir.name

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, patch_size[0], patch_size[1], 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")
        with tf.compat.v1.variable_scope(scope_name):
            model = ABlationEdgeFlowNetV1(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                num_out=4,
                variant_config=variant_config,
                init_neurons=int(meta_data.get("init_neurons", 32)),
                expansion_factor=float(meta_data.get("expansion_factor", 2.0)),
            )
            preds = model.build()
        preds_accumulated = accumulate_predictions(preds)
        scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_global_vars)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=eval_graph, config=config)
    ckpt_path = meta_data.get("checkpoint_path") or str(checkpoint_dir / "checkpoints" / f"{ckpt_name}.ckpt")
    with eval_graph.as_default():
        saver.restore(sess, ckpt_path)

    meta = dict(meta_data)
    meta["scope_name"] = scope_name
    meta["variant_config"] = variant_config
    meta["checkpoint_dir"] = str(checkpoint_dir)
    meta["checkpoint_path"] = str(ckpt_path)
    meta.setdefault("arch_code", [])
    return sess, input_ph, preds_accumulated, meta


def evaluate_ablation_checkpoint_dir_on_sintel(
    model_dir: Path,
    dataset_root: str,
    sintel_list: str,
    patch_size: Tuple[int, int],
    ckpt_name: str = "best",
    max_samples: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    import efnas.engine.retrain_v2_evaluator as retrain_eval

    original_setup = retrain_eval.setup_retrain_v2_eval_model
    retrain_eval.setup_retrain_v2_eval_model = setup_ablation_v1_eval_model
    try:
        return _eval_retrain_sintel(
            model_dir=model_dir,
            dataset_root=dataset_root,
            sintel_list=sintel_list,
            patch_size=patch_size,
            ckpt_name=ckpt_name,
            max_samples=max_samples,
            progress_desc=progress_desc,
        )
    finally:
        retrain_eval.setup_retrain_v2_eval_model = original_setup
