"""Evaluator for retrain_v2 fixed-architecture checkpoints."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf

from efnas.engine.eval_step import accumulate_predictions
from efnas.network.fixed_arch_models_v2 import FixedArchModelV2


def _ensure_graph_mode() -> None:
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()


def _load_checkpoint_meta(model_dir: Path, ckpt_name: str = "best") -> Dict[str, Any]:
    meta_path = model_dir / "checkpoints" / f"{ckpt_name}.ckpt.meta.json"
    if not meta_path.exists() and ckpt_name == "best":
        meta_path = model_dir / "checkpoints" / "last.ckpt.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint meta found for {model_dir} ({ckpt_name})")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_retrain_v2_eval_model(
    checkpoint_dir: Path,
    patch_size: Tuple[int, int],
    ckpt_name: str = "best",
):
    _ensure_graph_mode()
    meta_data = _load_checkpoint_meta(checkpoint_dir, ckpt_name)
    arch_code = meta_data["arch_code"]
    arch_list = [int(x) for x in arch_code] if not isinstance(arch_code, str) else [int(x) for x in arch_code.split(",") if str(x).strip()]
    scope_name = checkpoint_dir.name[len("model_") :] if checkpoint_dir.name.startswith("model_") else checkpoint_dir.name

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, patch_size[0], patch_size[1], 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")

        with tf.compat.v1.variable_scope(scope_name):
            model = FixedArchModelV2(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                arch_code=arch_list,
                num_out=4,
                init_neurons=32,
                expansion_factor=2.0,
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
    meta["checkpoint_dir"] = str(checkpoint_dir)
    meta["checkpoint_path"] = str(ckpt_path)
    return sess, input_ph, preds_accumulated, meta


def preprocess_eval_batch(input_batch):
    return (input_batch / 255.0) * 2.0 - 1.0
