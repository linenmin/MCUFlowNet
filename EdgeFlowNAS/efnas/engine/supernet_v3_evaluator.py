"""Utilities for evaluating inherited-weight V3 subnets from a supernet checkpoint."""

from pathlib import Path
from typing import Any, Dict


def setup_supernet_v3_eval_model(
    experiment_dir: Path,
    checkpoint_type: str = "best",
    flow_channels: int = 2,
    allow_growth: bool = True,
) -> Dict[str, Any]:
    """Build a dynamic-shape eval graph and restore one V3 supernet checkpoint."""
    import tensorflow as tf

    from efnas.engine.checkpoint_manager import build_checkpoint_paths
    from efnas.engine.eval_step import accumulate_predictions, build_epe_metric
    from efnas.nas.search_space_v3 import get_num_blocks
    from efnas.network.MultiScaleResNet_supernet_v3 import MultiScaleResNetSupernetV3

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    checkpoint_paths = build_checkpoint_paths(str(experiment_dir))
    key = "best" if str(checkpoint_type).strip().lower() == "best" else "last"
    checkpoint_prefix = checkpoint_paths[key]
    if not Path(str(checkpoint_prefix) + ".index").exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_prefix}")

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, int(flow_channels)], name="Label")
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[get_num_blocks()], name="ArchCode")
    is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="IsTraining")

    with tf.compat.v1.variable_scope("shared_supernet"):
        model = MultiScaleResNetSupernetV3(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=int(flow_channels) * 2,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
    preds_accum = accumulate_predictions(preds)
    epe_tensor = build_epe_metric(pred_tensor=preds_accum, label_ph=label_ph, num_out=int(flow_channels))

    update_ops = list(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=1)

    config = tf.compat.v1.ConfigProto()
    if allow_growth:
        config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, str(checkpoint_prefix))

    return {
        "sess": sess,
        "saver": saver,
        "checkpoint_path": str(checkpoint_prefix),
        "checkpoint_type": key,
        "input_ph": input_ph,
        "label_ph": label_ph,
        "arch_code_ph": arch_code_ph,
        "is_training_ph": is_training_ph,
        "preds_tensor": preds_accum,
        "epe_tensor": epe_tensor,
        "update_ops": update_ops,
        "flow_channels": int(flow_channels),
    }
