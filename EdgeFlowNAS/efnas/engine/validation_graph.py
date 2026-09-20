"""Inference at a fixed validation size, sharing all existing model variables."""
import tensorflow as tf
from efnas.network.fixed_arch_models import FixedArchModelV3
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric


def build_validation_graph(scope_name, arch_code, height, width, flow_channels=2):
    before = list(tf.compat.v1.global_variables())
    images = tf.compat.v1.placeholder(tf.float32, [None, height, width, 6], name='ValidationInput')
    labels = tf.compat.v1.placeholder(tf.float32, [None, height, width, flow_channels], name='ValidationLabel')
    with tf.compat.v1.variable_scope(scope_name, reuse=True):
        preds = FixedArchModelV3(images, False, arch_code, num_out=flow_channels*2).build()
        epe = build_epe_metric(accumulate_predictions(preds), labels, num_out=flow_channels)
    if [v.name for v in before] != [v.name for v in tf.compat.v1.global_variables()]:
        raise RuntimeError('Validation unexpectedly created model or optimizer variables')
    return {'epe': epe}, images, labels
