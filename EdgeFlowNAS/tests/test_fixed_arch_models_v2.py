"""Unit tests for V2 fixed-architecture retrain model helpers."""

import unittest

import tensorflow as tf

from efnas.engine.retrain_v2_trainer import _build_single_model_graph
from efnas.network.fixed_arch_models_v2 import FixedArchModelV2
from efnas.network.MultiScaleResNet_supernet_v2 import MultiScaleResNetSupernetV2


class TestFixedArchModelV2(unittest.TestCase):
    """Validate V2 fixed model basic contract."""

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

    def test_rejects_non_11d_arch_code(self) -> None:
        """V2 retrain model must require 11 architecture choices."""
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 32, 32, 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training_ph")
        with self.assertRaises(ValueError):
            FixedArchModelV2(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                arch_code=[0] * 9,
                num_out=4,
            )

    def test_builds_three_prediction_scales_for_valid_arch(self) -> None:
        """Selected fixed V2 subnet should still emit 1/4, 1/2 and 1/1 heads."""
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training_ph")
        model = FixedArchModelV2(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=[2, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1],
            num_out=4,
        )
        preds = model.build()
        self.assertEqual(len(preds), 3)

    def test_retrain_graph_variable_names_can_map_back_to_supernet(self) -> None:
        """Warm-start must be able to restore retrain vars from supernet checkpoint names."""
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training_ph")
        arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[11], name="arch_code_ph")

        with tf.compat.v1.variable_scope("candidate"):
            fixed_arch_code_ph = tf.constant([2, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1], dtype=tf.int32, name="FixedArchCode")
            retrain_model = MultiScaleResNetSupernetV2(
                input_ph=input_ph,
                arch_code_ph=fixed_arch_code_ph,
                is_training_ph=is_training_ph,
                num_out=4,
            )
            retrain_model.build()

        supernet = MultiScaleResNetSupernetV2(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=4,
        )
        supernet.build()

        fixed_names = {
            var.op.name[len("candidate/") :]
            for var in tf.compat.v1.global_variables()
            if var.op.name.startswith("candidate/")
        }
        supernet_names = {var.op.name for var in tf.compat.v1.global_variables() if not var.op.name.startswith("candidate/")}
        self.assertTrue(fixed_names)
        self.assertTrue(fixed_names.issubset(supernet_names))

    def test_warmstart_mapping_excludes_optimizer_slot_variables(self) -> None:
        """Warm-start restore should never require Adam slot tensors from source checkpoint."""
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 6], name="input_ph")
        label_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 2], name="label_ph")
        lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="lr_ph")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training_ph")

        graph = _build_single_model_graph(
            scope_name="candidate",
            arch_code=[2, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1],
            input_ph=input_ph,
            label_ph=label_ph,
            lr_ph=lr_ph,
            is_training_ph=is_training_ph,
            flow_channels=2,
            pred_channels=4,
            weight_decay=0.0,
            grad_clip_global_norm=0.0,
        )

        self.assertTrue(any("/Adam" in var.op.name for var in graph["scope_global_vars"]))
        self.assertTrue(graph["warmstart_keys"])
        self.assertFalse(any("/Adam" in key for key in graph["warmstart_keys"]))


if __name__ == "__main__":
    unittest.main()
