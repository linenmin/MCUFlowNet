import unittest

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


class TestAblationV1Model(unittest.TestCase):
    def setUp(self):
        if tf is None:
            self.skipTest("TensorFlow is required for graph construction tests")
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

    def test_all_variants_build_three_multiscale_outputs(self):
        from efnas.network.ablation_edgeflownet_v1 import ABlationEdgeFlowNetV1, build_ablation_variants

        variants = build_ablation_variants(None)
        self.assertEqual(
            [variant["name"] for variant in variants],
            [
                "edgeflownet_deconv",
                "edgeflownet_bilinear",
                "edgeflownet_bilinear_eca",
                "edgeflownet_bilinear_eca_gate4x",
            ],
        )

        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="is_training_ph")
        for variant in variants:
            with tf.compat.v1.variable_scope(variant["name"]):
                model = ABlationEdgeFlowNetV1(
                    input_ph=input_ph,
                    is_training_ph=is_training_ph,
                    num_out=4,
                    variant_config=variant,
                    init_neurons=32,
                    expansion_factor=2.0,
                )
                outputs = model.build()
            self.assertEqual(len(outputs), 3)
            self.assertEqual(outputs[-1].shape.as_list()[-1], 4)

    def test_variant_ops_are_distinguishable(self):
        from efnas.network.ablation_edgeflownet_v1 import ABlationEdgeFlowNetV1

        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="is_training_ph")
        variants = [
            {"name": "a0", "upsample_mode": "deconv", "bottleneck_eca": False, "gate_4x": False},
            {"name": "a3", "upsample_mode": "bilinear", "bottleneck_eca": True, "gate_4x": True},
        ]
        for variant in variants:
            with tf.compat.v1.variable_scope(variant["name"]):
                ABlationEdgeFlowNetV1(
                    input_ph=input_ph,
                    is_training_ph=is_training_ph,
                    num_out=4,
                    variant_config=variant,
                ).build()

        op_names = [op.name for op in tf.compat.v1.get_default_graph().get_operations()]
        self.assertTrue(any("a0" in name and "conv2d_transpose" in name for name in op_names))
        self.assertTrue(any("a3" in name and "resize" in name.lower() for name in op_names))
        self.assertTrue(any("a3" in name and "eca_bottleneck" in name for name in op_names))
        self.assertTrue(any("a3" in name and "global_gate_4x" in name for name in op_names))


if __name__ == "__main__":
    unittest.main()
