"""TensorFlow graph tests for Supernet V3 network."""

import unittest

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


class TestSupernetNetworkStructureV3(unittest.TestCase):
    """Validate V3 graph structure."""

    def setUp(self):
        if tf is None:
            self.skipTest("TensorFlow is required for graph construction tests")
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

    def test_v3_graph_contains_fixed_eca_and_global_gate(self):
        """The V3 supernet backbone should hard-wire bottleneck ECA and 1/4 gate."""
        from efnas.network.MultiScaleResNet_supernet_v3 import MultiScaleResNetSupernetV3

        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 6], name="input_ph")
        arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[11], name="arch_code_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="is_training_ph")
        model = MultiScaleResNetSupernetV3(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=4,
        )
        outputs = model.build()
        self.assertEqual(len(outputs), 3)
        op_names = [op.name for op in tf.compat.v1.get_default_graph().get_operations()]
        self.assertTrue(any("eca_bottleneck" in name for name in op_names))
        self.assertTrue(any("global_gate_4x" in name for name in op_names))

    def test_v3_arch_parallel_graph_exposes_three_arch_codes(self):
        """Arch-parallel mode should build a 3-subnet training graph."""
        from efnas.engine.supernet_trainer_v3 import _build_graph

        graph = _build_graph(
            {
                "train": {
                    "multi_gpu_mode": "arch_parallel",
                    "gpu_devices": "0,1,2",
                    "distill": {"enabled": False},
                    "grad_clip_global_norm": 200.0,
                },
                "data": {"input_height": 64, "input_width": 64, "flow_channels": 2},
            }
        )
        self.assertEqual(graph["multi_gpu_mode"], "arch_parallel")
        self.assertEqual(graph["arch_codes_ph"].shape.as_list(), [3, 11])


if __name__ == "__main__":
    unittest.main()
