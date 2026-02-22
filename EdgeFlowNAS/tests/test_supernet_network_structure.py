"""Unit tests for supernet network structure invariants."""

import unittest

import numpy as np

try:
    import tensorflow as tf
    from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet
    _TF_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - runtime env dependent
    tf = None
    MultiScaleResNetSupernet = None
    _TF_IMPORT_ERROR = str(exc)


@unittest.skipIf(tf is None, f"tensorflow unavailable: {_TF_IMPORT_ERROR}")
class TestSupernetNetworkStructure(unittest.TestCase):
    """Validate decoupled deep-choice block behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Disable eager mode once for TF1 graph tests."""
        tf.compat.v1.disable_eager_execution()

    def setUp(self) -> None:
        """Reset graph between test cases."""
        tf.compat.v1.reset_default_graph()

    def _build_model_graph(self, batch_size: int = 2):
        """Build one minimal graph for structure/forward checks."""
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 180, 240, 6], name="Input")
        arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[9], name="ArchCode")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")
        model = MultiScaleResNetSupernet(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=4,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
        return input_ph, arch_code_ph, is_training_ph, preds

    def test_deep_block_uses_decoupled_branch_variables(self) -> None:
        """Deep-choice block should expose independent branch variable scopes."""
        self._build_model_graph(batch_size=1)
        var_names = [var.name for var in tf.compat.v1.trainable_variables()]
        self.assertTrue(any("EB0/branch1_block1" in name for name in var_names))
        self.assertTrue(any("EB0/branch2_block2" in name for name in var_names))
        self.assertTrue(any("EB0/branch3_block3" in name for name in var_names))
        self.assertFalse(any("EB0/deep1" in name for name in var_names))
        self.assertFalse(any("EB0/deep2" in name for name in var_names))
        self.assertFalse(any("EB0/deep3" in name for name in var_names))

    def test_forward_outputs_keep_batch_and_channel_shape(self) -> None:
        """Forward pass should keep batch size and output channels unchanged."""
        input_ph, arch_code_ph, is_training_ph, preds = self._build_model_graph(batch_size=2)
        feed_input = np.random.randn(2, 180, 240, 6).astype(np.float32)
        arch_code = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_vals = sess.run(
                preds,
                feed_dict={
                    input_ph: feed_input,
                    arch_code_ph: arch_code,
                    is_training_ph: False,
                },
            )
        self.assertEqual(len(output_vals), 3)
        for output in output_vals:
            self.assertEqual(int(output.shape[0]), 2)
            self.assertEqual(int(output.shape[-1]), 4)
            self.assertGreater(int(output.shape[1]), 0)
            self.assertGreater(int(output.shape[2]), 0)


if __name__ == "__main__":
    unittest.main()
