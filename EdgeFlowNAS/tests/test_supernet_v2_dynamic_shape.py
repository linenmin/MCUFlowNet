"""Regression tests for dynamic-shape Supernet V2 graph construction."""

import unittest

try:
    import tensorflow as tf
    from efnas.nas.search_space_v2 import get_num_blocks
    from efnas.network.MultiScaleResNet_supernet_v2 import MultiScaleResNetSupernetV2

    _TF_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - runtime env dependent
    tf = None
    get_num_blocks = None
    MultiScaleResNetSupernetV2 = None
    _TF_IMPORT_ERROR = str(exc)


@unittest.skipIf(tf is None, f"tensorflow unavailable: {_TF_IMPORT_ERROR}")
class TestSupernetV2DynamicShape(unittest.TestCase):
    """Ensure the V2 graph can be built with dynamic spatial dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Disable eager execution once for TF1-style graph tests."""
        tf.compat.v1.disable_eager_execution()

    def setUp(self) -> None:
        """Reset the default graph between tests."""
        tf.compat.v1.reset_default_graph()

    def test_build_accepts_dynamic_spatial_input_shape(self) -> None:
        """Dynamic H/W placeholders should not fail during resize-conv graph construction."""
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 6], name="Input")
        arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[get_num_blocks()], name="ArchCode")
        is_training_ph = tf.compat.v1.placeholder_with_default(
            tf.constant(False, dtype=tf.bool),
            shape=[],
            name="IsTraining",
        )
        model = MultiScaleResNetSupernetV2(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=4,
            init_neurons=32,
            expansion_factor=2.0,
        )

        preds = model.build()

        self.assertEqual(len(preds), 3)


if __name__ == "__main__":
    unittest.main()
