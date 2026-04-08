"""Regression tests for dynamic-shape eval-step utilities."""

import unittest

import numpy as np

try:
    import tensorflow as tf
    from efnas.engine.eval_step import accumulate_predictions, build_epe_metric

    _TF_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - runtime env dependent
    tf = None
    accumulate_predictions = None
    build_epe_metric = None
    _TF_IMPORT_ERROR = str(exc)


@unittest.skipIf(tf is None, f"tensorflow unavailable: {_TF_IMPORT_ERROR}")
class TestEvalStepDynamicShape(unittest.TestCase):
    """Ensure eval helpers support dynamic spatial dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Disable eager execution once for TF1-style graph tests."""
        tf.compat.v1.disable_eager_execution()

    def setUp(self) -> None:
        """Reset the graph between tests."""
        tf.compat.v1.reset_default_graph()

    def test_accumulate_predictions_accepts_dynamic_spatial_shapes(self) -> None:
        """Prediction accumulation should resize using runtime H/W instead of static shape metadata."""
        pred_quarter = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 4], name="PredQuarter")
        pred_half = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 4], name="PredHalf")
        pred_full = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 4], name="PredFull")

        pred_accum = accumulate_predictions([pred_quarter, pred_half, pred_full])

        with tf.compat.v1.Session() as sess:
            output = sess.run(
                pred_accum,
                feed_dict={
                    pred_quarter: np.ones((1, 2, 2, 4), dtype=np.float32),
                    pred_half: np.ones((1, 4, 4, 4), dtype=np.float32) * 2.0,
                    pred_full: np.ones((1, 8, 8, 4), dtype=np.float32) * 3.0,
                },
            )

        self.assertEqual(output.shape, (1, 8, 8, 4))
        self.assertTrue(np.isfinite(output).all())

    def test_build_epe_metric_accepts_dynamic_spatial_shapes(self) -> None:
        """EPE graph construction should derive resize targets from runtime prediction shape."""
        pred_tensor = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 4], name="PredTensor")
        label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 2], name="Label")

        epe_tensor = build_epe_metric(pred_tensor=pred_tensor, label_ph=label_ph, num_out=2)

        with tf.compat.v1.Session() as sess:
            epe_val = sess.run(
                epe_tensor,
                feed_dict={
                    pred_tensor: np.zeros((1, 8, 8, 4), dtype=np.float32),
                    label_ph: np.zeros((1, 10, 10, 2), dtype=np.float32),
                },
            )

        self.assertTrue(np.isfinite(float(epe_val)))


if __name__ == "__main__":
    unittest.main()
