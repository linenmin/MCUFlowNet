"""Tests for the distill-or-not short FC2 retrain probe."""

import csv
import tempfile
import unittest
from pathlib import Path


class TestDistillOrNotShortRetrain(unittest.TestCase):
    def test_candidate_csv_loader_keeps_rank_gap_metadata(self):
        from wrappers.run_distill_or_not_fc2_batch import load_candidates

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "top10.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["candidate_id", "arch_code", "distill_rank", "no_distill_rank", "rank_gap"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_id": "T01",
                        "arch_code": "0,0,0,2,2,0,0,0,1,0,0",
                        "distill_rank": "2",
                        "no_distill_rank": "13",
                        "rank_gap": "11",
                    }
                )

            candidates = load_candidates(csv_path)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_id"], "T01")
        self.assertEqual(candidates[0]["arch_code"], "0,0,0,2,2,0,0,0,1,0,0")
        self.assertEqual(candidates[0]["rank_gap"], 11)

    def test_gpu_assignment_is_round_robin(self):
        from wrappers.run_distill_or_not_fc2_batch import assign_gpu

        self.assertEqual([assign_gpu(i, ["0", "1", "2", "3", "4"]) for i in range(7)], ["0", "1", "2", "3", "4", "0", "1"])

    def test_one_wrapper_parser_accepts_scratch_probe_controls(self):
        from wrappers.run_distill_or_not_fc2_one import _build_parser

        args = _build_parser().parse_args(
            [
                "--arch_code",
                "0,0,0,2,2,0,0,0,1,0,0",
                "--model_name",
                "T01",
                "--experiment_name",
                "probe",
                "--gpu_device",
                "0",
                "--num_epochs",
                "50",
                "--sintel_eval_every_epoch",
                "5",
                "--fc2_num_workers",
                "4",
                "--prefetch_batches",
                "2",
            ]
        )
        self.assertEqual(args.model_name, "T01")
        self.assertEqual(args.num_epochs, 50)
        self.assertEqual(args.sintel_eval_every_epoch, 5)
        self.assertEqual(args.fc2_num_workers, 4)

    def test_prefetch_wrapper_uses_existing_provider_signature(self):
        try:
            import tensorflow  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("TensorFlow is required to import trainer")
        from efnas.engine.distill_or_not_trainer import _wrap_prefetch

        class DummyProvider:
            source_dir = "dummy"

            def __len__(self):
                return 1

            def next_batch(self, batch_size):
                return batch_size

        wrapped = _wrap_prefetch(DummyProvider(), 1)
        self.assertEqual(wrapped.next_batch(4), 4)
        wrapped.close()

    def test_fixed_v3_model_builds_only_selected_stem_branch(self):
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            self.skipTest("TensorFlow is required")
        from efnas.network.fixed_arch_models_v3 import FixedArchModelV3

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 6], name="input_ph")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training_ph")
        with tf.compat.v1.variable_scope("candidate"):
            model = FixedArchModelV3(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                arch_code=[0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1],
                num_out=4,
            )
            preds = model.build()

        self.assertEqual(len(preds), 3)
        names = {var.op.name for var in tf.compat.v1.global_variables()}
        self.assertTrue(any("candidate/supernet_backbone/E0/" in name and "k3_" in name for name in names))
        self.assertFalse(any("candidate/supernet_backbone/E0/" in name and "k5_" in name for name in names))
        self.assertFalse(any("candidate/supernet_backbone/E0/" in name and "k7_" in name for name in names))
        self.assertTrue(any("candidate/supernet_backbone/eca_bottleneck" in name for name in names))
        self.assertTrue(any("global_gate_4x" in name for name in names))


if __name__ == "__main__":
    unittest.main()
