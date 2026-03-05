import os
import unittest

from efnas.search import eval_worker


class TestEvalWorkerCommand(unittest.TestCase):
    def test_build_eval_command_passes_through_eval_overrides(self) -> None:
        project_root = os.path.abspath(".")
        eval_cfg = {
            "eval_script": "wrappers/run_supernet_subnet_distribution.py",
            "supernet_config": "configs/supernet_fc2_180x240.yaml",
            "checkpoint_type": "best",
            "enable_vela": True,
            "vela_mode": "verbose",
            "vela_keep_artifacts": True,
            "batch_size": 16,
            "bn_recal_batches": 4,
            "eval_batches_per_arch": 8,
            "num_workers": 2,
            "cpu_only": True,
            "vela_optimise": "Size",
            "vela_limit": 1,
            "vela_rep_dataset_samples": 3,
            "vela_float32": True,
            "vela_verbose_log": True,
        }

        cmd = eval_worker._build_eval_command(
            project_root=project_root,
            eval_cfg=eval_cfg,
            arch_code_str="0,1,2,0,1,2,0,1,2",
            output_tag="agent_eval_012012012",
            run_output_dir=os.path.join(project_root, "outputs", "tmp_run"),
        )

        self.assertIn("--batch_size", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--bn_recal_batches", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--eval_batches_per_arch", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--num_workers", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--cpu_only", cmd)
        self.assertIn("--vela_optimise", cmd)
        self.assertIn("Size", cmd)
        self.assertIn("--vela_limit", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--vela_rep_dataset_samples", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--vela_float32", cmd)
        self.assertIn("--vela_verbose_log", cmd)

    def test_build_eval_command_omits_unset_eval_overrides(self) -> None:
        project_root = os.path.abspath(".")
        eval_cfg = {
            "eval_script": "wrappers/run_supernet_subnet_distribution.py",
            "supernet_config": "configs/supernet_fc2_180x240.yaml",
            "checkpoint_type": "best",
            "enable_vela": False,
            "vela_keep_artifacts": False,
        }

        cmd = eval_worker._build_eval_command(
            project_root=project_root,
            eval_cfg=eval_cfg,
            arch_code_str="0,0,0,0,0,0,0,0,0",
            output_tag="agent_eval_000000000",
            run_output_dir=os.path.join(project_root, "outputs", "tmp_run"),
        )

        self.assertNotIn("--batch_size", cmd)
        self.assertNotIn("--bn_recal_batches", cmd)
        self.assertNotIn("--eval_batches_per_arch", cmd)
        self.assertNotIn("--num_workers", cmd)
        self.assertNotIn("--cpu_only", cmd)
        self.assertNotIn("--vela_float32", cmd)
        self.assertNotIn("--vela_verbose_log", cmd)


if __name__ == "__main__":
    unittest.main()
