import os
import tempfile
import unittest
from pathlib import Path

from efnas.search import eval_worker


class TestEvalWorkerCommand(unittest.TestCase):
    def test_build_eval_command_passes_through_fc2_full_eval_cap_override(self) -> None:
        project_root = os.path.abspath(".")
        eval_cfg = {
            "eval_script": "wrappers/run_supernet_subnet_distribution_v2.py",
            "supernet_config": "configs/supernet_fc2_172x224_v2.yaml",
            "checkpoint_type": "best",
            "max_fc2_val_samples": 128,
        }

        cmd = eval_worker._build_eval_command(
            project_root=project_root,
            eval_cfg=eval_cfg,
            arch_code_str="0,1,2,0,1,2,0,1,0,1,0",
            output_tag="agent_eval_v2",
            run_output_dir=os.path.join(project_root, "outputs", "tmp_run"),
        )

        self.assertIn("--max_fc2_val_samples", cmd)
        self.assertIn("128", cmd)

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
            "eval_script": "wrappers/run_supernet_subnet_distribution_v2.py",
            "supernet_config": "configs/supernet_fc2_172x224_v2.yaml",
            "checkpoint_type": "best",
            "enable_vela": False,
            "vela_keep_artifacts": False,
            "max_fc2_val_samples": None,
        }

        cmd = eval_worker._build_eval_command(
            project_root=project_root,
            eval_cfg=eval_cfg,
            arch_code_str="0,0,0,0,0,0,0,0,0,0,0",
            output_tag="agent_eval_00000000000",
            run_output_dir=os.path.join(project_root, "outputs", "tmp_run"),
        )

        self.assertNotIn("--batch_size", cmd)
        self.assertNotIn("--bn_recal_batches", cmd)
        self.assertNotIn("--eval_batches_per_arch", cmd)
        self.assertNotIn("--max_fc2_val_samples", cmd)
        self.assertNotIn("--num_workers", cmd)
        self.assertNotIn("--cpu_only", cmd)
        self.assertNotIn("--vela_float32", cmd)
        self.assertNotIn("--vela_verbose_log", cmd)

    def test_parse_vela_summary_backfills_cycles_and_macs_from_nested_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            analysis = run_dir / "analysis"
            analysis.mkdir(parents=True, exist_ok=True)

            # Simulate old vela_metrics.csv without cycles/macs.
            (analysis / "vela_metrics.csv").write_text(
                "arch_code,sram_peak_mb,inference_ms,fps\n"
                "0,0,0,0,1.582031,120.801346,8.278300\n",
                encoding="utf-8",
            )

            # Vela raw summary stays under analysis/vela_tmp/<arch>/...
            summary_dir = analysis / "vela_tmp" / "arch_0000"
            summary_dir.mkdir(parents=True, exist_ok=True)
            (summary_dir / "arch_0000_summary_Grove_Sys_Config.csv").write_text(
                "inferences_per_second,sram_memory_used,cycles_npu,nn_macs\n"
                "8.278299714226886,1658880,48319101,2078899200\n",
                encoding="utf-8",
            )

            metrics = eval_worker._parse_vela_summary(str(run_dir))
            self.assertEqual(metrics.get("cycles_npu"), 48319101)
            self.assertEqual(metrics.get("macs"), 2078899200)


if __name__ == "__main__":
    unittest.main()
