"""Unit tests for retrain_v2 Sintel evaluation helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from efnas.engine.retrain_v2_eval_scaling import (
    _extract_processor_mean_epe,
    _resolve_prediction_flow_scale,
    _scale_prediction_for_sintel_eval,
)


class _FakeProcessor:
    def __init__(self, mean_epe=None, error_epes=None):
        self.MeanEPE = mean_epe
        self.errorEPEs = error_epes if error_epes is not None else []


class TestRetrainV2SintelTest(unittest.TestCase):
    def test_extract_processor_mean_epe_prefers_explicit_attribute(self) -> None:
        processor = _FakeProcessor(mean_epe=5.4321, error_epes=[np.array([1.0, 2.0])])
        self.assertAlmostEqual(_extract_processor_mean_epe(processor), 5.4321)

    def test_extract_processor_mean_epe_falls_back_to_error_epes(self) -> None:
        processor = _FakeProcessor(mean_epe=None, error_epes=[np.array([1.0, 3.0]), np.array([5.0])])
        self.assertAlmostEqual(_extract_processor_mean_epe(processor), 3.0)

    def test_extract_processor_mean_epe_returns_none_when_empty(self) -> None:
        processor = _FakeProcessor(mean_epe=None, error_epes=[])
        self.assertIsNone(_extract_processor_mean_epe(processor))

    def test_resolve_prediction_flow_scale_returns_one_for_fc2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiment_dir = Path(tmp_dir) / "outputs" / "retrain_v2_fc2" / "run1"
            model_dir = experiment_dir / "model_knee"
            model_dir.mkdir(parents=True, exist_ok=True)
            (experiment_dir / "run_manifest.json").write_text(
                json.dumps({"config": {"data": {"dataset": "FC2"}}}),
                encoding="utf-8",
            )
            self.assertEqual(_resolve_prediction_flow_scale(model_dir, {"metric": 1.23}), 1.0)

    def test_resolve_prediction_flow_scale_reads_ft3d_divisor_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiment_dir = Path(tmp_dir) / "outputs" / "retrain_v2_ft3d" / "run1"
            model_dir = experiment_dir / "model_knee"
            model_dir.mkdir(parents=True, exist_ok=True)
            (experiment_dir / "run_manifest.json").write_text(
                json.dumps({"config": {"data": {"dataset": "FT3D", "ft3d_flow_divisor": 12.5}}}),
                encoding="utf-8",
            )
            self.assertEqual(_resolve_prediction_flow_scale(model_dir, {"metric": 2.34}), 12.5)

    def test_resolve_prediction_flow_scale_prefers_explicit_meta_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "model_fast"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.assertEqual(_resolve_prediction_flow_scale(model_dir, {"flow_divisor": 7.5}), 7.5)

    def test_scale_prediction_for_sintel_eval_multiplies_flow_channels(self) -> None:
        preds = np.array([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=np.float32)
        scaled = _scale_prediction_for_sintel_eval(preds, 12.5)
        np.testing.assert_allclose(
            scaled,
            np.array([[[[12.5, -25.0], [37.5, -50.0]]]], dtype=np.float32),
        )

    def test_scale_prediction_for_sintel_eval_keeps_identity_for_scale_one(self) -> None:
        preds = np.array([[[[1.0, -2.0]]]], dtype=np.float32)
        scaled = _scale_prediction_for_sintel_eval(preds, 1.0)
        self.assertIsNot(scaled, preds)
        np.testing.assert_allclose(scaled, preds)


if __name__ == "__main__":
    unittest.main()
