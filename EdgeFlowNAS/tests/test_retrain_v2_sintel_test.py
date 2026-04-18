"""Unit tests for retrain_v2 Sintel evaluation result extraction."""

import unittest

import numpy as np

from wrappers.run_retrain_v2_sintel_test import _extract_processor_mean_epe


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


if __name__ == "__main__":
    unittest.main()
