"""CPU checks for the scientific grouping and averaging conventions."""
import unittest
import numpy as np
from evaluate_label_diagnostics import GROUPS, grouped_errors, means, merge


class GroupedErrorsTest(unittest.TestCase):
    def test_boundaries_and_component_threshold(self):
        truth = np.array([[[0, 0], [10, 0], [40, 0], [160, 0],
                           [50, 0], [50, 50], [51, 0]]], np.float32)
        result = grouped_errors(np.zeros_like(truth), truth)
        self.assertEqual([result[k]['pixels'] for k in GROUPS], [7, 1, 1, 4, 1, 2, 5])
        self.assertAlmostEqual(sum(result[k]['error_sum'] for k in GROUPS[1:5]), result['all']['error_sum'])

    def test_pixel_weighting_and_empty_group(self):
        target = {}
        merge(target, {'all': {'pixels': 1, 'error_sum': 10.0}})
        merge(target, {'all': {'pixels': 9, 'error_sum': 0.0}})
        self.assertEqual(means(target)['all']['epe'], 1.0)
        zero = np.zeros((1, 1, 2), np.float32)
        self.assertIsNone(means(grouped_errors(zero, zero))['motion_160_plus']['epe'])

    def test_nonfinite_predictions_rejected(self):
        with self.assertRaises(AssertionError):
            grouped_errors(np.full((1, 1, 2), np.nan), np.zeros((1, 1, 2)))


if __name__ == '__main__':
    unittest.main()
