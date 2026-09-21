import unittest
from check_saved_sintel import validate_scores


class SavedSintelTests(unittest.TestCase):
    def test_both_metrics_and_sample_count_must_reproduce(self):
        ref = dict(full_monitor_evaluated_samples='845', full_monitor_sintel_raw_epe='7',
                   full_monitor_sintel_legacy_epe='5')
        result = dict(samples=845, raw_epe=7+1e-7, legacy_epe=5)
        self.assertLess(abs(validate_scores(ref, result, 845)['raw']), 1e-5)
        for change in [dict(samples=76), dict(raw_epe=float('nan')), dict(legacy_epe=5.1)]:
            with self.assertRaises(ValueError): validate_scores(ref, {**result, **change}, 845)
