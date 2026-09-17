"""Regression checks for validation mode and a partial final batch."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.engine.retrain_trainer import _evaluate_with_progress


class Provider:
    def __len__(self): return 5
    def reset_cursor(self, index): self.cursor = index; self.sizes = []
    def next_batch(self, batch_size):
        self.sizes.append(batch_size)
        values = np.arange(self.cursor + 1, self.cursor + batch_size + 1, dtype=np.float32)
        self.cursor += batch_size
        return np.zeros((batch_size,1,1,6)), None, None, values


class Session:
    def run(self, op, feed_dict):
        assert feed_dict['mode'] is False
        return feed_dict['labels'].mean()


class ValidationTests(unittest.TestCase):
    def test_partial_batch_is_not_repeated_and_is_weighted(self):
        provider = Provider()
        result = _evaluate_with_progress(Session(), {'epe':'metric'}, 'input','labels','mode',
                                         provider,2,0,'validation test')
        self.assertEqual(provider.sizes,[2,2,1])
        self.assertEqual(result,3.0)


if __name__ == '__main__': unittest.main()
