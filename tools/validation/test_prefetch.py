"""Prefetch must not change consumed batches, epoch seeds, tails or errors."""
from pathlib import Path
import random
import sys
import time
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'EdgeFlowNAS'))
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.engine.stage_state import save_rng, restore_rng


class Provider:
    def __init__(self, numpy=False, delay=0):
        self.rng = np.random.RandomState(42) if numpy else random.Random(42)
        self._cursor = 0; self._order = list(range(100)); self.delay = delay
    def __len__(self): return 100
    def start_epoch(self, shuffle=True):
        self._order = list(range(100))
        if shuffle: self.rng.shuffle(self._order)
        self._cursor = 0
    def next_batch(self, batch_size):
        if self.delay: time.sleep(self.delay)
        batch = [(self._order[(self._cursor+i)%100], self.rng.random()) for i in range(batch_size)]
        self._cursor += batch_size
        return batch


class PrefetchTest(unittest.TestCase):
    def test_epochs_tail_pause_and_resume(self):
        for numpy in (False, True):
            for depth in (1,2):
                plain = Provider(numpy); wrapped = PrefetchBatchProvider(Provider(numpy), depth)
                for _ in range(3):
                    plain.start_epoch(); wrapped.start_epoch()
                    for batch_size in [4,4,1,4]:
                        self.assertEqual(plain.next_batch(batch_size), wrapped.next_batch(batch_size))
                        time.sleep(.01)  # Let the producer run ahead.
                    wrapped.pause()
                    self.assertEqual(plain._cursor, wrapped.provider._cursor)
                    self.assertEqual(save_rng(plain.rng), save_rng(wrapped.rng))
                fresh = Provider(numpy)
                restore_rng(fresh.rng, save_rng(wrapped.rng))
                plain.start_epoch(); fresh.start_epoch()
                self.assertEqual(plain.next_batch(4), fresh.next_batch(4))
                wrapped.close()

    def test_close_waits_for_slow_producer(self):
        wrapped = PrefetchBatchProvider(Provider(delay=1.2), 1)
        expected = wrapped.next_batch(2)
        producer = wrapped._thread
        time.sleep(.05)
        wrapped.close()
        self.assertFalse(producer.is_alive())
        self.assertIsNone(wrapped._thread)
        self.assertEqual(wrapped.provider._cursor, len(expected))

    def test_queued_exception_survives_dead_producer(self):
        class Broken(Provider):
            def next_batch(self, batch_size): raise ValueError('broken sample')
        wrapped = PrefetchBatchProvider(Broken(), 2)
        wrapped._start_prefetch(2)
        wrapped._thread.join()
        with self.assertRaisesRegex(ValueError, 'broken sample'):
            wrapped.next_batch(2)
        wrapped.close()


if __name__ == '__main__': unittest.main()
