"""Tests for bounded batch prefetch provider."""

import unittest

from efnas.data.prefetch_provider import PrefetchBatchProvider


class DummyProvider:
    """Small deterministic provider for prefetch tests."""

    def __init__(self):
        self.calls = []
        self.started = 0
        self.cursor = 0

    def __len__(self):
        return 5

    def start_epoch(self, shuffle=True):
        self.started += 1
        self.calls.append(("start_epoch", bool(shuffle)))

    def reset_cursor(self, index=0):
        self.cursor = int(index)
        self.calls.append(("reset_cursor", int(index)))

    def next_batch(self, batch_size):
        self.calls.append(("next_batch", int(batch_size)))
        return int(batch_size)


class TestPrefetchBatchProvider(unittest.TestCase):
    """Validate bounded prefetch wrapper behavior."""

    def test_delegates_len_and_epoch_reset(self):
        provider = DummyProvider()
        wrapped = PrefetchBatchProvider(provider, prefetch_batches=2)
        try:
            self.assertEqual(len(wrapped), 5)
            wrapped.start_epoch(shuffle=False)
            self.assertEqual(provider.calls[-1], ("start_epoch", False))
            wrapped.reset_cursor(3)
            self.assertEqual(provider.calls[-1], ("reset_cursor", 3))
        finally:
            wrapped.close()

    def test_prefetch_returns_batches_and_shutdown_is_idempotent(self):
        provider = DummyProvider()
        wrapped = PrefetchBatchProvider(provider, prefetch_batches=2)
        try:
            self.assertEqual(wrapped.next_batch(7), 7)
            self.assertEqual(wrapped.next_batch(3), 3)
        finally:
            wrapped.close()
            wrapped.close()


if __name__ == "__main__":
    unittest.main()
