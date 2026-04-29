"""Tests for bounded batch prefetch provider."""

import unittest
from threading import Event
import threading

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


class BlockingProvider:
    """Provider that lets tests close prefetch while a batch is in flight."""

    def __init__(self):
        self.release = Event()

    def __len__(self):
        return 1

    def next_batch(self, batch_size):
        self.release.wait(timeout=5.0)
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

    def test_close_does_not_clear_queue_before_worker_exits(self):
        provider = BlockingProvider()
        wrapped = PrefetchBatchProvider(provider, prefetch_batches=1)
        thread_errors = []
        old_hook = threading.excepthook

        def capture_thread_error(args):
            thread_errors.append(args.exc_value)

        threading.excepthook = capture_thread_error
        try:
            wrapped._start_prefetch(batch_size=4)
            worker = wrapped._thread
            wrapped.close()
            provider.release.set()
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(thread_errors, [])
        finally:
            threading.excepthook = old_hook
            provider.release.set()
            wrapped.close()


if __name__ == "__main__":
    unittest.main()
