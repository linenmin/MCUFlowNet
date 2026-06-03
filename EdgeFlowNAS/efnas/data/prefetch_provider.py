"""Bounded asynchronous batch prefetch wrapper."""

from queue import Full, Queue
from threading import Event, Thread
from typing import Any


class PrefetchBatchProvider:
    """Wrap a batch provider and prefetch future `next_batch` calls."""

    def __init__(self, provider: Any, prefetch_batches: int = 2):
        self.provider = provider
        self.prefetch_batches = max(0, int(prefetch_batches))
        self._queue = None
        self._stop = None
        self._thread = None
        self._current_batch_size = None
        self.source_dir = getattr(provider, "source_dir", "")
        self.sampling_mode = getattr(provider, "sampling_mode", "")
        self.crop_mode = getattr(provider, "crop_mode", "")

    def __len__(self):
        return len(self.provider)

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def _put_or_stop(self, queue: Queue, stop: Event, payload: Any) -> bool:
        while not stop.is_set():
            try:
                queue.put(payload, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _worker(self, queue: Queue, stop: Event, batch_size: int):
        while not stop.is_set():
            try:
                item = self.provider.next_batch(batch_size=batch_size)
            except Exception as exc:
                self._put_or_stop(queue, stop, (False, exc))
                return
            if not self._put_or_stop(queue, stop, (True, item)):
                return

    def _start_prefetch(self, batch_size: int) -> None:
        if self.prefetch_batches <= 0:
            return
        if self._thread is not None and self._thread.is_alive() and self._current_batch_size == int(batch_size):
            return
        self.close()
        self._current_batch_size = int(batch_size)
        self._stop = Event()
        self._queue = Queue(maxsize=self.prefetch_batches)
        self._thread = Thread(
            target=self._worker,
            args=(self._queue, self._stop, self._current_batch_size),
            name="batch_prefetch",
            daemon=True,
        )
        self._thread.start()

    def next_batch(self, batch_size: int):
        if self.prefetch_batches <= 0:
            return self.provider.next_batch(batch_size=batch_size)
        self._start_prefetch(batch_size=batch_size)
        ok, payload = self._queue.get()
        if not ok:
            raise payload
        return payload

    def start_epoch(self, shuffle=True):
        self.close()
        if hasattr(self.provider, "start_epoch"):
            return self.provider.start_epoch(shuffle=shuffle)
        return None

    def reset_cursor(self, index=0):
        self.close()
        if hasattr(self.provider, "reset_cursor"):
            return self.provider.reset_cursor(index)
        return None

    def close(self):
        stop = self._stop
        if stop is not None:
            stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._queue = None
        self._stop = None
