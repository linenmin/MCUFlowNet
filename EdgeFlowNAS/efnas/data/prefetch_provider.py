"""Bounded asynchronous batch prefetch wrapper."""

from queue import Full, Queue
from threading import Event, Thread
from typing import Any
import copy


class PrefetchBatchProvider:
    """Wrap a batch provider and prefetch future `next_batch` calls."""

    def __init__(self, provider: Any, prefetch_batches: int = 2):
        self.provider = provider
        self.prefetch_batches = max(0, int(prefetch_batches))
        self._queue = None
        self._stop = None
        self._thread = None
        self._current_batch_size = None
        self._consumed_state = None
        self.source_dir = getattr(provider, "source_dir", "")
        self.sampling_mode = getattr(provider, "sampling_mode", "")
        self.crop_mode = getattr(provider, "crop_mode", "")

    def __len__(self):
        return len(self.provider)

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def _capture_state(self):
        rng = getattr(self.provider, "rng", None)
        state = {}
        if rng is not None:
            method = "getstate" if hasattr(rng, "getstate") else "get_state"
            state["rng"] = (method, copy.deepcopy(getattr(rng, method)()))
        if hasattr(self.provider, "_cursor"):
            state["cursor"] = self.provider._cursor
        if hasattr(self.provider, "_order"):
            # FC2/FT3D never mutate this list while loading; start_epoch replaces it
            # only after pause() joins the producer. Avoid copying 80k IDs per batch.
            state["order"] = self.provider._order
        return state

    def _restore_state(self, state):
        if "rng" in state:
            method, value = state["rng"]
            getattr(self.provider.rng, "setstate" if method == "getstate" else "set_state")(value)
        if "cursor" in state:
            self.provider._cursor = state["cursor"]
        if "order" in state:
            self.provider._order = state["order"]

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
                state = self._capture_state()
            except Exception as exc:
                self._put_or_stop(queue, stop, (False, exc, None))
                return
            if not self._put_or_stop(queue, stop, (True, item, state)):
                return

    def _start_prefetch(self, batch_size: int) -> None:
        if self.prefetch_batches <= 0:
            return
        if self._thread is not None and self._current_batch_size == int(batch_size):
            return
        self.pause()
        self._consumed_state = self._capture_state()
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
        ok, payload, state = self._queue.get()
        if not ok:
            raise payload
        self._consumed_state = state
        return payload

    def start_epoch(self, shuffle=True):
        self.pause()
        if hasattr(self.provider, "start_epoch"):
            return self.provider.start_epoch(shuffle=shuffle)
        return None

    def reset_cursor(self, index=0):
        self.pause()
        if hasattr(self.provider, "reset_cursor"):
            return self.provider.reset_cursor(index)
        return None

    def pause(self):
        """Discard speculative work and restore the last delivered batch boundary."""
        stop = self._stop
        if stop is not None:
            stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Do not reset a provider while an old producer still accesses it.
            thread.join()
        if self._consumed_state is not None:
            self._restore_state(self._consumed_state)
        self._consumed_state = None
        self._thread = None
        self._queue = None
        self._stop = None

    def close(self):
        self.pause()
        if hasattr(self.provider, "close"):
            self.provider.close()
