"""Sliding-window rate limiter.

A token bucket with burst capacity C admits up to ``C + rpm`` calls inside some
60-second window, which is exactly the overshoot that gets an account throttled.
A sliding window admits ``rpm`` and not one more, and the invariant is directly
testable: no 60-second window ever contains more than ``rpm`` timestamps.

``clock`` and ``sleep`` are injected so the test can drive a fake clock and
assert the invariant over thousands of calls in microseconds.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        rpm: int,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self.rpm = rpm
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._marks: deque[float] = deque()
        self._lock = threading.Lock()
        self.total_waited_s = 0.0

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._marks and self._marks[0] <= cutoff:
            self._marks.popleft()

    def acquire(self) -> float:
        """Block until a slot is free. Returns how long it waited, in seconds."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._evict(now)
                if len(self._marks) < self.rpm:
                    self._marks.append(now)
                    self.total_waited_s += waited
                    return waited
                # Sleep just past the moment the oldest mark leaves the window.
                delay = self._marks[0] + self.window_s - now
            delay = max(delay, 1e-6)
            self._sleep(delay)
            waited += delay

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self._marks)
