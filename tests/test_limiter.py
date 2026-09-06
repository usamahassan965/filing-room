"""The M0 gate's rate-limit proof.

The claim being tested is the one that matters to the provider: *no 60-second
window ever contains more than `rpm` requests*. A fake clock lets us assert it
over thousands of calls without waiting an hour.
"""

from __future__ import annotations

import threading
import time

import pytest

from filing.llm.limiter import RateLimiter

RPM = 35
WINDOW = 60.0


class FakeClock:
    """Time only moves when someone sleeps. Deterministic, and instant."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.now += seconds


def max_in_any_window(marks: list[float], window: float) -> int:
    """Largest number of marks inside any half-open window of the given width."""
    worst = 0
    start = 0
    for end in range(len(marks)):
        while marks[end] - marks[start] >= window:
            start += 1
        worst = max(worst, end - start + 1)
    return worst


def test_never_exceeds_rpm_over_a_long_run():
    clock = FakeClock()
    limiter = RateLimiter(RPM, window_s=WINDOW, clock=clock.time, sleep=clock.sleep)

    marks = []
    for _ in range(500):
        limiter.acquire()
        marks.append(clock.now)

    assert max_in_any_window(marks, WINDOW) <= RPM


def test_first_burst_is_not_throttled():
    """The limiter must not add latency we do not owe."""
    clock = FakeClock()
    limiter = RateLimiter(RPM, window_s=WINDOW, clock=clock.time, sleep=clock.sleep)

    for _ in range(RPM):
        assert limiter.acquire() == 0.0
    assert clock.now == 0.0

    # The next one has to wait out the full window.
    waited = limiter.acquire()
    assert waited == pytest.approx(WINDOW, abs=1e-3)


def test_500_calls_take_at_least_the_arithmetic_minimum():
    clock = FakeClock()
    limiter = RateLimiter(RPM, window_s=WINDOW, clock=clock.time, sleep=clock.sleep)
    n = 500
    for _ in range(n):
        limiter.acquire()
    # n calls at rpm need at least floor((n-1)/rpm) full windows.
    assert clock.now >= ((n - 1) // RPM) * WINDOW - 1e-6


def test_holds_under_concurrency():
    """Real clock, short window -- catches a missing lock, runs in about a second."""
    limiter = RateLimiter(10, window_s=0.5)
    marks: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(6):
            limiter.acquire()
            with lock:
                marks.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    marks.sort()
    assert len(marks) == 30
    # One slot of slack for scheduler jitter on the observation timestamps.
    assert max_in_any_window(marks, 0.5) <= 11


def test_rejects_nonsense_config():
    with pytest.raises(ValueError):
        RateLimiter(0)
