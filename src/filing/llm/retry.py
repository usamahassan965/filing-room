"""Retry policy, in one place because three backends share it.

Two policies, not one, and the difference is deliberate:

``HOSTED`` is for a provider across the internet. It retries rate limits, and it
backs off for up to a minute, because a 429 on a free tier means *wait*, not
*fail*. ``LOCAL`` is for a server on this machine, which never rate-limits and
never has a transient upstream -- so it retries connection errors only, briefly,
and then gets out of the way rather than hiding a dead process behind a minute
of patient waiting.

Both exist so the SDK's own retry can stay switched off. A client retrying behind
our back would spend quota the limiter never counted, which is precisely the
budget bug this project cannot afford.
"""

from __future__ import annotations

import logging

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    RetryCallState,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

# 429 is the interesting one; the rest are ordinary transient server trouble.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    if isinstance(exc, httpx.TransportError):
        return True
    return False


def log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning("retry %s after %s: %s", state.attempt_number, type(exc).__name__, exc)


HOSTED = dict(
    retry=retry_if_exception(is_retryable),
    wait=wait_exponential_jitter(initial=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=log_retry,
    reraise=True,
)

LOCAL = dict(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError, httpx.TransportError)),
    wait=wait_exponential_jitter(initial=1, max=20),
    stop=stop_after_attempt(3),
    before_sleep=log_retry,
    reraise=True,
)
