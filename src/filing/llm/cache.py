"""Content-addressed cache for model calls.

The eval harness re-runs 150 questions many times over five weeks. Without this
the free tier is gone by week two. The key is a hash of everything that could
change the answer -- backend, model ID, call kind, and the full payload -- so a
cache hit is a guarantee, not a guess.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache

CACHE_SCHEMA = "v1"  # bump to invalidate every entry at once


def make_key(*, backend: str, model: str, kind: str, payload: Any) -> str:
    blob = json.dumps(
        {
            "schema": CACHE_SCHEMA,
            "backend": backend,
            "model": model,
            "kind": kind,
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CallCache:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self.hits = 0
        self.misses = 0
        self._cache: diskcache.Cache | None = None
        if enabled:
            path.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(path))

    def get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        value = self._cache.get(key, default=None)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if self._cache is not None:
            self._cache.set(key, value)

    def clear(self) -> int:
        if self._cache is None:
            return 0
        n = len(self._cache)
        self._cache.clear()
        return n

    def __len__(self) -> int:
        return len(self._cache) if self._cache is not None else 0

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
