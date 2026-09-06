"""Reranking that runs on this machine.

Gemini serves no reranker, so this fills the gap -- and it fills it better than
the hosted option it replaces. The Ollama backend "reranks" by embedding cosine,
which is not reranking at all: it is the retriever's own opinion, asked a second
time, so it can never correct the retriever's mistakes. A cross-encoder reads
the query and the passage *together* in one forward pass, which is why it can.

The cost is that it is not free at inference the way an API call is free at rest:
a 6-layer MiniLM scoring 50 passages takes a beat on a laptop CPU. That is the
right trade here -- there is no quota, no key, and no network, so the evaluation
sweep in M8 can rerank as many times as it likes.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from filing.llm.base import Ranking

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load(model_id: str, device: str, max_length: int):
    """Load once per process.

    Loading a cross-encoder costs a couple of seconds and a few hundred MB, so a
    CLI command that reranks twice must not pay for it twice. lru_cache is doing
    real work here, not tidiness.
    """
    started = time.perf_counter()
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise ImportError(
            "local reranking needs sentence-transformers. Install it with:\n"
            "  pip install sentence-transformers"
        ) from exc

    model = CrossEncoder(model_id, device=device, max_length=max_length)
    log.info("loaded %s on %s in %.1fs", model_id, device, time.perf_counter() - started)
    return model


class LocalReranker:
    """A cross-encoder behind the same ``rank`` call every backend uses."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def warm(self) -> None:
        """Pay the load cost on purpose, at a moment of your choosing."""
        _load(self.model_id, self.device, self.max_length)

    def rank(self, query: str, passages: list[str]) -> list[Ranking]:
        if not passages:
            return []
        model = _load(self.model_id, self.device, self.max_length)
        scores = model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return sorted(
            (Ranking(index=i, score=float(s)) for i, s in enumerate(scores)),
            key=lambda r: r.score,
            reverse=True,
        )
