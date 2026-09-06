"""Embeddings that run on this machine.

Why this exists, in one measurement. Gemini's free embedding tier is metered
three ways -- 100 requests a minute, 30,000 tokens a minute, and **1,000
documents a day** -- and only the last one matters. ``batchEmbedContents``
counts each text in the batch as its own request, so a 32-chunk batch spends 32
of the day's thousand. The narrative half of this corpus is 32,218 chunks. That
is thirty-two days of indexing, and the quota string says so in as many words::

    Quota exceeded for metric: embed_content_free_tier_requests, limit: 1000

No amount of pacing fixes a daily cap; it is not a throughput problem. So the
vector space is built here instead, by the same sentence-transformers runtime
that already serves the cross-encoder in ``rerank_local``. There is no key, no
quota and no network, which also means the M8 evaluation sweep can re-embed as
often as it likes. Gemini stays where it earns its keep -- synthesis, where a
local 33M-parameter model would be a real downgrade and a hosted call is one
request per answer rather than one per chunk.

**BGE is asymmetric, and its asymmetry is a prompt, not a parameter.** A passage
is embedded bare; a query is embedded behind a fixed instruction. Get that
backwards and retrieval still works, just worse -- the failure mode this project
keeps trying to make impossible -- so the prefix lives beside the model ID here
and is applied from ``input_type`` alone.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

log = logging.getLogger(__name__)

# The instruction BAAI trained the query side on, verbatim. Passages get no
# prefix at all -- that asymmetry is the model, not a convention.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Per-family query prefixes. A model that is not listed is treated as symmetric.
QUERY_PREFIXES: dict[str, str] = {
    "bge": BGE_QUERY_PREFIX,
    "e5": "query: ",
    "gte": "",
}

PASSAGE_PREFIXES: dict[str, str] = {"e5": "passage: "}


def _family(model_id: str) -> str:
    name = model_id.rsplit("/", 1)[-1].lower()
    for fam in QUERY_PREFIXES:
        if name.startswith(fam):
            return fam
    return ""


@lru_cache(maxsize=2)
def _load(model_id: str, device: str, max_length: int):  # noqa: ANN202
    """Load once per process.

    Same reasoning as the reranker: a CLI command that embeds a query and then
    reranks must not pay the load cost twice, and an index build must not pay
    it once per batch.
    """
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise ImportError(
            "local embeddings need sentence-transformers. Install it with:\n"
            "  pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = max_length
    log.info("loaded %s on %s in %.1fs", model_id, device, time.perf_counter() - started)
    return model


class LocalEmbedder:
    """A bi-encoder behind the same ``embed`` call every backend uses."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.family = _family(model_id)

    def warm(self) -> None:
        """Pay the load cost on purpose, at a moment of your choosing."""
        _load(self.model_id, self.device, self.max_length)

    def prefixed(self, text: str, input_type: str) -> str:
        table = QUERY_PREFIXES if input_type == "query" else PASSAGE_PREFIXES
        return table.get(self.family, "") + text

    def encode(self, texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
        if not texts:
            return []
        model = _load(self.model_id, self.device, self.max_length)
        vectors = model.encode(
            [self.prefixed(t, input_type) for t in texts],
            batch_size=self.batch_size,
            # Cosine is the distance Qdrant is configured with, and a
            # unit-norm vector is what makes the dot product cosine.
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        return int(
            _load(self.model_id, self.device, self.max_length).get_sentence_embedding_dimension()
        )
