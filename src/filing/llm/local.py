"""The backend that owns the vector space.

Everything retrieval needs -- embedding and reranking -- runs on this machine;
generation does not, and this backend refuses it rather than pretending. That
split is deliberate and it is the answer to a measured wall, not a preference:
Gemini's free embedding tier serves 1,000 documents a *day* (see
``filing.llm.embed_local``), and this corpus has 32,218 narrative chunks. One
hosted call per answer is affordable; one per chunk is not.

The important consequence is that the vector space stops being rented. There is
no key, no quota, no network and no per-call cost, so re-embedding the corpus
costs one afternoon of four CPU threads -- the full 32,218-chunk build measured
230 chunks a minute, 140.6 minutes end to end -- instead of a month of somebody
else's free-tier allowance. Two and a half hours is a real cost, but it is a
cost you can choose to pay; a daily cap is not. That is what makes it possible
to change the chunker and actually find out whether retrieval got better.

``collection_name`` puts this backend's name and its model's width into the
Qdrant collection, exactly as it does for the hosted ones. Pointing the
retriever at a gemini-built index with this backend selected does not return
worse neighbours; it fails to find a collection and says which one it wanted.
"""

from __future__ import annotations

import logging

from filing.config import ModelSpec, Settings, model_for, settings
from filing.llm.base import InputType, Ranking, Usage
from filing.llm.cache import CallCache, make_key
from filing.llm.embed_local import LocalEmbedder
from filing.llm.errors import LLMError
from filing.llm.rerank_local import LocalReranker
from filing.tracing import get_tracer

log = logging.getLogger(__name__)


class LocalBackend:
    name = "local"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        espec: ModelSpec = model_for("embed", "local")
        rspec: ModelSpec = model_for("rerank", "local")
        self._embedder = LocalEmbedder(
            espec.id,
            device=self.cfg.embed_device,
            batch_size=self.cfg.embed_batch_size,
            max_length=self.cfg.embed_max_tokens,
        )
        self._reranker = LocalReranker(
            rspec.id,
            device=self.cfg.rerank_device,
            batch_size=self.cfg.rerank_batch_size,
        )
        self.cache = CallCache(self.cfg.cache_dir, enabled=self.cfg.cache_enabled)
        self.tracer = get_tracer("filing.llm")
        # Zero, always, and that is the point of the number rather than an
        # oversight: nothing here leaves the machine.
        self.http_calls = 0

    # ------------------------------------------------------------------ chat

    def chat(self, messages, **kwargs) -> str:  # noqa: ANN001, ANN003
        raise LLMError(
            "the local backend serves embeddings and reranking, not chat. "
            "Set LLM_BACKEND=gemini (or ollama) for generation; the retrieval "
            "half stays local either way -- see EMBED_BACKEND."
        )

    # ------------------------------------------------------------- embedding

    def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType = "passage",
        model_id: str | None = None,
    ) -> list[list[float]]:
        """Encode a batch on the local bi-encoder.

        Queries are cached, passages are not. A passage's vector already lives
        in Qdrant, so a second copy on disk would be 32,000 duplicates of data
        the index holds -- and recomputing one costs milliseconds and no quota.
        A query, on the other hand, is asked again by every eval sweep, and
        caching it is what keeps a sweep's cost in the reranker rather than the
        encoder.
        """
        if not texts:
            return []
        spec = model_for("embed", "local")
        mid = model_id or spec.id

        with self.tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("openinference.span.kind", "EMBEDDING")
            span.set_attribute("llm.model_name", mid)
            span.set_attribute("filing.input_type", input_type)
            span.set_attribute("filing.batch_size", len(texts))
            span.set_attribute("filing.local", True)

            if input_type != "query":
                return self._embedder.encode(texts, input_type=input_type)

            keys = [
                make_key(
                    backend=self.name,
                    model=mid,
                    kind="embed",
                    payload={"text": t, "input_type": input_type, "dim": spec.dim},
                )
                for t in texts
            ]
            out: list[list[float] | None] = [self.cache.get(k) for k in keys]
            missing = [i for i, v in enumerate(out) if v is None]
            span.set_attribute("filing.cache_hits", len(texts) - len(missing))
            if missing:
                vectors = self._embedder.encode([texts[i] for i in missing], input_type=input_type)
                for i, vec in zip(missing, vectors, strict=True):
                    out[i] = vec
                    self.cache.set(keys[i], vec)
            return [v for v in out if v is not None]

    # ---------------------------------------------------------------- rerank

    def rerank(
        self,
        query: str,
        passages: list[str],
        *,
        top_n: int | None = None,
        model_id: str | None = None,
    ) -> list[Ranking]:
        if not passages:
            return []
        spec = model_for("rerank", "local")
        mid = model_id or spec.id
        key = make_key(
            backend=self.name,
            model=mid,
            kind="rerank",
            payload={"query": query, "passages": passages},
        )

        with self.tracer.start_as_current_span("llm.rerank") as span:
            span.set_attribute("openinference.span.kind", "RERANKER")
            span.set_attribute("reranker.model_name", mid)
            span.set_attribute("reranker.query", query[:500])
            span.set_attribute("reranker.top_k", top_n or len(passages))
            span.set_attribute("filing.local", True)

            cached = self.cache.get(key)
            span.set_attribute("filing.cache_hit", cached is not None)
            if cached is not None:
                rankings = [Ranking(index=r["index"], score=r["score"]) for r in cached]
            else:
                reranker = (
                    self._reranker
                    if mid == spec.id
                    else LocalReranker(
                        mid, device=self.cfg.rerank_device, batch_size=self.cfg.rerank_batch_size
                    )
                )
                rankings = reranker.rank(query, passages)
                self.cache.set(key, [{"index": r.index, "score": r.score} for r in rankings])
            return rankings[:top_n] if top_n is not None else rankings

    # ------------------------------------------------------------------ misc

    def warm(self) -> None:
        self._embedder.warm()
        self._reranker.warm()

    def usage(self) -> Usage:
        return Usage(http_calls=0, cache_hits=self.cache.hits)

    def close(self) -> None:
        self.cache.close()
