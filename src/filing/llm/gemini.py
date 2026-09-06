"""Google Gemini backend.

Same contract as every other backend -- chat, embed, rerank, usage -- and the
same four responsibilities concentrated in one place: rate limit, retry, cache,
spans.

Shape of the provider, which is why the code looks like it does:
  * chat is OpenAI-compatible, so it reuses the SDK and the OpenAI instrumentor
  * embeddings are *not*: taskType (query vs passage) has no OpenAI equivalent,
    and the compatibility shim drops unknown fields silently, so embeddings go
    through the native REST API where taskType is a first-class parameter
  * there is no reranker at all, so reranking runs locally on a cross-encoder

Rate limits are per model, not per provider. On the free tier the chat model
allows roughly a tenth of what the embedding model does, and sharing one limiter
would drag indexing down to the speed of the slowest model in the registry.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx
from openai import NotFoundError, OpenAI
from tenacity import retry

from filing.config import ModelSpec, Settings, model_for, settings
from filing.llm.base import InputType, Ranking, Usage
from filing.llm.cache import CallCache, make_key
from filing.llm.errors import MissingCredentials, ModelUnavailable
from filing.llm.limiter import RateLimiter
from filing.llm.rerank_local import LocalReranker
from filing.llm.retry import HOSTED as _RETRY
from filing.tracing import get_tracer

log = logging.getLogger(__name__)

# Gemini's spelling of the asymmetric-embedding distinction. The concept is the
# provider-independent thing; the vocabulary is not.
_TASK_TYPE: dict[InputType, str] = {
    "query": "RETRIEVAL_QUERY",
    "passage": "RETRIEVAL_DOCUMENT",
}


def _l2_normalize(vec: list[float]) -> list[float]:
    """Renormalize a truncated embedding.

    gemini-embedding-001 emits unit-norm vectors at its full 3072 dimensions
    only. Ask for fewer and you get a Matryoshka prefix, which is *not* unit
    norm -- so cosine similarity silently stops being cosine similarity. This is
    the kind of bug that shows up as a mediocre retrieval score and never as an
    error.
    """
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class GeminiBackend:
    name = "gemini"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        if not self.cfg.gemini_api_key:
            raise MissingCredentials(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and paste a key "
                "from https://aistudio.google.com/apikey (Create API key)."
            )
        # max_retries=0: tenacity owns retries, so every retry also passes the
        # limiter. The SDK retrying behind our back would spend unbudgeted quota.
        self._client = OpenAI(
            base_url=self.cfg.gemini_openai_base_url,
            api_key=self.cfg.gemini_api_key.get_secret_value(),
            timeout=self.cfg.request_timeout_s,
            max_retries=0,
        )
        self._http = httpx.Client(
            timeout=self.cfg.request_timeout_s,
            headers={"x-goog-api-key": self.cfg.gemini_api_key.get_secret_value()},
        )
        self._limiters: dict[str, RateLimiter] = {}
        self.cache = CallCache(self.cfg.cache_dir, enabled=self.cfg.cache_enabled)
        self.tracer = get_tracer("filing.llm")
        self.http_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

        rspec = model_for("rerank", "gemini")
        self._reranker = LocalReranker(
            rspec.id,
            device=self.cfg.rerank_device,
            batch_size=self.cfg.rerank_batch_size,
        )

    # ---------------------------------------------------------------- limiting

    def _limiter(self, model_id: str, spec: ModelSpec) -> RateLimiter:
        """One limiter per model ID, created on first use."""
        if model_id not in self._limiters:
            self._limiters[model_id] = RateLimiter(spec.rpm or self.cfg.rate_limit_rpm)
        return self._limiters[model_id]

    @property
    def limiter(self) -> RateLimiter:
        """The chat limiter, for callers that just want to inspect one."""
        spec = model_for("chat", "gemini")
        return self._limiter(spec.id, spec)

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "chat",
        model_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        spec = model_for(role, "gemini")
        mid = model_id or spec.id
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key = make_key(backend=self.name, model=mid, kind="chat", payload=payload)

        with self.tracer.start_as_current_span("llm.chat") as span:
            # CHAIN, not LLM: this span is the cache-and-limit wrapper. The
            # actual model call is the auto-instrumented child underneath, and
            # labelling both LLM would double-count every call in Phoenix.
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("llm.model_name", mid)
            span.set_attribute("filing.role", role)
            cached = self.cache.get(key)
            span.set_attribute("filing.cache_hit", cached is not None)
            if cached is not None:
                return cached

            text = self._chat_call(mid, role, spec, payload)
            self.cache.set(key, text)
            return text

    @retry(**_RETRY)
    def _chat_call(self, model_id: str, role: str, spec: ModelSpec, payload: dict[str, Any]) -> str:
        self._limiter(model_id, spec).acquire()
        self.http_calls += 1
        try:
            resp = self._client.chat.completions.create(model=model_id, **payload)
        except NotFoundError as exc:
            raise ModelUnavailable(role, model_id, spec.alternates, str(exc)) from exc
        if resp.usage:
            self._prompt_tokens += resp.usage.prompt_tokens or 0
            self._completion_tokens += resp.usage.completion_tokens or 0
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------- embedding

    def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType = "passage",
        model_id: str | None = None,
    ) -> list[list[float]]:
        """Embed a batch. Cached per text, so re-indexing an unchanged corpus is free."""
        if not texts:
            return []
        spec = model_for("embed", "gemini")
        mid = model_id or spec.id

        with self.tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("openinference.span.kind", "EMBEDDING")
            span.set_attribute("llm.model_name", mid)
            span.set_attribute("filing.input_type", input_type)
            span.set_attribute("filing.batch_size", len(texts))

            keys = [
                make_key(
                    backend=self.name,
                    model=mid,
                    kind="embed",
                    # dim is part of the key: the same text at 768 and at 1536
                    # dimensions are different vectors, not the same one twice.
                    payload={"text": t, "input_type": input_type, "dim": spec.dim},
                )
                for t in texts
            ]
            out: list[list[float] | None] = [self.cache.get(k) for k in keys]
            missing = [i for i, v in enumerate(out) if v is None]
            span.set_attribute("filing.cache_hits", len(texts) - len(missing))

            if missing:
                vectors = self._embed_call(mid, spec, [texts[i] for i in missing], input_type)
                for i, vec in zip(missing, vectors, strict=True):
                    out[i] = vec
                    self.cache.set(keys[i], vec)

            return [v for v in out if v is not None]

    @retry(**_RETRY)
    def _embed_call(
        self,
        model_id: str,
        spec: ModelSpec,
        texts: list[str],
        input_type: InputType,
    ) -> list[list[float]]:
        self._limiter(model_id, spec).acquire()
        self.http_calls += 1

        request: dict[str, Any] = {
            "model": f"models/{model_id}",
            "taskType": _TASK_TYPE[input_type],
        }
        # Only the gemini-embedding family accepts a truncated width.
        truncated = spec.dim is not None and model_id.startswith("gemini-embedding")
        if truncated:
            request["outputDimensionality"] = spec.dim

        resp = self._http.post(
            f"{self.cfg.gemini_base_url.rstrip('/')}/models/{model_id}:batchEmbedContents",
            json={"requests": [{**request, "content": {"parts": [{"text": t}]}} for t in texts]},
        )
        if resp.status_code == 404:
            raise ModelUnavailable("embed", model_id, spec.alternates, resp.text[:300])
        resp.raise_for_status()

        vectors = [list(e.get("values", [])) for e in resp.json().get("embeddings", [])]
        if len(vectors) != len(texts):  # pragma: no cover - provider contract
            raise ValueError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return [_l2_normalize(v) for v in vectors] if truncated else vectors

    # --------------------------------------------------------------- rerank

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
        spec = model_for("rerank", "gemini")
        mid = model_id or spec.id
        key = make_key(
            backend=self.name,
            model=mid,
            kind="rerank",
            payload={"query": query, "passages": passages},
        )

        with self.tracer.start_as_current_span("llm.rerank") as span:
            # Nothing auto-instruments a local forward pass, so the reranker span
            # is annotated by hand with the names Phoenix expects.
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
                # No limiter and no http_calls bump: this call never leaves the
                # machine, and counting it as network traffic would make the
                # budget numbers lie.
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

    def usage(self) -> Usage:
        return Usage(
            http_calls=self.http_calls,
            cache_hits=self.cache.hits,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

    def close(self) -> None:
        self._http.close()
        self.cache.close()
