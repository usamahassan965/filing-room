"""Local Ollama backend -- the credits-ran-out escape hatch.

Wired in at M0 rather than "when we need it", because the moment you need it is
the moment you cannot afford to spend two days building it. It implements the
same three verbs, so switching is one environment variable:

    LLM_BACKEND=ollama

One honest caveat, stated at runtime as well as here: Ollama serves no
cross-encoder. ``rerank`` here scores by embedding cosine, which is a weaker
signal than a real reranker. Numbers produced under this backend are labelled
as such and never go in the results table.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from openai import NotFoundError, OpenAI
from tenacity import retry

from filing.config import Settings, model_for, settings
from filing.llm.base import InputType, Ranking, Usage
from filing.llm.cache import CallCache, make_key
from filing.llm.errors import ModelUnavailable
from filing.llm.limiter import RateLimiter
from filing.llm.retry import LOCAL as _RETRY
from filing.tracing import get_tracer

log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


class OllamaBackend:
    name = "ollama"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        self._client = OpenAI(
            base_url=f"{self.cfg.ollama_base_url.rstrip('/')}/v1",
            api_key="ollama",  # unused by Ollama, required by the SDK
            timeout=self.cfg.request_timeout_s,
            max_retries=0,
        )
        # No provider quota locally, but the same limiter stays in the path so
        # the two backends have identical timing behaviour under test.
        self.limiter = RateLimiter(max(self.cfg.rate_limit_rpm, 600))
        self.cache = CallCache(self.cfg.cache_dir, enabled=self.cfg.cache_enabled)
        self.tracer = get_tracer("filing.llm")
        self.http_calls = 0
        self._warned_rerank = False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "chat",
        model_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        spec = model_for(role, "ollama")
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
            text = self._chat_call(mid, role, payload)
            self.cache.set(key, text)
            return text

    @retry(**_RETRY)
    def _chat_call(self, model_id: str, role: str, payload: dict[str, Any]) -> str:
        self.limiter.acquire()
        self.http_calls += 1
        try:
            resp = self._client.chat.completions.create(model=model_id, **payload)
        except NotFoundError as exc:
            raise ModelUnavailable(role, model_id, (), f"run: ollama pull {model_id}") from exc
        return (resp.choices[0].message.content or "").strip()

    def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType = "passage",
        model_id: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        spec = model_for("embed", "ollama")
        mid = model_id or spec.id
        with self.tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("openinference.span.kind", "EMBEDDING")
            span.set_attribute("llm.model_name", mid)
            span.set_attribute("filing.batch_size", len(texts))
            keys = [
                make_key(backend=self.name, model=mid, kind="embed", payload={"text": t})
                for t in texts
            ]
            out: list[list[float] | None] = [self.cache.get(k) for k in keys]
            missing = [i for i, v in enumerate(out) if v is None]
            span.set_attribute("filing.cache_hits", len(texts) - len(missing))
            if missing:
                vectors = self._embed_call(mid, [texts[i] for i in missing])
                for i, vec in zip(missing, vectors, strict=True):
                    out[i] = vec
                    self.cache.set(keys[i], vec)
            return [v for v in out if v is not None]

    @retry(**_RETRY)
    def _embed_call(self, model_id: str, texts: list[str]) -> list[list[float]]:
        self.limiter.acquire()
        self.http_calls += 1
        try:
            resp = self._client.embeddings.create(model=model_id, input=texts)
        except NotFoundError as exc:
            raise ModelUnavailable("embed", model_id, (), f"run: ollama pull {model_id}") from exc
        return [list(d.embedding) for d in sorted(resp.data, key=lambda d: d.index)]

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
        if not self._warned_rerank:
            log.warning(
                "ollama backend has no cross-encoder; reranking by embedding cosine. "
                "Do not report retrieval numbers produced this way."
            )
            self._warned_rerank = True
        with self.tracer.start_as_current_span("llm.rerank") as span:
            span.set_attribute("openinference.span.kind", "RERANKER")
            span.set_attribute(
                "reranker.model_name", f"{model_for('rerank', 'ollama').id} (cosine)"
            )
            span.set_attribute("reranker.query", query[:500])
            span.set_attribute("filing.degraded", True)
            qv = self.embed([query], input_type="query", model_id=model_id)[0]
            pvs = self.embed(passages, model_id=model_id)
            ranked = sorted(
                (Ranking(index=i, score=_cosine(qv, pv)) for i, pv in enumerate(pvs)),
                key=lambda r: r.score,
                reverse=True,
            )
            return ranked[:top_n] if top_n is not None else ranked

    def usage(self) -> Usage:
        return Usage(http_calls=self.http_calls, cache_hits=self.cache.hits)

    def close(self) -> None:
        self.cache.close()
