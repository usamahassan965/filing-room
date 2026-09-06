"""NVIDIA NIM backend.

Every model call in this project goes through this class -- no exceptions, and
that rule is the whole design. It is the only place that knows about the rate
limit, the retry policy, the cache, and the span boundaries, so those four
things are impossible to forget at a call site.

Shape of the provider, which is why the code looks like it does:
  * chat and embeddings are OpenAI-compatible at integrate.api.nvidia.com/v1
  * embeddings are *asymmetric* -- input_type must say query or passage
  * reranking is not OpenAI-shaped at all: own host, own body, own response
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from openai import NotFoundError, OpenAI
from tenacity import retry

from filing.config import Settings, model_for, settings
from filing.llm.base import InputType, Ranking, Usage
from filing.llm.cache import CallCache, make_key
from filing.llm.errors import MissingCredentials, ModelUnavailable
from filing.llm.limiter import RateLimiter
from filing.llm.retry import HOSTED as _RETRY
from filing.tracing import get_tracer

log = logging.getLogger(__name__)


class NvidiaBackend:
    name = "nvidia"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        if not self.cfg.nvidia_api_key:
            raise MissingCredentials(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and paste a key "
                "from https://build.nvidia.com (any model page -> Get API Key)."
            )
        # max_retries=0: tenacity owns the retry policy, so every retry also
        # passes through the rate limiter. The SDK retrying behind our back
        # would spend budget we never accounted for.
        self._client = OpenAI(
            base_url=self.cfg.nvidia_base_url,
            api_key=self.cfg.nvidia_api_key.get_secret_value(),
            timeout=self.cfg.request_timeout_s,
            max_retries=0,
        )
        self._http = httpx.Client(timeout=self.cfg.request_timeout_s)
        self.limiter = RateLimiter(self.cfg.rate_limit_rpm)
        self.cache = CallCache(self.cfg.cache_dir, enabled=self.cfg.cache_enabled)
        self.tracer = get_tracer("filing.llm")
        self.http_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

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
        spec = model_for(role, "nvidia")
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

            text = self._chat_call(mid, role, spec.alternates, payload)
            self.cache.set(key, text)
            return text

    @retry(**_RETRY)
    def _chat_call(
        self, model_id: str, role: str, alternates: tuple[str, ...], payload: dict[str, Any]
    ) -> str:
        self.limiter.acquire()
        self.http_calls += 1
        try:
            resp = self._client.chat.completions.create(model=model_id, **payload)
        except NotFoundError as exc:
            raise ModelUnavailable(role, model_id, alternates, str(exc)) from exc
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
        spec = model_for("embed", "nvidia")
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
                    payload={"text": t, "input_type": input_type},
                )
                for t in texts
            ]
            out: list[list[float] | None] = [self.cache.get(k) for k in keys]
            missing = [i for i, v in enumerate(out) if v is None]
            span.set_attribute("filing.cache_hits", len(texts) - len(missing))

            if missing:
                vectors = self._embed_call(
                    mid, spec.alternates, [texts[i] for i in missing], input_type
                )
                for i, vec in zip(missing, vectors, strict=True):
                    out[i] = vec
                    self.cache.set(keys[i], vec)

            return [v for v in out if v is not None]

    @retry(**_RETRY)
    def _embed_call(
        self,
        model_id: str,
        alternates: tuple[str, ...],
        texts: list[str],
        input_type: InputType,
    ) -> list[list[float]]:
        self.limiter.acquire()
        self.http_calls += 1
        try:
            resp = self._client.embeddings.create(
                model=model_id,
                input=texts,
                encoding_format="float",
                # input_type is not an OpenAI parameter; NIM reads it off the body.
                extra_body={"input_type": input_type, "truncate": "END"},
            )
        except NotFoundError as exc:
            raise ModelUnavailable("embed", model_id, alternates, str(exc)) from exc
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]

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
        spec = model_for("rerank", "nvidia")
        mid = model_id or spec.id
        payload = {"query": query, "passages": passages}
        key = make_key(backend=self.name, model=mid, kind="rerank", payload=payload)

        with self.tracer.start_as_current_span("llm.rerank") as span:
            # The OpenAI instrumentor cannot see this call -- it is raw httpx --
            # so the reranker span is annotated by hand, with the OpenInference
            # attribute names Phoenix expects.
            span.set_attribute("openinference.span.kind", "RERANKER")
            span.set_attribute("reranker.model_name", mid)
            span.set_attribute("reranker.query", query[:500])
            span.set_attribute("reranker.top_k", top_n or len(passages))
            cached = self.cache.get(key)
            span.set_attribute("filing.cache_hit", cached is not None)
            if cached is not None:
                rankings = [Ranking(index=r["index"], score=r["score"]) for r in cached]
            else:
                rankings = self._rerank_call(
                    mid, spec.endpoint_for(mid), spec.alternates, query, passages
                )
                self.cache.set(key, [{"index": r.index, "score": r.score} for r in rankings])
            return rankings[:top_n] if top_n is not None else rankings

    @retry(**_RETRY)
    def _rerank_call(
        self,
        model_id: str,
        endpoint: str | None,
        alternates: tuple[str, ...],
        query: str,
        passages: list[str],
    ) -> list[Ranking]:
        if not endpoint:  # pragma: no cover - registry invariant
            raise ValueError(f"rerank spec {model_id!r} has no endpoint")
        self.limiter.acquire()
        self.http_calls += 1
        resp = self._http.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.cfg.nvidia_api_key.get_secret_value()}",
                "Accept": "application/json",
            },
            json={
                "model": model_id,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
                "truncate": "END",
            },
        )
        if resp.status_code == 404:
            raise ModelUnavailable("rerank", model_id, alternates, resp.text[:300])
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
        # NIM returns them sorted already, but not depending on that is free.
        return sorted(
            (Ranking(index=int(r["index"]), score=float(r["logit"])) for r in rankings),
            key=lambda r: r.score,
            reverse=True,
        )

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
