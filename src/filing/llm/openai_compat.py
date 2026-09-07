"""One backend for every provider that speaks OpenAI.

The registry indirection this project has carried since M0 was justified on the
claim that swapping providers is an entry, not a refactor. This file is where
that claim gets tested: Cohere, Groq and OVHcloud are three different companies
on three continents with three different business models, and all three are
reachable through ``OpenAI(base_url=...)``. What differs between them is a URL,
a credential and a rate limit -- which is to say, a registry entry.

So there is no CohereBackend and no GroqBackend. There is one class, keyed by
provider name, and adding a fourth provider is a dict entry plus two lines in
``Settings``.

The two places providers genuinely differ, both learned by probing rather than
by reading docs:

  * **Groq sits behind Cloudflare bot protection.** A default Python
    ``User-Agent`` earns HTTP 403 "Error 1010: access denied based on your
    browser's signature" *before* the key is ever checked -- which reads exactly
    like a bad credential and is not one. It wants a browser UA string.
  * **Nobody reranks here.** Cohere serves a reranker, and a good one, but
    wiring it would confound the comparison: the point of these configs is to
    change the *generator* and hold retrieval fixed. So reranking runs on the
    same local cross-encoder every other backend uses.

Embeddings go through the standard ``/embeddings`` route. They are not used by
the eval configs -- ``EMBED_BACKEND=local`` owns the vector space and the Qdrant
collection is named after it -- but a backend that could not embed would be a
backend this project's own interface does not accept.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from openai import NotFoundError, OpenAI
from tenacity import retry

from filing.config import Backend, ModelSpec, Settings, model_for, settings
from filing.llm.base import InputType, Ranking, Usage
from filing.llm.cache import CallCache, make_key
from filing.llm.errors import MissingCredentials, ModelUnavailable
from filing.llm.limiter import RateLimiter
from filing.llm.rerank_local import LocalReranker
from filing.llm.retry import HOSTED as _RETRY
from filing.tracing import get_tracer

log = logging.getLogger(__name__)

# Cloudflare fingerprints the client before the origin sees the request, so this
# is not politeness -- it is the difference between 200 and 403 on Groq.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _strip_authorization(request: httpx.Request) -> None:
    """Remove the Authorization header entirely before the request goes out.

    An absent header and an empty one are not the same thing. OVHcloud serves
    anonymous requests -- no header at all is answered normally, and throttled
    with 429 when the shared per-IP quota runs out. Send it a placeholder
    instead and it stops reading the request as anonymous and starts reading it
    as a *failed* credential: `403 Forbidden: authentication failed`, on every
    question, which looks exactly like a key problem and is the opposite of one.

    The OpenAI SDK requires an api_key string and turns it into a header
    unconditionally, so the only place to undo it is on the way to the socket.
    """
    request.headers.pop("authorization", None)


def _anonymous_client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, event_hooks={"request": [_strip_authorization]})


class OpenAICompatBackend:
    """A hosted OpenAI-compatible provider, named by ``provider``."""

    def __init__(self, provider: Backend, cfg: Settings | None = None) -> None:
        self.name = provider
        self.cfg = cfg or settings()
        base_url, secret, where = self.cfg.provider_credentials(provider)
        if where and not secret:
            raise MissingCredentials(
                f"{provider.upper()}_API_KEY is not set. Paste a key from {where} "
                "into .env -- the line is already there, under 'provider bake-off'."
            )
        self._client = OpenAI(
            base_url=base_url,
            # The SDK insists on a non-empty string even where the provider
            # wants no credential at all. For those, the placeholder never
            # reaches the wire -- see _strip_authorization.
            api_key=secret or "anonymous",
            timeout=self.cfg.request_timeout_s,
            # max_retries=0: tenacity owns retries, so every retry also passes
            # the limiter. The SDK retrying behind our back spends unbudgeted
            # quota, which on an 8,000-token-per-minute tier is the whole budget.
            max_retries=0,
            default_headers={"User-Agent": _BROWSER_UA},
            http_client=_anonymous_client(self.cfg.request_timeout_s) if not where else None,
        )
        self._limiters: dict[str, RateLimiter] = {}
        self.cache = CallCache(self.cfg.cache_dir, enabled=self.cfg.cache_enabled)
        self.tracer = get_tracer("filing.llm")
        self.http_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

        rspec = model_for("rerank", provider)
        self._reranker = LocalReranker(
            rspec.id, device=self.cfg.rerank_device, batch_size=self.cfg.rerank_batch_size
        )

    # ---------------------------------------------------------------- limiting

    def _limiter(self, model_id: str, spec: ModelSpec) -> RateLimiter:
        if model_id not in self._limiters:
            self._limiters[model_id] = RateLimiter(spec.rpm or self.cfg.rate_limit_rpm)
        return self._limiters[model_id]

    @property
    def limiter(self) -> RateLimiter:
        spec = model_for("chat", self.name)
        return self._limiter(spec.id, spec)

    # -------------------------------------------------------------------- chat

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "chat",
        model_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        spec = model_for(role, self.name)
        mid = model_id or spec.id
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key = make_key(backend=self.name, model=mid, kind="chat", payload=payload)

        with self.tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.model_name", mid)
            span.set_attribute("llm.provider", self.name)
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
        choice = resp.choices[0].message
        # A reasoning model can spend the whole max_tokens budget thinking and
        # return content="" with the answer stranded in `reasoning`. Falling
        # back to it is more honest than scoring the model as having said
        # nothing at all.
        text = choice.content or getattr(choice, "reasoning", None) or ""
        return text.strip()

    # --------------------------------------------------------------- embedding

    def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType = "passage",
        model_id: str | None = None,
    ) -> list[list[float]]:
        spec = model_for("embed", self.name)
        mid = model_id or spec.id
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(
                make_key(
                    backend=self.name,
                    model=mid,
                    kind="embed",
                    payload={"text": t, "input_type": input_type},
                )
            )
            if hit is None:
                missing.append(i)
            else:
                out[i] = hit

        with self.tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("llm.model_name", mid)
            span.set_attribute("filing.cache_hit", not missing)
            if missing:
                vectors = self._embed_call(mid, spec, [texts[i] for i in missing], input_type)
                for i, vec in zip(missing, vectors, strict=True):
                    out[i] = vec
                    self.cache.set(
                        make_key(
                            backend=self.name,
                            model=mid,
                            kind="embed",
                            payload={"text": texts[i], "input_type": input_type},
                        ),
                        vec,
                    )
        return [v for v in out if v is not None]

    @retry(**_RETRY)
    def _embed_call(
        self, model_id: str, spec: ModelSpec, texts: list[str], input_type: InputType
    ) -> list[list[float]]:
        self._limiter(model_id, spec).acquire()
        self.http_calls += 1
        try:
            resp = self._client.embeddings.create(model=model_id, input=texts)
        except NotFoundError as exc:
            raise ModelUnavailable("embed", model_id, spec.alternates, str(exc)) from exc
        if resp.usage:
            self._prompt_tokens += resp.usage.prompt_tokens or 0
        return [d.embedding for d in resp.data]

    # ------------------------------------------------------------------ rerank

    def rerank(self, query: str, passages: list[str], *, top_n: int | None = None) -> list[Ranking]:
        """Local cross-encoder, deliberately -- see the module docstring."""
        spec = model_for("rerank", self.name)
        mid = spec.id
        key = make_key(
            backend="local",
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
                rankings = self._reranker.rank(query, passages)
                self.cache.set(key, [{"index": r.index, "score": r.score} for r in rankings])
            return rankings[:top_n] if top_n is not None else rankings

    # -------------------------------------------------------------------- misc

    def usage(self) -> Usage:
        return Usage(
            http_calls=self.http_calls,
            cache_hits=self.cache.hits,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

    def close(self) -> None:
        self.cache.close()
