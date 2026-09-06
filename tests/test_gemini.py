"""The Gemini backend's three provider-specific hazards, asserted offline.

None of these need a key or a network. They exist because each one fails
silently in production: a dropped taskType degrades retrieval without erroring,
an unnormalised vector makes cosine similarity stop meaning cosine similarity,
and counting a local rerank as an HTTP call makes the budget numbers lie.
"""

from __future__ import annotations

import math

import pytest

from filing.config import Settings, model_for
from filing.llm.base import Ranking
from filing.llm.errors import MissingCredentials
from filing.llm.gemini import GeminiBackend


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def backend(tmp_path):
    cfg = Settings(
        gemini_api_key="test-key",
        llm_backend="gemini",
        cache_dir=tmp_path / "llm",
        cache_enabled=True,
        tracing_enabled=False,
    )
    b = GeminiBackend(cfg)
    yield b
    b.close()


def _stub_embeddings(backend, vectors: list[list[float]]) -> list[dict]:
    """Capture request bodies; reply with `vectors`."""
    sent: list[dict] = []

    def fake_post(url, json=None, **kw):  # noqa: A002 - httpx's parameter name
        sent.append({"url": url, "body": json})
        return _FakeResponse({"embeddings": [{"values": v} for v in vectors]})

    backend._http.post = fake_post  # noqa: SLF001 - deliberate seam
    return sent


def test_embed_sends_gemini_task_type_not_openai_input_type(backend):
    """The concept is provider-independent; the vocabulary is not.

    Google's OpenAI shim drops fields it does not recognise without complaining,
    so `input_type` would vanish and every query would be embedded as a
    document. Hence the native endpoint and the explicit translation.
    """
    sent = _stub_embeddings(backend, [[1.0, 0.0]])

    backend.embed(["what is a 10-K?"], input_type="query")
    backend.embed(["Item 1A. Risk Factors"], input_type="passage")

    assert sent[0]["body"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert sent[1]["body"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert sent[0]["url"].endswith(":batchEmbedContents")


def test_embed_requests_the_registered_width(backend):
    sent = _stub_embeddings(backend, [[1.0, 0.0]])
    backend.embed(["x"])
    request = sent[0]["body"]["requests"][0]
    assert request["outputDimensionality"] == model_for("embed", "gemini").dim


def test_truncated_embeddings_are_renormalised(backend):
    """gemini-embedding-001 is unit-norm only at its full width.

    Ask for fewer dimensions and you get a Matryoshka prefix whose norm is less
    than one. Feed that to a cosine index and the scores are quietly wrong.
    """
    _stub_embeddings(backend, [[3.0, 4.0]])  # norm 5, nowhere near unit
    vec = backend.embed(["x"])[0]
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)
    assert math.isclose(vec[0], 0.6) and math.isclose(vec[1], 0.8)


def test_query_and_passage_embeddings_are_cached_separately(backend):
    """Same text, different task type, different vector -- so different key."""
    calls = _stub_embeddings(backend, [[1.0, 0.0]])
    backend.embed(["supply chain"], input_type="query")
    backend.embed(["supply chain"], input_type="passage")
    assert len(calls) == 2, "the two task types collided in the cache"
    backend.embed(["supply chain"], input_type="query")
    assert len(calls) == 2, "a repeat should have been served from cache"


def test_local_rerank_is_not_counted_as_network_traffic(backend):
    """It never leaves the machine, so it must not appear in the HTTP budget."""

    class _FakeReranker:
        calls = 0

        def rank(self, query, passages):
            type(self).calls += 1
            return [Ranking(index=i, score=float(len(passages) - i)) for i in range(len(passages))]

    backend._reranker = _FakeReranker()  # noqa: SLF001
    before = backend.http_calls

    ranked = backend.rerank("supply chain risk", ["a", "b", "c"], top_n=2)

    assert [r.index for r in ranked] == [0, 1]
    assert backend.http_calls == before, "a local forward pass was billed as an HTTP call"
    assert backend.usage().http_calls == before

    backend.rerank("supply chain risk", ["a", "b", "c"], top_n=2)
    assert _FakeReranker.calls == 1, "the second rerank should have hit the cache"


def test_each_model_gets_its_own_limiter(backend):
    """A shared limiter would throttle embeddings to the chat model's 10 rpm."""
    chat = model_for("chat", "gemini")
    embed = model_for("embed", "gemini")

    chat_limiter = backend._limiter(chat.id, chat)  # noqa: SLF001
    embed_limiter = backend._limiter(embed.id, embed)  # noqa: SLF001

    assert chat_limiter is not embed_limiter
    assert chat_limiter.rpm == chat.rpm
    assert embed_limiter.rpm == embed.rpm
    assert embed_limiter.rpm > chat_limiter.rpm
    # And the same model asks twice for the same limiter, or the window resets.
    assert backend._limiter(chat.id, chat) is chat_limiter  # noqa: SLF001


def test_missing_key_names_the_page_that_issues_one(tmp_path):
    with pytest.raises(MissingCredentials) as exc:
        GeminiBackend(Settings(gemini_api_key="", cache_dir=tmp_path, tracing_enabled=False))
    assert "aistudio.google.com" in str(exc.value)


def test_spans_declare_a_kind_phoenix_understands(backend, monkeypatch):
    """Untyped spans render as UNKNOWN, and an unreadable trace is a dead feature.

    Tracing is the transparency story this project is built to show, so the span
    kinds are asserted rather than eyeballed in the UI once and forgotten.
    """
    seen: dict[str, dict[str, object]] = {}

    class _Span:
        def __init__(self, name: str) -> None:
            self.attrs = seen.setdefault(name, {})

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(backend.tracer, "start_as_current_span", lambda n, **kw: _Span(n))
    _stub_embeddings(backend, [[1.0, 0.0]])
    backend._reranker = type("R", (), {"rank": lambda self, q, p: []})()  # noqa: SLF001

    backend.embed(["x"])
    backend.rerank("q", ["a"])

    assert seen["llm.embed"]["openinference.span.kind"] == "EMBEDDING"
    assert seen["llm.rerank"]["openinference.span.kind"] == "RERANKER"
