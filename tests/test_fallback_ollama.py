"""The escape hatch's contract, asserted without Ollama running.

This backend exists to be switched on during an outage, which is the worst
possible time to discover it drifted out of the interface. Everything here runs
offline against a fake OpenAI-compatible server, so the contract is checked on
every commit rather than on the day the quota runs out.

What this file cannot prove is that Ollama itself answers -- that is what
``filing smoke --backend ollama`` is for, and the two checks are not
substitutes. This one catches drift in our code; that one catches a model that
was never pulled.
"""

from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from filing.config import Settings, model_for
from filing.llm.base import LLMBackend, Ranking
from filing.llm.errors import ModelUnavailable
from filing.llm.fallback_ollama import OllamaBackend


@pytest.fixture
def cfg(tmp_path):
    return Settings(
        llm_backend="ollama",
        cache_dir=tmp_path / "llm",
        cache_enabled=True,
        tracing_enabled=False,
    )


def backend_with(cfg, handler) -> OllamaBackend:
    """An OllamaBackend whose transport is a function, not a socket.

    The seam is the http client rather than the SDK method, on purpose: the
    real ``openai`` client still parses the response and still raises
    ``NotFoundError`` from a 404, which is the exact conversion
    ``ModelUnavailable`` depends on.
    """
    b = OllamaBackend(cfg)
    b._client = OpenAI(  # noqa: SLF001 - deliberate seam
        base_url=f"{cfg.ollama_base_url.rstrip('/')}/v1",
        api_key="ollama",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return b


def chat_reply(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "llama3.2:3b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def embed_reply(vectors: list[list[float]], *, reverse: bool = False) -> httpx.Response:
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "model": "nomic-embed-text",
            "data": list(reversed(data)) if reverse else data,
        },
    )


# ------------------------------------------------------------------ interface


def test_the_backend_still_implements_the_interface(cfg):
    """The drift check.

    A verb added to ``LLMBackend`` for the Gemini path and not added here would
    otherwise surface as an AttributeError during an outage.
    """
    assert isinstance(OllamaBackend(cfg), LLMBackend)


def test_nothing_leaves_the_machine(cfg):
    """The whole point of the fallback is that it needs no provider."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return chat_reply("a 10-K is an annual report")

    backend_with(cfg, handler).chat([{"role": "user", "content": "hi"}], role="chat_fast")
    assert seen == ["http://localhost:11434/v1/chat/completions"]


# ---------------------------------------------------------------------- chat


def test_chat_returns_the_message_text(cfg):
    b = backend_with(cfg, lambda r: chat_reply("  an annual report  "))
    assert b.chat([{"role": "user", "content": "hi"}]) == "an annual report"


def test_the_same_prompt_does_not_reach_the_network_twice(cfg):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return chat_reply("cached me")

    b = backend_with(cfg, handler)
    msgs = [{"role": "user", "content": "hi"}]
    first = b.chat(msgs, role="chat_fast")
    second = b.chat(msgs, role="chat_fast")

    assert first == second
    assert len(calls) == 1
    assert b.usage().http_calls == 1


def test_a_model_that_was_never_pulled_says_how_to_pull_it(cfg):
    """Ollama's 404 for an absent model is indistinguishable from a retired
    hosted ID unless the error says so. The fix here is a shell command, not a
    registry edit, so the message carries that instead."""
    b = backend_with(cfg, lambda r: httpx.Response(404, json={"error": {"message": "not found"}}))
    with pytest.raises(ModelUnavailable) as exc:
        b.chat([{"role": "user", "content": "hi"}], role="chat_fast")
    assert f"ollama pull {model_for('chat_fast', 'ollama').id}" in str(exc.value)


# --------------------------------------------------------------------- embed


def test_embeddings_come_back_in_submitted_order(cfg):
    """The API returns an ``index`` per row and is not obliged to return them
    sorted. Trusting arrival order would pair every chunk with another chunk's
    vector -- a corpus-wide corruption that raises nothing and shows up only as
    bad retrieval."""
    b = backend_with(cfg, lambda r: embed_reply([[1.0, 0.0], [0.0, 1.0]], reverse=True))
    assert b.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_of_nothing_calls_nothing(cfg):
    b = backend_with(cfg, lambda r: pytest.fail("should not have been called"))
    assert b.embed([]) == []


def test_only_uncached_texts_are_sent(cfg):
    """Indexing re-runs are the expensive case; this is what makes them cheap."""
    sent: list[list[str]] = []

    def handler(request):
        texts = json.loads(request.content)["input"]
        sent.append(texts)
        return embed_reply([[float(len(t)), 0.0] for t in texts])

    b = backend_with(cfg, handler)
    b.embed(["alpha"])
    b.embed(["alpha", "beta"])

    assert sent == [["alpha"], ["beta"]]  # the second call asked for one text


# -------------------------------------------------------------------- rerank


def test_rerank_orders_by_cosine_and_honours_top_n(cfg):
    """The stand-in has to be *right*, not merely present: a reranker that
    returns the input order is worse than no reranker, because it looks like
    one in the trace."""
    vectors = {
        "query": [1.0, 0.0],
        "far": [0.0, 1.0],
        "near": [0.9, 0.1],
        "middling": [0.6, 0.6],
    }

    def handler(request):
        texts = json.loads(request.content)["input"]
        return embed_reply([vectors[t] for t in texts])

    b = backend_with(cfg, handler)
    ranked = b.rerank("query", ["far", "near", "middling"], top_n=2)

    assert [r.index for r in ranked] == [1, 2]
    assert isinstance(ranked[0], Ranking)
    assert ranked[0].score > ranked[1].score


def test_rerank_of_nothing_is_empty(cfg):
    b = backend_with(cfg, lambda r: pytest.fail("should not have been called"))
    assert b.rerank("q", []) == []


def test_the_degraded_rerank_warns_once(cfg, caplog):
    """Once, not per call: a warning on every rerank in an eval sweep is a
    warning nobody reads."""
    b = backend_with(cfg, lambda r: embed_reply([[1.0, 0.0]]))
    with caplog.at_level("WARNING"):
        b.rerank("q", ["a"])
        b.rerank("q", ["a"])
    assert sum("no cross-encoder" in m for m in caplog.messages) == 1


# ------------------------------------------------------- what a swap costs


def test_the_fallback_is_not_index_compatible_with_gemini():
    """Encoded here because prose in a risk register does not fail a build.

    Flipping ``LLM_BACKEND`` keeps chat working and cannot keep retrieval
    working: these two models emit different widths, and even at equal width
    they would be different vector spaces, where cosine is meaningless. A
    backend swap for retrieval means re-embedding into that backend's own
    collection -- so the store keys its collection by backend, model and dim,
    and a mismatch is a startup error rather than bad neighbours.
    """
    assert model_for("embed", "gemini").dim != model_for("embed", "ollama").dim
