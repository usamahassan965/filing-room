"""The M0 gate, and the thing it could not previously see.

`filing smoke` used to pass with `http calls: 0, cache hits: 4` -- the second run
of the day replayed the first one's answers, so a revoked key, a retired model ID
or an exhausted free tier still printed 5/5. The one failure a smoke test exists
to catch was the one failure it was blind to.

These tests hold the fix in place from both sides: the marked payload must reach
the backend (the gate does real work), and a backend that serves the first chat
from cache must fail the gate rather than pass it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from filing.cli import app
from filing.llm.base import Ranking, Usage

runner = CliRunner()


@dataclass
class Spec:
    id: str
    dim: int | None = None


class StubBackend:
    """A backend whose cache behaviour is dialled in by the test.

    ``cached_kinds`` names the verbs that answer from cache. That is the whole
    point of the double: a real warm cache is exactly what the gate has to stop
    accepting as proof of life.
    """

    name = "stub"

    def __init__(self, cached_kinds: frozenset[str] = frozenset()) -> None:
        self.cached = cached_kinds
        self.http_calls = 0
        self.cache_hits = 0
        self.chats: list[list[dict[str, str]]] = []
        self.embedded: list[str] = []
        self.seen: dict[str, str] = {}

    def _serve(self, kind: str, payload: str) -> bool:
        """True when this call is a cache hit -- either forced, or a real repeat."""
        if kind in self.cached or self.seen.get(kind) == payload:
            self.cache_hits += 1
            return True
        self.seen[kind] = payload
        self.http_calls += 1
        return False

    def chat(self, messages, *, role="chat", max_tokens=1024, **kw):  # noqa: ANN001, ARG002
        self.chats.append(messages)
        self._serve("chat", str(messages))
        return "A 10-K is an annual report."

    def embed(self, texts, *, input_type="passage", **kw):  # noqa: ANN001, ARG002
        self.embedded.extend(texts)
        self._serve("embed", str(texts))
        return [[0.0] * 8 for _ in texts]

    def rerank(self, query, passages, *, top_n=None, **kw):  # noqa: ANN001, ARG002
        return [Ranking(index=i, score=1.0 - i / 10) for i in (1, 2, 0)][: top_n or 3]

    def usage(self) -> Usage:
        return Usage(http_calls=self.http_calls, cache_hits=self.cache_hits)


@pytest.fixture
def stub(monkeypatch):
    """Install a StubBackend and neutralise everything the gate does around it."""

    def install(cached_kinds: frozenset[str] = frozenset()) -> StubBackend:
        backend = StubBackend(cached_kinds)
        monkeypatch.setattr("filing.cli.build_backend", lambda *a, **k: backend)
        monkeypatch.setattr("filing.cli.model_for", lambda role, backend: Spec(f"stub-{role}"))
        monkeypatch.setattr("filing.cli.setup_tracing", lambda cfg: True)
        monkeypatch.setattr("filing.cli.flush_tracing", lambda: True)
        monkeypatch.setattr("filing.cli.span_count", lambda: 3)
        monkeypatch.setattr("filing.cli.span_names", lambda: ["llm.chat", "llm.embed"])
        return backend

    return install


def test_the_gate_passes_when_the_calls_are_live(stub):
    backend = stub()
    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 0, result.output
    assert "5/5 passed" in result.output
    assert backend.http_calls == 2  # one chat, one embed; the repeat is the cache check


def test_a_cached_first_chat_fails_the_gate(stub):
    """The regression. A warm cache used to be indistinguishable from a live key."""
    stub(frozenset({"chat"}))
    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 1
    assert "CACHED -- not a check" in result.output


def test_a_cached_first_embed_fails_the_gate(stub):
    stub(frozenset({"embed"}))
    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 1
    assert "live=False" in result.output


def test_every_run_sends_a_payload_no_other_run_sent(stub):
    """Two runs in one process must not collide, or run two grades run one."""
    first = stub()
    runner.invoke(app, ["smoke"])
    second = stub()
    runner.invoke(app, ["smoke"])

    assert first.chats[0] != second.chats[0]
    assert first.embedded[0] != second.embedded[0]


def test_the_dedup_check_repeats_the_marked_prompt(stub):
    """Otherwise check 4 proves dedup on some other run's entry, not on this one's."""
    backend = stub()
    runner.invoke(app, ["smoke"])

    assert len(backend.chats) == 2
    assert backend.chats[0] == backend.chats[1]
    assert backend.cache_hits == 1  # the repeat, and only the repeat
