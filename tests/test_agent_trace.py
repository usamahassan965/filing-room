"""The exported trace, and the two ways an export like this is usually a lie.

The gate asks for "one full trace exported to docs/trace_example.json". The easy
way to satisfy it is to open the Phoenix UI, press export, and commit whatever
comes out. That file would be a screenshot in JSON's clothing: no one could
regenerate it, and if the graph stopped emitting spans tomorrow nothing in the
repository would notice.

So the export is code, and code can be wrong in two quiet ways that these tests
exist to catch. The first is capturing nothing and reporting success -- spans go
to a no-op tracer, the file is written with an empty span list, and the run
looks clean. The second is capturing everything forever: a recorder attached and
never detached would hold all ten spans of every one of 150 questions to write
one file. Both are tested here by observation rather than by inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from filing.agent.nodes import Tools
from filing.agent.trace import SpanRecorder, capture_question, to_tree, write_trace

# ------------------------------------------------------------------- doubles


@dataclass
class Usage:
    http_calls: int = 0


class StubBackend:
    """Answers a plan and then a synthesis, so one question runs end to end."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, *, role="chat", temperature=0.0, max_tokens=512, **kw):  # noqa: ANN001, ARG002
        self.calls += 1
        return self.replies[(self.calls - 1) % len(self.replies)]

    def rerank(self, query, passages, *, top_n=None, **kw):  # noqa: ANN001, ARG002
        return [_Rank(i, 5.0 - i) for i in range(len(passages))][: top_n or 5]

    def usage(self) -> Usage:
        return Usage(http_calls=self.calls)


@dataclass
class _Rank:
    index: int
    score: float


@dataclass
class FakeChunk:
    chunk_id: str = "c1"
    ticker: str = "NVDA"
    form: str = "10-K"
    period_end: str = "2024-01-28"
    accn: str = "0000000000-00-000000"
    text: str = "Revenue rose on data centre demand. " * 30
    section: str = "Item 7"
    char_start: int = 0
    char_end: int = 100


@dataclass
class FakeHit:
    chunk: Any
    fused_score: float = 0.016
    rerank_score: Any = None

    @property
    def citation(self) -> str:
        return f"{self.chunk.ticker} {self.chunk.form} {self.chunk.period_end}"


class FakeRetriever:
    def search(self, query, *, k=50, where=None, rerank=True, top_n=5):  # noqa: ANN001, ARG002
        return [FakeHit(FakeChunk())]


def plan(route: str, **kw: str) -> str:
    return json.dumps({"route": route, **kw})


@pytest.fixture
def tools() -> Tools:
    backend = StubBackend([plan("text", ticker="NVDA"), "Data centre demand rose [1]"])
    return Tools(backend=backend, retriever=FakeRetriever())


def span(name: str, sid: str, parent: str = "", start: int = 0, **kw: Any) -> dict[str, Any]:
    base = {
        "name": name,
        "span_id": sid,
        "parent_id": parent,
        "trace_id": "t",
        "start_ns": start,
        "ms": 1.0,
        "status": "UNSET",
        "attributes": {},
    }
    return base | kw


# -------------------------------------------------------------------- to_tree


def test_a_child_span_nests_under_its_parent():
    tree = to_tree([span("agent.plan", "b", "a", 1), span("agent.question", "a", "", 0)])
    assert [n["name"] for n in tree] == ["agent.question"]
    assert [c["name"] for c in tree[0]["children"]] == ["agent.plan"]


def test_siblings_come_back_in_start_order_not_completion_order():
    """The reason the export is a tree at all.

    Spans reach ``on_end`` when they *finish*, so a flat dump of a nested trace
    puts the innermost span first and the node containing it last. A reader
    seeing `synthesise` above `question` would reasonably conclude the graph ran
    backwards.
    """
    root = span("agent.question", "a", "", 0)
    late = span("agent.synthesise", "c", "a", 9)
    early = span("agent.plan", "b", "a", 1)
    tree = to_tree([late, early, root])  # completion order: innermost first
    assert [c["name"] for c in tree[0]["children"]] == ["agent.plan", "agent.synthesise"]


def test_the_ids_are_spent_on_the_shape_and_then_dropped():
    """They exist to rebuild the nesting; keeping them puts thirty-two
    characters of hex in front of every line of a file meant to be read."""
    node = to_tree([span("agent.plan", "b", "", 0, attributes={"route": "text"})])[0]
    assert set(node) == {"name", "ms", "status", "attributes"}
    assert node["attributes"] == {"route": "text"}


def test_an_orphan_span_is_still_reported_rather_than_dropped():
    """A parent outside the prefix filter is normal, not a reason to lose a node."""
    tree = to_tree([span("agent.plan", "b", "missing", 0)])
    assert [n["name"] for n in tree] == ["agent.plan"]


# -------------------------------------------------------------- the recorder


def test_the_recorder_keeps_only_the_names_it_was_asked_for():
    rec = SpanRecorder(prefix="agent.")

    class S:
        def __init__(self, name: str) -> None:
            self.name = name

    rec.on_end(S("agent.plan"))
    rec.on_end(S("openai.chat"))
    assert [s.name for s in rec.spans] == ["agent.plan"]


# ---------------------------------------------------------- capture_question


def test_a_capture_records_the_nodes_the_graph_actually_ran(tools):
    """The test that fails if the tracer binding regresses.

    ``Tools.tracer`` is resolved by a default_factory when the tools are built,
    which on a machine with no collector is before any provider exists. Capture
    without rebinding it and the nodes write to the no-op global tracer: the
    call succeeds, the file is written, and the span list is empty.
    """
    state, spans = capture_question("What drove NVDA revenue?", tools=tools, qid="nar-001")
    names = {s["name"] for s in spans}
    assert {"agent.question", "agent.plan", "agent.route", "agent.retrieve.text"} <= names
    assert state["answer"]


def test_the_node_spans_hang_off_the_question_span(tools):
    _, spans = capture_question("What drove NVDA revenue?", tools=tools)
    tree = to_tree(spans)
    assert [n["name"] for n in tree] == ["agent.question"]
    assert len(tree[0]["children"]) >= 4


def test_the_recorder_stops_collecting_once_the_capture_is_over(tools):
    """A recorder left attached would hold 1,500 spans to write one file.

    The SDK has no ``remove_span_processor``, so the detach is the recorder
    being told to stop matching -- which means it has to be observed, not
    assumed.
    """
    _, first = capture_question("What drove NVDA revenue?", tools=tools)
    _, second = capture_question("What drove NVDA revenue?", tools=tools)
    assert len(second) == len(first)  # not first + second


def test_the_tools_get_their_own_tracer_back(tools):
    before = tools.tracer
    capture_question("What drove NVDA revenue?", tools=tools)
    assert tools.tracer is before


def test_a_question_that_throws_still_returns_its_spans(tools):
    class Boom(FakeRetriever):
        def search(self, *a, **kw):  # noqa: ANN002, ANN003
            raise RuntimeError("qdrant is down")

    tools.retriever = Boom()
    state, spans = capture_question("What drove NVDA revenue?", tools=tools)
    assert "qdrant is down" in state["error"]
    assert any(s["name"] == "agent.plan" for s in spans)


# --------------------------------------------------------------- write_trace


def test_the_file_carries_the_answer_next_to_the_spans(tmp_path, tools):
    """Both, because a span tree with no answer under it shows that the graph
    ran, not that it worked."""
    q = "What drove NVDA revenue?"
    state, spans = capture_question(q, tools=tools, qid="nar-001")
    out = write_trace(tmp_path / "t.json", question=q, qid="nar-001", state=state, spans=spans)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["qid"] == "nar-001"
    assert doc["route"] == "text"
    assert doc["answer"] == state["answer"]
    assert doc["evidence"][0]["citation"]
    assert doc["spans"][0]["name"] == "agent.question"


def test_the_evidence_body_is_trimmed_so_the_file_stays_readable(tmp_path, tools):
    state, spans = capture_question("What drove NVDA revenue?", tools=tools)
    out = write_trace(tmp_path / "t.json", question="q", qid="x", state=state, spans=spans)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["evidence"][0]["body"]) <= 400


def test_the_note_says_how_to_regenerate_the_file(tmp_path):
    """The whole difference between this file and an export from a UI."""
    out = write_trace(tmp_path / "t.json", question="q", qid="x", state={}, spans=[])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "python -m filing.eval trace --qid" in doc["note"]
