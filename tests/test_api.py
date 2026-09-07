"""The transparency surface, tested without a corpus and without a model.

Every test here hands :func:`filing.api.payload_for` a state the graph could
have produced and asserts on the payload, or drives the routes with an engine
that returns a canned one. That is not a shortcut around the real thing -- it is
the property the gate asks for, stated as a test: if the payload can be built
from a finished state alone, then the interface cannot be getting anything from
anywhere else.

The four outcome states get a test each, because they are the states a demo
skips. An abstention and a block both print the same token, and the only place
they are distinguishable is in this payload.
"""

from __future__ import annotations

import json

import pytest

from filing.agent.state import Evidence, Grade, SubQuestion
from filing.agent.verify import verify_answer
from filing.api import (
    DEFAULT_CONFIG,
    STAGES,
    Answer,
    AskEngine,
    EngineNotReady,
    create_app,
    payload_for,
)
from filing.eval.runner import REFUSAL

pytest.importorskip("fastapi")

FACT = Evidence(
    kind="fact",
    body=(
        "NVDA reported Revenues of 60,922,000,000 USD for the period ending "
        "2024-01-28 (tag Revenues)."
    ),
    citation="NVDA 10-K 2024-01-28 [0001045810-24-000029 Revenues]",
    accn="0001045810-24-000029",
    value=60_922_000_000.0,
    unit="USD",
    tag="Revenues",
    period_end="2024-01-28",
    ticker="NVDA",
    score=1.0,
)

CHUNK = Evidence(
    kind="text",
    body="Revenue for fiscal 2024 was $60,922 million, up 126% from fiscal 2023.",
    citation="NVDA 10-K 2024-01-28 Item 7 [0001045810-24-000029 1200:1400]",
    accn="0001045810-24-000029",
    chunk_id="c1",
    char_start=1200,
    char_end=1400,
    period_end="2024-01-28",
    ticker="NVDA",
    score=7.2,
)

# No accession, no span: a record that looks like a citation until it is
# followed. `unlocatable` is the field that exists to say so.
ROOTLESS = Evidence(kind="text", body="Revenue grew.", citation="somewhere", accn="")

QUESTION = "What was NVDA's revenue in fiscal 2024?"


def state_for(answer: str, evidence: list[Evidence], **extra: object) -> dict:
    verdict = verify_answer(answer, evidence, question=QUESTION, route="sql")
    base = {
        "answer": answer,
        "evidence": evidence,
        "plan": [
            SubQuestion(
                text=QUESTION,
                route="sql",
                ticker="NVDA",
                concept="revenue",
                period_end="2024-01-28",
                why="a single reported figure",
            )
        ],
        "plan_note": "",
        "route": "sql",
        "grade": Grade(ok=True, reason="1 fact row", detail={"rows": 1}),
        "repairs": 0,
        "repair_log": [],
        "verdict": verdict,
        "flagged": not verdict.ok,
        "llm_calls": 2,
        "seconds": 3.0,
    }
    base.update(extra)
    return base


def build(answer: str, evidence: list[Evidence], **extra: object) -> Answer:
    return payload_for(
        state_for(answer, evidence, **extra),
        question=QUESTION,
        qid="t",
        config=DEFAULT_CONFIG,
        guard="block",
    )


# --------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------


def test_evidence_is_numbered_by_the_marker_that_cites_it():
    p = build("Revenue was $60.9 billion [1], up 126% [2].", [FACT, CHUNK])
    assert [e.marker for e in p.evidence] == [1, 2]
    assert all(e.cited for e in p.evidence)
    assert all(e.locatable for e in p.evidence)


def test_uncited_evidence_is_reported_as_shown_but_not_used():
    p = build("Revenue was $60.9 billion [1].", [FACT, CHUNK])
    assert [e.cited for e in p.evidence] == [True, False]


def test_a_record_with_nothing_to_look_up_is_marked_unlocatable():
    p = build("Revenue grew [1].", [ROOTLESS])
    assert p.evidence[0].locatable is False
    assert p.verification.unlocatable == [1]


def test_claims_carry_the_markers_and_figures_of_their_own_sentence():
    p = build("Revenue was $60.9 billion [1]. It rose 126% [2].", [FACT, CHUNK])
    assert len(p.claims) == 2
    assert p.claims[0].markers == [1]
    assert p.claims[1].markers == [2]
    assert [f.text for f in p.claims[0].figures] == ["$60.9"]
    assert [f.text for f in p.claims[1].figures] == ["126"]


def test_the_claim_split_is_the_one_the_guard_checked():
    """A figure and a dollar amount must not split "$60.9 billion" in two."""
    p = build("Revenue was $60.9 billion [1].", [FACT])
    assert len(p.claims) == 1
    assert p.claims[0].text.endswith("[1].")


def test_a_supported_figure_names_the_evidence_it_matched():
    p = build("Revenue was $60.9 billion [1].", [FACT])
    figure = next(n for n in p.verification.numbers if n.text == "$60.9")
    assert figure.status == "supported"
    assert figure.matched == FACT.citation
    assert figure.marker == 1
    assert figure.derivation == ""


def test_a_derived_figure_shows_the_computed_value_beside_its_inputs():
    """The M7 criterion, in one assertion: the arithmetic is on the payload."""
    other = Evidence(
        kind="fact",
        body="NVDA reported Revenues of 26,974,000,000 USD for the period ending 2023-01-29.",
        citation="NVDA 10-K 2023-01-29 [0001045810-23-000017 Revenues]",
        accn="0001045810-23-000017",
        value=26_974_000_000.0,
        unit="USD",
        tag="Revenues",
        period_end="2023-01-29",
        ticker="NVDA",
    )
    p = build("Revenue grew 125.85% year over year [1][2].", [FACT, other])
    figure = next(n for n in p.verification.numbers if n.text.startswith("125"))
    assert figure.status == "derived"
    assert figure.how == "recomputed"
    assert "26,974,000,000" in figure.derivation
    assert "60,922,000,000" in figure.derivation
    assert figure.matched == ""


def test_an_unsupported_figure_survives_into_the_payload():
    p = build("Revenue was $71.4 billion [1].", [FACT])
    bad = next(n for n in p.verification.numbers if n.text == "$71.4")
    assert bad.status == "unsupported"
    assert bad.matched == "" and bad.derivation == ""
    assert p.verification.ok is False
    assert p.verification.unsupported == 1
    assert p.verification.reasons


def test_verification_off_is_not_the_same_as_verification_clean():
    p = payload_for(
        {"answer": "Revenue was $60.9 billion [1].", "evidence": [FACT], "verdict": None},
        question=QUESTION,
        qid="t",
        guard="",
    )
    assert p.verification.enabled is False
    assert p.verification.numbers == []
    assert p.verification.mode == ""


# --------------------------------------------------------------------------
# the four outcomes
# --------------------------------------------------------------------------


def test_outcome_answered():
    assert build("Revenue was $60.9 billion [1].", [FACT]).outcome == "answered"


def test_outcome_abstained_is_the_agent_declining():
    p = build(REFUSAL, [], refused=True, route="refuse")
    assert p.outcome == "abstained"
    assert p.refused is True
    assert p.verification.blocked is False


def test_outcome_blocked_is_the_verifier_overruling():
    """Same text as an abstention, and a different state. The guard's whole visibility."""
    verdict = verify_answer("Revenue was $71.4 billion [1].", [FACT], question=QUESTION)
    p = payload_for(
        {
            "answer": REFUSAL,
            "evidence": [FACT],
            "verdict": verdict,
            "refused": True,
            "blocked": True,
        },
        question=QUESTION,
        qid="t",
        guard="block",
    )
    assert p.outcome == "blocked"
    assert p.verification.blocked is True
    assert p.verification.ok is False
    assert p.verification.reasons


def test_outcome_error_carries_the_exception_and_no_answer():
    p = payload_for(
        {"error": "RuntimeError: qdrant is not running", "answer": ""},
        question=QUESTION,
        qid="t",
    )
    assert p.outcome == "error"
    assert p.answer == ""
    assert "qdrant" in p.error


# --------------------------------------------------------------------------
# the trace link
# --------------------------------------------------------------------------


def test_no_collector_means_no_link_rather_than_a_dead_one(monkeypatch):
    monkeypatch.setattr("filing.tracing.tracing_is_live", lambda: False)
    p = payload_for({"answer": "x"}, question=QUESTION, qid="t", trace_id="ab" * 16)
    assert p.trace.trace_id == "ab" * 16
    assert p.trace.url == ""
    assert p.trace.live is False


def test_a_live_collector_produces_a_project_scoped_url(monkeypatch):
    monkeypatch.setattr("filing.tracing.tracing_is_live", lambda: True)
    p = payload_for({"answer": "x"}, question=QUESTION, qid="t", trace_id="ab" * 16)
    assert p.trace.live is True
    assert p.trace.url.endswith(f"/traces/{'ab' * 16}")
    assert p.trace.project in p.trace.url


# --------------------------------------------------------------------------
# the routes
# --------------------------------------------------------------------------


class FakeEngine(AskEngine):
    """An engine with no stores behind it, so the routes can be tested alone."""

    def __init__(self, *, payload: Answer | None = None, fail: bool = False) -> None:
        super().__init__()
        self.payload = payload
        self.fail = fail
        self.asked: list[str] = []

    def warm(self) -> FakeEngine:
        if self.fail:
            raise EngineNotReady("no index on disk")
        return self

    @property
    def ready(self) -> bool:
        return not self.fail

    @property
    def guard(self) -> str:
        return "block"

    def ask(self, question: str, *, qid: str = "") -> Answer:
        self.warm()
        self.asked.append(question)
        return self.payload or build("Revenue was $60.9 billion [1].", [FACT])

    def stream(self, question: str, *, qid: str = ""):
        self.warm()
        yield {"event": "start", "data": {"qid": "t", "question": question}}
        yield {"event": "stage", "data": {"node": "plan", "label": "reading the question"}}
        yield {"event": "stage", "data": {"node": "verify", "label": "verifying the answer"}}
        yield {"event": "answer", "data": self.ask(question).model_dump()}


def client_for(engine: AskEngine):
    from fastapi.testclient import TestClient

    return TestClient(create_app(engine))


def test_health_answers_before_the_stores_are_open():
    engine = FakeEngine(fail=True)
    body = client_for(engine).get("/health").json()
    assert body["ok"] is True and body["ready"] is False


def test_ask_returns_the_evidence_records_not_just_prose():
    """The definition of done, as a test: the UI could not be the source of truth."""
    engine = FakeEngine()
    body = client_for(engine).post("/ask", json={"question": QUESTION}).json()
    assert body["answer"]
    assert len(body["evidence"]) == 1
    assert body["evidence"][0]["citation"] == FACT.citation
    assert body["claims"][0]["markers"] == [1]
    assert body["verification"]["enabled"] is True
    assert engine.asked == [QUESTION]


def test_a_missing_corpus_is_a_503_with_the_command_that_fixes_it():
    r = client_for(FakeEngine(fail=True)).post("/ask", json={"question": QUESTION})
    assert r.status_code == 503
    assert "index" in r.json()["detail"]


def test_asking_for_another_config_is_refused_rather_than_silently_ignored():
    r = client_for(FakeEngine()).post("/ask", json={"question": QUESTION, "config": "agent"})
    assert r.status_code == 400
    assert "agent-guarded" in r.json()["detail"]


def test_an_empty_question_is_rejected_by_the_schema():
    assert client_for(FakeEngine()).post("/ask", json={"question": ""}).status_code == 422


def test_the_stream_reports_stages_and_ends_with_the_payload():
    r = client_for(FakeEngine()).post("/ask/stream", json={"question": QUESTION})
    assert r.status_code == 200
    events = [line for line in r.text.splitlines() if line.startswith("event:")]
    assert "event: stage" in events
    last = [line for line in r.text.splitlines() if line.startswith("data:")][-1]
    payload = json.loads(last[len("data:") :].strip())
    assert payload["evidence"][0]["citation"] == FACT.citation


def test_the_index_names_the_ui_command():
    body = client_for(FakeEngine()).get("/").json()
    assert "streamlit" in body["ui"]


def test_the_schema_publishes_the_evidence_contract():
    schema = client_for(FakeEngine()).get("/openapi.json").json()
    props = schema["components"]["schemas"]["Answer"]["properties"]
    assert {"evidence", "claims", "verification", "trace", "outcome"} <= set(props)


# --------------------------------------------------------------------------
# the real streamer
# --------------------------------------------------------------------------


class StubGraph:
    """A graph that emits what langgraph's two stream modes emit, and nothing else."""

    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom

    def stream(self, state, config=None, stream_mode=None):
        yield "updates", {"plan": {}}
        yield "values", {**state, "route": "sql"}
        if self.boom:
            raise RuntimeError("qdrant went away")
        yield "updates", {"synthesise": {}}
        yield (
            "values",
            {
                **state,
                "route": "sql",
                "answer": "Revenue was $60.9 billion [1].",
                "evidence": [FACT],
            },
        )


def streamer(*, boom: bool = False) -> AskEngine:
    """A real ``AskEngine`` -- its own ``stream``, on its own thread -- over a stub graph."""
    engine = AskEngine()
    engine.graph = StubGraph(boom=boom)
    engine.tools = object()
    engine.warm = lambda: engine  # type: ignore[method-assign]
    return engine


def test_the_streamer_names_each_node_then_hands_over_the_payload():
    events = list(streamer().stream(QUESTION))
    assert [e["event"] for e in events] == ["start", "stage", "stage", "answer"]
    assert [e["data"]["label"] for e in events[1:3]] == [
        STAGES["plan"],
        STAGES["synthesise"],
    ]
    payload = events[-1]["data"]
    assert payload["outcome"] == "answered"
    assert payload["evidence"][0]["citation"] == FACT.citation


def test_a_graph_that_dies_mid_stream_still_ends_with_a_payload():
    """The client has a plan and a route on screen by then; a dropped connection
    would leave them under a spinner that never resolves."""
    events = list(streamer(boom=True).stream(QUESTION))
    assert events[-1]["event"] == "answer"
    payload = events[-1]["data"]
    assert payload["outcome"] == "error"
    assert "qdrant went away" in payload["error"]


def test_the_span_is_opened_and_closed_on_one_thread():
    """An opentelemetry context token cannot be reset on a thread that did not take it.

    A span held open across the ``yield`` is entered on whichever starlette
    worker pulled the first item and exited on whichever pulled the last, which
    logs ``Failed to detach context`` on every request and leaves the trace's
    parenting to luck. So the invariant is not "no error was logged" -- with no
    collector running there is no context to detach and nothing would be -- it
    is that both ends of the span happen in the same place.
    """
    import contextlib
    import threading as th

    from opentelemetry import trace

    seen: list[int] = []

    @contextlib.contextmanager
    def span():
        seen.append(th.get_ident())
        try:
            yield trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        finally:
            seen.append(th.get_ident())

    engine = streamer()
    engine._span = span  # type: ignore[method-assign]
    for _ in engine.stream(QUESTION):  # drained a step at a time, as starlette drains it
        pass
    assert len(seen) == 2
    assert seen[0] == seen[1] != th.get_ident()
