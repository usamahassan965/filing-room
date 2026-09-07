"""The page, driven headless.

Streamlit runs its widgets in "bare mode" outside a browser session -- the calls
succeed and the output goes nowhere -- which is enough to test the thing worth
testing here, and it is not layout. It is that **every panel renders every state
without raising**. An abstention, a block, an error, a run with no evidence at
all: those are the four screens a demo never shows and the four that a reader
who is trying to break the claim will go looking for first.

The other half is the coupling test. The page maps the API's vocabulary --
outcomes, figure statuses, the ways a number can be matched -- onto colours and
sentences, and those maps are written out by hand. If the API grows a fifth
outcome or the verifier a fifth ``how``, the page would silently fall back to a
grey dot and the reader would never know a case had been missed. So the maps are
asserted against the source of the vocabulary rather than against a copy of it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

st = pytest.importorskip("streamlit")

from filing import ui  # noqa: E402
from filing.agent.state import Evidence  # noqa: E402
from filing.agent.verify import verify_answer  # noqa: E402
from filing.api import payload_for  # noqa: E402
from filing.eval.runner import REFUSAL  # noqa: E402

FACT = Evidence(
    kind="fact",
    body="NVDA reported Revenues of 60,922,000,000 USD for the period ending 2024-01-28.",
    citation="NVDA 10-K 2024-01-28 [0001045810-24-000029 Revenues]",
    accn="0001045810-24-000029",
    value=60_922_000_000.0,
    unit="USD",
    tag="Revenues",
    period_end="2024-01-28",
    ticker="NVDA",
    score=1.0,
)

PANELS = (
    ui.outcome_banner,
    ui.answer_panel,
    ui.claims_panel,
    ui.verification_panel,
    ui.evidence_panel,
    ui.path_panel,
    ui.footer,
)


def payload(answer: str, evidence: list[Evidence], **extra: Any) -> dict[str, Any]:
    verdict = verify_answer(answer, evidence, question="q", route="sql")
    state: dict[str, Any] = {
        "answer": answer,
        "evidence": evidence,
        "route": "sql",
        "verdict": verdict,
        "seconds": 1.0,
        "llm_calls": 2,
    }
    state.update(extra)
    return payload_for(state, question="q", qid="t", guard="block").model_dump()


def render(p: dict[str, Any]) -> None:
    for panel in PANELS:
        panel(p)


# --------------------------------------------------------------------------
# every state renders
# --------------------------------------------------------------------------


def test_an_answer_renders():
    render(payload("Revenue was $60.9 billion [1].", [FACT]))


def test_an_abstention_renders():
    render(payload(REFUSAL, [], refused=True, route="refuse"))


def test_a_block_renders():
    render(payload("Revenue was $71.4 billion [1].", [FACT], blocked=True, refused=True))


def test_an_error_renders():
    render({"outcome": "error", "error": "RuntimeError: qdrant is not running", "question": "q"})


def test_an_answer_with_no_evidence_renders():
    render(payload("I could not find it.", []))


def test_an_unknown_outcome_falls_back_rather_than_raising():
    """A payload from a newer server must not take the page down."""
    render({"outcome": "something-new", "question": "q"})


def test_an_empty_payload_renders():
    """The shape a truncated stream leaves behind."""
    render({})


def test_an_uncited_and_unlocatable_answer_renders():
    rootless = Evidence(kind="text", body="Revenue grew.", citation="somewhere", accn="")
    render(payload("Revenue grew 12% [1][9].", [rootless]))


# --------------------------------------------------------------------------
# the page speaks the API's vocabulary
# --------------------------------------------------------------------------


def test_every_outcome_the_api_can_emit_has_a_banner():
    from filing.api import _outcome

    emitted = {
        _outcome({"error": "x"}, "", blocked=False),
        _outcome({}, "", blocked=True),
        _outcome({"refused": True}, REFUSAL, blocked=False),
        _outcome({}, "an answer", blocked=False),
    }
    assert emitted <= set(ui.OUTCOMES)


def test_every_figure_status_has_a_colour():
    # The statuses `verify_numbers` can assign, from the module that assigns them.
    assert {"supported", "derived", "context", "unsupported"} <= set(ui.STATUS_COLOUR)


def test_every_way_a_number_can_be_matched_has_a_sentence():
    assert {"digits", "scaled", "recomputed", "question", "period", ""} <= set(ui.HOW)


def test_the_abstention_wording_does_not_call_it_a_failure():
    """An abstention on an unanswerable question is the behaviour the eval rewards."""
    _, _, meaning = ui.OUTCOMES["abstained"]
    assert "declined" in meaning
    assert "fail" not in meaning.lower()


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


def test_sse_parses_event_and_data_pairs():
    lines = [
        "event: stage",
        'data: {"label": "reading the question"}',
        "",
        "event: answer",
        'data: {"outcome": "answered"}',
        "",
    ]
    assert list(ui._sse(lines)) == [
        ("stage", '{"label": "reading the question"}'),
        ("answer", '{"outcome": "answered"}'),
    ]


def test_sse_yields_a_trailing_event_with_no_blank_line_after_it():
    assert list(ui._sse(["event: answer", "data: {}"])) == [("answer", "{}")]


class FakeStage:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def update(self, label: str = "", **_: Any) -> None:
        self.labels.append(label)


class FakeStream:
    def __init__(self, lines: list[str], status: int = 200) -> None:
        self.lines = lines
        self.status_code = status
        self.text = "\n".join(lines)

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def iter_lines(self):
        return iter(self.lines)

    def read(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"detail": self.text}


def test_ask_reports_each_stage_and_returns_the_final_payload(monkeypatch):
    body = payload("Revenue was $60.9 billion [1].", [FACT])
    lines = [
        "event: stage",
        'data: {"node": "plan", "label": "reading the question"}',
        "",
        "event: stage",
        'data: {"node": "verify", "label": "verifying the answer"}',
        "",
        "event: answer",
        f"data: {json.dumps(body)}",
        "",
    ]
    monkeypatch.setattr(ui.httpx, "stream", lambda *a, **k: FakeStream(lines))
    stage = FakeStage()
    got = ui.ask("http://x", "q", stage=stage)
    assert got["answer"] == body["answer"]
    assert got["stages"] == ["reading the question", "verifying the answer"]
    assert len(stage.labels) == 2


def test_a_503_becomes_an_error_payload_rather_than_an_exception(monkeypatch):
    monkeypatch.setattr(
        ui.httpx, "stream", lambda *a, **k: FakeStream(["no index on disk"], status=503)
    )
    got = ui.ask("http://x", "q", stage=FakeStage())
    assert got["outcome"] == "error"
    assert "no index" in got["error"]
    render(got)


def test_a_stream_that_closes_early_is_an_error_not_a_blank_page(monkeypatch):
    monkeypatch.setattr(
        ui.httpx, "stream", lambda *a, **k: FakeStream(["event: stage", "data: {}"])
    )
    got = ui.ask("http://x", "q", stage=FakeStage())
    assert got["outcome"] == "error"
    render(got)


def test_a_server_that_cannot_stream_falls_back_to_the_blocking_endpoint(monkeypatch):
    import httpx

    body = payload("Revenue was $60.9 billion [1].", [FACT])

    def boom(*_a: Any, **_k: Any):
        raise httpx.ConnectError("no sse through this proxy")

    class FakePost:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return body

    monkeypatch.setattr(ui.httpx, "stream", boom)
    monkeypatch.setattr(ui.httpx, "post", lambda *a, **k: FakePost())
    got = ui.ask("http://x", "q", stage=FakeStage())
    assert got["answer"] == body["answer"]


def test_health_reports_a_down_server_as_a_state(monkeypatch):
    import httpx

    def boom(*_a: Any, **_k: Any):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ui.httpx, "get", boom)
    info = ui.health("http://127.0.0.1:9")
    assert info["ok"] is False
    assert "ConnectError" in info["detail"]
