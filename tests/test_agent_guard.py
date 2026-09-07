"""The guard in place: the verify node, the graph it sits in, and what it blocks.

`tests/test_agent_verify.py` holds the checker itself -- what counts as a
supported figure, what a dangling citation is. This file holds the wiring, which
fails in a different way: a verifier that works perfectly and is never called
produces exactly the same results file as no verifier at all.

Three properties, then:

* verification is **off unless asked for**, because M5's published numbers were
  measured without it and a default that quietly changed them would make the
  two gates incomparable;
* a bad figure is **blocked** in block mode and **kept and flagged** in flag
  mode -- the bail-out clause the gate card writes down in advance, exercised
  rather than described;
* the verdict reaches the state, and from there the results file, because the
  failure taxonomy is only a gallery if something records it.
"""

from __future__ import annotations

from filing.agent.graph import build_graph, mermaid, run_question
from filing.agent.nodes import Tools
from filing.eval.runner import REFUSAL
from test_agent_graph import (
    FakeRetriever,
    FakeRow,
    FakeSql,
    StubBackend,
    plan_reply,
)

QUESTION = "What did TST report for Revenues for the fiscal year ended 2024-12-31?"


class RecheckingSql(FakeSql):
    """A facts store that also answers the verifier's primary-key read.

    Separate from the plain double on purpose: the recheck is a second entry
    point into the store, and a test that stubbed it onto the same object would
    not notice if the verifier started calling `lookup` instead -- which would
    put the concept resolver back inside its own audit.
    """

    def __init__(self, rows=(), known: bool = True, value: float | None = None) -> None:
        super().__init__(rows=rows, known=known)
        self.value = value
        self.rechecks: list[tuple] = []

    def recheck(self, ticker, tag, *, period_end=None, unit="USD", accn=""):  # noqa: ANN001
        self.rechecks.append((ticker, tag, period_end, unit, accn))
        if self.value is None:
            return None
        return FakeRow(val=self.value)


def _tools(answer: str, *, verify: bool = True, guard: str = "block", value: float = 200.0):
    backend = StubBackend(
        [plan_reply("sql", ticker="TST", concept="Revenues", period_end="2024-12-31"), answer]
    )
    return Tools(
        backend=backend,
        sql=RecheckingSql(rows=(FakeRow(),), value=value),
        retriever=FakeRetriever(),
        verify=verify,
        guard=guard,
    )


# --- the topology -----------------------------------------------------------


def test_verify_is_a_node_and_not_a_wrapper():
    """A step that has no span is a step nobody can audit after the fact."""
    graph = build_graph(Tools()).get_graph()
    assert "verify" in set(graph.nodes)
    text = mermaid()
    assert "synthesise --> verify" in text


def test_the_check_runs_on_every_answer_including_refusals():
    """One unconditional edge, so the graph's shape does not depend on a flag."""
    backend = StubBackend([plan_reply("refuse", why="no filing says")])
    state = run_question(
        "What will TST's share price be next week?",
        tools=Tools(backend=backend, retriever=FakeRetriever(), verify=True),
    )
    assert state["refused"]
    assert state["verdict"] is not None
    assert state["verdict"].ok  # a refusal states no figure, so there is nothing to fail


# --- the default ------------------------------------------------------------


def test_verification_is_off_unless_asked_for():
    """M5's numbers were measured without it; the default has to keep them measurable."""
    tools = _tools("TST reported revenue of 999 USD [1]", verify=False)
    state = run_question(QUESTION, tools=tools)
    assert "verdict" not in state
    assert state["answer"] == "TST reported revenue of 999 USD [1]"


# --- blocking ---------------------------------------------------------------


def test_a_supported_answer_passes_untouched():
    tools = _tools("TST reported revenue of 200 USD [1]")
    state = run_question(QUESTION, tools=tools)
    assert state["verdict"].ok
    assert not state.get("flagged")
    assert not state.get("blocked")
    assert state["answer"] == "TST reported revenue of 200 USD [1]"
    # The recheck went back to the store by identity, not through the resolver.
    assert tools.sql.rechecks == [("TST", "Revenues", "2024-12-31", "USD", "a-2")]


def test_a_figure_that_is_not_in_the_evidence_does_not_ship():
    state = run_question(QUESTION, tools=_tools("TST reported revenue of 999 USD [1]"))
    assert state["blocked"]
    assert state["refused"]
    assert state["answer"] == REFUSAL
    assert "synthesis_drift" in state["verdict"].taxonomy


def test_a_citation_pointing_past_the_evidence_does_not_ship():
    state = run_question(QUESTION, tools=_tools("TST reported revenue of 200 USD [7]"))
    assert state["blocked"]
    assert state["verdict"].dangling == (7,)


def test_a_fact_the_store_no_longer_holds_is_a_failure():
    """A restated figure is not a figure anyone should still be quoting."""
    state = run_question(QUESTION, tools=_tools("TST reported revenue of 200 USD [1]", value=None))
    assert state["blocked"]
    assert state["verdict"].unrechecked


# --- flagging: the bail-out clause ------------------------------------------


def test_flag_mode_keeps_the_answer_and_still_records_the_finding():
    """The gate's own escape hatch: a system that surfaces its uncertainty
    beats one that suppresses it, so the finding survives the answer shipping."""
    tools = _tools("TST reported revenue of 999 USD [1]", guard="flag")
    state = run_question(QUESTION, tools=tools)
    assert state["answer"] == "TST reported revenue of 999 USD [1]"
    assert not state.get("blocked")
    assert state["flagged"]
    assert state["verdict"].unsupported


def test_off_mode_computes_the_verdict_without_acting_on_it():
    state = run_question(QUESTION, tools=_tools("TST reported revenue of 999 USD [1]", guard="off"))
    assert state["answer"] == "TST reported revenue of 999 USD [1]"
    assert state["flagged"]
