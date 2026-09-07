"""The graph: the routes it takes, the calls it spends, and the loop it cannot exceed.

The gate asks for a repair loop provably bounded at two, and "provably" is doing
work there. A unit test that runs one question and counts two repairs proves the
budget for that question. The test here does better: it drives a question that
can *never* be repaired successfully -- every store returns nothing, every
repair strategy fails -- so the loop runs until something stops it, and asserts
what stopped it.

The other thing held here is the call budget, because it is the design decision
the whole node layout was made for. Two hosted calls per answered question, and
zero for a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from filing.agent.graph import build_graph, mermaid, run_question
from filing.agent.nodes import Nodes, Tools, grade_evidence, heuristic_plan
from filing.agent.state import REPAIR_BUDGET, Evidence, Grade, SubQuestion
from filing.eval.runner import REFUSAL

# --- doubles ---------------------------------------------------------------


@dataclass
class Ranking:
    index: int
    score: float


@dataclass
class Usage:
    http_calls: int = 0
    cache_hits: int = 0


class StubBackend:
    """Replies from a script; counts its calls the way a real backend does."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls = 0
        self.prompts: list[list[dict[str, str]]] = []

    def chat(self, messages, *, role="chat", temperature=0.0, max_tokens=512, **kw):  # noqa: ANN001, ARG002
        self.calls += 1
        self.prompts.append(messages)
        return self.replies.pop(0) if self.replies else "an answer [1]"

    def rerank(self, query, passages, *, top_n=None, **kw):  # noqa: ANN001, ARG002
        # Descending, so index 0 stays first and the score is above the floor.
        return [Ranking(index=i, score=5.0 - i) for i in range(len(passages))][: top_n or 5]

    def usage(self) -> Usage:
        return Usage(http_calls=self.calls)


@dataclass
class FakeRow:
    ticker: str = "TST"
    tag: str = "Revenues"
    label: str = "Revenues"
    unit: str = "USD"
    span: str = "FY"
    period_start: Any = None
    period_end: str = "2024-12-31"
    val: float = 200.0
    accn: str = "a-2"
    form: str = "10-K"

    @property
    def citation(self) -> str:
        return f"{self.ticker} {self.form} {self.period_end} {self.tag} [{self.accn}]"


@dataclass
class FakeAnswer:
    rows: tuple
    concept: Any = None
    reason: str = ""


class FakeConcept:
    tag = "Revenues"


class FakeResolver:
    def __init__(self, known: bool = True) -> None:
        self.known = known

    def resolve(self, phrase):  # noqa: ANN001
        return FakeConcept() if self.known and phrase else None


class FakeSql:
    """A facts store that either has the row or does not, and says which."""

    def __init__(self, rows=(), known: bool = True) -> None:
        self.rows = tuple(rows)
        self.resolver = FakeResolver(known)
        self.calls: list[tuple] = []

    def lookup(self, ticker, concept, *, period_end=None, **kw):  # noqa: ANN001, ARG002
        self.calls.append((ticker, concept, period_end))
        reason = "" if self.rows else "empty"
        return FakeAnswer(rows=self.rows, concept=FakeConcept(), reason=reason)


@dataclass
class FakeChunk:
    chunk_id: str = "c1"
    accn: str = "a-1"
    ticker: str = "TST"
    form: str = "10-K"
    period_end: str = "2024-12-31"
    item_key: str = "7"
    item: str = "7"
    char_start: int = 0
    char_end: int = 100
    text: str = "Revenue increased because of higher unit volumes."


@dataclass
class FakeHit:
    chunk: FakeChunk
    fused_score: float = 0.016
    rerank_score: Any = None

    @property
    def citation(self) -> str:
        return f"{self.chunk.ticker} {self.chunk.form} {self.chunk.period_end} [{self.chunk.accn}]"


class FakeRetriever:
    def __init__(self, hits=None) -> None:
        self.hits = list(hits) if hits is not None else [FakeHit(FakeChunk())]
        self.queries: list[str] = []

    def search(self, query, *, k=50, where=None, rerank=True, top_n=5):  # noqa: ANN001, ARG002
        self.queries.append(query)
        return list(self.hits)


def plan_reply(route: str, **kw: str) -> str:
    import json

    return json.dumps({"route": route, **kw})


# --- the topology ----------------------------------------------------------


def test_the_diagram_comes_from_the_compiled_graph():
    """The architecture picture is the code, not a drawing beside it."""
    text = mermaid()
    for node in ("plan", "route", "retrieve_sql", "rerank", "grade", "repair", "synthesise"):
        assert node in text
    assert "repair --> route" in text  # a repair may change the route, so it re-enters there


def test_every_named_step_is_its_own_node():
    """One node per step is what makes one span per step possible."""
    graph = build_graph(Tools()).get_graph()
    assert {
        "plan",
        "route",
        "retrieve_sql",
        "retrieve_text",
        "retrieve_graph",
        "refuse",
        "rerank",
        "grade",
        "synthesise",
        "repair",
    } <= set(graph.nodes)


# --- routing ---------------------------------------------------------------


def test_a_numeric_question_reaches_the_facts_store_and_costs_two_calls():
    backend = StubBackend(
        [
            plan_reply("sql", ticker="TST", concept="Revenues", period_end="2024-12-31"),
            "TST reported revenue of 200 USD [1]",
        ]
    )
    sql = FakeSql(rows=(FakeRow(),))
    state = run_question(
        "What did TST report for Revenues for the fiscal year ended 2024-12-31?",
        tools=Tools(backend=backend, sql=sql, retriever=FakeRetriever()),
    )
    assert state["route"] == "sql"
    assert sql.calls == [("TST", "Revenues", "2024-12-31")]
    assert state["evidence"][0].value == 200.0
    assert state["llm_calls"] == 2
    assert not state["refused"]


def test_a_refusal_costs_one_call_not_two():
    """Paying a model to write "insufficient evidence" is paying for a foregone conclusion."""
    backend = StubBackend([plan_reply("refuse", why="no filing contains tomorrow's price")])
    state = run_question(
        "What will TST's share price be next week?",
        tools=Tools(backend=backend, retriever=FakeRetriever()),
    )
    assert state["route"] == "refuse"
    assert state["answer"] == REFUSAL
    assert state["refused"]
    assert state["llm_calls"] == 1


def test_a_narrative_question_goes_through_the_reranker():
    backend = StubBackend([plan_reply("text", ticker="TST"), "Because volumes rose [1]"])
    retriever = FakeRetriever([FakeHit(FakeChunk(chunk_id=f"c{i}")) for i in range(10)])
    state = run_question("Why did revenue rise?", tools=Tools(backend=backend, retriever=retriever))
    assert state["route"] == "text"
    assert len(state["candidates"]) == 10  # fused
    assert len(state["evidence"]) == 5  # reranked
    assert state["evidence"][0].score == 5.0  # the cross-encoder's, not RRF's 0.016


def test_an_unroutable_sql_concept_is_downgraded_rather_than_failed():
    """The model may name a concept the registry does not have. That is a downgrade."""
    backend = StubBackend(
        [plan_reply("sql", ticker="TST", concept="vibes"), "answer [1]"],
    )
    state = run_question(
        "What were TST's vibes in 2024?",
        tools=Tools(backend=backend, sql=FakeSql(known=False), retriever=FakeRetriever()),
    )
    assert state["route"] == "text"
    assert any("not a registry concept" in note for note in state["repair_log"])


def test_a_plan_reply_that_is_not_json_falls_back_to_the_baseline_behaviour():
    """One malformed reply costs one question's accuracy, not the run."""
    backend = StubBackend(["I think you should look at the 10-K.", "answer [1]"])
    state = run_question(
        "Why did TST's margin fall?",
        tools=Tools(backend=backend, retriever=FakeRetriever()),
    )
    assert state["route"] == "text"
    assert state["plan_note"] == "plan reply did not parse as JSON"


def test_json_survives_a_code_fence():
    """Fencing is a provider habit, not a routing failure -- cf. the bracket bug in M4.5."""
    backend = StubBackend(
        ['```json\n{"route": "text", "ticker": "TST"}\n```', "answer [1]"],
    )
    state = run_question("Why?", tools=Tools(backend=backend, retriever=FakeRetriever()))
    assert state["route"] == "text"
    assert state["plan_note"] == ""


# --- the bounded loop ------------------------------------------------------


def test_the_repair_loop_stops_at_the_budget_when_nothing_can_ever_succeed():
    """The gate's requirement, driven by a question no repair can fix.

    Both stores are empty and stay empty, so every grade fails and every repair
    strategy fails after it. The loop therefore runs until the *budget* stops
    it, which is the thing being asserted -- not that this particular question
    happened to need two passes.
    """
    ask = plan_reply("sql", ticker="TST", concept="Revenues", period_end="2024-12-31")
    backend = StubBackend([ask])
    state = run_question(
        "What did TST report for Revenues for the fiscal year ended 2024-12-31?",
        tools=Tools(backend=backend, sql=FakeSql(rows=()), retriever=FakeRetriever(hits=[])),
    )
    assert state["repairs"] == REPAIR_BUDGET == 2
    assert state["answer"] == REFUSAL
    assert state["llm_calls"] == 1  # plan only; the abstention is free


def test_the_sql_branch_repairs_by_dropping_the_period_first():
    """The first repair is the one a person would try: the year end may have moved."""
    sql = FakeSql(rows=())
    ask = plan_reply("sql", ticker="TST", concept="Revenues", period_end="2024-12-31")
    backend = StubBackend([ask])
    run_question("q", tools=Tools(backend=backend, sql=sql, retriever=FakeRetriever(hits=[])))
    assert [c[2] for c in sql.calls] == ["2024-12-31", None]


def test_the_second_sql_repair_hands_the_question_to_the_text_retriever():
    sql = FakeSql(rows=())
    retriever = FakeRetriever([FakeHit(FakeChunk())])
    backend = StubBackend(
        [
            plan_reply("sql", ticker="TST", concept="Revenues", period_end="2024-12-31"),
            "found it in the text [1]",
        ]
    )
    state = run_question("q", tools=Tools(backend=backend, sql=sql, retriever=retriever))
    assert state["route"] == "text"
    assert state["repairs"] == 2
    assert not state["refused"]


def test_a_text_repair_re_queries_with_the_ticker_when_the_company_is_wrong():
    """The commonest real failure: the right kind of paragraph, the wrong company."""
    retriever = FakeRetriever([FakeHit(FakeChunk(ticker="OTHER"))])
    backend = StubBackend([plan_reply("text", ticker="TST")])
    state = run_question("Why did revenue rise?", tools=Tools(backend=backend, retriever=retriever))
    assert retriever.queries[1].startswith("TST ")
    assert state["repairs"] == 2


# --- the deterministic pieces, driven directly -----------------------------


def _sub(**kw):
    return SubQuestion(text=kw.pop("text", "q"), **kw)


def test_grading_a_fact_asks_only_whether_the_row_came_back():
    row = Evidence.from_fact(FakeRow())
    assert grade_evidence("sql", [row], _sub(route="sql")).ok
    assert not grade_evidence("sql", [], _sub(route="sql")).ok


def test_grading_text_uses_the_classifiers_own_boundary_not_a_fitted_number():
    hot = Evidence(kind="text", body="b", citation="c", accn="a", score=0.01, ticker="TST")
    cold = Evidence(kind="text", body="b", citation="c", accn="a", score=-4.0, ticker="TST")
    assert grade_evidence("text", [hot], _sub(ticker="TST")).ok
    bad = grade_evidence("text", [cold], _sub(ticker="TST"))
    assert not bad.ok
    assert bad.missing == "relevance"


def test_the_wrong_company_is_its_own_diagnosis():
    """ "No evidence from TST" and "nothing was relevant" want different repairs."""
    off = Evidence(kind="text", body="b", citation="c", accn="a", score=9.0, ticker="OTHER")
    grade = grade_evidence("text", [off], _sub(ticker="TST"))
    assert not grade.ok
    assert grade.missing == "company"


def test_a_refusal_route_grades_as_ok_because_it_is_a_decision():
    assert grade_evidence("refuse", [], _sub(route="refuse")).ok


def test_the_heuristic_plan_reads_a_ticker_and_a_date_and_nothing_else():
    sub = heuristic_plan("What did ABBV report for Revenues for the year ended 2019-12-31?")
    assert (sub.ticker, sub.period_end, sub.route) == ("ABBV", "2019-12-31", "text")


def test_retrieval_only_mode_makes_no_calls_at_all():
    """The M4 ablation shape: retrieval measured without a generator in the loop."""
    backend = StubBackend()
    tools = Tools(backend=backend, retriever=FakeRetriever(), generate=False)
    state = run_question("Why did revenue rise?", tools=tools)
    assert backend.calls == 0
    assert state["llm_calls"] == 0
    assert state["answer"] == ""
    assert state["evidence"]  # retrieval still happened


def test_one_bad_question_does_not_end_the_run():
    class Exploding(FakeRetriever):
        def search(self, *a, **kw):  # noqa: ANN001, ARG002
            raise RuntimeError("qdrant is down")

    backend = StubBackend([plan_reply("text", ticker="TST")])
    state = run_question("q", tools=Tools(backend=backend, retriever=Exploding()))
    assert "RuntimeError: qdrant is down" in state["error"]
    assert state["answer"] == ""


@pytest.mark.parametrize("route", ["sql", "text", "graph", "refuse"])
def test_a_node_runs_for_every_route_without_a_store_present(route):
    """Every branch degrades rather than raising when its store is missing."""
    nodes = Nodes(Tools())
    out = nodes.route({"question": "q", "plan": [_sub(route=route)], "repairs": 0})
    assert out["route"] in {"text", "refuse"} or route == "refuse"


def test_the_grade_record_carries_what_the_repair_needs():
    grade = Grade(ok=False, reason="r", missing="company", detail={"top_score": 1.0})
    assert grade.missing == "company"
    assert grade.detail["top_score"] == 1.0
