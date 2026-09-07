"""The agent inside M4's harness: same questions, same cache, same scoring rule.

The point of running the agent through the baseline's runner is that neither
system gets to choose how it is measured. So these tests are mostly about the
seams -- the fingerprint that keeps the two caches apart, the route an
abstention reports, and the one place where the metric genuinely does not fit
and the harness says "unmeasured" instead of "zero".

That last one is worth stating plainly, because it is the tempting place to
cheat. M4 defines a supported citation as one whose chunk overlaps a gold
character span. An XBRL fact has no character span: the value in DuckDB and the
number printed in the filing are the same fact reached two different ways, and
only one of them carries offsets. Inventing offsets for the fact so the existing
metric would score it would turn a real limitation into a green column, so the
numeric slice's retrieval and citation figures come back empty for the agent and
its numeric claim rests on exact-match alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from conftest import make_chunk
from filing.agent.nodes import Tools
from filing.agent.state import Evidence
from filing.config import Settings
from filing.eval import dataset, runner
from filing.eval.dataset import EvalQuestion, Span
from filing.eval.runner import CONFIGS, EvalConfig, answer_agent

# ------------------------------------------------------------------- doubles


@dataclass
class Usage:
    http_calls: int = 0


class FakeBackend:
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
class FakeRow:
    ticker: str = "NVDA"
    tag: str = "Revenues"
    label: str = "Revenues"
    unit: str = "USD"
    span: str = "FY"
    period_start: Any = None
    period_end: str = "2024-01-28"
    val: float = 1000.0
    accn: str = "0000000000-00-000000"
    form: str = "10-K"

    @property
    def citation(self) -> str:
        return f"{self.ticker} {self.form} {self.period_end} {self.tag} [{self.accn}]"


@dataclass
class FakeAnswer:
    rows: tuple
    concept: Any = None
    reason: str = ""


class _Concept:
    tag = "Revenues"


class FakeSql:
    def __init__(self, rows=None) -> None:
        self.rows = (FakeRow(),) if rows is None else tuple(rows)
        self.resolver = _Resolver()

    def lookup(self, ticker, concept, *, period_end=None, **kw):  # noqa: ANN001, ARG002
        return FakeAnswer(rows=self.rows, concept=_Concept())


class _Resolver:
    def resolve(self, phrase):  # noqa: ANN001
        return _Concept() if phrase else None


@dataclass
class FakeHit:
    chunk: Any
    fused_score: float = 0.016
    rerank_score: Any = None

    @property
    def citation(self) -> str:
        return f"{self.chunk.ticker} {self.chunk.form} {self.chunk.period_end} [{self.chunk.accn}]"


class FakeRetriever:
    def __init__(self, chunks=None) -> None:
        self.chunks = chunks if chunks is not None else [make_chunk("a body", char_start=0)]

    def search(self, query, *, k=50, where=None, rerank=True, top_n=5):  # noqa: ANN001, ARG002
        return [FakeHit(c) for c in self.chunks[:k]]


def plan_reply(route: str, **kw: str) -> str:
    import json

    return json.dumps({"route": route, **kw})


def question(qid="num-001", slice_="numeric", **over) -> EvalQuestion:
    base = dict(
        id=qid,
        slice=slice_,
        question="What did NVDA report for Revenues?",
        value=1000.0 if slice_ == "numeric" else None,
        unit="USD" if slice_ == "numeric" else "",
        tag="Revenues" if slice_ == "numeric" else "",
        spans=() if slice_ == "unanswerable" else (Span("0000000000-00-000000", 0, 6, "a body"),),
        origin="test",
        gold_source="test",
    )
    return EvalQuestion(**(base | over))


def tools(backend, *, sql=None, retriever=None, **kw) -> Tools:
    return Tools(backend=backend, sql=sql, retriever=retriever or FakeRetriever(), **kw)


@pytest.fixture
def env(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data", llm_backend="gemini", embed_backend="local")
    dataset.write(
        [question(), question("nar-001", "narrative"), question("una-001", "unanswerable")],
        dataset.dataset_path(cfg.data_dir),
    )
    return cfg


AGENT = CONFIGS["agent"]


# ---------------------------------------------------------------- the config


def test_the_agent_is_a_config_the_harness_can_be_asked_for():
    assert CONFIGS["agent"].system == "agent"
    assert CONFIGS["agent-retrieval"].generate is False


def test_the_agent_and_the_baseline_do_not_share_a_cache(env):
    """Different systems, same questions: two experiments, never one directory."""
    sha = "0" * 64
    assert CONFIGS["agent"].resolved(env).fingerprint(sha) != CONFIGS["baseline"].resolved(
        env
    ).fingerprint(sha)


def test_an_unknown_system_lists_the_ones_that_exist(env):
    bad = EvalConfig(name="x", system="oracle")
    with pytest.raises(ValueError, match="agent, naive"):
        runner.run(env, config=bad, write=False)


def test_the_agent_holds_the_baselines_generator_so_the_comparison_is_about_the_agent():
    """Same chat role as `baseline`. A faster model on one side is not a finding."""
    assert CONFIGS["agent"].chat_role == CONFIGS["baseline"].chat_role


# ------------------------------------------------------- one question's shape


def test_a_text_answer_carries_its_chunks_and_its_citations():
    backend = FakeBackend([plan_reply("text", ticker="NVDA"), "Because volumes rose [1]"])
    out = answer_agent(question("nar-001", "narrative"), tools=tools(backend), config=AGENT)
    assert out.route == "text"
    assert [c.chunk_id for c in out.retrieved] == [make_chunk("a body", char_start=0).chunk_id]
    assert out.citations == (make_chunk("a body", char_start=0).chunk_id,)
    assert out.llm_calls == 2


def test_a_fact_answer_reports_no_chunks_and_no_citations(env):
    """The honest empty. See this module's docstring -- a fact has no span.

    The alternative was to give the fact a made-up character range so that
    `citations_supported` would have something to overlap. That is the one move
    that would turn a stated limitation into a green column, which is why the
    assertion here is that the columns stay empty.
    """
    backend = FakeBackend(
        [
            plan_reply("sql", ticker="NVDA", concept="Revenues"),
            "NVDA reported revenue of 1,000 USD [1]",
        ]
    )
    out = answer_agent(question(), tools=tools(backend, sql=FakeSql()), config=AGENT)
    assert out.route == "sql"
    assert "1,000" in out.answer
    assert out.retrieved == ()
    assert out.citations == ()


def test_an_abstention_reports_the_route_it_ended_on_not_the_one_it_wanted(env):
    """A SQL question that found nothing did, in the end, refuse -- and the
    baseline is scored by exactly that rule."""
    backend = FakeBackend([plan_reply("sql", ticker="NVDA", concept="Revenues")])
    out = answer_agent(
        question(),
        tools=tools(backend, sql=FakeSql(rows=()), retriever=FakeRetriever(chunks=[])),
        config=AGENT,
    )
    assert out.refused
    assert out.route == "refuse"
    assert out.llm_calls == 1


def test_a_deliberate_refusal_is_the_same_shape_as_a_failed_one():
    backend = FakeBackend([plan_reply("refuse", why="no filing holds a future price")])
    out = answer_agent(question("una-001", "unanswerable"), tools=tools(backend), config=AGENT)
    assert (out.route, out.refused, out.citations) == ("refuse", True, ())


def test_retrieval_only_reports_no_route_rather_than_a_guessed_one(env):
    """With no generator there is no router, and "" is how the scorer is told."""
    out = answer_agent(
        question("nar-001", "narrative"),
        tools=tools(None, generate=False),
        config=replace(AGENT, generate=False),
    )
    assert out.route == ""
    assert out.retrieved  # retrieval still ran
    assert out.llm_calls == 0


def test_a_thrown_exception_becomes_an_error_on_the_outcome():
    class Boom(FakeRetriever):
        def search(self, *a, **kw):  # noqa: ANN002, ANN003
            raise RuntimeError("qdrant is down")

    backend = FakeBackend([plan_reply("text", ticker="NVDA")])
    out = answer_agent(
        question("nar-001", "narrative"), tools=tools(backend, retriever=Boom()), config=AGENT
    )
    assert "qdrant is down" in out.error
    assert out.answer == ""


# ------------------------------------------------------------------- the run


def test_a_run_answers_scores_and_writes(env):
    backend = FakeBackend([plan_reply("text", ticker="NVDA"), "Revenue was 1,000 [1]"])
    report = runner.run(
        env, config="agent", agent_tools=tools(backend, sql=FakeSql()), use_cache=False, write=False
    )
    assert report.answered == 3
    assert report.card is not None
    assert report.errors == 0


def test_a_second_run_costs_nothing(env):
    backend = FakeBackend([plan_reply("text", ticker="NVDA"), "Revenue was 1,000 [1]"])
    first = runner.run(env, config="agent", agent_tools=tools(backend, sql=FakeSql()), write=False)
    spent = backend.calls
    second = runner.run(env, config="agent", agent_tools=None, write=False)
    assert first.answered == 3
    assert second.from_cache == 3
    assert second.llm_calls == 0
    assert backend.calls == spent  # nothing reached for the model again


def test_the_graph_is_compiled_once_for_the_run_not_once_per_question(env, monkeypatch):
    """Rebuilding a pure value 150 times buys nothing but wall clock."""
    import filing.agent.graph as g

    built = []
    real = g.build_graph

    def counting(t):  # noqa: ANN001
        built.append(t)
        return real(t)

    monkeypatch.setattr(g, "build_graph", counting)
    backend = FakeBackend([plan_reply("text", ticker="NVDA"), "Revenue was 1,000 [1]"])
    runner.run(
        env, config="agent", agent_tools=tools(backend, sql=FakeSql()), use_cache=False, write=False
    )
    assert len(built) == 1


def test_a_fact_citation_is_dropped_rather_than_counted_as_unresolvable():
    """`parse_citations` skips evidence with no chunk id, which is the seam."""
    fact = Evidence(kind="fact", body="b", citation="c", accn="a", value=1.0)
    chunk = Evidence(kind="text", body="b", citation="c", accn="a", chunk_id="c1")
    assert runner.parse_citations("both [1] and [2]", [fact, chunk]) == ("c1",)
