"""The state the graph carries, and the one evidence type every store speaks.

Three stores answer questions here and they have nothing in common: DuckDB
returns a row, the retriever returns a chunk with a rerank score, the graph
returns an edge with a sentence. If each reached the synthesiser in its own
shape, the prompt would have three branches, the grader would have three code
paths, and "cite your evidence" would mean three different things.

So they are normalised at the edge of each retrieval node into one
:class:`Evidence` record. The record keeps what makes a claim checkable -- an
accession number, and either a character span or a tag and period -- and throws
away everything a synthesiser cannot use. That is the "unified evidence schema"
the M5 gate asks for, and its real job is that **an answer citing a fact and an
answer citing a paragraph are audited the same way**.

The state itself is a ``TypedDict`` because LangGraph merges partial updates
into it: a node returns the keys it changed and the framework does the rest. It
is total=False for the same reason -- a node that only sets ``evidence`` should
not have to restate the question.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

# Two, and the bound is structural rather than advisory: the graph's conditional
# edge reads the counter, so a repair loop that wanted to run three times cannot
# -- there is no edge for it to take. The number is small on purpose. A repair
# that has not helped twice is not going to help on the third pass; it is going
# to spend budget converting a clean abstention into a confident wrong answer.
REPAIR_BUDGET = 2

#: Which store answers. ``refuse`` is a route, not a failure: deciding that no
#: store can answer is a decision the router makes on purpose, and the eval set
#: scores it as such.
Route = Literal["sql", "text", "graph", "refuse"]

#: Where a piece of evidence came from, and therefore how it is checked.
EvidenceKind = Literal["fact", "text", "edge"]


@dataclass(frozen=True, slots=True)
class SubQuestion:
    """One answerable piece of the question, with the store that should answer it.

    The planner emits these. A single-company point lookup produces exactly one;
    "how did X's margin compare with Y's" produces two, each with its own route,
    which is what makes the shape worth having even though the frozen v1.0 eval
    set never asks such a question.
    """

    text: str
    route: Route = "text"
    ticker: str = ""
    concept: str = ""
    period_end: str = ""
    why: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "route": self.route,
            "ticker": self.ticker,
            "concept": self.concept,
            "period_end": self.period_end,
            "why": self.why,
        }


@dataclass(frozen=True, slots=True)
class Evidence:
    """One checkable thing, whichever store it came from.

    ``body`` is what the synthesiser reads. Everything else exists so that a
    reader can go back to the filing and see for themselves -- which is the
    whole claim this project makes, and it survives only if every branch
    carries the same locators.
    """

    kind: EvidenceKind
    body: str
    citation: str
    accn: str
    score: float = 0.0
    # text and edge evidence: the span in the filing
    chunk_id: str = ""
    char_start: int = 0
    char_end: int = 0
    # fact evidence: the number and what it is
    value: float | None = None
    unit: str = ""
    tag: str = ""
    period_end: str = ""
    ticker: str = ""

    @classmethod
    def from_fact(cls, row: Any) -> Evidence:
        """A :class:`filing.agent.sql.FactRow`. Score 1.0 -- a row is not ranked.

        There is no "how relevant is this fact": it is the fact that was asked
        for, or the query returned nothing. Giving it a similarity score would
        put a made-up number where the grader looks for a real one.
        """
        return cls(
            kind="fact",
            body=f"{row.ticker} reported {row.label} of {row.val:,.0f} {row.unit} "
            f"for the period ending {row.period_end} (tag {row.tag}).",
            citation=row.citation,
            accn=row.accn,
            score=1.0,
            value=row.val,
            unit=row.unit,
            tag=row.tag,
            period_end=str(row.period_end),
            ticker=row.ticker,
        )

    @classmethod
    def from_hit(cls, hit: Any) -> Evidence:
        """A :class:`filing.stores.retrieve.Hit`.

        Score is the cross-encoder's where there is one and the fusion score
        otherwise, and the grader is told which -- an RRF score near 0.016 and a
        rerank score near 8 are not on the same scale and must never be compared
        as though they were.
        """
        chunk = hit.chunk
        return cls(
            kind="text",
            body=chunk.text,
            citation=hit.citation,
            accn=chunk.accn,
            score=float(hit.rerank_score if hit.rerank_score is not None else hit.fused_score),
            chunk_id=chunk.chunk_id,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            period_end=str(chunk.period_end),
            ticker=chunk.ticker,
        )

    @classmethod
    def from_edge(cls, edge: Any, *, score: float = 0.0) -> Evidence:
        """A :class:`filing.stores.graph.Edge`, cited by the sentence it came from."""
        item = f" Item {edge.item_key}" if edge.item_key else ""
        return cls(
            kind="edge",
            body=f"{edge.source} --{edge.kind}--> {edge.target}: {edge.sentence}",
            citation=(
                f"{edge.ticker} {edge.form} {edge.period_end}{item} "
                f"[{edge.accn} {edge.char_start}:{edge.char_end}]"
            ),
            accn=edge.accn,
            score=score,
            char_start=edge.char_start,
            char_end=edge.char_end,
            period_end=str(edge.period_end),
            ticker=edge.ticker,
        )


@dataclass(frozen=True, slots=True)
class Grade:
    """Whether the evidence supports an answer, and if not, what is missing.

    Deterministic by construction. M4's rule is that no language model sits in
    the metric path; an agent that asks a model "is this good enough?" puts one
    back, and worse, puts it in the loop that decides whether to spend more
    budget. A SQL row either came back or it did not. A text hit carries a
    cross-encoder score from a local model. Both are checkable without a
    network, which is also why the repair decision is reproducible.
    """

    ok: bool
    reason: str
    missing: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """What flows through the graph.

    Two keys accumulate rather than being overwritten, and they are annotated
    with a reducer so that the framework does it rather than each node
    remembering to. Both are keys that more than one node writes: the call
    counter is incremented by ``plan`` and again by ``synthesise``, and a
    last-writer-wins merge would report a two-call question as a one-call
    question -- the budget number quietly halved, in the direction that
    flatters. The repair log has the same shape and the same failure.
    """

    # set by the caller
    qid: str
    question: str

    # plan / route
    plan: list[SubQuestion]
    route: Route
    plan_note: str

    # retrieve / rerank
    evidence: list[Evidence]
    candidates: list[Evidence]  # pre-rerank, so the node's effect is measurable

    # grade / repair
    grade: Grade
    repairs: int
    repair_log: Annotated[list[str], operator.add]

    # synthesise
    answer: str
    refused: bool

    # bookkeeping the runner reads back
    llm_calls: Annotated[int, operator.add]
    seconds: float
    error: str
