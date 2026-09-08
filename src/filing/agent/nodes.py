"""The nodes. One span each, and a hosted model in two of them.

The budget is the design. A five-call agent -- plan, route, grade, repair,
synthesise -- costs 750 calls over the 150-question eval set, which is three
quarters of a Cohere free month for one run of one experiment. This graph
spends two, and the savings are not shortcuts:

* **plan and route are one call.** Deciding what the question is asking and
  deciding which store answers it are the same act of reading. Splitting them
  buys a second opinion from the same model on its own output, which is not an
  independent check.
* **grading is deterministic.** M4's rule is that no language model sits in the
  metric path. An agent that grades itself with a model puts one back, and puts
  it in the loop that decides whether to spend more budget. A SQL row either
  came back or it did not; a text hit carries a cross-encoder logit from a local
  model; both are checkable offline, which also makes the repair decision
  reproducible.
* **repair is deterministic.** It picks the next thing to try from what the
  grade said was missing, and there are only a few things to try.
* **an abstention costs nothing.** When the grade is still bad after the repair
  budget, the answer is the refusal token, written without a call. Paying a
  model to say "insufficient evidence" is paying for a foregone conclusion.

That leaves the two places where judgement is actually needed: reading the
question, and writing the answer from evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from filing.agent.state import (
    REPAIR_BUDGET,
    AgentState,
    Evidence,
    Grade,
    Route,
    SubQuestion,
)

# The refusal token is M4's, imported rather than restated: an agent that
# refuses in different words than the baseline would be scored by a different
# rule, and comparing the two is the entire point of running both. The import
# goes this direction only -- the runner reaches the agent through a function-
# local import, so there is no cycle.
from filing.agent.verify import guard_answer, verify_answer
from filing.eval.runner import REFUSAL
from filing.tracing import get_tracer

ROUTES: tuple[Route, ...] = ("sql", "text", "graph", "refuse")

# The plan reply is about eighty tokens of JSON. This is not eighty.
#
# A reasoning model spends its output budget thinking before it writes, and it
# is one budget: the thinking is drawn from the same allowance as the text. On
# 2026-09-08 the chat model spent all but roughly ten tokens of a 256 budget on
# this prompt and returned an opening fence, a route and half a key -- valid
# JSON up to the cut, and worthless after it. Measured across seven eval
# questions, 256 failed to parse seven times out of seven and 1024 parsed seven
# out of seven, which is not a marginal call.
#
# The failure is silent by construction. `heuristic_plan` catches it and returns
# route="text", so a dead planner is indistinguishable from a working one that
# always chooses text: nothing raises, no route goes missing, and the only trace
# is a `why` field nobody reads. That is why this is a named constant with a
# comment rather than a literal -- the next model with a longer thinking habit
# will reintroduce the bug, and it will not announce itself either.
PLAN_MAX_TOKENS = 1024

# The cross-encoder is a binary relevance classifier trained with a logistic
# loss, so zero is its own decision boundary -- above it the model says
# "relevant", below it "not". Using that rather than a percentile of this
# corpus's scores keeps the floor from being a number fitted to the eval set,
# which is the same reason RRF's k was left at 60 in M3.
TEXT_SCORE_FLOOR = 0.0

# How many fused candidates the rerank node reads. Matches M3's FUSED_K so the
# agent's text branch is the retriever M3 measured, not a different one.
CANDIDATE_K = 50
TOP_N = 5

_TICKER = re.compile(r"\b([A-Z]{1,5})\b")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

PLAN_SYSTEM = (
    "You read a question about SEC filings and decide which store can answer it.\n"
    "Reply with JSON only, no prose, in this shape:\n"
    '{"route": "sql|text|graph|refuse", "ticker": "", "concept": "", '
    '"period_end": "YYYY-MM-DD", "why": ""}\n'
    "Rules:\n"
    "1. route=sql when the question asks for a single reported figure a company "
    "filed -- revenue, net income, total assets, and the like. Put the company's "
    "ticker in `ticker`, the financial concept in `concept` exactly as the "
    "question words it, and the period end date in `period_end` if the question "
    "gives one.\n"
    "2. route=text when the answer is prose in a filing: risks, strategy, "
    "management's discussion, why something changed.\n"
    "3. route=graph when the question is about a relationship between named "
    "organisations -- who supplies, competes with, or acquired whom.\n"
    "4. route=refuse when no SEC filing could contain the answer -- a share "
    "price today, a competitor's private data, a prediction about next year.\n"
    "5. `concept` is copied from the question, not translated. If the question "
    'says "Assets, Current", write that.'
)

SYNTH_SYSTEM = (
    "You answer questions about SEC filings using only the numbered evidence provided.\n"
    "Rules:\n"
    f"1. If the evidence does not contain the answer, reply with exactly: {REFUSAL}\n"
    "2. Never use knowledge from outside the evidence, and never estimate a number.\n"
    "3. Cite the evidence you used as bracketed numbers, e.g. [2] or [1][3].\n"
    "4. Be brief: two sentences at most, and give figures exactly as the "
    "evidence states them, with their units."
)


@dataclass
class Tools:
    """Everything the nodes reach for, injected rather than imported.

    A node that constructs its own store cannot be tested without one, and the
    repair loop is exactly the thing that most needs testing with stores that
    return nothing on purpose.
    """

    backend: Any = None
    sql: Any = None  # filing.agent.sql.SqlTool
    retriever: Any = None  # filing.stores.retrieve.Retriever
    graph: Any = None  # filing.agent.entities.GraphTool
    k: int = CANDIDATE_K
    top_n: int = TOP_N
    # Whether the cross-encoder runs at all. A switch rather than a constant
    # because the stage has to be ablatable: it is free (a local forward pass)
    # and therefore never questioned, which is exactly the kind of stage that
    # earns its place by assumption. Off, the node passes the fused order
    # through and the agent's text branch is RRF alone.
    rerank: bool = True
    # M6's verifier and guard. Off by default, and that default is load-bearing:
    # `agent` is the config M5's committed numbers were measured under, and a
    # gate that silently changed the behaviour of the previous gate's headline
    # run would make the comparison it exists to support meaningless. The
    # guarded config turns them on beside it.
    verify: bool = False
    guard: str = "block"
    temperature: float = 0.0
    max_tokens: int = 512
    chat_role: str = "chat"
    generate: bool = True
    tracer: Any = field(default_factory=lambda: get_tracer("filing.agent"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _json_object(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a model's reply, however it wrapped it.

    Models fence JSON, prefix it with "Here is the JSON:", or emit it bare, and
    which one is a property of the provider rather than of the question. A
    parser that only accepts the bare form would report a provider's formatting
    habit as a routing failure -- the same class of measurement error as the
    lenticular-bracket citation bug in M4.5.
    """
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        got = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return got if isinstance(got, dict) else {}


def heuristic_plan(question: str) -> SubQuestion:
    """The plan used when there is no generator, and when the model's reply is junk.

    Not a router -- it is a floor. It reads a ticker and a date out of the
    question with regexes and guesses ``text``, which is the baseline's
    behaviour. Its purpose is that a parse failure degrades to the M4 system
    rather than to a crash, so one malformed reply costs one question's accuracy
    instead of the run.
    """
    tickers = [t for t in _TICKER.findall(question) if len(t) >= 2]
    date = _DATE.search(question)
    return SubQuestion(
        text=question,
        route="text",
        ticker=tickers[0] if tickers else "",
        period_end=date.group(1) if date else "",
        why="heuristic: no plan call, or the plan reply did not parse",
    )


def build_synthesis_prompt(question: str, evidence: list[Evidence]) -> list[dict[str, str]]:
    """Numbered evidence, then the question. Same contract as M4's baseline.

    Numbered rather than cited by id for M4's reason -- a model cannot reproduce
    a UUID reliably, so asking for one measures transcription -- and the numbers
    are mapped back to citations after the call.
    """
    blocks = []
    for i, ev in enumerate(evidence, start=1):
        blocks.append(f"[{i}] {ev.citation}\n{ev.body}")
    body = "\n\n".join(blocks) if blocks else "(no evidence was retrieved)"
    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": f"Evidence:\n\n{body}\n\nQuestion: {question}"},
    ]


# --------------------------------------------------------------------------
# the nodes
# --------------------------------------------------------------------------


class Nodes:
    """The node bodies, bound to a set of tools.

    Methods rather than closures so each one is importable and callable on its
    own: every test below drives a single node with a hand-built state, which is
    the only way to test the repair loop without also testing three stores.
    """

    def __init__(self, tools: Tools) -> None:
        self.t = tools

    # -- plan ------------------------------------------------------------
    def plan(self, state: AgentState) -> dict[str, Any]:
        """One hosted call: what is being asked, and which store answers it."""
        question = state["question"]
        with self.t.tracer.start_as_current_span("agent.plan") as span:
            span.set_attribute("filing.question", question[:500])
            if not self.t.generate or self.t.backend is None:
                sub = heuristic_plan(question)
                span.set_attribute("filing.plan.route", sub.route)
                span.set_attribute("filing.plan.heuristic", True)
                return {"plan": [sub], "llm_calls": 0, "plan_note": sub.why}

            before = self.t.backend.usage().http_calls
            reply = self.t.backend.chat(
                [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {"role": "user", "content": question},
                ],
                role=self.t.chat_role,
                temperature=self.t.temperature,
                max_tokens=PLAN_MAX_TOKENS,
            )
            calls = self.t.backend.usage().http_calls - before
            got = _json_object(reply)
            if not got:
                sub = heuristic_plan(question)
                note = "plan reply did not parse as JSON"
            else:
                route = str(got.get("route", "")).strip().lower()
                sub = SubQuestion(
                    text=question,
                    route=route if route in ROUTES else "text",  # type: ignore[arg-type]
                    ticker=str(got.get("ticker") or "").strip().upper(),
                    concept=str(got.get("concept") or "").strip(),
                    period_end=str(got.get("period_end") or "").strip(),
                    why=str(got.get("why") or "").strip()[:200],
                )
                note = "" if route in ROUTES else f"unknown route {route!r}, fell back to text"
            span.set_attribute("filing.plan.route", sub.route)
            span.set_attribute("filing.plan.heuristic", False)
            return {"plan": [sub], "llm_calls": calls, "plan_note": note}

    # -- route -----------------------------------------------------------
    def route(self, state: AgentState) -> dict[str, Any]:
        """Deterministic. The plan proposes; availability disposes.

        The model can ask for a store that is not loaded, or for the SQL branch
        with a concept the registry does not know. Both are downgrades to text
        rather than errors, and both are recorded -- a router accuracy figure
        that silently included repairs the router did not make would be a
        flattering number about the wrong component.
        """
        plan = state.get("plan") or [heuristic_plan(state["question"])]
        sub = plan[0]
        route: Route = sub.route
        notes: list[str] = []
        with self.t.tracer.start_as_current_span("agent.route") as span:
            if route == "sql":
                if self.t.sql is None:
                    notes.append("no facts store; sql -> text")
                    route = "text"
                elif not sub.concept or self.t.sql.resolver.resolve(sub.concept) is None:
                    notes.append(f"{sub.concept!r} is not a registry concept; sql -> text")
                    route = "text"
            elif route == "graph" and (self.t.graph is None or not self.t.graph.available):
                notes.append("no graph store; graph -> text")
                route = "text"
            if route == "text" and self.t.retriever is None:
                notes.append("no retriever; text -> refuse")
                route = "refuse"
            span.set_attribute("filing.route", route)
            span.set_attribute("filing.route.downgraded", route != sub.route)
            return {"route": route, "repair_log": notes}

    # -- retrieve --------------------------------------------------------
    def retrieve_sql(self, state: AgentState) -> dict[str, Any]:
        sub = state["plan"][0]
        with self.t.tracer.start_as_current_span("agent.retrieve.sql") as span:
            period = sub.period_end or None
            answer = self.t.sql.lookup(sub.ticker, sub.concept, period_end=period)
            span.set_attribute("filing.sql.tag", answer.concept.tag if answer.concept else "")
            span.set_attribute("filing.sql.rows", len(answer.rows))
            evidence = [Evidence.from_fact(r) for r in answer.rows]
            return {
                "candidates": evidence,
                "evidence": evidence,
                "repair_log": [answer.reason] if answer.reason else [],
            }

    def retrieve_text(self, state: AgentState) -> dict[str, Any]:
        """Fused candidates only. The cross-encoder is the next node's job.

        M3's ``Retriever.search`` reranks by default; here it is asked not to,
        so that the rerank node is a node and its effect on the answer is
        visible in the trace rather than hidden inside a retriever call.
        """
        sub = state["plan"][0]
        with self.t.tracer.start_as_current_span("agent.retrieve.text") as span:
            hits = self.t.retriever.search(sub.text, k=self.t.k, rerank=False)
            span.set_attribute("filing.text.candidates", len(hits))
            return {"candidates": [Evidence.from_hit(h) for h in hits], "evidence": []}

    def retrieve_graph(self, state: AgentState) -> dict[str, Any]:
        sub = state["plan"][0]
        with self.t.tracer.start_as_current_span("agent.retrieve.graph") as span:
            found = self.t.graph.search(sub.text, ticker=sub.ticker, limit=self.t.top_n)
            span.set_attribute("filing.graph.edges", len(found))
            return {"candidates": found, "evidence": found}

    def refuse(self, state: AgentState) -> dict[str, Any]:
        """The router decided no store can answer. That is an answer, not a failure."""
        with self.t.tracer.start_as_current_span("agent.refuse"):
            return {"candidates": [], "evidence": []}

    # -- rerank ----------------------------------------------------------
    def rerank(self, state: AgentState) -> dict[str, Any]:
        """Cross-encoder over the fused candidates. A no-op for facts and edges.

        A fact is not more or less relevant than itself and an edge is already
        selected by the entity it names; running a text reranker over either
        would invent an ordering and, worse, a score the grader might read as a
        confidence.
        """
        candidates = list(state.get("candidates") or [])
        with self.t.tracer.start_as_current_span("agent.rerank") as span:
            span.set_attribute("filing.rerank.in", len(candidates))
            skip = not self.t.rerank or self.t.backend is None
            if not candidates or candidates[0].kind != "text" or skip:
                span.set_attribute("filing.rerank.applied", False)
                return {"evidence": candidates[: self.t.top_n]}
            rankings = self.t.backend.rerank(
                state["plan"][0].text,
                [c.body for c in candidates],
                top_n=self.t.top_n,
            )
            ranked = [
                Evidence(
                    kind=candidates[r.index].kind,
                    body=candidates[r.index].body,
                    citation=candidates[r.index].citation,
                    accn=candidates[r.index].accn,
                    score=float(r.score),
                    chunk_id=candidates[r.index].chunk_id,
                    char_start=candidates[r.index].char_start,
                    char_end=candidates[r.index].char_end,
                    period_end=candidates[r.index].period_end,
                    ticker=candidates[r.index].ticker,
                )
                for r in rankings[: self.t.top_n]
            ]
            span.set_attribute("filing.rerank.applied", True)
            span.set_attribute("filing.rerank.top_score", ranked[0].score if ranked else 0.0)
            return {"evidence": ranked}

    # -- grade -----------------------------------------------------------
    def grade(self, state: AgentState) -> dict[str, Any]:
        """Deterministic, and different per route because the evidence differs."""
        route = state.get("route", "text")
        evidence = list(state.get("evidence") or [])
        sub = state["plan"][0]
        with self.t.tracer.start_as_current_span("agent.grade") as span:
            grade = grade_evidence(route, evidence, sub)
            span.set_attribute("filing.grade.ok", grade.ok)
            span.set_attribute("filing.grade.reason", grade.reason)
            return {"grade": grade}

    # -- repair ----------------------------------------------------------
    def repair(self, state: AgentState) -> dict[str, Any]:
        """One bounded attempt to fix what the grade said was missing.

        Every strategy here is a thing a person would try next, and each is
        tried once. The counter is incremented here and read by the graph's
        conditional edge, so the bound is a property of the graph rather than of
        anybody remembering to check it.
        """
        n = state.get("repairs", 0) + 1
        grade = state.get("grade")
        sub = state["plan"][0]
        route = state.get("route", "text")
        with self.t.tracer.start_as_current_span("agent.repair") as span:
            span.set_attribute("filing.repair.attempt", n)
            span.set_attribute("filing.repair.missing", grade.missing if grade else "")
            new_sub, new_route, note = plan_repair(route, sub, grade, attempt=n, tools=self.t)
            span.set_attribute("filing.repair.action", note)
            return {
                "repairs": n,
                "plan": [new_sub],
                "route": new_route,
                "repair_log": [f"repair {n}: {note}"],
            }

    # -- synthesise ------------------------------------------------------
    def synthesise(self, state: AgentState) -> dict[str, Any]:
        """One hosted call, or none at all when the honest answer is a refusal."""
        grade = state.get("grade")
        evidence = list(state.get("evidence") or [])
        route = state.get("route", "text")
        with self.t.tracer.start_as_current_span("agent.synthesise") as span:
            span.set_attribute("filing.evidence.count", len(evidence))
            if route == "refuse" or grade is None or not grade.ok or not evidence:
                span.set_attribute("filing.synthesise.called", False)
                return {"answer": REFUSAL, "refused": True, "llm_calls": 0}
            if not self.t.generate or self.t.backend is None:
                span.set_attribute("filing.synthesise.called", False)
                return {"answer": "", "refused": False, "llm_calls": 0}
            before = self.t.backend.usage().http_calls
            text = self.t.backend.chat(
                build_synthesis_prompt(state["question"], evidence),
                role=self.t.chat_role,
                temperature=self.t.temperature,
                max_tokens=self.t.max_tokens,
            ).strip()
            calls = self.t.backend.usage().http_calls - before
            span.set_attribute("filing.synthesise.called", True)
            return {
                "answer": text,
                "refused": REFUSAL.lower() in text.lower(),
                "llm_calls": calls,
            }

    # -- verify ----------------------------------------------------------
    def verify(self, state: AgentState) -> dict[str, Any]:
        """Check the prose against the evidence it was written from. No call.

        The last node, and the only one that reads the model's output rather
        than feeding it. Everything it does is arithmetic and string matching,
        so it costs nothing and cannot itself hallucinate -- which is the whole
        argument for putting a deterministic verifier here instead of a second
        model asked whether the first one was telling the truth.

        The verdict is computed whatever the guard mode is. Blocking is a
        decision about what to *do* with the finding; the finding itself goes on
        the span either way, because a run that suppressed its own failure
        taxonomy when the guard was off would have no failure gallery to filter.
        """
        answer = state.get("answer", "")
        evidence = list(state.get("evidence") or [])
        with self.t.tracer.start_as_current_span("agent.verify") as span:
            if not self.t.verify:
                span.set_attribute("filing.verify.enabled", False)
                return {}
            verdict = verify_answer(
                answer,
                evidence,
                question=state.get("question", ""),
                route=state.get("route", ""),
                grade=state.get("grade"),
                repairs=state.get("repairs", 0),
                repair_log=list(state.get("repair_log") or []),
                sql=self.t.sql,
                refused=bool(state.get("refused")),
            )
            guarded, blocked = guard_answer(answer, verdict, mode=self.t.guard, refusal=REFUSAL)
            span.set_attribute("filing.verify.enabled", True)
            span.set_attribute("filing.verify.ok", verdict.ok)
            span.set_attribute("filing.verify.mode", self.t.guard)
            span.set_attribute("filing.verify.blocked", blocked)
            span.set_attribute("filing.verify.figures", len(verdict.checked))
            span.set_attribute("filing.verify.unsupported", len(verdict.unsupported))
            span.set_attribute("filing.verify.citations", len(verdict.markers))
            span.set_attribute("filing.verify.dangling", len(verdict.dangling))
            span.set_attribute("filing.verify.unlocatable", len(verdict.unlocatable))
            span.set_attribute("filing.verify.uncited", len(verdict.uncited))
            # One list attribute and one joined string. Phoenix filters on the
            # string ("failure.kind contains synthesis_drift"), which is what
            # makes the gallery a filter rather than a manual read; the list is
            # for anything reading the span programmatically.
            span.set_attribute("filing.failure.kind", ",".join(verdict.taxonomy))
            span.set_attribute("filing.failure.kinds", list(verdict.taxonomy))
            span.set_attribute("filing.verify.reason", "; ".join(verdict.reasons)[:400])
            out: dict[str, Any] = {"verdict": verdict, "flagged": not verdict.ok}
            if blocked:
                out["answer"] = guarded
                out["refused"] = True
                out["blocked"] = True
            return out


# --------------------------------------------------------------------------
# the deterministic parts, as free functions so they can be tested alone
# --------------------------------------------------------------------------


def grade_evidence(route: str, evidence: list[Evidence], sub: SubQuestion) -> Grade:
    """Is this evidence enough, and if not, which repair does that imply?

    ``missing`` is the whole reason this returns a record instead of a boolean:
    "the company reports no such tag" and "the retriever came back with the
    wrong company" want completely different next moves, and a grader that only
    says "no" leaves the repair node guessing.
    """
    if route == "refuse":
        return Grade(ok=True, reason="router declined: no store can answer this")
    if not evidence:
        return Grade(ok=False, reason="no evidence", missing="anything")

    if route == "sql":
        top = evidence[0]
        if top.value is None:
            return Grade(ok=False, reason="row without a value", missing="value")
        return Grade(
            ok=True,
            reason=f"fact found: {top.tag} {top.period_end}",
            detail={"tag": top.tag, "period_end": top.period_end},
        )

    if route == "graph":
        return Grade(ok=True, reason=f"{len(evidence)} edges", detail={"edges": len(evidence)})

    top = max(e.score for e in evidence)
    if sub.ticker and not any(e.ticker == sub.ticker for e in evidence):
        # Scale-free and the most common real failure: the right kind of
        # paragraph from the wrong company. Worth its own repair.
        return Grade(
            ok=False,
            reason=f"no evidence from {sub.ticker}",
            missing="company",
            detail={"top_score": top, "tickers": sorted({e.ticker for e in evidence})},
        )
    if top < TEXT_SCORE_FLOOR:
        return Grade(
            ok=False,
            reason=f"top rerank score {top:.2f} below the classifier's own boundary",
            missing="relevance",
            detail={"top_score": top},
        )
    return Grade(ok=True, reason=f"top rerank score {top:.2f}", detail={"top_score": top})


def plan_repair(
    route: str,
    sub: SubQuestion,
    grade: Grade | None,
    *,
    attempt: int,
    tools: Tools,
) -> tuple[SubQuestion, Route, str]:
    """What to try next. Pure, so the loop's behaviour is a table, not a mystery.

    The new route is written onto the sub-question rather than returned beside
    it, because the ``route`` node re-derives the route from ``plan[0].route``
    every time it runs and a repair re-enters there. Returning a route the
    sub-question contradicts would let the two disagree, and the node would win
    -- which is how a "fall back to the text retriever" repair can be reported
    in the log and never actually happen.
    """
    missing = grade.missing if grade else "anything"

    def amend(note: str, new_route: Route, **fields: str) -> tuple[SubQuestion, Route, str]:
        changed = {**sub.as_dict(), **fields, "route": new_route}
        return SubQuestion(**changed), new_route, note  # type: ignore[arg-type]

    if route == "sql":
        # A concept the company reports under a sibling tag, or a period the
        # question named that the company's fiscal calendar does not have.
        if attempt == 1 and sub.period_end:
            return amend("drop the period and take the most recent filing", "sql", period_end="")
        return amend("sql found nothing; fall back to the text retriever", "text")

    if route == "graph":
        return amend("graph found nothing; fall back to the text retriever", "text")

    # text
    if missing == "company" and sub.ticker:
        return amend(
            f"re-query with the ticker in front: {sub.ticker}",
            "text",
            text=f"{sub.ticker} {sub.text}",
        )
    if sub.concept:
        return amend("re-query with the concept appended", "text", text=f"{sub.text} {sub.concept}")
    return amend("no repair left to try", "text")


def should_repair(state: AgentState) -> str:
    """The conditional edge. The budget lives here, in the graph, once."""
    grade = state.get("grade")
    if grade is not None and grade.ok:
        return "synthesise"
    if state.get("repairs", 0) >= REPAIR_BUDGET:
        return "synthesise"
    return "repair"


def route_branch(state: AgentState) -> str:
    """Which retrieval node the router chose."""
    route = state.get("route", "text")
    return {
        "sql": "retrieve_sql",
        "text": "retrieve_text",
        "graph": "retrieve_graph",
        "refuse": "refuse",
    }[route]
