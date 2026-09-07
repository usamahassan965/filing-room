"""The answer, and everything a reader needs to disbelieve it.

Phoenix is the developer's view of this system: spans, latencies, the shape of
the graph. It is the wrong surface for the question a user actually has, which
is not "how long did the rerank take" but "why should I believe that number?"
This module is the other half -- the same run, described in terms of the claim
rather than the machinery.

The design rule is one sentence: **the API carries the evidence, the UI renders
it.** Every panel in :mod:`filing.ui` is a field in :class:`Answer`, and there
is nothing the interface knows that a ``curl`` of ``/ask`` does not. That is not
tidiness. A transparency surface whose explanation is assembled in the front end
is a surface that can drift from the run it claims to describe, and no reviewer
would be able to tell -- the screenshot would look identical either way. Keeping
the payload authoritative makes the claim checkable: pipe the JSON to a file and
every citation, figure and verdict shown on screen is in it.

So the response is a record of the decisions, in the order the graph made them:

* the **plan** the model wrote, and the **route** the deterministic router
  actually took -- separately, because a downgrade from ``sql`` to ``text`` is
  the most interesting thing that happens on a bad question and collapsing the
  two would hide it;
* the **evidence** the synthesiser was shown, each item numbered with the marker
  the answer cites it by, and flagged for whether it carries a locator;
* the answer split into **claims**, each with the markers it cites and the
  figures it states -- the same sentence split the guard's uncited-claim check
  uses, so the panel is showing the decomposition that was checked rather than
  one invented for display;
* the **verification**, figure by figure: what was stated, whether the evidence
  carried it, and *how* it was matched -- printed digits, a rescaled reading, or
  arithmetic over two facts, in which case the derivation is spelled out so the
  computed value sits beside its inputs;
* the **outcome**, which distinguishes four states rather than two. ``answered``
  and ``error`` are obvious; ``abstained`` is the agent declining, and
  ``blocked`` is the guard overruling an answer the agent was willing to give.
  Both print the refusal token, and treating them as one state would make the
  guard invisible in exactly the runs where it did something.

``/ask/stream`` is the same payload arriving in stages. It exists because the
plan and the route are known a second in, and the evidence a few seconds later,
while the answer takes as long as the generator takes -- a UI that waits for all
of it shows a spinner during the part of the run that is most worth watching.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from filing.agent.verify import figures_in, has_locator, markers_in, sentences
from filing.config import Settings, settings
from filing.eval.runner import REFUSAL

__all__ = [
    "DEFAULT_CONFIG",
    "Answer",
    "Ask",
    "AskEngine",
    "EngineNotReady",
    "create_app",
    "payload_for",
    "serve",
]

#: The guard on, in blocking mode. M6's `agent` config leaves verification off
#: because it is the run M5's committed numbers were measured under; a user
#: facing surface has no such obligation, and shipping the unverified config
#: behind a page whose whole subject is verification would be absurd.
DEFAULT_CONFIG = "agent-guarded"

#: Node name -> what the interface should say is happening. The graph's node
#: names are the vocabulary of the trace, so the stream speaks them rather than
#: inventing a parallel set of stage names that would then have to be kept in
#: step with the graph by hand.
STAGES: dict[str, str] = {
    "plan": "reading the question",
    "route": "choosing a store",
    "retrieve_sql": "querying the facts store",
    "retrieve_text": "searching the filings",
    "retrieve_graph": "walking the entity graph",
    "refuse": "no store can answer this",
    "rerank": "reranking the candidates",
    "grade": "checking the evidence",
    "repair": "repairing and re-routing",
    "synthesise": "writing the answer",
    "verify": "verifying the answer",
}


class EngineNotReady(RuntimeError):
    """The stores this needs are not on disk. A 503, not a 500."""


# --------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------


class PlanStep(BaseModel):
    """One piece of the question and the store the planner wanted for it."""

    text: str = ""
    route: str = ""
    ticker: str = ""
    concept: str = ""
    period_end: str = ""
    why: str = ""


class EvidenceRecord(BaseModel):
    """One thing the synthesiser was shown, numbered by the marker that cites it.

    ``locatable`` is the field that matters and the one a demo would omit: it
    says whether a reader can get from this record back to a span of a filing.
    A citation that cannot be followed is the failure this project is built to
    not commit, so the surface reports it per item rather than in aggregate.
    """

    marker: int
    kind: str
    body: str
    citation: str
    accn: str = ""
    score: float = 0.0
    cited: bool = False
    locatable: bool = False
    chunk_id: str = ""
    char_start: int = 0
    char_end: int = 0
    value: float | None = None
    unit: str = ""
    tag: str = ""
    period_end: str = ""
    ticker: str = ""


class NumberCheck(BaseModel):
    """One figure the answer states, and what the verifier made of it.

    ``matched`` and ``derivation`` are separate fields because they are
    different claims. ``matched`` names the evidence whose digits or rescaled
    reading the figure landed on. ``derivation`` is prose the verifier wrote
    describing the arithmetic it performed -- "change from 60,922 to 65,000" --
    and is the only place the computed value and its inputs appear together.
    """

    text: str
    value: float
    status: str
    how: str = ""
    matched: str = ""
    derivation: str = ""
    marker: int = 0


class Claim(BaseModel):
    """One sentence of the answer, with its citations and its figures."""

    text: str
    markers: list[int] = Field(default_factory=list)
    figures: list[NumberCheck] = Field(default_factory=list)
    cited: bool = False


class Verification(BaseModel):
    """The verdict, in the shape a panel can render without arithmetic.

    ``enabled`` is not decoration. When the guard is off the payload still has
    this object, with every list empty -- and a UI that could not tell "nothing
    was wrong" from "nothing was checked" would print a green tick over an
    unverified answer.
    """

    enabled: bool = False
    ok: bool = True
    mode: str = ""
    blocked: bool = False
    numbers: list[NumberCheck] = Field(default_factory=list)
    figures_checked: int = 0
    supported: int = 0
    unsupported: int = 0
    markers: list[int] = Field(default_factory=list)
    dangling: list[int] = Field(default_factory=list)
    unlocatable: list[int] = Field(default_factory=list)
    uncited: list[str] = Field(default_factory=list)
    unrechecked: list[str] = Field(default_factory=list)
    taxonomy: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class GradeReport(BaseModel):
    ok: bool = False
    reason: str = ""
    missing: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class TraceLink(BaseModel):
    """Where this run is in Phoenix, when Phoenix is listening.

    ``live`` is false when nothing is exporting, and then ``url`` is empty
    rather than a plausible link to a page that will report the trace as not
    found. A dead link on a transparency surface is worse than no link.
    """

    trace_id: str = ""
    url: str = ""
    project: str = ""
    live: bool = False


class Answer(BaseModel):
    """One question, answered, with the whole decision path attached."""

    qid: str
    question: str
    answer: str = ""
    outcome: str = "answered"  # answered | abstained | blocked | error
    refused: bool = False
    config: str = DEFAULT_CONFIG
    plan: list[PlanStep] = Field(default_factory=list)
    plan_note: str = ""
    route: str = ""
    grade: GradeReport = Field(default_factory=GradeReport)
    repairs: int = 0
    repair_log: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verification: Verification = Field(default_factory=Verification)
    trace: TraceLink = Field(default_factory=TraceLink)
    llm_calls: int = 0
    seconds: float = 0.0
    error: str = ""


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    #: Which of the eval configs to run under. Named rather than a pile of
    #: knobs, so a screenshot of the UI names an experiment that
    #: ``python -m filing.eval run`` can reproduce.
    config: str = DEFAULT_CONFIG


# --------------------------------------------------------------------------
# state -> payload
# --------------------------------------------------------------------------


def _outcome(state: dict[str, Any], answer: str, *, blocked: bool) -> str:
    if state.get("error"):
        return "error"
    if blocked:
        return "blocked"
    if state.get("refused") or (answer and REFUSAL.lower() in answer.lower()):
        return "abstained"
    return "answered"


def _number_checks(numbers: tuple[Any, ...], by_citation: dict[str, int]) -> list[NumberCheck]:
    out: list[NumberCheck] = []
    for n in numbers:
        derived = n.how == "recomputed"
        out.append(
            NumberCheck(
                text=n.text,
                value=n.value,
                status=n.status,
                how=n.how,
                matched="" if derived else n.source,
                derivation=n.source if derived else "",
                marker=0 if derived else by_citation.get(n.source, 0),
            )
        )
    return out


def _claims(answer: str, checks: list[NumberCheck]) -> list[Claim]:
    """Split the answer, and hand each sentence the figures it states.

    Matched by the text of the figure rather than by an offset, because the
    verifier reads the whole answer at once and does not keep spans. A figure
    is consumed from the pool as it is placed, so "revenue rose to $60.9
    billion from $60.9 billion" puts one in each sentence rather than the same
    one in both; a figure that cannot be placed is simply left out of the panel,
    which is the failure that shows too little rather than the wrong thing.
    """
    pool = list(checks)
    out: list[Claim] = []
    for sentence in sentences(answer):
        figures: list[NumberCheck] = []
        for text in figures_in(sentence):
            i = next((j for j, c in enumerate(pool) if c.text == text), None)
            if i is not None:
                figures.append(pool.pop(i))
        markers = markers_in(sentence)
        out.append(
            Claim(text=sentence, markers=list(markers), figures=figures, cited=bool(markers))
        )
    return out


def payload_for(
    state: dict[str, Any],
    *,
    question: str,
    qid: str,
    config: str = DEFAULT_CONFIG,
    guard: str = "",
    trace_id: str = "",
    cfg: Settings | None = None,
) -> Answer:
    """Everything the run decided, in one value. No model, no network.

    Pure over the finished state, which is what makes it testable without a
    corpus: every test of this surface hands it a hand-built state and asserts
    on the payload, and the four outcome states are reachable that way.
    """
    cfg = cfg or settings()
    answer = str(state.get("answer") or "")
    blocked = bool(state.get("blocked"))
    verdict = state.get("verdict")

    evidence_in = list(state.get("evidence") or [])
    cited = set(markers_in(answer))
    records: list[EvidenceRecord] = []
    by_citation: dict[str, int] = {}
    for i, e in enumerate(evidence_in, start=1):
        citation = getattr(e, "citation", "")
        by_citation.setdefault(citation, i)
        records.append(
            EvidenceRecord(
                marker=i,
                kind=getattr(e, "kind", ""),
                body=getattr(e, "body", ""),
                citation=citation,
                accn=getattr(e, "accn", ""),
                score=float(getattr(e, "score", 0.0) or 0.0),
                cited=i in cited,
                locatable=has_locator(e),
                chunk_id=getattr(e, "chunk_id", ""),
                char_start=int(getattr(e, "char_start", 0) or 0),
                char_end=int(getattr(e, "char_end", 0) or 0),
                value=getattr(e, "value", None),
                unit=getattr(e, "unit", ""),
                tag=getattr(e, "tag", ""),
                period_end=str(getattr(e, "period_end", "") or ""),
                ticker=getattr(e, "ticker", ""),
            )
        )

    checks = _number_checks(tuple(getattr(verdict, "numbers", ()) or ()), by_citation)
    verification = Verification(enabled=verdict is not None, mode=guard, blocked=blocked)
    if verdict is not None:
        verification = Verification(
            enabled=True,
            ok=bool(verdict.ok),
            mode=guard,
            blocked=blocked,
            numbers=checks,
            figures_checked=len(verdict.checked),
            supported=sum(1 for n in verdict.checked if n.status != "unsupported"),
            unsupported=len(verdict.unsupported),
            markers=list(verdict.markers),
            dangling=list(verdict.dangling),
            unlocatable=list(verdict.unlocatable),
            uncited=list(verdict.uncited),
            unrechecked=list(verdict.unrechecked),
            taxonomy=list(verdict.taxonomy),
            reasons=list(verdict.reasons),
        )

    grade = state.get("grade")
    grade_report = GradeReport()
    if grade is not None:
        grade_report = GradeReport(
            ok=bool(getattr(grade, "ok", False)),
            reason=str(getattr(grade, "reason", "")),
            missing=str(getattr(grade, "missing", "")),
            detail=dict(getattr(grade, "detail", {}) or {}),
        )

    return Answer(
        qid=qid,
        question=question,
        answer=answer,
        outcome=_outcome(state, answer, blocked=blocked),
        refused=bool(state.get("refused")),
        config=config,
        plan=[PlanStep(**p.as_dict()) for p in (state.get("plan") or [])],
        plan_note=str(state.get("plan_note") or ""),
        route=str(state.get("route") or ""),
        grade=grade_report,
        repairs=int(state.get("repairs") or 0),
        repair_log=list(state.get("repair_log") or []),
        evidence=records,
        claims=_claims(answer, checks),
        verification=verification,
        trace=_trace_link(trace_id, cfg),
        llm_calls=int(state.get("llm_calls") or 0),
        seconds=round(float(state.get("seconds") or 0.0), 3),
        error=str(state.get("error") or ""),
    )


def _trace_link(trace_id: str, cfg: Settings) -> TraceLink:
    from filing.gallery import trace_url
    from filing.tracing import tracing_is_live

    live = bool(trace_id) and tracing_is_live()
    return TraceLink(
        trace_id=trace_id,
        url=trace_url(trace_id, cfg=cfg) if live else "",
        project=cfg.phoenix_project,
        live=live,
    )


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------


@dataclass
class AskEngine:
    """The graph, built once and kept.

    Building it opens the Qdrant collection, loads the BM25 index and pulls the
    cross-encoder into memory -- seconds of work that must not happen per
    request. It is deliberately *not* done at import: a missing corpus should
    make ``/ask`` answer 503 with a sentence saying which command builds it, not
    make the process fail to start and take the health check down with it.
    """

    config: str = DEFAULT_CONFIG
    cfg: Settings = field(default_factory=settings)
    tools: Any = None
    graph: Any = None
    resolved: Any = None

    @property
    def ready(self) -> bool:
        return self.graph is not None

    def warm(self) -> AskEngine:
        """Open the stores and compile the graph. Idempotent."""
        if self.ready:
            return self
        from filing.agent.graph import build_graph
        from filing.eval.runner import build_agent_tools, get_config
        from filing.llm.factory import build_backend
        from filing.tracing import setup_tracing

        setup_tracing(self.cfg)
        try:
            ec = get_config(self.config).resolved(self.cfg)
            backend = build_backend(self.cfg, ec.chat_backend or None) if ec.generate else None
            self.tools = build_agent_tools(self.cfg, backend=backend, config=ec)
        except Exception as exc:  # noqa: BLE001 - reported, not raised through
            raise EngineNotReady(
                f"{type(exc).__name__}: {exc}. Build the corpus first: "
                "`filing ingest`, `filing facts`, `filing chunks`, `filing index`."
            ) from exc
        self.resolved = ec
        self.graph = build_graph(self.tools)
        return self

    @property
    def guard(self) -> str:
        """The guard mode in force, or empty when verification is off.

        Empty rather than the tools' default, because ``Tools.guard`` holds
        ``"block"`` whether or not the verifier runs, and reporting that as the
        mode would put "blocking" on a payload where nothing was checked.
        """
        return str(getattr(self.tools, "guard", "")) if getattr(self.tools, "verify", False) else ""

    def _span(self) -> Any:
        """One span per request, so the trace id on the payload names this run.

        The graph opens ``agent.question`` inside it, which makes the whole
        request a single trace rather than one root per node -- and the link the
        UI shows lands on a tree, not on a fragment.
        """
        from filing.tracing import get_tracer

        return get_tracer("filing.api").start_as_current_span("api.ask")

    def ask(self, question: str, *, qid: str = "") -> Answer:
        """One question, start to finish, with the trace id captured."""
        from filing.agent.graph import run_question

        self.warm()
        qid = qid or uuid.uuid4().hex[:12]
        with self._span() as span:
            span.set_attribute("filing.api.qid", qid)
            span.set_attribute("filing.api.question", question[:500])
            span.set_attribute("filing.api.config", self.config)
            trace_id = _trace_id(span)
            state = run_question(question, tools=self.tools, qid=qid, graph=self.graph)
        return payload_for(
            dict(state),
            question=question,
            qid=qid,
            config=self.config,
            guard=self.guard,
            trace_id=trace_id,
            cfg=self.cfg,
        )

    def stream(self, question: str, *, qid: str = "") -> Iterator[dict[str, Any]]:
        """The same run, one event per node, then the payload.

        ``updates`` names the node that just finished and ``values`` carries the
        state as the reducers have merged it -- both are asked for, because
        neither alone is enough: the node names have no accumulated state
        attached and the accumulated state does not say what produced it.

        A failure mid-stream is an ``error`` event rather than a dropped
        connection. The client has already rendered a plan and a route by then,
        and leaving that on screen under a spinner that never resolves is the
        error state this gate says must not happen.

        The run happens on a thread of its own and this generator only drains a
        queue, which is not an optimisation. An OpenTelemetry context token is a
        ``contextvars`` reset token, and a token taken on one thread cannot be
        reset on another. Starlette pumps a synchronous SSE generator through
        its thread pool one ``__next__`` at a time, so a span opened *around*
        the ``yield`` gets entered on one worker and exited on whichever worker
        happens to take the last step -- which logs ``Failed to detach context``
        on every single request and leaves the span's parenting to luck. Owning
        the span on one thread makes both ends of it happen in one context.
        """
        self.warm()
        qid = qid or uuid.uuid4().hex[:12]
        started = time.monotonic()
        state: dict[str, Any] = {
            "qid": qid,
            "question": question,
            "repairs": 0,
            "llm_calls": 0,
            "repair_log": [],
            "evidence": [],
            "candidates": [],
        }
        stage: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
        box: dict[str, Any] = {"final": dict(state), "trace_id": ""}

        def run() -> None:
            try:
                with self._span() as span:
                    span.set_attribute("filing.api.qid", qid)
                    span.set_attribute("filing.api.question", question[:500])
                    span.set_attribute("filing.api.config", self.config)
                    span.set_attribute("filing.api.stream", True)
                    box["trace_id"] = _trace_id(span)
                    try:
                        for mode, chunk in self.graph.stream(
                            state, {"recursion_limit": 40}, stream_mode=["updates", "values"]
                        ):
                            if mode == "values":
                                box["final"] = dict(chunk)
                                continue
                            for node in chunk:
                                stage.put(
                                    ("stage", {"node": node, "label": STAGES.get(node, node)})
                                )
                    except Exception as exc:  # noqa: BLE001 - one bad question is an event
                        box["final"]["error"] = f"{type(exc).__name__}: {exc}"
                        box["final"].setdefault("answer", "")
            finally:
                stage.put(None)

        yield {"event": "start", "data": {"qid": qid, "question": question, "config": self.config}}
        worker = threading.Thread(target=run, name=f"ask-{qid}", daemon=True)
        worker.start()
        while True:
            item = stage.get()
            if item is None:
                break
            kind, data = item
            yield {"event": kind, "data": data}
        worker.join()

        final: dict[str, Any] = box["final"]
        final["seconds"] = time.monotonic() - started
        payload = payload_for(
            final,
            question=question,
            qid=qid,
            config=self.config,
            guard=self.guard,
            trace_id=str(box["trace_id"]),
            cfg=self.cfg,
        )
        yield {"event": "answer", "data": payload.model_dump()}


def _trace_id(span: Any) -> str:
    try:
        ctx = span.get_span_context()
    except Exception:  # noqa: BLE001 - a no-op span has no context
        return ""
    tid = getattr(ctx, "trace_id", 0)
    return format(tid, "032x") if tid else ""


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------


def create_app(engine: AskEngine | None = None, *, warm: bool = False) -> Any:
    """The FastAPI app, with the engine injectable.

    Injectable because every test of this module drives it with an engine that
    returns a canned payload: the routes, the status codes and the error shapes
    are worth testing, and none of them need a corpus or a model to be tested.
    """
    from fastapi import FastAPI, HTTPException

    engine = engine or AskEngine()
    app = FastAPI(
        title="Filing Room",
        version="0.1.0",
        summary="Agentic RAG over SEC filings, with the evidence attached.",
        description=__doc__,
    )
    app.state.engine = engine
    if warm:  # pragma: no cover - exercised by `filing serve`, not by tests
        engine.warm()

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Up, and whether the stores are open yet. Never raises."""
        return {
            "ok": True,
            "ready": engine.ready,
            "config": engine.config,
            "guard": engine.guard,
            "project": engine.cfg.phoenix_project,
        }

    @app.post("/ask", response_model=Answer)
    def ask(body: Ask) -> Answer:
        if body.config and body.config != engine.config:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"this server runs the {engine.config!r} config; "
                    f"restart it with --config {body.config} to change that"
                ),
            )
        try:
            return engine.ask(body.question)
        except EngineNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/ask/stream")
    def ask_stream(body: Ask) -> Any:
        """The same answer, staged. Server-sent events, one JSON object each."""
        from sse_starlette.sse import EventSourceResponse

        if body.config and body.config != engine.config:
            raise HTTPException(
                status_code=400,
                detail=f"this server runs the {engine.config!r} config",
            )
        try:
            engine.warm()
        except EngineNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def events() -> Iterator[dict[str, str]]:
            for item in engine.stream(body.question):
                yield {"event": item["event"], "data": json.dumps(item["data"], default=str)}

        return EventSourceResponse(events())

    # Annotated `dict[str, str]` rather than a response class: this module has
    # `from __future__ import annotations`, so every annotation reaches FastAPI
    # as a string, and a starlette response type it cannot resolve into a
    # pydantic model takes /openapi.json down with it -- which is a broken
    # schema on the surface whose whole argument is its published contract.
    @app.get("/")
    def index() -> dict[str, str]:
        return {
            "name": "Filing Room",
            "ask": "POST /ask",
            "stream": "POST /ask/stream",
            "health": "GET /health",
            "schema": "GET /docs",
            "ui": "streamlit run src/filing/ui.py",
        }

    return app


def serve(
    *,
    config: str = DEFAULT_CONFIG,
    host: str = "127.0.0.1",
    port: int = 8000,
    warm: bool = True,
) -> None:  # pragma: no cover - the process, not the app
    """Run it. Bound to localhost by default; this API has no authentication."""
    import uvicorn

    uvicorn.run(create_app(AskEngine(config=config), warm=warm), host=host, port=port)
