"""One question's spans, captured in-process and written out as a tree.

The gate asks for "one full trace exported to docs/trace_example.json", and the
obvious way to produce that file is to open Phoenix, find the run, and use its
export button. That file would be a screenshot in JSON's clothing: nobody
reading the repository could regenerate it, and nothing would notice if the
graph stopped emitting a span.

So the trace is captured the same way the M0 smoke gate counts spans -- with a
``SpanProcessor`` attached to our own provider, collecting finished spans in
memory. It works whether or not a collector is running, which matters because
the claim being made is about the code's instrumentation and not about whether
Phoenix happened to be up on the afternoon the file was written.

What comes out is a tree, not a list. Spans arrive in *completion* order, so a
flat dump puts the innermost span first and the node that contains it last,
which reads backwards. Re-parenting by span id and sorting by start time gives
the reader the shape they expected: one span per node, in the order the graph
ran them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import SpanProcessor

__all__ = ["SpanRecorder", "capture_question", "to_tree", "write_trace"]


def _hex(value: int | None, width: int) -> str:
    return format(value, f"0{width}x") if value else ""


@dataclass
class SpanRecorder(SpanProcessor):
    """Collects finished spans. Subclasses the SDK type for the M0 reason.

    ``phoenix.otel``'s provider calls private hooks on its processors, so a
    duck-typed object works until it does not. This is the same trap
    :class:`filing.tracing.SpanCounter` documents.
    """

    prefix: str = ""
    spans: list[Any] = field(default_factory=list)

    def on_end(self, span: Any) -> None:
        if not self.prefix or span.name.startswith(self.prefix):
            self.spans.append(span)

    def as_dicts(self) -> list[dict[str, Any]]:
        out = []
        for s in self.spans:
            ctx = s.get_span_context()
            out.append(
                {
                    "name": s.name,
                    "span_id": _hex(ctx.span_id, 16),
                    "parent_id": _hex(s.parent.span_id if s.parent else None, 16),
                    "trace_id": _hex(ctx.trace_id, 32),
                    "start_ns": s.start_time,
                    "ms": round((s.end_time - s.start_time) / 1e6, 3),
                    "status": s.status.status_code.name,
                    "attributes": {k: v for k, v in (s.attributes or {}).items()},
                }
            )
        return out


def to_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nest by parent id, order by start time, and drop the ids once used.

    The ids exist to rebuild the shape; keeping them in the output would put
    thirty-two characters of hex in front of every node name in a file whose
    whole purpose is to be read.
    """
    by_id = {s["span_id"]: dict(s, children=[]) for s in spans}
    roots: list[dict[str, Any]] = []
    for span in by_id.values():
        parent = by_id.get(span["parent_id"])
        (parent["children"] if parent else roots).append(span)

    def tidy(node: dict[str, Any]) -> dict[str, Any]:
        children = sorted(node.pop("children"), key=lambda c: c["start_ns"])
        node.pop("span_id", None)
        node.pop("parent_id", None)
        node.pop("start_ns", None)
        node.pop("trace_id", None)
        out = {k: v for k, v in node.items() if v not in ("", {}, None)}
        if children:
            out["children"] = [tidy(c) for c in children]
        return out

    return [tidy(r) for r in sorted(roots, key=lambda c: c["start_ns"])]


def capture_question(question: str, *, tools: Any, qid: str = "") -> tuple[Any, list[dict]]:
    """Run one question with a recorder attached, and detach it afterwards.

    Attached and detached around this one call rather than left on for the
    process: a recorder that outlived the capture would hold every span of a
    150-question run in memory to write one file.
    """
    from filing import tracing
    from filing.agent.graph import run_question

    tracing.setup_tracing()
    provider = tracing._provider  # noqa: SLF001 - the module owns it; see get_tracer
    if provider is None:
        # No Phoenix and no SDK provider: stand one up locally so the capture
        # works on a machine with the collector switched off. The file is
        # evidence about the graph's instrumentation, not about the collector.
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        tracing._provider = provider  # noqa: SLF001
        tracing._configured = True  # noqa: SLF001

    recorder = SpanRecorder(prefix="agent.")
    tracing._attach_counter(provider, recorder)  # noqa: SLF001

    # Rebind the tools' tracer. ``Tools.tracer`` is a default_factory, so it was
    # resolved when the caller built the tools -- which, on a machine with no
    # collector, was before there was a provider to resolve it against, leaving
    # the nodes writing into the no-op global tracer. The capture would then
    # succeed and produce zero spans, which is the failure that looks like a
    # working feature.
    previous, tools.tracer = tools.tracer, provider.get_tracer("filing.agent")
    try:
        state = run_question(question, tools=tools, qid=qid)
    finally:
        tools.tracer = previous
        # There is no remove_span_processor in the SDK, so the recorder stays
        # attached and is simply told to stop matching. Cheaper than shutting
        # the provider down, and it leaves tracing working for whatever runs
        # next in the process.
        recorder.prefix = "\0"
    return state, recorder.as_dicts()


def write_trace(
    path: Path,
    *,
    question: str,
    qid: str,
    state: Any,
    spans: list[dict[str, Any]],
    note: str = "",
) -> Path:
    """Write the trace beside the answer it produced.

    Both, because a span tree with no answer at the bottom of it does not show
    that the graph worked -- it shows that it ran.
    """
    evidence = list(state.get("evidence") or [])
    plan = list(state.get("plan") or [])
    doc = {
        "note": note
        or (
            "One question through the M5 agent graph, captured in-process by "
            "filing.agent.trace and regenerable with "
            "`python -m filing.eval trace --qid <id>`."
        ),
        "qid": qid,
        "question": question,
        "route": state.get("route", ""),
        "plan": [p.as_dict() for p in plan],
        "repairs": state.get("repairs", 0),
        "repair_log": list(state.get("repair_log") or []),
        "llm_calls": state.get("llm_calls", 0),
        "seconds": round(float(state.get("seconds") or 0.0), 3),
        "answer": state.get("answer", ""),
        "refused": bool(state.get("refused")),
        "evidence": [
            {
                "kind": e.kind,
                "citation": e.citation,
                "score": e.score,
                "value": e.value,
                "unit": e.unit,
                "tag": e.tag,
                "body": e.body[:400],
            }
            for e in evidence
        ],
        "spans": to_tree(spans),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path
