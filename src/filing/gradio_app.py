"""The same reader's view, on a surface that will run for free.

:mod:`filing.ui` is the Streamlit page and is still the one to run locally. This
is the deployed one, and it exists because of a hosting constraint rather than a
design preference:

* Streamlit Community Cloud gives a free app 1 GB of RAM. The API process
  measures 849 MiB resident with the models loaded, which leaves 175 MiB for
  Streamlit itself and everything the runtime holds. It would be tuned to the
  edge of an OOM on every cold start.
* A Hugging Face Docker Space would take the compose file unchanged, but Docker
  Spaces now require a paid plan to create.
* A free Hugging Face **Gradio** Space on CPU basic gets 2 vCPU and 16 GB of
  RAM, unmetered and unlimited in number. That is the one free door left, and
  it is a Gradio-shaped door.

The hardware to ask for is CPU basic, not ZeroGPU. ZeroGPU is the more generous
tier and the wrong one: it meters five GPU-minutes a day, which this would spend
in about twenty questions, and it is gated on account age -- both prices paid for
a device this workload never touches. The embedder is a 384-dimension bge-small
over 32k points and the reranker is a MiniLM cross-encoder over five passages;
neither is interesting to a GPU. Sixteen gigabytes of RAM is the resource that
was actually scarce, and CPU basic is where it is.

Two differences from the Streamlit page follow from a Space running one process:

1. There is no HTTP hop. The page holds an :class:`~filing.api.AskEngine` and
   calls it directly. The constraint the Streamlit page is written under
   survives intact, because what it renders is still *only* the ``/ask``
   payload: ``engine.stream`` ends by yielding the value ``payload_for``
   built, the very object FastAPI would have serialised. This module never
   opens a store, never reads the graph, and never decides whether a figure was
   supported. It renders a document that was finished before it arrived.
2. There is no Qdrant server. ``QDRANT_PATH`` points at a directory that
   ``qdrant-client`` opens in-process, written by ``filing pack``.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Iterator
from typing import Any

import gradio as gr

from filing.render import (
    EXAMPLES,
    answer_panel,
    claims_panel,
    esc,
    evidence_panel,
    outcome_banner,
    path_panel,
    verification_panel,
)

#: A Space has one CPU allocation and no queue discipline of its own worth
#: relying on, and a single question costs seconds of model time. Two people
#: arriving at once should wait in line rather than halve each other's speed.
CONCURRENCY = 1

TITLE = "Filing Room"
SUBTITLE = "Agentic RAG over SEC filings — with the evidence attached."

#: Written on the page rather than left to the reader to discover, because the
#: interesting behaviour of this system is the one nobody clicks on purpose.
INTRO = """
Ask about a figure or a policy in a **10-K**. The agent plans a route, queries
either the XBRL facts store or the filing text, grades what came back, repairs
its own query up to twice, writes an answer, and then a verifier checks every
number in that answer against the evidence and can refuse to ship it.

Everything below the answer is that machinery's own record of what it did. The
last example question is unanswerable on purpose — the system abstaining is the
behaviour the evaluation set rewards, and a demo that only showed answers would
be hiding half the design.
"""


def _placeholder() -> str:
    return (
        "<div style='opacity:.6;padding:1.2rem 0'>Ask something, or pick one of the examples.</div>"
    )


def _stage_line(seen: list[str], done: str = "") -> str:
    """The nodes that have finished, in order, as one line."""
    if done:
        return f"<div style='opacity:.7;font-size:.85rem'>{esc(done)}</div>"
    if not seen:
        return "<div style='opacity:.7;font-size:.85rem'>asking…</div>"
    trail = " → ".join(esc(s) for s in seen)
    return f"<div style='opacity:.7;font-size:.85rem'>{trail} …</div>"


def _footer(payload: dict[str, Any]) -> str:
    trace = payload.get("trace") or {}
    tiles = (
        ("Seconds", f"{payload.get('seconds', 0):.1f}"),
        ("LLM calls", payload.get("llm_calls", 0)),
        ("Config", payload.get("config", "?")),
    )
    out = ["<div style='display:flex;gap:2rem;flex-wrap:wrap;margin:.6rem 0'>"]
    for label, value in tiles:
        out.append(
            f"<div><div style='font-size:.78rem;opacity:.7'>{esc(label)}</div>"
            f"<div style='font-size:1.3rem;font-variant-numeric:tabular-nums'>"
            f"{esc(value)}</div></div>"
        )
    out.append("</div>")
    if trace.get("url"):
        out.append(
            f"<div style='font-size:.85rem'><a href='{esc(trace['url'])}'>"
            f"Open this run in Phoenix</a> · <code>{esc(trace['trace_id'][:16])}…</code></div>"
        )
    elif trace.get("trace_id"):
        out.append(
            f"<div style='font-size:.82rem;opacity:.7'>Trace "
            f"<code>{esc(trace['trace_id'][:16])}…</code> — no collector is listening "
            f"on this host, so there is nothing to link to.</div>"
        )
    else:
        out.append(
            "<div style='font-size:.82rem;opacity:.7'>Tracing is off, so this run "
            "has no trace to link to.</div>"
        )
    return "".join(out)


def run(engine: Any, question: str) -> Iterator[tuple[Any, ...]]:
    """One question, streamed. Yields the whole page on every step.

    Gradio wants one value per output component per yield, so every yield here
    is the entire page. The intermediate ones clear the panels on purpose: the
    previous question's verification table left standing beside the new
    question's progress line is a page that appears to have answered something
    it has not looked at yet.
    """
    question = (question or "").strip()
    if not question:
        yield ("", "", _placeholder(), "", "", "", "", "", {})
        return

    seen: list[str] = []
    yield (_stage_line(seen), "", "", "", "", "", "", "", {})

    payload: dict[str, Any] = {}
    try:
        for event in engine.stream(question):
            kind, data = event.get("event"), event.get("data") or {}
            if kind == "stage":
                seen.append(str(data.get("label") or data.get("node") or ""))
                yield (_stage_line(seen), "", "", "", "", "", "", "", {})
            elif kind == "answer":
                payload = dict(data)
    except Exception as exc:  # noqa: BLE001 - a failed run is a state to render
        traceback.print_exc()
        payload = {
            "outcome": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "question": question,
        }

    if not payload:
        payload = {
            "outcome": "error",
            "error": "the run ended without producing a payload",
            "question": question,
        }

    yield (
        _stage_line(seen, done=f"{len(seen)} step(s) · {payload.get('outcome', '')}"),
        outcome_banner(payload),
        answer_panel(payload),
        claims_panel(payload),
        verification_panel(payload),
        path_panel(payload),
        evidence_panel(payload),
        _footer(payload),
        payload,
    )


def build(engine: Any = None) -> gr.Blocks:
    """The page, with the engine injectable.

    Injectable for the same reason :func:`filing.api.create_app` is: the tests
    drive this with an engine that yields a canned payload, and every question
    worth asking of a renderer -- does an abstention print the refusal, does a
    block say who blocked it, does a figure get the colour its status calls for
    -- is answerable without a corpus or a model.
    """
    if engine is None:  # pragma: no cover - the real thing, not the tested path
        from filing.api import AskEngine

        engine = AskEngine()

    # ``theme`` moved from the constructor to ``launch`` in Gradio 6, and is
    # passed there rather than here so this keeps building under both.
    with gr.Blocks(title=TITLE, fill_width=True) as page:
        gr.Markdown(f"# {TITLE}\n{SUBTITLE}")
        with gr.Accordion("What this is", open=False):
            gr.Markdown(INTRO)

        with gr.Row():
            box = gr.Textbox(
                label="Question",
                placeholder=EXAMPLES[0],
                lines=2,
                scale=5,
                autofocus=True,
            )
            go = gr.Button("Ask", variant="primary", scale=1)

        gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[box], label="Try one")

        status = gr.HTML()
        banner = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                answer = gr.HTML(_placeholder())
                claims = gr.HTML()
                verification = gr.HTML()
            with gr.Column(scale=2):
                path = gr.HTML()
                evidence = gr.HTML()

        footer = gr.HTML()

        with gr.Accordion("The payload this page was built from", open=False):
            gr.Markdown(
                "Everything above is a rendering of this object. Nothing on the page "
                "was decided here — a reader who suspects the interface of flattering "
                "the system can check the two against each other in one screenful."
            )
            raw = gr.JSON()

        outs = [status, banner, answer, claims, verification, path, evidence, footer, raw]

        def handler(q: str) -> Iterator[tuple[Any, ...]]:
            return run(engine, q)

        go.click(handler, inputs=[box], outputs=outs, concurrency_limit=CONCURRENCY)
        box.submit(handler, inputs=[box], outputs=outs, concurrency_limit=CONCURRENCY)

    return page


def main() -> None:  # pragma: no cover - the entry point, not a tested path
    """Warm the stores, then serve.

    Warming before ``launch`` rather than on the first question is the whole
    difference between a Space that looks broken and one that looks slow. The
    build takes tens of seconds -- opening the embedded store, loading BM25,
    pulling the cross-encoder -- and a visitor who asks during it would watch a
    spinner with no explanation. Hugging Face shows its own "Building" state
    until the port is listening, which is the honest place for that wait.
    """
    from filing.api import AskEngine

    engine = AskEngine()
    try:
        engine.warm()
    except Exception:  # noqa: BLE001 - a cold page beats no page
        traceback.print_exc()

    build(engine).queue(max_size=16).launch(
        server_name="0.0.0.0",  # noqa: S104 - a container binds to its own network
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(),
        show_api=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
