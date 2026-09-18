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

The page is also the public face of the project, visited by people who have not
read the README. So it says what it is before it asks for anything: the headline
results, which companies are in the corpus, and a tracker that draws the
pipeline as it runs. Every one of those is either a number the README reports or
a rendering of the stream; none of it is decided here.
"""

from __future__ import annotations

import inspect
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

REPO_URL = "https://github.com/usamahassan965/filing-room"
REPORT_URL = "https://usamahassan965.github.io/filingroom.html"

#: The headline rows of the README, and nothing that is not in it. A number on
#: this page that the repository could not reproduce would be the one claim on
#: the page with no evidence attached.
HEADLINE: tuple[tuple[str, str], ...] = (
    ("98.8%", "numeric answers exact; a naive RAG baseline gets 7.5%"),
    ("0 / 133", "figures shipped without support in the evidence"),
    ("426 / 427", "deliberately corrupted answers caught by the verifier"),
    ("407", "10-K and 10-Q filings from 20 companies"),
)

#: The corpus, by sector. Mirrors ``universe.yaml``, which the Space does not
#: ship; ``tests/test_deploy.py`` fails if the two drift apart.
COVERAGE: dict[str, tuple[str, ...]] = {
    "Semiconductors": ("NVDA", "AMD", "INTC", "AVGO", "QCOM"),
    "Big-box retail": ("WMT", "TGT", "COST", "HD", "LOW"),
    "Pharmaceuticals": ("JNJ", "PFE", "MRK", "ABBV", "LLY"),
    "Oil & gas": ("XOM", "CVX", "COP", "SLB", "PSX"),
}

#: What each example exercises, in the order of ``EXAMPLES``. The labels name
#: the route the question is meant to take, so a visitor can predict the
#: pipeline before clicking and then check the prediction against the tracker.
EXAMPLE_KINDS: tuple[tuple[str, str], ...] = (
    ("Figure", "facts store"),
    ("Accounting policy", "filing text"),
    ("Narrative", "filing text"),
    ("Unanswerable", "refuses on purpose"),
)

#: The pipeline as the tracker draws it. Graph nodes fold onto these steps:
#: the three retrievers and the refusal are one step with the store named
#: under it, because which store answered is the fact, not three boxes.
STEPS: tuple[tuple[str, str], ...] = (
    ("plan", "Plan"),
    ("route", "Route"),
    ("retrieve", "Retrieve"),
    ("rerank", "Rerank"),
    ("grade", "Grade"),
    ("repair", "Repair"),
    ("synthesise", "Write"),
    ("verify", "Verify"),
)

_STORE = {
    "retrieve_sql": "facts",
    "retrieve_text": "text",
    "retrieve_graph": "graph",
    "refuse": "refused",
}

#: Written on the page rather than left to the reader to discover, because the
#: interesting behaviour of this system is the one nobody clicks on purpose.
INTRO = """
Ask about a figure or a policy in a **10-K or 10-Q**. The agent plans a route,
queries either the XBRL facts store or the filing text, grades what came back,
repairs its own query up to twice, writes an answer, and then a verifier checks
every number in that answer against the evidence and can refuse to ship it.

Everything below the answer is that machinery's own record of what it did. The
last example question is unanswerable on purpose — the system abstaining is the
behaviour the evaluation set rewards, and a demo that only showed answers would
be hiding half the design.
"""

#: Written against Gradio's own theme variables, so the page follows the
#: visitor's light or dark setting without a second palette to keep in step.
CSS = """
.fr-askrow { align-items: flex-end; }
.fr-ask { flex-grow: 0 !important; height: 3rem; margin-bottom: 2px; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.fr-hero { padding: 6px 2px 2px; }
.fr-eyebrow { font-family: var(--font-mono); font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--color-accent); font-weight: 600; }
.fr-hero h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); font-weight: 700; letter-spacing: -.02em;
  margin: .25rem 0 .3rem; line-height: 1.1; }
.fr-sub { font-size: 1.04rem; color: var(--body-text-color-subdued); margin: 0 0 1rem;
  max-width: 64ch; line-height: 1.5; }
.fr-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.fr-stat { border: 1px solid var(--border-color-primary); border-radius: 10px;
  padding: 10px 14px; background: var(--block-background-fill); }
.fr-stat b { display: block; font-family: var(--font-mono); font-size: 1.35rem;
  font-variant-numeric: tabular-nums; color: var(--body-text-color); }
.fr-stat span { display: block; font-size: .78rem; line-height: 1.35;
  color: var(--body-text-color-subdued); margin-top: 2px; }
.fr-links { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 12px; font-size: .88rem; }
.fr-links a { color: var(--color-accent); text-decoration: none; font-weight: 600; }
.fr-links a:hover, .fr-links a:focus-visible { text-decoration: underline; }
.fr-note { color: var(--body-text-color-subdued); }
@media (max-width: 760px) { .fr-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.fr-label { font-family: var(--font-mono); font-size: .72rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--body-text-color-subdued); margin: 2px 0 -6px; }
.fr-ex { text-align: left !important; justify-content: flex-start !important;
  white-space: pre-line !important; line-height: 1.4 !important; font-weight: 400 !important;
  padding: 10px 12px !important; min-height: 64px; }

.fr-coverage { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px;
  margin: 4px 0 10px; }
.fr-coverage h6 { font-size: .74rem; letter-spacing: .08em; text-transform: uppercase;
  margin: 0 0 6px; color: var(--body-text-color-subdued); }
.fr-tick { display: inline-block; font-family: var(--font-mono); font-size: .8rem;
  border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 1px 7px;
  margin: 0 4px 4px 0; }
.fr-hint { font-size: .86rem; color: var(--body-text-color-subdued); }
@media (max-width: 760px) { .fr-coverage { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.fr-track { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 6px; }
.fr-step { flex: 1 1 84px; min-width: 76px; border: 1px solid var(--border-color-primary);
  border-radius: 9px; padding: 7px 10px; background: var(--block-background-fill); }
.fr-step .n { font-family: var(--font-mono); font-size: .66rem; letter-spacing: .08em;
  color: var(--body-text-color-subdued); }
.fr-step .t { display: block; font-weight: 600; font-size: .9rem; }
.fr-step .s { display: block; font-family: var(--font-mono); font-size: .72rem;
  color: var(--body-text-color-subdued); min-height: 1.1em; }
.fr-step.done { border-color: #1a7f4b; }
.fr-step.done .n { color: #1a7f4b; }
.fr-step.skip { opacity: .45; border-style: dashed; }
.fr-step.live { border-color: var(--color-accent); box-shadow: 0 0 0 2px var(--color-accent-soft); }
.fr-step.live .n { color: var(--color-accent); }
.fr-step.live .n::after { content: " \\25CF"; animation: fr-pulse 1.1s ease-in-out infinite; }
@keyframes fr-pulse { 50% { opacity: .15; } }
@media (prefers-reduced-motion: reduce) { .fr-step.live .n::after { animation: none; } }
.fr-trail { font-size: .82rem; color: var(--body-text-color-subdued); }

.fr-card { border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: 14px 18px; background: var(--block-background-fill); margin-bottom: 12px; }
.fr-card h5 { font-size: .74rem !important; letter-spacing: .1em; text-transform: uppercase;
  color: var(--body-text-color-subdued) !important; margin: 0 0 .6rem !important;
  font-weight: 600 !important; }
.fr-card h5 ~ h5 { margin-top: 1rem !important; }
.fr-answer { font-size: 1.08rem; line-height: 1.6; }
.fr-empty { color: var(--body-text-color-subdued); padding: 18px 2px; }
.fr-runstats { display: flex; flex-wrap: wrap; gap: 28px; }
.fr-runstats > div > div:first-child { font-size: .72rem; letter-spacing: .06em;
  text-transform: uppercase; color: var(--body-text-color-subdued); }
.fr-runstats > div > div:last-child { font-family: var(--font-mono); font-size: 1.1rem; }
.fr-foot { font-size: .82rem; color: var(--body-text-color-subdued); text-align: center;
  padding: 10px 0 4px; }
.fr-foot a { color: var(--color-accent); }
"""


def _theme() -> gr.themes.Base:
    """Ink and ledger blue, set in a face made for financial documents."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="#f4f6f9",
        body_background_fill_dark="#0d1420",
        block_background_fill="#ffffff",
        block_background_fill_dark="#141d2b",
        color_accent="#1f4f99",
        color_accent_soft="rgba(31, 79, 153, 0.18)",
        color_accent_soft_dark="rgba(122, 162, 230, 0.22)",
        button_primary_background_fill="#1f3f73",
        button_primary_background_fill_hover="#18335e",
        button_primary_background_fill_dark="#4f7fd0",
        button_primary_background_fill_hover_dark="#6a93da",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#0d1420",
    )


def _hero() -> str:
    stats = "".join(
        f"<div class='fr-stat'><b>{esc(v)}</b><span>{esc(k)}</span></div>" for v, k in HEADLINE
    )
    return (
        "<div class='fr-hero'>"
        "<div class='fr-eyebrow'>Agentic RAG · SEC 10-K / 10-Q · fiscal 2020 – 2024</div>"
        f"<h1>{esc(TITLE)}</h1>"
        "<p class='fr-sub'>Ask a question about a public company's filings. A LangGraph agent "
        "picks the store that holds the answer, repairs its own retrieval when the evidence "
        "is thin, answers with citations &mdash; and a verifier checks every number before "
        "it ships.</p>"
        f"<div class='fr-stats'>{stats}</div>"
        "<div class='fr-links'>"
        f"<a href='{REPO_URL}' target='_blank' rel='noopener'>Source on GitHub ↗</a>"
        f"<a href='{REPORT_URL}' target='_blank' rel='noopener'>Engineering report ↗</a>"
        "<span class='fr-note'>Free CPU hardware · one question at a time · "
        "usually 5&ndash;15 s per answer</span>"
        "</div></div>"
    )


def _coverage() -> str:
    cols = "".join(
        f"<div><h6>{esc(sector)}</h6>"
        + "".join(f"<span class='fr-tick'>{esc(t)}</span>" for t in tickers)
        + "</div>"
        for sector, tickers in COVERAGE.items()
    )
    return (
        f"<div class='fr-coverage'>{cols}</div>"
        "<p class='fr-hint'>Periods ending January 2020 through February 2025. A figure "
        "question works best when it names the company, the metric and the period end, the "
        "way the first example does. A question about anything outside these companies "
        "should be refused, not guessed &mdash; try one.</p>"
    )


def _card(body: str, extra: str = "") -> str:
    return f"<div class='fr-card {extra}'>{body}</div>" if body else ""


def _placeholder() -> str:
    return (
        "<div class='fr-empty'>Ask something, or pick one of the examples. The tracker "
        "above lights up as each step of the pipeline finishes.</div>"
    )


def _track(nodes: list[str], labels: list[str], done: str = "") -> str:
    """The pipeline as a row of steps, lit as the graph's nodes finish.

    ``nodes`` are graph node names in the order they finished and ``labels``
    the stream's sentence for each. While running, every step seen is done and
    a trailing cell says the next one is under way. Once finished, the steps
    the run never touched are drawn as skipped rather than pending, so a
    refusal does not look like a run that stalled before retrieval.
    """
    folded: dict[str, int] = {}
    store = ""
    for node in nodes:
        step = "retrieve" if node in _STORE else node
        folded[step] = folded.get(step, 0) + 1
        store = _STORE.get(node, store)

    cells = []
    for i, (key, name) in enumerate(STEPS, 1):
        count = folded.get(key, 0)
        sub = ""
        if key == "retrieve" and count:
            sub = store + (f" ×{count}" if count > 1 else "")
        elif count > 1:
            sub = f"×{count}"
        if count:
            state = "done"
        elif done and nodes:
            state, sub = "skip", "skipped"
        else:
            state = ""
        cells.append(
            f"<div class='fr-step {state}'><span class='n'>{i:02d}</span>"
            f"<span class='t'>{esc(name)}</span><span class='s'>{esc(sub)}</span></div>"
        )
    if not done:
        cells.append(
            "<div class='fr-step live'><span class='n'>NOW</span>"
            "<span class='t'>working</span><span class='s'></span></div>"
        )
        trail = (" → ".join(esc(s) for s in labels) + " …") if labels else "asking…"
    else:
        trail = esc(done)
    return f"<div class='fr-track'>{''.join(cells)}</div><div class='fr-trail'>{trail}</div>"


def _footer(payload: dict[str, Any]) -> str:
    trace = payload.get("trace") or {}
    tiles = (
        ("Seconds", f"{payload.get('seconds', 0):.1f}"),
        ("LLM calls", payload.get("llm_calls", 0)),
        ("Repairs", payload.get("repairs", 0)),
        ("Config", payload.get("config", "?")),
    )
    out = ["<h5>This run</h5><div class='fr-runstats'>"]
    for label, value in tiles:
        out.append(f"<div><div>{esc(label)}</div><div>{esc(value)}</div></div>")
    out.append("</div>")
    note = "font-size:.82rem;opacity:.75;margin-top:.6rem"
    if trace.get("url"):
        out.append(
            f"<div style='{note}'><a href='{esc(trace['url'])}'>Open this run in Phoenix</a>"
            f" · <code>{esc(trace['trace_id'][:16])}…</code></div>"
        )
    elif trace.get("trace_id"):
        out.append(
            f"<div style='{note}'>Trace <code>{esc(trace['trace_id'][:16])}…</code> — no "
            f"collector is listening on this host, so there is nothing to link to.</div>"
        )
    else:
        out.append(
            f"<div style='{note}'>Tracing is off on this host, so this run has no trace "
            f"to link to.</div>"
        )
    return _card("".join(out))


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

    nodes: list[str] = []
    labels: list[str] = []
    yield (_track(nodes, labels), "", "", "", "", "", "", "", {})

    payload: dict[str, Any] = {}
    try:
        for event in engine.stream(question):
            kind, data = event.get("event"), event.get("data") or {}
            if kind == "stage":
                nodes.append(str(data.get("node") or ""))
                labels.append(str(data.get("label") or data.get("node") or ""))
                yield (_track(nodes, labels), "", "", "", "", "", "", "", {})
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

    seconds = payload.get("seconds")
    summary = f"{len(nodes)} step(s) · {payload.get('outcome', '')}"
    if isinstance(seconds, int | float):
        summary += f" · {seconds:.1f} s"
    yield (
        _track(nodes, labels, done=summary),
        outcome_banner(payload),
        _card(f"<h5>Answer</h5><div class='fr-answer'>{answer_panel(payload)}</div>"),
        _card(claims_panel(payload)),
        _card(verification_panel(payload)),
        _card(path_panel(payload)),
        _card(evidence_panel(payload)),
        _footer(payload),
        payload,
    )


def _styling() -> tuple[dict[str, Any], dict[str, Any]]:
    """Where ``theme`` and ``css`` go: the constructor before Gradio 6, ``launch`` after.

    Asked of the installed version rather than pinned, so the page builds the
    same under both and a Space that resolves a newer Gradio does not lose its
    styling to a keyword the constructor stopped reading.
    """
    style = {"theme": _theme(), "css": CSS}
    if "css" in inspect.signature(gr.Blocks.launch).parameters:
        return {}, style
    return style, {}


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

    at_build, _ = _styling()
    with gr.Blocks(title=f"{TITLE} — {SUBTITLE}", fill_width=True, **at_build) as page:
        gr.HTML(_hero())

        with gr.Row(equal_height=False, elem_classes=["fr-askrow"]):
            box = gr.Textbox(
                label="Your question",
                placeholder=EXAMPLES[0],
                lines=2,
                scale=6,
                autofocus=True,
            )
            go = gr.Button(
                "Ask", variant="primary", scale=1, min_width=110, elem_classes=["fr-ask"]
            )

        gr.HTML("<div class='fr-label'>Try one — each takes a different path</div>")
        with gr.Row(equal_height=True):
            picks = []
            for question, (kind, route) in zip(EXAMPLES, EXAMPLE_KINDS, strict=True):
                picks.append(
                    (
                        question,
                        gr.Button(
                            f"{kind} · {route}\n{question}",
                            size="sm",
                            variant="secondary",
                            elem_classes=["fr-ex"],
                        ),
                    )
                )

        with gr.Accordion("What's in the corpus — 20 companies, 4 sectors", open=False):
            gr.HTML(_coverage())
        with gr.Accordion("How it works", open=False):
            gr.Markdown(INTRO)

        status = gr.HTML(_track([], [], done="Waiting for a question."))
        banner = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                answer = gr.HTML(_card(_placeholder()))
                claims = gr.HTML()
                verification = gr.HTML()
            with gr.Column(scale=2):
                path = gr.HTML()
                evidence = gr.HTML()
                footer = gr.HTML()

        with gr.Accordion("The raw payload this page was built from", open=False):
            gr.Markdown(
                "Everything above is a rendering of this object. Nothing on the page "
                "was decided here — a reader who suspects the interface of flattering "
                "the system can check the two against each other in one screenful."
            )
            raw = gr.JSON()

        gr.HTML(
            "<div class='fr-foot'>Built by Usama · "
            f"<a href='{REPO_URL}' target='_blank' rel='noopener'>GitHub</a> · "
            f"<a href='{REPORT_URL}' target='_blank' rel='noopener'>Engineering report</a>"
            " · The Space sleeps when idle; the first question after a wake takes longer.</div>"
        )

        outs = [status, banner, answer, claims, verification, path, evidence, footer, raw]

        def handler(q: str) -> Iterator[tuple[Any, ...]]:
            # ``yield from``, not ``return``. Gradio decides whether an event
            # streams by asking ``inspect.isgeneratorfunction`` of the function
            # it was handed, and a plain function that returns a generator is
            # not one. Written with ``return`` this passes every test of ``run``
            # -- ``run`` is still a generator, still yields nine values -- and
            # then fails in the browser only, with Gradio reporting one output
            # value where nine were needed and naming the generator object as
            # the value. The keyword is the wiring.
            yield from run(engine, q)

        ask = {"fn": handler, "outputs": outs, "concurrency_limit": CONCURRENCY}
        go.click(inputs=[box], **ask)
        box.submit(inputs=[box], **ask)
        for question, button in picks:
            button.click(lambda q=question: q, outputs=[box], queue=False).then(inputs=[box], **ask)

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

    _, at_launch = _styling()
    build(engine).queue(max_size=16).launch(
        server_name="0.0.0.0",  # noqa: S104 - a container binds to its own network
        server_port=int(os.environ.get("PORT", "7860")),
        **at_launch,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
