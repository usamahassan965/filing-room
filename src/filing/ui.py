"""The reader's view. A client of :mod:`filing.api`, and nothing more.

Run it with::

    pip install -e ".[ui]"             # streamlit is an extra, not a dependency
    filing serve                       # terminal one -- the API
    streamlit run src/filing/ui.py     # terminal two -- this

The constraint this file is written under is that it holds no knowledge of its
own. It never imports the graph, never opens a store, never decides whether a
figure was supported; every judgement on screen was made by the run and arrived
over HTTP in the ``/ask`` payload. The page is a rendering of that JSON, which
is why the last panel on it *is* that JSON -- a reader who suspects the
interface of flattering the system can open the raw payload and check the two
against each other in the same screenful.

That constraint is what makes the surface worth anything. It is easy to build a
page that shows citations; it is the page whose citations cannot disagree with
the run that is the deliverable.

Four states, not two. An answer, an abstention (the agent declined), a block
(the verifier overruled an answer the agent was willing to give), and an error.
The middle two both print the refusal token and would look identical if this
page collapsed them -- and the guard, which is the entire subject of M6, would
become invisible in exactly the runs where it acted.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

#: Overridable because ``filing serve`` takes a ``--port`` and the box running
#: the page is not always the box running the model budget. The sidebar can
#: still be pointed anywhere at runtime; this is only what it starts on.
DEFAULT_API = os.environ.get("FILING_API", "http://127.0.0.1:8000")
TIMEOUT = 300.0

#: Questions that exercise different paths through the graph, so the first
#: thing a visitor clicks is not necessarily the one branch that looks best.
#: Every one of them names a company and a period this corpus actually holds:
#: an example that abstains because the ticker was never ingested teaches the
#: visitor nothing about the system and everything about the demo. The last one
#: abstains on purpose -- it is the state the eval set rewards, and a surface
#: that only ever showed answers would be hiding half the design.
EXAMPLES = [
    "What revenues did NVIDIA report for the fiscal year ended January 29, 2023?",
    "Which costs does NVIDIA include in cost of revenue when it computes gross profit?",
    "What did AMD point to in 2020 as evidence that it could sustain profitability "
    "for purposes of its deferred tax assets?",
    "What was the population of France in 1780?",
]

#: Outcome -> (badge, colour, what it means). The wording is deliberate: the
#: abstention line says the system declined, not that it failed, because an
#: abstention on an unanswerable question is the behaviour the eval set rewards
#: and a UI that apologised for it would be lying about the design.
OUTCOMES: dict[str, tuple[str, str, str]] = {
    "answered": ("Answered", "#1a7f4b", "Every figure below was checked against the evidence."),
    "abstained": (
        "Abstained",
        "#8a6d1f",
        "The agent declined: the stores it searched did not carry the answer.",
    ),
    "blocked": (
        "Blocked by the verifier",
        "#a13d2d",
        "The agent produced an answer and the verifier refused to ship it.",
    ),
    "error": ("Error", "#a13d2d", "The run did not finish. Nothing below is an answer."),
}

STATUS_COLOUR = {
    "supported": "#1a7f4b",
    "derived": "#2f6f9f",
    "context": "#6b6b6b",
    "unsupported": "#a13d2d",
}

HOW = {
    "digits": "its digits are printed in the evidence",
    "scaled": "a rescaled reading of it matches a value in the evidence",
    "recomputed": "recomputed from two facts",
    "question": "echoed from the question, not a claim of its own",
    "period": "a period in the evidence, not a figure",
    "": "nothing in the evidence carries it",
}


# --------------------------------------------------------------------------
# talking to the API
# --------------------------------------------------------------------------


def health(base: str) -> dict[str, Any]:
    try:
        r = httpx.get(f"{base.rstrip('/')}/health", timeout=5.0)
        r.raise_for_status()
        return dict(r.json())
    except Exception as exc:  # noqa: BLE001 - a down server is a state, not a crash
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def _sse(lines: Any) -> Any:
    """Parse ``event:``/``data:`` pairs out of a server-sent event stream.

    Hand-rolled rather than pulled in as a dependency: the server emits one
    ``data:`` line per event and nothing else, so a parser for the general case
    would be more code than this whole function.
    """
    event, data = "", ""
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if event or data:
                yield event or "message", data
            event, data = "", ""
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data += line[5:].strip()
    if event or data:
        yield event or "message", data


def ask(base: str, question: str, *, stage: Any) -> dict[str, Any]:
    """Stream the run, reporting each stage, and return the finished payload.

    Falls back to the blocking endpoint if the stream cannot be opened, because
    a proxy that does not pass server-sent events should cost the reader the
    progress display and nothing else.
    """
    url = f"{base.rstrip('/')}/ask/stream"
    payload: dict[str, Any] = {}
    seen: list[str] = []
    try:
        with httpx.stream("POST", url, json={"question": question}, timeout=TIMEOUT) as r:
            if r.status_code >= 400:
                r.read()
                return {"outcome": "error", "error": _detail(r), "question": question}
            for event, data in _sse(r.iter_lines()):
                if event == "stage":
                    label = json.loads(data).get("label", "")
                    seen.append(label)
                    stage.update(label=f"{label}  ({len(seen)})")
                elif event == "answer":
                    payload = json.loads(data)
    except httpx.HTTPError:
        try:
            r = httpx.post(f"{base.rstrip('/')}/ask", json={"question": question}, timeout=TIMEOUT)
            if r.status_code >= 400:
                return {"outcome": "error", "error": _detail(r), "question": question}
            payload = dict(r.json())
        except Exception as exc:  # noqa: BLE001
            return {
                "outcome": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "question": question,
            }
    if not payload:
        return {
            "outcome": "error",
            "error": "the stream closed before the answer arrived",
            "question": question,
        }
    payload["stages"] = seen
    return payload


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except Exception:  # noqa: BLE001
        return r.text or f"HTTP {r.status_code}"


# --------------------------------------------------------------------------
# the panels
# --------------------------------------------------------------------------


def outcome_banner(payload: dict[str, Any]) -> None:
    label, colour, meaning = OUTCOMES.get(payload.get("outcome", ""), OUTCOMES["error"])
    st.markdown(
        f"<div style='border-left:4px solid {colour};padding:.55rem .9rem;margin:.2rem 0 1rem;'>"
        f"<b style='color:{colour}'>{label}</b><br>"
        f"<span style='opacity:.75;font-size:.9rem'>{meaning}</span></div>",
        unsafe_allow_html=True,
    )


def answer_panel(payload: dict[str, Any]) -> None:
    outcome = payload.get("outcome", "")
    if outcome == "error":
        st.error(payload.get("error") or "the run did not finish")
        return
    answer = payload.get("answer") or ""
    if outcome in ("abstained", "blocked"):
        st.markdown(f"> **{answer or 'INSUFFICIENT EVIDENCE'}**")
        reasons = (payload.get("verification") or {}).get("reasons") or []
        grade = payload.get("grade") or {}
        if outcome == "blocked" and reasons:
            st.markdown("**Why the verifier stopped it**")
            for reason in reasons:
                st.markdown(f"- {reason}")
        elif not grade.get("ok", True) and grade.get("reason"):
            st.markdown(f"**Why**  {grade['reason']}")
        return
    st.markdown(answer)


def claims_panel(payload: dict[str, Any]) -> None:
    """The answer, sentence by sentence, with what backs each one."""
    claims = payload.get("claims") or []
    if not claims:
        return
    st.markdown("##### Claims")
    for claim in claims:
        markers = claim.get("markers") or []
        # This line ends up inside an ``unsafe_allow_html`` block below, where a
        # browser reads tags and not markdown, so the emphasis has to be tags.
        cites = " ".join(f"<code>[{m}]</code>" for m in markers) if markers else "<em>nothing</em>"
        st.markdown(f"{claim.get('text', '')}")
        bits = [f"cites {cites}"]
        for fig in claim.get("figures") or []:
            colour = STATUS_COLOUR.get(fig.get("status", ""), "#6b6b6b")
            bits.append(
                f"<span style='color:{colour}'>&#9679;</span> "
                f"<code>{fig.get('text', '')}</code> {fig.get('status', '')}"
            )
        st.markdown(
            "<div style='font-size:.82rem;opacity:.8;margin:-.5rem 0 .9rem'>"
            + " &nbsp;·&nbsp; ".join(bits)
            + "</div>",
            unsafe_allow_html=True,
        )


def verification_panel(payload: dict[str, Any]) -> None:
    v = payload.get("verification") or {}
    st.markdown("##### Verification")
    if not v.get("enabled"):
        st.info(
            "Verification is off in this config. Nothing below was checked -- "
            "which is not the same as nothing being wrong."
        )
        return

    checked = v.get("figures_checked", 0)
    cols = st.columns(4)
    cols[0].metric("Figures checked", checked)
    cols[1].metric("Unsupported", v.get("unsupported", 0))
    cols[2].metric("Citations", len(v.get("markers") or []))
    cols[3].metric("Dangling", len(v.get("dangling") or []) + len(v.get("unlocatable") or []))

    numbers = [n for n in (v.get("numbers") or []) if n.get("status") != "context"]
    if numbers:
        st.markdown(
            "<div style='font-size:.85rem;opacity:.7;margin-bottom:.4rem'>"
            "Every figure the answer states, and what it was matched against."
            "</div>",
            unsafe_allow_html=True,
        )
    for n in numbers:
        colour = STATUS_COLOUR.get(n.get("status", ""), "#6b6b6b")
        against = n.get("derivation") or n.get("matched") or "&mdash;"
        marker = n.get("marker") or 0
        tail = f" &nbsp;<code>[{marker}]</code>" if marker else ""
        st.markdown(
            f"<div style='border-left:3px solid {colour};padding:.35rem .7rem;margin:.3rem 0;"
            f"background:rgba(128,128,128,.06)'>"
            f"<code style='font-size:1rem'>{n.get('text', '')}</code> "
            f"<span style='color:{colour};font-weight:600'>{n.get('status', '')}</span>"
            f"<div style='font-size:.82rem;opacity:.8;margin-top:.2rem'>"
            f"{HOW.get(n.get('how', ''), n.get('how', ''))} &nbsp;&rarr;&nbsp; "
            f"<span style='font-family:monospace'>{against}</span>{tail}</div></div>",
            unsafe_allow_html=True,
        )

    for label, key in (
        ("Citations pointing past the evidence", "dangling"),
        ("Citations with nothing to look up", "unlocatable"),
        ("Sentences stating a figure and citing nothing", "uncited"),
        ("Contradicted by the facts store", "unrechecked"),
    ):
        if items := v.get(key):
            st.warning(f"**{label}:** " + ", ".join(str(i) for i in items))
    if tags := v.get("taxonomy"):
        st.caption("Failure tags on this run: " + ", ".join(tags))


def evidence_panel(payload: dict[str, Any]) -> None:
    records = payload.get("evidence") or []
    # Plain markdown, no inline tags: ``st.markdown`` escapes HTML unless it is
    # asked not to, so a ``<small>`` here prints itself on the page.
    st.markdown("##### Evidence")
    st.caption(f"{len(records)} record(s) shown to the writer.")
    if not records:
        st.caption("No store returned anything, which is why there is nothing to cite.")
        return
    for e in records:
        cited = "cited" if e.get("cited") else "not cited"
        head = f"[{e['marker']}]  {e.get('citation') or '(no citation)'}  ·  {cited}"
        with st.expander(head, expanded=bool(e.get("cited"))):
            if not e.get("locatable"):
                st.error("This record carries nothing a reader could look up.")
            meta = [f"**{e.get('kind', '')}**", f"score `{e.get('score', 0):.3f}`"]
            if e.get("value") is not None:
                meta.append(f"value `{e['value']:,.0f} {e.get('unit', '')}`")
            if e.get("tag"):
                meta.append(f"tag `{e['tag']}`")
            if e.get("char_end", 0) > e.get("char_start", 0):
                meta.append(f"chars `{e['char_start']}:{e['char_end']}`")
            if e.get("accn"):
                meta.append(f"accession `{e['accn']}`")
            st.markdown(" · ".join(meta))
            st.markdown(
                f"<div style='font-size:.88rem;white-space:pre-wrap'>{e.get('body', '')}</div>",
                unsafe_allow_html=True,
            )


def path_panel(payload: dict[str, Any]) -> None:
    """Plan, route, grade, repairs -- the decisions, in the order taken."""
    st.markdown("##### How it got there")
    plan = payload.get("plan") or []
    route = payload.get("route") or "?"
    if plan:
        p = plan[0]
        wanted = p.get("route") or "?"
        arrow = f"`{wanted}`" if wanted == route else f"`{wanted}` &rarr; `{route}`"
        st.markdown(f"**Planned** {arrow}", unsafe_allow_html=True)
        detail = [p[k] for k in ("ticker", "concept", "period_end") if p.get(k)]
        if detail:
            st.caption(" · ".join(detail))
        if p.get("why"):
            st.caption(f"“{p['why']}”")
    else:
        st.markdown(f"**Routed to** `{route}`")
    if note := payload.get("plan_note"):
        st.caption(note)

    grade = payload.get("grade") or {}
    if grade.get("reason"):
        mark = "passed" if grade.get("ok") else "failed"
        st.markdown(f"**Grade** {mark} — {grade['reason']}")
        if grade.get("missing"):
            st.caption(f"missing: {grade['missing']}")

    repairs = payload.get("repairs") or 0
    st.markdown(f"**Repairs** {repairs} of 2")
    for line in payload.get("repair_log") or []:
        st.caption(f"· {line}")


def footer(payload: dict[str, Any]) -> None:
    trace = payload.get("trace") or {}
    cols = st.columns(3)
    cols[0].metric("Seconds", f"{payload.get('seconds', 0):.1f}")
    cols[1].metric("LLM calls", payload.get("llm_calls", 0))
    cols[2].metric("Config", payload.get("config", "?"))
    if trace.get("url"):
        st.markdown(f"[Open this run in Phoenix]({trace['url']})  ·  `{trace['trace_id'][:16]}…`")
    elif trace.get("trace_id"):
        st.caption(
            f"Trace `{trace['trace_id'][:16]}…` — no collector is listening, so there is "
            "nothing to link to. Start Phoenix and ask again."
        )
    else:
        st.caption("Tracing is off, so this run has no trace to link to.")
    with st.expander("The payload this page was built from"):
        st.caption(
            "Everything above is a rendering of this object. Nothing on the page was decided here."
        )
        st.json(payload)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - the page, driven by streamlit
    st.set_page_config(page_title="Filing Room", page_icon="§", layout="wide")
    st.title("Filing Room")
    st.caption("Agentic RAG over SEC filings — with the evidence attached.")

    with st.sidebar:
        st.subheader("Server")
        base = st.text_input("API", value=DEFAULT_API)
        info = health(base)
        if info.get("ok"):
            st.success(f"up · `{info.get('config', '?')}`")
            st.caption(
                f"guard: `{info.get('guard') or 'off'}` · "
                f"stores: {'open' if info.get('ready') else 'not opened yet'}"
            )
        else:
            st.error("no server")
            st.caption(str(info.get("detail", "")))
            st.code("filing serve", language="bash")
        st.subheader("Try one")
        for i, q in enumerate(EXAMPLES):
            if st.button(q, key=f"ex{i}", use_container_width=True):
                st.session_state["question"] = q

    question = st.text_input("Question", key="question", placeholder=EXAMPLES[0])
    go = st.button("Ask", type="primary", disabled=not question.strip())

    if go and question.strip():
        with st.status("asking", expanded=True) as stage:
            payload = ask(base, question.strip(), stage=stage)
            stage.update(label=payload.get("outcome", "done"), state="complete", expanded=False)
        st.session_state["payload"] = payload

    payload = st.session_state.get("payload")
    if not payload:
        st.info("Ask something, or pick one of the examples in the sidebar.")
        return

    outcome_banner(payload)
    left, right = st.columns([3, 2], gap="large")
    with left:
        answer_panel(payload)
        claims_panel(payload)
        verification_panel(payload)
    with right:
        path_panel(payload)
        evidence_panel(payload)
    st.divider()
    footer(payload)


# `streamlit run` executes this file as a script, so the guard is what makes the
# page render there while leaving the module importable by the tests below it.
if __name__ == "__main__":  # pragma: no cover
    main()
