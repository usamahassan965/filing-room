"""The payload, rendered. Shared by both surfaces, owned by neither.

:mod:`filing.ui` renders the ``/ask`` payload with Streamlit; :mod:`filing.gradio_app`
renders the same payload with Gradio, because a Hugging Face Space runs one
process and Streamlit is not one of the SDKs it will run. Two renderers of one
document is exactly the situation in which a colour, a label or a piece of
wording drifts on one surface and not the other, and the drift is invisible
until someone opens both at once.

So the judgements live here, once: which outcomes exist, what each one is
called, what colour it wears, and what every figure status and match kind
means in a sentence. Both surfaces import them, and the renderers below turn a
payload into HTML that either can display.

The constraint the surfaces are written under still holds and is the reason
this module has no imports from the rest of the package: it holds no knowledge
of its own. Nothing here opens a store, calls a model, or decides whether a
figure was supported. Every judgement on screen was made by the run and
arrived in the payload; these functions choose typography for it. A renderer
that could disagree with the run would make the whole surface worthless, and
the way to keep that impossible is to give it nothing to disagree with.

One thing this module does that the Streamlit page got for free: it escapes.
``st.markdown`` escapes HTML unless asked not to, but a string built here goes
to the browser as written, and evidence bodies are spans of SEC filings --
text that genuinely contains ``<``, ``&`` and the occasional stray tag. Every
value interpolated below goes through :func:`esc`.
"""

from __future__ import annotations

import html
import json
from typing import Any

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


def esc(value: Any) -> str:
    """Everything interpolated into HTML goes through here.

    Evidence bodies are spans of SEC filings and answers are model output;
    both routinely contain characters a browser reads as markup.
    """
    return html.escape("" if value is None else str(value), quote=True)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------


def outcome_banner(payload: dict[str, Any]) -> str:
    label, colour, meaning = OUTCOMES.get(payload.get("outcome", ""), OUTCOMES["error"])
    return (
        f"<div style='border-left:4px solid {colour};padding:.55rem .9rem;margin:.2rem 0 1rem'>"
        f"<b style='color:{colour}'>{esc(label)}</b><br>"
        f"<span style='opacity:.75;font-size:.9rem'>{esc(meaning)}</span></div>"
    )


def answer_panel(payload: dict[str, Any]) -> str:
    outcome = payload.get("outcome", "")
    if outcome == "error":
        detail = payload.get("error") or "the run did not finish"
        return (
            f"<div style='border-left:4px solid #a13d2d;background:rgba(161,61,45,.08);"
            f"padding:.6rem .9rem'>{esc(detail)}</div>"
        )
    answer = payload.get("answer") or ""
    if outcome in ("abstained", "blocked"):
        out = [
            f"<blockquote style='margin:0 0 .8rem;padding:.5rem .9rem;"
            f"border-left:3px solid #8a6d1f'><b>{esc(answer or 'INSUFFICIENT EVIDENCE')}</b>"
            f"</blockquote>"
        ]
        reasons = (payload.get("verification") or {}).get("reasons") or []
        grade = payload.get("grade") or {}
        if outcome == "blocked" and reasons:
            out.append("<p><b>Why the verifier stopped it</b></p><ul>")
            out += [f"<li>{esc(r)}</li>" for r in reasons]
            out.append("</ul>")
        elif not grade.get("ok", True) and grade.get("reason"):
            out.append(f"<p><b>Why</b>&nbsp; {esc(grade['reason'])}</p>")
        return "".join(out)
    return f"<div style='white-space:pre-wrap'>{esc(answer)}</div>"


def claims_panel(payload: dict[str, Any]) -> str:
    """The answer, sentence by sentence, with what backs each one."""
    claims = payload.get("claims") or []
    if not claims:
        return ""
    out = ["<h5>Claims</h5>"]
    for claim in claims:
        markers = claim.get("markers") or []
        cites = (
            " ".join(f"<code>[{esc(m)}]</code>" for m in markers) if markers else "<em>nothing</em>"
        )
        bits = [f"cites {cites}"]
        for fig in claim.get("figures") or []:
            colour = STATUS_COLOUR.get(fig.get("status", ""), "#6b6b6b")
            bits.append(
                f"<span style='color:{colour}'>&#9679;</span> "
                f"<code>{esc(fig.get('text', ''))}</code> {esc(fig.get('status', ''))}"
            )
        out.append(f"<p style='margin:.2rem 0'>{esc(claim.get('text', ''))}</p>")
        out.append(
            "<div style='font-size:.82rem;opacity:.8;margin:0 0 .9rem'>"
            + " &nbsp;·&nbsp; ".join(bits)
            + "</div>"
        )
    return "".join(out)


def verification_panel(payload: dict[str, Any]) -> str:
    v = payload.get("verification") or {}
    out = ["<h5>Verification</h5>"]
    if not v.get("enabled"):
        return "".join(
            out
            + [
                "<div style='border-left:4px solid #2f6f9f;background:rgba(47,111,159,.08);"
                "padding:.6rem .9rem'>Verification is off in this config. Nothing below was "
                "checked &mdash; which is not the same as nothing being wrong.</div>"
            ]
        )

    tiles = (
        ("Figures checked", v.get("figures_checked", 0)),
        ("Unsupported", v.get("unsupported", 0)),
        ("Citations", len(v.get("markers") or [])),
        ("Dangling", len(v.get("dangling") or []) + len(v.get("unlocatable") or [])),
    )
    out.append("<div style='display:flex;gap:1.5rem;flex-wrap:wrap;margin:.4rem 0 1rem'>")
    for label, value in tiles:
        out.append(
            f"<div><div style='font-size:.78rem;opacity:.7'>{esc(label)}</div>"
            f"<div style='font-size:1.5rem;font-variant-numeric:tabular-nums'>"
            f"{esc(value)}</div></div>"
        )
    out.append("</div>")

    numbers = [n for n in (v.get("numbers") or []) if n.get("status") != "context"]
    if numbers:
        out.append(
            "<div style='font-size:.85rem;opacity:.7;margin-bottom:.4rem'>"
            "Every figure the answer states, and what it was matched against.</div>"
        )
    for n in numbers:
        colour = STATUS_COLOUR.get(n.get("status", ""), "#6b6b6b")
        against = n.get("derivation") or n.get("matched") or "&mdash;"
        marker = n.get("marker") or 0
        tail = f" &nbsp;<code>[{esc(marker)}]</code>" if marker else ""
        out.append(
            f"<div style='border-left:3px solid {colour};padding:.35rem .7rem;margin:.3rem 0;"
            f"background:rgba(128,128,128,.06)'>"
            f"<code style='font-size:1rem'>{esc(n.get('text', ''))}</code> "
            f"<span style='color:{colour};font-weight:600'>{esc(n.get('status', ''))}</span>"
            f"<div style='font-size:.82rem;opacity:.8;margin-top:.2rem'>"
            f"{esc(HOW.get(n.get('how', ''), n.get('how', '')))} &nbsp;&rarr;&nbsp; "
            f"<span style='font-family:monospace'>{esc(against)}</span>{tail}</div></div>"
        )

    for label, key in (
        ("Citations pointing past the evidence", "dangling"),
        ("Citations with nothing to look up", "unlocatable"),
        ("Sentences stating a figure and citing nothing", "uncited"),
        ("Contradicted by the facts store", "unrechecked"),
    ):
        if items := v.get(key):
            joined = ", ".join(esc(i) for i in items)
            out.append(
                f"<div style='border-left:4px solid #8a6d1f;background:rgba(138,109,31,.08);"
                f"padding:.5rem .9rem;margin:.3rem 0'><b>{esc(label)}:</b> {joined}</div>"
            )
    if tags := v.get("taxonomy"):
        out.append(
            "<div style='font-size:.82rem;opacity:.7;margin-top:.5rem'>Failure tags on this run: "
            + ", ".join(esc(t) for t in tags)
            + "</div>"
        )
    return "".join(out)


def evidence_panel(payload: dict[str, Any]) -> str:
    records = payload.get("evidence") or []
    out = [
        "<h5>Evidence</h5>",
        f"<div style='font-size:.82rem;opacity:.7'>{len(records)} record(s) "
        f"shown to the writer.</div>",
    ]
    if not records:
        out.append(
            "<div style='font-size:.82rem;opacity:.7'>No store returned anything, "
            "which is why there is nothing to cite.</div>"
        )
        return "".join(out)
    for e in records:
        cited = "cited" if e.get("cited") else "not cited"
        head = f"[{esc(e.get('marker', ''))}]  {esc(e.get('citation') or '(no citation)')}"
        meta = [f"<b>{esc(e.get('kind', ''))}</b>", f"score <code>{e.get('score', 0):.3f}</code>"]
        if e.get("value") is not None:
            meta.append(f"value <code>{e['value']:,.0f} {esc(e.get('unit', ''))}</code>")
        if e.get("tag"):
            meta.append(f"tag <code>{esc(e['tag'])}</code>")
        if e.get("char_end", 0) > e.get("char_start", 0):
            meta.append(f"chars <code>{esc(e['char_start'])}:{esc(e['char_end'])}</code>")
        if e.get("accn"):
            meta.append(f"accession <code>{esc(e['accn'])}</code>")
        warn = (
            ""
            if e.get("locatable")
            else (
                "<div style='border-left:4px solid #a13d2d;background:rgba(161,61,45,.08);"
                "padding:.4rem .7rem;margin:.3rem 0'>This record carries nothing a reader "
                "could look up.</div>"
            )
        )
        out.append(
            f"<details {'open' if e.get('cited') else ''} "
            f"style='border:1px solid rgba(128,128,128,.25);border-radius:6px;"
            f"padding:.4rem .7rem;margin:.4rem 0'>"
            f"<summary style='cursor:pointer'>{head} &nbsp;·&nbsp; "
            f"<span style='opacity:.7'>{esc(cited)}</span></summary>"
            f"{warn}"
            f"<div style='font-size:.85rem;margin:.3rem 0'>{' · '.join(meta)}</div>"
            f"<div style='font-size:.88rem;white-space:pre-wrap'>{esc(e.get('body', ''))}</div>"
            f"</details>"
        )
    return "".join(out)


def path_panel(payload: dict[str, Any]) -> str:
    """Plan, route, grade, repairs -- the decisions, in the order taken."""
    out = ["<h5>How it got there</h5>"]
    for p in payload.get("plan") or []:
        arrow = esc(p.get("route", ""))
        out.append(f"<p style='margin:.2rem 0'><b>Planned</b> {arrow}</p>")
        detail = [esc(p[k]) for k in ("ticker", "concept", "period_end") if p.get(k)]
        if detail:
            out.append(f"<div style='font-size:.82rem;opacity:.7'>{' · '.join(detail)}</div>")
        if p.get("why"):
            out.append(
                f"<div style='font-size:.82rem;opacity:.7'>&ldquo;{esc(p['why'])}&rdquo;</div>"
            )
    if route := payload.get("route"):
        out.append(f"<p style='margin:.4rem 0'><b>Routed to</b> <code>{esc(route)}</code></p>")
    if note := payload.get("plan_note"):
        out.append(f"<div style='font-size:.82rem;opacity:.7'>{esc(note)}</div>")
    grade = payload.get("grade") or {}
    if grade:
        mark = "&#10003;" if grade.get("ok") else "&#10007;"
        out.append(
            f"<p style='margin:.4rem 0'><b>Grade</b> {mark} &mdash; "
            f"{esc(grade.get('reason', ''))}</p>"
        )
        if grade.get("missing"):
            out.append(
                f"<div style='font-size:.82rem;opacity:.7'>missing: {esc(grade['missing'])}</div>"
            )
    repairs = esc(payload.get("repairs", 0))
    out.append(f"<p style='margin:.4rem 0'><b>Repairs</b> {repairs} of 2</p>")
    for line in payload.get("repair_log") or []:
        out.append(f"<div style='font-size:.82rem;opacity:.7'>· {esc(line)}</div>")
    return "".join(out)


def payload_json(payload: dict[str, Any]) -> str:
    """The document the page was built from, verbatim.

    The last panel is the raw payload for a reason: a reader who suspects the
    interface of flattering the system can check the two against each other in
    one screenful.
    """
    return json.dumps(payload, indent=1, default=str)
