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
import re
from typing import Any
from urllib.parse import quote

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


_FORM = re.compile(r"\b(10-K|10-Q|8-K|20-F)(/A)?\b")
_PLAIN = re.compile(r"^[A-Za-z0-9$%,.;:'()]+$")

_MARK = "background:rgba(234,179,8,.32);color:inherit;border-radius:3px;padding:0 2px"
_LINK = "font-size:.85rem;font-weight:600;text-decoration:none"
_ROW = "padding:.3rem .6rem;border-top:1px solid rgba(128,128,128,.2);vertical-align:top"
_KICKER = (
    "font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;opacity:.7;margin:.6rem 0 .3rem"
)


def _human(value: float, unit: str) -> str:
    """26,974,000,000 USD as a reader says it: $26.97 billion."""
    if unit != "USD":
        return ""
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(value) >= size:
            return f"${value / size:,.2f} {word}"
    return ""


def _fragment(body: str, figures: list[str]) -> str:
    """A text fragment that lands the reader on the cited sentence.

    ``#:~:text=`` scrolls a browser to the first match and highlights it; a
    browser that does not support it, or a fragment the page does not match,
    just opens the document at the top, so a miss costs nothing. The start of
    the sentence carrying a cited figure is used rather than the figure
    itself: "60,922" appears in a filing a dozen times, the words leading into
    it once. Only plain words go in -- a stray symbol from the text extraction
    is enough to make the match fail.
    """
    sentences = re.split(r"(?<=[.;])\s+", body)
    chosen = next((s for s in sentences if any(f and f in s for f in figures)), "") or body
    words: list[str] = []
    for word in chosen.split():
        if not _PLAIN.match(word):
            break
        words.append(word.rstrip(".,;:"))
        if len(words) == 8:
            break
    if len(words) < 4:
        return ""
    return "#:~:text=" + quote(" ".join(words), safe="").replace("-", "%2D")


def _mark(body: str, figures: list[str]) -> str:
    """The escaped body with every cited figure highlighted where it appears."""
    out = esc(body)
    for fig in sorted({esc(f) for f in figures if f}, key=len, reverse=True):
        out = out.replace(fig, f"<mark style='{_MARK}'>{fig}</mark>")
    return out


def _source(e: dict[str, Any], figures: list[str]) -> str:
    """One reference, drawn as what it is: a fact table or a filing excerpt."""
    citation = e.get("citation") or "(no citation)"
    form = _FORM.search(citation)
    url = e.get("source_url") or ""
    state = "cited in the answer" if e.get("cited") else "retrieved, not cited"
    head = (
        "<div style='display:flex;justify-content:space-between;gap:.6rem;flex-wrap:wrap;"
        "align-items:baseline'><span style='font-family:monospace;font-size:.85rem'>"
        f"<b>[{esc(e.get('marker', ''))}]</b> {esc(citation)}</span>"
        f"<span style='font-size:.78rem;opacity:.7'>{state} · "
        f"score {float(e.get('score') or 0):.3f}</span></div>"
    )
    warn = (
        ""
        if e.get("locatable")
        else (
            "<div style='border-left:4px solid #a13d2d;background:rgba(161,61,45,.08);"
            "padding:.4rem .7rem;margin:.4rem 0'>This record carries nothing a reader "
            "could look up.</div>"
        )
    )

    if e.get("kind") == "fact" and e.get("value") is not None:
        value = float(e["value"])
        unit = e.get("unit", "")
        human = _human(value, unit)
        filing = " ".join(x for x in (form.group(0) if form else "", e.get("accn", "")) if x)
        rows = (
            ("Company", e.get("ticker", "")),
            ("Reported line item", e.get("tag", "")),
            ("Value", f"{value:,.0f} {unit}" + (f"  ({human})" if human else "")),
            ("Period ending", e.get("period_end", "")),
            ("Filing", filing),
        )
        cells = "".join(
            f"<tr><td style='{_ROW};opacity:.7;white-space:nowrap'>{esc(k)}</td>"
            f"<td style='{_ROW};font-family:monospace'>{esc(v)}</td></tr>"
            for k, v in rows
            if v
        )
        body = (
            f"<div style='{_KICKER}'>XBRL fact, as tagged in the filing</div>"
            "<table style='border-collapse:collapse;width:100%;font-size:.88rem;"
            f"font-variant-numeric:tabular-nums'>{cells}</table>"
        )
        href, link_text = url, "Open the filing on SEC.gov ↗"
    else:
        span = ""
        if e.get("char_end", 0) > e.get("char_start", 0):
            span = (
                f"<div style='font-size:.75rem;opacity:.6;margin-top:.35rem'>characters "
                f"{int(e['char_start']):,}–{int(e['char_end']):,} of the filing's text</div>"
            )
        body = (
            f"<div style='{_KICKER}'>Excerpt from the filing</div>"
            "<div style='border-left:3px solid rgba(128,128,128,.45);padding:.5rem .8rem;"
            "background:rgba(128,128,128,.07);border-radius:0 6px 6px 0;font-size:.9rem;"
            f"line-height:1.55;white-space:pre-wrap'>{_mark(e.get('body', ''), figures)}</div>"
            f"{span}"
        )
        fragment = _fragment(e.get("body", ""), figures) if url else ""
        href = url + fragment
        link_text = "Find this passage on SEC.gov ↗" if fragment else "Open the filing on SEC.gov ↗"
    link = (
        f"<div style='margin-top:.5rem'><a href='{esc(href)}' target='_blank' "
        f"rel='noopener' style='{_LINK}'>{esc(link_text)}</a></div>"
        if href
        else ""
    )
    return (
        "<div style='border:1px solid rgba(128,128,128,.28);border-radius:8px;"
        f"padding:.6rem .8rem;margin:.5rem 0'>{head}{warn}{body}{link}</div>"
    )


def evidence_panel(payload: dict[str, Any]) -> str:
    """The references: every record the writer was shown, cited ones first.

    Each is drawn as the kind of thing it is -- an XBRL fact as the rows of a
    table, a passage as an excerpt with the figures the answer took from it
    highlighted -- and linked to the filing on sec.gov, so a reader can check
    the answer against the primary source rather than against this page.
    """
    records = payload.get("evidence") or []
    out = [
        "<h5>Sources</h5>",
        f"<div style='font-size:.82rem;opacity:.7'>{len(records)} record(s) "
        f"shown to the writer.</div>",
    ]
    if not records:
        out.append(
            "<div style='font-size:.82rem;opacity:.7'>No store returned anything, "
            "which is why there is nothing to cite.</div>"
        )
        return "".join(out)

    figures: dict[Any, list[str]] = {}
    for n in (payload.get("verification") or {}).get("numbers") or []:
        if n.get("marker") and n.get("status") in ("supported", "derived"):
            figures.setdefault(n["marker"], []).append(str(n.get("text", "")))

    cited = [e for e in records if e.get("cited")]
    rest = [e for e in records if not e.get("cited")]
    out += [_source(e, figures.get(e.get("marker"), [])) for e in cited]
    if rest:
        inner = "".join(_source(e, []) for e in rest)
        out.append(
            f"<details {'' if cited else 'open'} style='margin-top:.4rem'>"
            "<summary style='cursor:pointer;font-size:.85rem;opacity:.8'>"
            f"Also retrieved, not cited ({len(rest)})</summary>{inner}</details>"
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
