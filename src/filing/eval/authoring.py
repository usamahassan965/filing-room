"""How the eval set was made. Run once; the output is the artefact.

This module is the datasheet's source code. Every question in
``questions_v1.jsonl`` came out of one of the three functions below, and the
question's ``origin`` and ``gold_source`` fields name which one and how its gold
was established, so a reader can audit the set without reading this file --
though the file is here for the reader who wants to.

The rule the whole module obeys: **gold is found, not asserted.** A numeric
question exists only if the value the XBRL store holds is also printed in the
filing that reported it, at offsets this code located; a narrative question's
span is a stretch of text that was read before the question was written for it.
Nothing here asks a language model what the answer is, because a set whose gold
is a model's opinion measures agreement with that model and calls it accuracy.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb

from filing.config import Settings
from filing.eval.dataset import EvalQuestion, Span
from filing.stores.parse import ParsedFiling, parse_filing
from filing.stores.verify import candidate_strings

# --------------------------------------------------------------------------
# numeric
# --------------------------------------------------------------------------

# Tags a person might actually ask about, which is a smaller set than the tags
# a filer reports. "Deferred Tax Assets, Other" is a real fact and a fake
# question; asking it would inflate the slice with prose nobody searches for.
# Kept as a list because the order is the sampler's tie-break, so the headline
# figures are reached for first.
HEADLINE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "GrossProfit",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "IncomeTaxExpenseBenefit",
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet",
    "Goodwill",
    "PropertyPlantAndEquipmentNet",
    "RetainedEarningsAccumulatedDeficit",
    "LongTermDebtNoncurrent",
    "OperatingLeaseLiability",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
]

# No more than this many questions may share a company or a concept. Without
# the caps the sampler returns eighty variations of "what were revenues", which
# would make the slice's mean a measurement of one question asked repeatedly.
MAX_PER_TICKER = 5
MAX_PER_TAG = 5

# A rendering that appears this often in a document is not evidence of anything
# -- "1,000" in a filing that reports in thousands hits a table of round
# numbers, and a gold span that matches half the balance sheet makes recall
# meaningless. Such a rendering is skipped and a more specific one tried.
MAX_HITS_PER_RENDERING = 8

# Four significant digits before a rendering is allowed to be gold, for the
# same reason. "349,585" is a fingerprint; "12" is a coincidence.
MIN_RENDERED_DIGITS = 4

NUMERIC_SQL = f"""
SELECT f.ticker, f.tag, c.label, f.unit, f.span,
       CAST(f.period_start AS VARCHAR), CAST(f.period_end AS VARCHAR),
       f.val, f.accn, g.form, g.path
FROM facts_current f
JOIN filings g ON g.accn = f.accn
LEFT JOIN concepts c ON c.taxonomy = f.taxonomy AND c.tag = f.tag
WHERE f.unit = 'USD'
  AND f.span IN ('FY', 'instant')
  AND abs(f.val) > 1e6
  AND g.form = '10-K'
  AND f.tag IN ({",".join("?" * len(HEADLINE_TAGS))})
ORDER BY hash(f.accn || f.tag || CAST(f.period_end AS VARCHAR))
"""


@dataclass(frozen=True, slots=True)
class _Fact:
    ticker: str
    tag: str
    label: str
    unit: str
    span: str
    period_start: str
    period_end: str
    val: float
    accn: str
    form: str
    path: str


def numeric_question_text(fact: _Fact) -> str:
    """A question a person would type, built from the concept's own label.

    Two templates, because a duration and an instant are different questions in
    English and asking "as of" about a year's revenue is the kind of wrongness
    a reader notices immediately and stops trusting the rest of the set for.
    """
    label = (fact.label or fact.tag).rstrip(".")
    if fact.span == "instant":
        return f"What was {fact.ticker}'s {label} as of {fact.period_end}?"
    return f"What did {fact.ticker} report for {label} for the fiscal year ended {fact.period_end}?"


def _digits(rendered: str) -> int:
    return sum(ch.isdigit() for ch in rendered)


def find_value_spans(text: str, val: float, accn: str) -> tuple[list[Span], str]:
    """Locate every printing of ``val`` in a filing's flattened text.

    Tries the same renderings M2's verifier tries -- units, thousands, millions,
    grouped or plain, parenthesised when negative -- and returns the offsets of
    the first rendering that hits a small, specific number of places. The match
    must be delimited: ``349,585`` inside ``1,349,585`` is a different number,
    and a span that lands on it is a wrong label that no later system can argue
    its way out of.
    """
    for rendered, how in candidate_strings(val):
        if _digits(rendered) < MIN_RENDERED_DIGITS:
            continue
        pattern = re.compile(rf"(?<![\d.,]){re.escape(rendered)}(?![\d,]*\d)")
        hits = list(pattern.finditer(text))
        if not hits or len(hits) > MAX_HITS_PER_RENDERING:
            continue
        return [
            Span(accn=accn, char_start=m.start(), char_end=m.end(), quote=_around(text, m))
            for m in hits
        ], how
    return [], ""


def _around(text: str, match: re.Match[str], width: int = 110) -> str:
    """The number with enough of its row to see what it is a number *of*."""
    lo = max(0, match.start() - width)
    hi = min(len(text), match.end() + width)
    return ("..." if lo else "") + text[lo:hi] + ("..." if hi < len(text) else "")


def build_numeric(cfg: Settings, *, target: int = 80) -> list[EvalQuestion]:
    """Sample headline facts, keep the ones the filing itself prints.

    A fact that the store holds but the document never renders is dropped, not
    kept as an unanswerable: it is answerable, from the XBRL attachments, and
    labelling it otherwise would teach the router the wrong lesson. M2 measured
    that miss rate; this is the same phenomenon seen from the other side.
    """
    con = duckdb.connect(str(cfg.facts_path), read_only=True)
    try:
        rows = con.execute(NUMERIC_SQL, HEADLINE_TAGS).fetchall()
    finally:
        con.close()

    facts = [_Fact(*r) for r in rows]
    order = {t: i for i, t in enumerate(HEADLINE_TAGS)}
    facts.sort(key=lambda f: (order.get(f.tag, 99), f.ticker, f.period_end))

    per_ticker: Counter[str] = Counter()
    per_tag: Counter[str] = Counter()
    parsed: dict[str, ParsedFiling] = {}
    out: list[EvalQuestion] = []

    for fact in facts:
        if len(out) >= target:
            break
        if per_ticker[fact.ticker] >= MAX_PER_TICKER or per_tag[fact.tag] >= MAX_PER_TAG:
            continue
        if fact.accn not in parsed:
            parsed[fact.accn] = parse_filing(
                cfg.data_dir / fact.path, accn=fact.accn, form=fact.form
            )
        pf = parsed[fact.accn]
        if not pf.ok:
            continue
        spans, how = find_value_spans(pf.text, fact.val, fact.accn)
        if not spans:
            continue
        per_ticker[fact.ticker] += 1
        per_tag[fact.tag] += 1
        out.append(
            EvalQuestion(
                id="",
                slice="numeric",
                question=numeric_question_text(fact),
                value=fact.val,
                unit=fact.unit,
                tag=fact.tag,
                spans=tuple(spans),
                tickers=(fact.ticker,),
                forms=(fact.form,),
                period_end=fact.period_end,
                accn=fact.accn,
                origin=(
                    "generated from facts_current: headline us-gaap tag, USD, "
                    "FY or instant, |val| > 1e6, capped at "
                    f"{MAX_PER_TICKER}/company and {MAX_PER_TAG}/concept"
                ),
                gold_source=(
                    f"value from XBRL, located in the filing text as {how}; "
                    f"{len(spans)} printing(s)"
                ),
            )
        )
    return out


# --------------------------------------------------------------------------
# narrative
# --------------------------------------------------------------------------

# Risk factors and MD&A are where a filing says something rather than reports
# something, and they are the two items a person reads. Item 3 is mostly
# cross-references and Item 1 is boilerplate about the industry.
NARRATIVE_ITEMS = ("1A", "7")

# A subsection heading in a 10-K is short, has no terminal full stop, and is
# followed by prose. These bounds are the operational version of that sentence.
HEAD_MIN_CHARS, HEAD_MAX_CHARS = 24, 170
HEAD_MIN_WORDS, HEAD_MAX_WORDS = 4, 24
BODY_MIN_CHARS = 500

# How much of the following prose is gold. A subsection often runs for pages;
# the first stretch is what answers the question the heading poses, and calling
# five pages gold would let a system score by returning any of them.
GOLD_MAX_CHARS = 2_400

_HEAD_BAD = re.compile(
    r"^(item|part|table of contents|index|see |the following|as of |for the )", re.I
)
_HEAD_OK = re.compile(r"[A-Za-z]")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A located subsection: where it is, what it is headed, what it says."""

    accn: str
    ticker: str
    form: str
    period_end: str
    item_key: str
    heading: str
    char_start: int
    char_end: int
    body: str


def _looks_like_heading(s: str) -> bool:
    s = s.strip()
    if not (HEAD_MIN_CHARS <= len(s) <= HEAD_MAX_CHARS):
        return False
    words = s.split()
    if not (HEAD_MIN_WORDS <= len(words) <= HEAD_MAX_WORDS):
        return False
    if s.endswith((".", ";", ",", ":")) and not s.endswith("Inc."):
        return False
    if _HEAD_BAD.match(s) or not _HEAD_OK.search(s):
        return False
    # A heading is words, not a row of a table that lost its columns.
    return sum(ch.isdigit() for ch in s) <= len(s) // 8


def find_candidates(
    pf: ParsedFiling, *, ticker: str, form: str, period_end: str
) -> list[Candidate]:
    """Every heading-and-prose pair inside the narrative items of one filing."""
    out: list[Candidate] = []
    for section in pf.sections:
        if section.item.upper() not in NARRATIVE_ITEMS:
            continue
        blocks = [b for b in pf.blocks if b[0] >= section.body and b[1] <= section.end]
        for i, (start, end) in enumerate(blocks[:-1]):
            head = pf.text[start:end]
            if not _looks_like_heading(head):
                continue
            b_start, b_end = blocks[i + 1]
            if b_end - b_start < BODY_MIN_CHARS:
                continue
            gold_end = min(b_end, start + GOLD_MAX_CHARS)
            out.append(
                Candidate(
                    accn=pf.accn,
                    ticker=ticker,
                    form=form,
                    period_end=period_end,
                    item_key=section.key,
                    heading=head.strip(),
                    char_start=start,
                    char_end=gold_end,
                    body=pf.text[start:gold_end],
                )
            )
    return out


def narrative_question(
    cand: Candidate,
    question: str,
    *,
    note: str = "",
) -> EvalQuestion:
    """Attach a hand-written question to a located span.

    The question is written after reading ``cand.body`` and is deliberately not
    the heading with a question mark on it: a question that quotes its own gold
    measures string matching, and BM25 would win the slice on wording alone.
    """
    return EvalQuestion(
        id="",
        slice="narrative",
        question=question,
        spans=(
            Span(
                accn=cand.accn,
                char_start=cand.char_start,
                char_end=cand.char_end,
                quote=cand.body[:240],
            ),
        ),
        tickers=(cand.ticker,),
        forms=(cand.form,) if cand.form else (),
        period_end=cand.period_end,
        accn=cand.accn,
        item_key=cand.item_key,
        origin=(
            "span located by subsection-heading detection inside Item "
            f"{cand.item_key}; question hand-written after reading the span"
        ),
        gold_source=f"hand-checked; heading: {cand.heading[:120]}",
        note=note,
    )


def verify_spans(cfg: Settings, questions: list[EvalQuestion]) -> list[str]:
    """Re-parse every cited filing and confirm each span still says what it said.

    Cheap insurance against the failure that would quietly poison everything: a
    span recorded against one parse and scored against another. Returns a list
    of complaints, empty when the set is sound.
    """
    from filing.ingest.manifest import Manifest

    with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
        rows = manifest.con.execute("SELECT accn, form, path FROM filings").fetchall()
    meta = {r[0]: (r[1], r[2]) for r in rows}

    problems: list[str] = []
    by_accn: dict[str, list[tuple[str, Span]]] = {}
    for q in questions:
        for s in q.spans:
            by_accn.setdefault(s.accn, []).append((q.id, s))

    for accn, items in sorted(by_accn.items()):
        if accn not in meta:
            problems.append(f"{accn}: cited by the eval set but not in the manifest")
            continue
        form, path = meta[accn]
        pf = parse_filing(Path(cfg.data_dir) / path, accn=accn, form=form)
        for qid, span in items:
            if span.char_end > len(pf.text):
                problems.append(f"{qid}: span {span.char_start}:{span.char_end} runs past the text")
                continue
            got = pf.text[span.char_start : span.char_end]
            want = span.quote.strip(".")
            head = want.split("...")[0] if want.startswith("...") else want
            if head and head.strip() and head.strip()[:60] not in got and got[:60] not in want:
                problems.append(f"{qid}: span no longer matches its recorded quote")
    return problems
