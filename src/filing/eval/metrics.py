"""Scoring. Deterministic, arithmetic, and free -- no model is asked anything.

The rule this module exists to enforce: **nothing here calls a language model.**
Not to judge an answer, not to decide whether two phrasings mean the same thing,
not to check a citation. An LLM judge would make every number in the results
table a function of a model that changes underneath it, cost a thousand calls a
run against a 1,000-call-a-day budget, and -- worst of the three -- grade the
system being measured with a sibling of the system being measured. So the eval
set was built to be scorable by string and number comparison, and this is where
that decision gets paid off.

That constraint shapes what can be measured, and the honest response is to say
which questions each metric answers and which it does not:

``exact_match``
    Numeric only. The answer text is scanned for numbers, every plausible scale
    is tried (a filing says "16,434" and means millions; an answer may say
    "$16.4 billion"), and the question is scored right if any reading lands
    within :data:`~filing.eval.dataset.NUMERIC_TOLERANCE` of the XBRL value.
    It cannot tell a right number stated for the wrong reason from a right one.

``hit_rate@k`` and ``recall@k``
    Both are reported because they are different questions and conflating them
    is the standard way retrieval numbers get inflated. Gold spans are
    *alternatives* -- a passage repeated across three annual reports is three
    spans, and finding any one of them answers the question -- so ``hit_rate``
    (did any gold span appear in the top k) is the metric that matches the task.
    ``recall`` (what fraction of the gold spans appeared) is the stricter one,
    and on repeated disclosures it is bounded well below 1 by k itself.

``ndcg@k``
    Rank-sensitive, so a system that puts the evidence fifth is separated from
    one that puts it first. Binary gains; the ideal ranking is
    ``min(len(spans), k)`` hits at the top.

``router_accuracy``
    Did the system send a numeric question to SQL, a narrative one to text, and
    an unanswerable one to a refusal. The single number that says whether the
    agentic part is doing anything.

``abstention``
    Unanswerable only. The share of the ten that were refused. Reported beside
    ``over_answered`` -- the same number from the other side -- because a system
    that answers all ten confabulated ten times and no other metric shows it.

``citation_resolution``
    Of the chunk ids an answer cited, how many exist at all (``resolvable``) and
    how many overlap a gold span (``supported``). The first is a formatting
    property, the second is the one that matters, and M3 measured 31% on the
    first, which is why both are here.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from filing.eval.dataset import NUMERIC_TOLERANCE, SLICES, EvalQuestion

# A run is scored at these depths. 1 and 5 because 5 is what the baseline
# returns; 10 because the real system fuses deeper and the difference between
# "found it" and "ranked it" only shows up when k moves.
KS = (1, 5, 10)


# --------------------------------------------------------------------------
# what a run produces
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk, reduced to what scoring needs: where it came from.

    Deliberately not the ``Chunk`` -- storing 2,000 characters of text per
    result per question per config turns a results file into a corpus, and the
    only thing the scorer asks of a chunk is which stretch of which filing it
    covers.
    """

    chunk_id: str
    accn: str
    char_start: int
    char_end: int
    score: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RetrievedChunk:
        return cls(**d)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one system did with one question."""

    qid: str
    route: str = ""
    answer: str = ""
    refused: bool = False
    retrieved: tuple[RetrievedChunk, ...] = ()
    citations: tuple[str, ...] = ()
    llm_calls: int = 0
    seconds: float = 0.0
    error: str = ""
    # M6. The verifier's finding, carried on the outcome rather than recomputed
    # at scoring time -- it has to be, because a results file holds answers and
    # citations but not the evidence bodies or the fact values the check reads,
    # and re-deriving it later would mean re-running the questions. `blocked`
    # says the guard acted; `verdict` says what it found, whether it acted or
    # not. An outcome from a run with verification off carries neither.
    verdict: dict[str, Any] | None = None
    blocked: bool = False

    @property
    def flagged(self) -> bool:
        """The verifier failed this answer -- whatever the guard did about it."""
        return self.verdict is not None and not self.verdict.get("ok", True)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["retrieved"] = [c.to_json() for c in self.retrieved]
        d["citations"] = list(self.citations)
        if self.verdict is None:
            d.pop("verdict")
            d.pop("blocked")
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Outcome:
        d = dict(d)
        d["retrieved"] = tuple(RetrievedChunk.from_json(c) for c in d.get("retrieved", ()))
        d["citations"] = tuple(d.get("citations", ()))
        return cls(**d)


# --------------------------------------------------------------------------
# numeric answers
# --------------------------------------------------------------------------

# Matches 1,234 / 1234.5 / .5 -- and nothing else. Percentages, dates and share
# counts all match too; that is fine, because every candidate is tried and the
# question is only scored right if one of them is the gold value.
#
# Both ways a filing writes a negative are read: accounting parentheses, and a
# leading minus. The minus was missing until M6's verifier flagged num-015 --
# an answer stating COP's -2,701,000,000 loss, matching the store exactly, and
# scored wrong because the sign was dropped on the way in and a loss was
# compared against a profit. The leading `-` can only start a match where the
# character before it is not a word character, which is what keeps it from
# eating the hyphens in "2022-08-28" or the range in "10-15%".
_NUMBER = re.compile(r"(?<![\w.])([-(]?)\$?\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)(\)?)")


def _signed(value: float, opener: str, closer: str) -> float:
    """Apply whichever negative notation the text used."""
    if opener == "-" or (opener == "(" and closer == ")"):
        return -value
    return value


# A filing's income statement is in millions and its answer usually is too, so
# a bare "16,434" has to be allowed to mean 16.4 billion. Every scale is tried
# rather than guessed, which is loose -- but the alternative is a scale
# heuristic that silently marks correct answers wrong.
_SCALES: tuple[tuple[str, float], ...] = (
    ("", 1.0),
    ("thousand", 1e3),
    ("million", 1e6),
    ("billion", 1e9),
    ("trillion", 1e12),
)

_WORD_SCALE = re.compile(r"\b(thousand|million|billion|trillion)s?\b", re.I)


def parse_numbers(text: str) -> list[float]:
    """Every number an answer could be claiming, at every plausible scale.

    A word scale immediately after a number ("2.4 billion") is applied to that
    number specifically; beyond that every number is also offered at each of the
    five scales, because "we recognised 16,434" in a filing means millions and
    an answer that repeats it means the same thing.
    """
    out: list[float] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(2).replace(",", "")
        try:
            val = float(raw)
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
        val = _signed(val, m.group(1), m.group(3))
        tail = text[m.end() : m.end() + 14]
        word = _WORD_SCALE.match(tail.strip())
        scales = [s for _, s in _SCALES]
        if word:
            named = dict(_SCALES)[word.group(1).lower()]
            out.append(val * named)
        out.extend(val * s for s in scales)
    return out


def numeric_match(answer: str, gold: float, *, tol: float = NUMERIC_TOLERANCE) -> bool:
    """Does any reading of the answer land within tolerance of the gold value?"""
    if gold is None:
        return False
    window = tol * max(1.0, abs(gold))
    return any(abs(v - gold) <= window for v in parse_numbers(answer))


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def gains(question: EvalQuestion, retrieved: tuple[RetrievedChunk, ...]) -> list[int]:
    """1 where the retrieved chunk overlaps a gold span, 0 elsewhere, in rank order."""
    return [int(question.is_gold(c.accn, c.char_start, c.char_end)) for c in retrieved]


def hit_rate(question: EvalQuestion, retrieved: tuple[RetrievedChunk, ...], k: int) -> float:
    return float(any(gains(question, retrieved)[:k]))


def recall(question: EvalQuestion, retrieved: tuple[RetrievedChunk, ...], k: int) -> float:
    """Fraction of the question's gold spans that a top-k chunk overlaps."""
    if not question.spans:
        return 0.0
    found = {
        i
        for c in retrieved[:k]
        for i, s in enumerate(question.spans)
        if s.overlaps(c.accn, c.char_start, c.char_end)
    }
    return len(found) / len(question.spans)


def ndcg(question: EvalQuestion, retrieved: tuple[RetrievedChunk, ...], k: int) -> float:
    g = gains(question, retrieved)[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(g))
    ideal = min(len(question.spans), k)
    if not ideal:
        return 0.0
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg


# --------------------------------------------------------------------------
# the scorecard
# --------------------------------------------------------------------------


@dataclass
class SliceScore:
    name: str
    n: int = 0
    # How many of `n` the retrieval columns were computed over. Its own column
    # in the table rather than a footnote, because a hit@5 over a subset the
    # system chose for itself needs its denominator visible next to it.
    retrieval_n: int = 0
    exact_match: float | None = None
    hit_rate: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    router_accuracy: float | None = None
    abstention: float | None = None
    over_answered: float | None = None
    citations_made: int = 0
    citations_resolvable: float | None = None
    citations_supported: float | None = None
    # M6 verification, counted only over outcomes that carried a verdict. A run
    # with the verifier off leaves every one of these None and prints no
    # verification table at all -- which is the honest rendering, because a
    # check that never ran is not a check that found nothing. `figures_checked`
    # is the denominator of `hallucinated` and sits beside it for the same
    # reason `retrieval_n` sits beside hit@k.
    verified_n: int = 0
    figures_checked: int = 0
    hallucinated: float | None = None
    flag_rate: float | None = None
    block_rate: float | None = None
    locator_rate: float | None = None
    errors: int = 0
    llm_calls: int = 0
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("hit_rate", "recall", "ndcg"):
            d[key] = {str(k): v for k, v in d[key].items()}
        return d


@dataclass
class Scorecard:
    slices: dict[str, SliceScore]
    overall: SliceScore

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_json(),
            "slices": {k: v.to_json() for k, v in self.slices.items()},
        }


# Routes whose evidence carries no character span. An XBRL fact reached through
# DuckDB and the same number printed in the filing are one fact by two paths,
# and only one path has offsets -- so a question answered from `sql` or `graph`
# is not a retrieval failure, it is outside what span overlap can score. The
# alternative was to invent a span for the fact, which would turn a stated
# limitation into a green column.
SPANLESS_ROUTES = frozenset({"sql", "graph"})


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _score_group(
    name: str,
    questions: list[EvalQuestion],
    outcomes: dict[str, Outcome],
    known_chunks: set[str] | None,
    generated: bool = True,
) -> SliceScore:
    s = SliceScore(name=name, n=len(questions))
    if not questions:
        return s

    routed: list[float] = []
    em: list[float] = []
    abstained: list[float] = []
    per_k: dict[str, dict[int, list[float]]] = {"hit": {}, "rec": {}, "ndcg": {}}
    resolvable: list[float] = []
    supported: list[float] = []
    flags: list[float] = []
    blocks: list[float] = []
    bad_figures = 0
    located = 0
    markers = 0

    for q in questions:
        o = outcomes.get(q.id)
        if o is None:
            o = Outcome(qid=q.id, error="not attempted")
        if o.error:
            s.errors += 1
        s.llm_calls += o.llm_calls
        s.seconds += o.seconds
        if generated:
            routed.append(float(o.route == q.route))
            if q.slice == "numeric":
                em.append(float(not o.refused and numeric_match(o.answer, q.value or 0.0)))
            if q.slice == "unanswerable":
                abstained.append(float(o.refused))

        # Retrieval is scored on every answerable question the system answered
        # from text, numeric ones included -- that is the whole point of
        # indexing the financial statements in the baseline, and without it the
        # numeric slice would be a router test only. A question answered from a
        # spanless store is excluded rather than scored zero: see
        # SPANLESS_ROUTES. The baseline has no such route, so nothing about its
        # published numbers moves.
        if q.spans and o.route not in SPANLESS_ROUTES:
            s.retrieval_n += 1
            for k in KS:
                per_k["hit"].setdefault(k, []).append(hit_rate(q, o.retrieved, k))
                per_k["rec"].setdefault(k, []).append(recall(q, o.retrieved, k))
                per_k["ndcg"].setdefault(k, []).append(ndcg(q, o.retrieved, k))

        by_id = {c.chunk_id: c for c in o.retrieved}
        for cid in o.citations:
            s.citations_made += 1
            known = cid in by_id or (known_chunks is not None and cid in known_chunks)
            resolvable.append(float(known))
            c = by_id.get(cid)
            supported.append(float(c is not None and q.is_gold(c.accn, c.char_start, c.char_end)))

        # The verdict is read, never recomputed: the check needs the evidence
        # bodies, and those are gone by the time a results file is scored.
        if o.verdict is not None:
            s.verified_n += 1
            flags.append(float(o.flagged))
            blocks.append(float(o.blocked))
            for claim in o.verdict.get("numbers", ()):
                if claim.get("status") == "context":
                    continue
                s.figures_checked += 1
                bad_figures += int(claim.get("status") == "unsupported")
            made = len(o.verdict.get("markers", ()))
            located += made
            markers += (
                made + len(o.verdict.get("dangling", ())) + len(o.verdict.get("unlocatable", ()))
            )

    if routed:
        s.router_accuracy = _mean(routed)
    if em:
        s.exact_match = _mean(em)
    if abstained:
        s.abstention = _mean(abstained)
        s.over_answered = 1.0 - s.abstention
    for k in KS:
        if k in per_k["hit"]:
            s.hit_rate[k] = _mean(per_k["hit"][k])
            s.recall[k] = _mean(per_k["rec"][k])
            s.ndcg[k] = _mean(per_k["ndcg"][k])
    if resolvable:
        s.citations_resolvable = _mean(resolvable)
        s.citations_supported = _mean(supported)
    if s.verified_n:
        s.flag_rate = _mean(flags)
        s.block_rate = _mean(blocks)
        s.hallucinated = bad_figures / s.figures_checked if s.figures_checked else 0.0
        if markers:
            s.locator_rate = located / markers
    return s


def score(
    questions: list[EvalQuestion],
    outcomes: list[Outcome],
    *,
    known_chunks: set[str] | None = None,
    generated: bool = True,
) -> Scorecard:
    """Score a whole run. ``known_chunks`` lets a citation resolve to a chunk the
    answer did not retrieve -- a hallucinated id and a real-but-unretrieved id
    are different failures and the results file should be able to tell them
    apart.

    ``generated=False`` scores a run on retrieval alone: exact match, routing
    and abstention come back ``None`` rather than zero, because the difference
    between "the system routed every question wrongly" and "the system does not
    route" is the whole finding, and a table printing 0.0% for both lies about
    one of them. It is a parameter and not something inferred from the outcomes
    on purpose -- a run whose answers are all empty because every call failed
    looks identical from here, and that one must score as the zero it is.
    """
    by_id = {o.qid: o for o in outcomes}
    return Scorecard(
        slices={
            name: _score_group(
                name, [q for q in questions if q.slice == name], by_id, known_chunks, generated
            )
            for name in SLICES
        },
        overall=_score_group("overall", questions, by_id, known_chunks, generated),
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}%"


def to_markdown(card: Scorecard, *, title: str = "") -> str:
    """The results table. One row per slice, so a slice that collapsed is visible.

    A single headline number would hide the finding this whole gate exists to
    expose -- that a system can score respectably overall while answering every
    unanswerable question.
    """
    rows = [*card.slices.values(), card.overall]
    lines: list[str] = []
    if title:
        lines += [f"### {title}", ""]
    head = ["slice", "n", "exact", "router", "ret n"]
    head += [f"hit@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
    head += ["abstain", "cite ok", "cite gold", "LLM calls"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        cells = [r.name, str(r.n), _pct(r.exact_match), _pct(r.router_accuracy)]
        cells += [str(r.retrieval_n) if r.retrieval_n else "--"]
        cells += [_pct(r.hit_rate.get(k)) if r.hit_rate else "--" for k in KS]
        cells += [f"{r.ndcg[k]:.3f}" if k in r.ndcg else "--" for k in KS]
        cells += [
            _pct(r.abstention),
            _pct(r.citations_resolvable),
            _pct(r.citations_supported),
            str(r.llm_calls),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += _verification_table(rows)
    return "\n".join(lines) + "\n"


def _verification_table(rows: list[SliceScore]) -> list[str]:
    """The M6 columns, as their own table under the main one.

    Six more columns on a table already fourteen wide would be unreadable, and
    they answer a different question: the first table asks whether the answer
    was right, this one asks whether it was checkable. It is absent entirely
    when no row carries a verdict -- a run with the verifier off has nothing to
    say here, and saying it in zeroes would read as a perfect score.
    """
    if not any(r.verified_n for r in rows):
        return []
    head = ["slice", "verified", "figures", "hallucinated", "locator", "flagged", "blocked"]
    out = ["", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = [
            r.name,
            str(r.verified_n),
            str(r.figures_checked),
            _pct(r.hallucinated),
            _pct(r.locator_rate),
            _pct(r.flag_rate),
            _pct(r.block_rate),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return out
