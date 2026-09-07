"""The last unchecked step, checked.

By the time the graph reaches ``synthesise`` everything is verifiable: the
router's choice is a label, the SQL row came from DuckDB, the text hits carry
character offsets, and the grader is arithmetic. Then a language model reads all
of that and writes prose, and nothing downstream looks at what it wrote. That is
the only place in the pipeline where a number can appear that no store ever
produced -- and M4 measured it happening. The naive baseline scored 7.5% exact
on numbers while retrieving the right evidence 2.5% of the time, which is not a
bonus: those answers came from the model's memory of the company, not from the
filing in front of it.

So this module asks three questions of a finished answer, all of them
deterministic and none of them requiring a model:

**Is every number in it in the evidence?** Every figure the answer states is
matched against the numbers the synthesiser was actually shown -- a fact's value
for the SQL branch, the printed digits of a chunk for the text branch. A figure
that matches nothing is *unsupported*, which is this project's word for what is
usually called a hallucination. The word matters: the check proves the number is
absent from the evidence, not that it is false, and those are different claims.

**Does every claim carry a locator?** A sentence stating a figure has to cite a
marker, the marker has to point at evidence that exists, and that evidence has
to carry something a reader can look up -- an accession plus a character span,
or an accession plus a tag and period. A confident sentence with no way back to
a filing is the failure this whole project is built to not commit.

**Should it ship?** The guard turns the two findings above into one decision.
Its default is to *block* -- an answer with an unsupported figure is replaced by
the refusal token -- because the alternative is a system that knows it printed
an unverifiable number and prints it anyway.

Two limits, stated here rather than discovered by a reader:

*The number check is a floor, not a proof.* Matching is tried at every scale, so
"$60.9 billion" matches a stored 60,922 (millions) -- necessary, because filings
and answers rarely agree on units, and the cost is that a wrong number close to
a right one at some scale can pass. Every match records *how* it matched, and
the report separates matches on the exact digit string from matches that needed
rescaling and rounding, because the second tier is the weaker evidence and
should be counted as such.

*A derived figure is only as checkable as its inputs.* A percentage the model
computed from two evidence values is recomputed here and confirmed or rejected;
a percentage computed from anything else cannot be, and is reported as derived
rather than quietly counted as supported.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from filing.eval.dataset import NUMERIC_TOLERANCE

# Imported rather than restated. An answer's numbers get read by one rule
# whether they are being scored or verified -- a verifier that parses
# differently from the scorer can pass a figure the scorer marks wrong, and then
# the two disagree about the same answer with no way to say which is right. The
# names are private to `metrics` and the coupling is deliberate.
from filing.eval.metrics import _NUMBER, _SCALES, _WORD_SCALE, _signed

__all__ = [
    "GUARD_MODES",
    "TAXONOMY",
    "GuardReport",
    "NumberClaim",
    "Verdict",
    "classify",
    "figures_in",
    "guard_answer",
    "has_locator",
    "markers_in",
    "recheck_facts",
    "resolve_citations",
    "sentences",
    "uncited_claims",
    "verify_answer",
    "verify_numbers",
]

# The guard's three settings. `block` is the default and the one the gate is
# measured under; `flag` is M6's pre-decided bail-out -- if blocking rejects too
# many valid answers, the response is to log the flag rate rather than to
# quietly loosen the check until it passes.
GUARD_MODES = ("off", "flag", "block")

# Citation markers must come out before any number is read, or `[1]` is scored
# as the figure one and every cited answer acquires a hallucinated number. The
# fullwidth pair is M4.5's finding: gpt-oss-120b cites with U+3010/U+3011.
#
# A marker can name several pieces of evidence at once -- `[2, 3, 5]` -- and
# reading only the single-marker form was worse than reading none. The whole
# bracket survived the strip, so its digits were read as figures and reported
# as three hallucinations; the sentence looked uncited because no marker was
# found in it; and the citations themselves went uncounted. That one omission
# produced every narrative flag in the first M6 run.
_CITATION = re.compile(r"[\[【［]\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*[\]】］]")
_MARKER = re.compile(r"\d{1,3}")

_SCALE_BY_WORD: dict[str, float] = dict(_SCALES)
_ALL_SCALES: tuple[float, ...] = tuple(s for _, s in _SCALES)

# A bare four-digit integer in this range, appearing in the question or in an
# evidence period, is a year rather than a figure. Narrow on purpose: the
# exemption also requires that nothing marks the number as money (no currency
# symbol in front, no scale word behind), because "$2,024 million" is a figure
# that happens to look like a year and must not be waved through.
_YEAR_LO, _YEAR_HI = 1900, 2100

# Split on terminal punctuation only where a sentence really ends -- followed by
# space or end of line. Splitting on every "." cuts "$60.9 billion" in half and
# leaves a fragment carrying a figure and no citation marker, which makes the
# uncited-claim check fire on well-formed answers. That is not a hypothetical:
# it was the first thing the check did.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# ISO dates in evidence bodies, removed before digit runs are read. "2024-01-28"
# otherwise contributes the digit strings 2024, 1 and 28, which would let an
# answer's "28" match a period boundary and count as supported.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True, slots=True)
class NumberClaim:
    """One figure an answer states, and what became of it.

    ``how`` is kept rather than collapsed into the status because the two ways a
    number can be supported are not equally strong, and a report that prints one
    percentage for both is hiding the weaker half.
    """

    text: str
    value: float
    status: str  # supported | derived | context | unsupported
    how: str = ""  # digits | scaled | recomputed | question | period
    source: str = ""  # the citation of the evidence that carried it

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the verifier found, in enough detail to act on and to publish."""

    numbers: tuple[NumberClaim, ...] = ()
    markers: tuple[int, ...] = ()
    dangling: tuple[int, ...] = ()
    unlocatable: tuple[int, ...] = ()
    uncited: tuple[str, ...] = ()
    unrechecked: tuple[str, ...] = ()
    taxonomy: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def unsupported(self) -> tuple[NumberClaim, ...]:
        return tuple(n for n in self.numbers if n.status == "unsupported")

    @property
    def checked(self) -> tuple[NumberClaim, ...]:
        """Figures the check applies to. Years and question echoes are not
        claims the system is making, so they are not in the denominator."""
        return tuple(n for n in self.numbers if n.status != "context")

    @property
    def ok(self) -> bool:
        """Nothing unsupported, dangling, uncited, or contradicted by the store."""
        return not (
            self.unsupported
            or self.dangling
            or self.unlocatable
            or self.uncited
            or self.unrechecked
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "numbers": [n.to_json() for n in self.numbers],
            "markers": list(self.markers),
            "dangling": list(self.dangling),
            "unlocatable": list(self.unlocatable),
            "uncited": list(self.uncited),
            "unrechecked": list(self.unrechecked),
            "taxonomy": list(self.taxonomy),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Verdict:
        return cls(
            numbers=tuple(NumberClaim(**n) for n in d.get("numbers", ())),
            markers=tuple(d.get("markers", ())),
            dangling=tuple(d.get("dangling", ())),
            unlocatable=tuple(d.get("unlocatable", ())),
            uncited=tuple(d.get("uncited", ())),
            unrechecked=tuple(d.get("unrechecked", ())),
            taxonomy=tuple(d.get("taxonomy", ())),
            reasons=tuple(d.get("reasons", ())),
        )


# --------------------------------------------------------------------------
# reading numbers
# --------------------------------------------------------------------------


def strip_citations(text: str) -> str:
    """Blank the markers and any ISO date, keeping the length so offsets line up.

    Dates go for the same reason markers do. "for the period ending 2022-08-28"
    otherwise states the figures eight and twenty-eight, neither of which any
    filing contains, and the answer is reported as carrying two hallucinated
    numbers because it said when the period ended. The year is left alone: it
    is caught downstream by the year rule, which can tell a period from a
    figure by looking at the evidence, and blanking it here would hide a
    genuine "$2,024 million" from that check.
    """
    blanked = _CITATION.sub(lambda m: " " * len(m.group(0)), text)
    return _ISO_DATE.sub(lambda m: m.group(0)[:4] + " " * 6, blanked)


@dataclass(frozen=True, slots=True)
class _Stated:
    text: str
    face: float
    readings: tuple[float, ...]
    digits: str
    money: bool
    bare_int: bool


def _read(text: str) -> list[_Stated]:
    """Every number in a stretch of prose, with the readings it could carry.

    A scale word directly after the number pins it (``2.4 billion`` is one
    value, not five). Without one, every scale is offered, because a filing's
    income statement is in millions and an answer that repeats a figure from it
    means millions without saying so.
    """
    out: list[_Stated] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(2).replace(",", "")
        try:
            face = float(raw)
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
        face = _signed(face, m.group(1), m.group(3))
        word = _WORD_SCALE.match(text[m.end() : m.end() + 14].strip())
        if word:
            readings = (face * _SCALE_BY_WORD[word.group(1).lower()],)
        else:
            readings = tuple(face * s for s in _ALL_SCALES)
        before = text[max(0, m.start() - 2) : m.start() + 1]
        out.append(
            _Stated(
                text=m.group(0).strip(),
                face=face,
                readings=readings,
                digits=raw.lstrip("0") or "0",
                money="$" in m.group(0) or "$" in before,
                bare_int="." not in raw and not word,
            )
        )
    return out


def sentences(answer: str) -> tuple[str, ...]:
    """The answer split the way the uncited-claim check splits it.

    Public because M7's evidence panel shows the answer claim by claim, and a
    panel that split sentences its own way would be showing the reader a
    decomposition the guard never looked at. One definition, two consumers.
    """
    return tuple(s for s in (part.strip() for part in _SENTENCE.split(answer)) if s)


def markers_in(text: str) -> tuple[int, ...]:
    """The citation markers in a stretch of prose, in order, without repeats."""
    out: list[int] = []
    for bracket in _CITATION.finditer(text):
        for marker in _MARKER.finditer(bracket.group(0)):
            i = int(marker.group(0))
            if i not in out:
                out.append(i)
    return tuple(out)


def figures_in(text: str) -> tuple[str, ...]:
    """The figures a stretch of prose states, as they are written in it.

    The same reading :func:`verify_numbers` does, exposed without the
    classification, so a caller can ask *which sentence* a checked figure was
    stated in without re-implementing how a number is found.
    """
    return tuple(s.text for s in _read(strip_citations(text)))


def _digit_strings(text: str) -> set[str]:
    """The digit runs in a body of text, comma grouping removed.

    This is the strong half of the number check: a match here means the answer
    reproduced digits that are printed in the evidence, with no rescaling and no
    rounding standing between the two.
    """
    return {m.group(2).replace(",", "").lstrip("0") or "0" for m in _NUMBER.finditer(text)}


def _close(a: float, b: float, *, tol: float = NUMERIC_TOLERANCE) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# --------------------------------------------------------------------------
# the number check
# --------------------------------------------------------------------------


def verify_numbers(
    answer: str,
    evidence: list[Any],
    *,
    question: str = "",
    tol: float = NUMERIC_TOLERANCE,
) -> tuple[NumberClaim, ...]:
    """Classify every figure the answer states against the evidence it was shown.

    The order of the checks is the order of their strength, and the first one
    that fires wins, so ``how`` always names the *best* justification a figure
    has rather than the first one tried.
    """
    body = strip_citations(answer)
    stated = _read(body)
    if not stated:
        return ()

    # What the evidence carries, in the forms a stated figure can match.
    #
    # `values` is exact: a fact's stored value, one number with one meaning.
    # `readings` is the text side, and it is looser by necessity -- a chunk that
    # prints "60,922" inside an income statement means 60.9 billion, and nothing
    # in the chunk says so. So a printed number is offered at every scale unless
    # a scale word pins it, which is the same generosity the scorer applies to
    # answers and for the same reason. The cost is real and worth naming: with
    # enough numbers in the evidence, a fabricated figure can land within
    # tolerance of one of them at some power of a thousand. That is why the
    # digit-string tier is reported separately -- it is the half of the check
    # that cannot be passed by coincidence of scale.
    values: list[tuple[float, str]] = []
    readings: list[tuple[float, str]] = []
    digits: dict[str, str] = {}
    years: set[int] = set()
    for e in evidence:
        citation = getattr(e, "citation", "")
        value = getattr(e, "value", None)
        if value is not None:
            values.append((float(value), citation))
            for d in _digit_strings(f"{float(value):,.0f}"):
                digits.setdefault(d, citation)
        text = _ISO_DATE.sub(" ", getattr(e, "body", "") or "")
        for m in _ISO_DATE.finditer(getattr(e, "body", "") or ""):
            years.add(int(m.group(0)[:4]))
        for d in _digit_strings(text):
            digits.setdefault(d, citation)
        for printed in _read(text):
            readings.extend((r, citation) for r in printed.readings)
        period = str(getattr(e, "period_end", "") or "")
        if len(period) >= 4 and period[:4].isdigit():
            years.add(int(period[:4]))

    q_digits = _digit_strings(question)
    q_years = {int(d) for d in q_digits if len(d) == 4 and d.isdigit()}

    claims: list[NumberClaim] = []
    for s in stated:
        # 1. Not a figure at all: a year the question or the evidence names.
        #    Checked first, because a fiscal year is also a four-digit number
        #    printed in the evidence and would otherwise be counted as a
        #    supported *figure* -- inflating the denominator the hallucination
        #    rate is measured over with numbers nobody is claiming. Guarded so a
        #    money figure shaped like a year ("$2,024 million") is not exempted.
        year = int(s.face)
        if (
            s.bare_int
            and not s.money
            and _YEAR_LO <= year <= _YEAR_HI
            and (year in years or year in q_years)
        ):
            claims.append(
                NumberClaim(s.text, s.face, "context", "period" if year in years else "question")
            )
            continue

        # 2. The digits are printed in the evidence. Nothing was rescaled.
        if s.digits in digits:
            claims.append(NumberClaim(s.text, s.face, "supported", "digits", digits[s.digits]))
            continue

        # 3. A reading of it lands on a value the evidence carries. This is the
        #    tier that permits "$60.9 billion" for a stored 60,922 million, and
        #    the tier whose looseness the module docstring owns up to.
        hit = next(
            (
                (v, cite)
                for r in s.readings
                for v, cite in (*values, *readings)
                if _close(r, v, tol=tol)
            ),
            None,
        )
        if hit is not None:
            claims.append(NumberClaim(s.text, s.face, "supported", "scaled", hit[1]))
            continue

        # 4. Echoed straight from the question. Not the system's claim to make
        #    or to be wrong about -- it is repeating what it was asked.
        if s.digits in q_digits:
            claims.append(NumberClaim(s.text, s.face, "context", "question"))
            continue

        # 5. Arithmetic on two evidence values -- a growth rate, a share, a
        #    difference. Recomputed rather than trusted, and labelled `derived`
        #    rather than `supported`, because the inputs are verified and the
        #    operation is inferred.
        how = _derives(s, values, tol=tol)
        if how:
            claims.append(NumberClaim(s.text, s.face, "derived", "recomputed", how))
            continue

        claims.append(NumberClaim(s.text, s.face, "unsupported"))
    return tuple(claims)


def _derives(s: _Stated, values: list[tuple[float, str]], *, tol: float) -> str:
    """Can this figure be recomputed from two numbers the evidence carries?

    Three operations, because three are what a two-sentence answer about
    filings actually performs: a difference, a ratio expressed as a percentage,
    and a period-over-period change. Anything else is left unsupported rather
    than guessed at -- a verifier that keeps inventing operations until one
    fits is not checking anything.

    Only *fact* values are eligible as inputs, never numbers read out of chunk
    text. A printed number carries no unit, so arithmetic over the text pool
    would be arithmetic over five candidate readings each, and something would
    always come out within tolerance of something. Restricting derivation to
    values that arrived with a unit attached is what keeps this a check.
    """
    if len(values) < 2:
        return ""
    nums = [v for v, _ in values]
    for i, a in enumerate(nums):
        for b in nums[i + 1 :]:
            for reading in s.readings:
                if _close(reading, a - b, tol=tol) or _close(reading, b - a, tol=tol):
                    return f"difference of {a:,.0f} and {b:,.0f}"
            if b:
                pct = 100.0 * a / b
                change = 100.0 * (a - b) / abs(b)
                if _close(s.face, pct, tol=tol):
                    return f"{a:,.0f} as a percentage of {b:,.0f}"
                if _close(s.face, change, tol=tol):
                    return f"change from {b:,.0f} to {a:,.0f}"
            if a:
                change = 100.0 * (b - a) / abs(a)
                if _close(s.face, 100.0 * b / a, tol=tol):
                    return f"{b:,.0f} as a percentage of {a:,.0f}"
                if _close(s.face, change, tol=tol):
                    return f"change from {a:,.0f} to {b:,.0f}"
    return ""


def recheck_facts(
    answer: str,
    evidence: list[Any],
    *,
    sql: Any,
    tol: float = NUMERIC_TOLERANCE,
) -> tuple[str, ...]:
    """Re-read every fact the answer rests on, and confirm the prose kept it.

    This is the gate's "recomputed in DuckDB from source facts" line. For each
    piece of fact evidence, :meth:`SqlTool.recheck` goes back to the store by
    (ticker, tag, period, unit) -- not through the concept resolver, which is
    one of the components under test -- and the value it returns has to be
    findable in the answer at some scale. A synthesiser shown 60,922 that writes
    69,022 produces evidence and store agreeing with each other and disagreeing
    with the prose, which is exactly the shape this catches.

    Returns one string per disagreement, empty when the answer is clean. Facts
    the store no longer holds are reported too: an answer citing a figure that
    has since been restated away is not a figure anyone should be quoting.
    """
    if sql is None:
        return ()
    body = strip_citations(answer)
    stated = [r for s in _read(body) for r in s.readings]
    # An answer that states no figure has kept nothing to disagree with. The
    # refusal token is the case that matters: it was reaching here with the
    # fact evidence still attached and being reported as contradicting the
    # store, which made every abstention -- the behaviour the gate rewards --
    # look like a verification failure. `verify_answer` promises refusals pass
    # trivially, and this is where that promise is kept.
    if not stated:
        return ()
    problems: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for e in evidence:
        if getattr(e, "kind", "") != "fact" or getattr(e, "value", None) is None:
            continue
        key = (getattr(e, "ticker", ""), getattr(e, "tag", ""), str(getattr(e, "period_end", "")))
        if key in seen or not all(key[:2]):
            continue
        seen.add(key)
        row = sql.recheck(
            key[0],
            key[1],
            period_end=key[2] or None,
            unit=getattr(e, "unit", "") or "USD",
            accn=getattr(e, "accn", ""),
        )
        if row is None:
            problems.append(f"{key[0]} {key[1]} {key[2]} is no longer in the store")
            continue
        if not any(_close(r, float(row.val), tol=tol) for r in stated):
            problems.append(
                f"{key[0]} {key[1]} {key[2]} is {float(row.val):,.0f} in the store, "
                f"and no figure in the answer matches it"
            )
    return tuple(problems)


# --------------------------------------------------------------------------
# the citation resolver
# --------------------------------------------------------------------------


def has_locator(e: Any) -> bool:
    """Can a reader get back to the filing from this piece of evidence?

    Two shapes qualify, and they are the two the unified evidence schema
    produces: an accession with a character span, and an accession with a tag
    and a period. Both name something a person can open. An accession on its
    own does not -- it names a document, and "it is somewhere in this 300-page
    10-K" is the kind of citation this project exists to not accept.
    """
    if not getattr(e, "accn", ""):
        return False
    if getattr(e, "char_end", 0) > getattr(e, "char_start", 0):
        return True
    return bool(getattr(e, "tag", "") and getattr(e, "period_end", ""))


def resolve_citations(answer: str, evidence: list[Any]) -> tuple[tuple[int, ...], ...]:
    """Split an answer's markers into (made, dangling, unlocatable).

    Dangling means the marker points past the end of the evidence list -- the
    model cited [7] when it was shown five things. Unlocatable means it points
    at real evidence that carries nothing to look up, which is rarer and worse,
    because it looks like a citation all the way until someone tries to follow
    it.
    """
    made: list[int] = []
    dangling: list[int] = []
    unlocatable: list[int] = []
    for i in markers_in(answer):
        if not 1 <= i <= len(evidence):
            dangling.append(i)
        elif not has_locator(evidence[i - 1]):
            unlocatable.append(i)
        else:
            made.append(i)
    return tuple(made), tuple(dangling), tuple(unlocatable)


def uncited_claims(answer: str) -> tuple[str, ...]:
    """Sentences that state a figure and cite nothing, plus the answer that cites nothing at all.

    The per-sentence check is scoped to sentences carrying a number on purpose.
    Requiring a marker on every sentence would flag "Revenue rose year over year
    [1]. This was driven by data centre demand." -- a second sentence that adds
    no checkable claim -- and a guard that fires on good prose gets switched
    off, which is the failure mode the M6 bail-out clause was written for.

    That scoping leaves a hole, and the hole was found by measurement rather
    than by reading: stripping every marker from the agent's real answers was
    caught 98 times out of 116, and all eighteen escapes were narrative answers
    that state no figure. Being figure-scoped, the check had nothing to look at,
    so a purely qualitative answer citing nothing passed -- which is precisely
    the claim-without-a-locator this gate exists to stop. Hence the second rule:
    an answer written from evidence has to cite *something*. It is a far weaker
    demand than a marker per sentence, and the run shows it costs nothing --
    every one of the 116 real answers already satisfies it.
    """
    out: list[str] = []
    for sentence in sentences(answer):
        if _CITATION.search(sentence):
            continue
        if figures_in(sentence):
            out.append(sentence)
    # Only when the per-sentence pass found nothing: an uncited answer that
    # states a figure is already reported sentence by sentence, and adding the
    # whole text again would count one failure twice.
    if not out and not _CITATION.search(answer) and strip_citations(answer).strip():
        out.append(answer.strip())
    return tuple(out)


# --------------------------------------------------------------------------
# the whole verdict, and the guard
# --------------------------------------------------------------------------


def verify_answer(
    answer: str,
    evidence: list[Any],
    *,
    question: str = "",
    route: str = "",
    grade: Any = None,
    repairs: int = 0,
    repair_log: list[str] | None = None,
    sql: Any = None,
    refused: bool = False,
    tol: float = NUMERIC_TOLERANCE,
) -> Verdict:
    """Everything the verifier can say about a finished answer.

    Refusals are verified too, and pass trivially: an abstention makes no claim,
    so there is nothing in it to be unsupported. That is worth being explicit
    about, because the opposite convention -- treating a refusal as a failed
    verification -- would make the guard's own output look like the thing it
    guards against.
    """
    numbers = verify_numbers(answer, evidence, question=question, tol=tol)
    made, dangling, unlocatable = resolve_citations(answer, evidence)
    # An abstention is excused from the citation rules, not from the number
    # ones: it makes no claim to cite, and demanding a marker on the refusal
    # token would score the behaviour the gate rewards as a failure.
    uncited = uncited_claims(answer) if evidence and not refused else ()
    unrechecked = recheck_facts(answer, evidence, sql=sql, tol=tol) if sql is not None else ()

    reasons: list[str] = []
    if bad := [n for n in numbers if n.status == "unsupported"]:
        shown = ", ".join(n.text for n in bad[:3])
        reasons.append(f"{len(bad)} figure(s) not in the evidence: {shown}")
    if dangling:
        reasons.append(f"citation(s) past the end of the evidence: {list(dangling)}")
    if unlocatable:
        reasons.append(f"citation(s) with nothing to look up: {list(unlocatable)}")
    if uncited:
        reasons.append(f"{len(uncited)} sentence(s) state a figure and cite nothing")
    reasons.extend(unrechecked)

    return Verdict(
        numbers=numbers,
        markers=made,
        dangling=dangling,
        unlocatable=unlocatable,
        uncited=uncited,
        unrechecked=unrechecked,
        taxonomy=classify(
            route=route,
            grade=grade,
            evidence=evidence,
            numbers=numbers,
            dangling=dangling,
            repairs=repairs,
            repair_log=repair_log or [],
            unrechecked=unrechecked,
        ),
        reasons=tuple(reasons),
    )


def guard_answer(
    answer: str,
    verdict: Verdict,
    *,
    mode: str = "block",
    refusal: str = "",
) -> tuple[str, bool]:
    """Decide whether the answer ships. Returns ``(answer, blocked)``.

    ``block`` replaces a failing answer with the refusal token, which is the
    strong reading of the gate: a claim without a resolvable locator does not
    ship. ``flag`` computes the identical verdict and lets the answer through,
    which is the pre-decided response if blocking turns out to reject too many
    valid answers -- the point of having written the bail-out down in advance is
    that switching to it is a reported decision rather than a quiet loosening of
    the check.
    """
    if mode not in GUARD_MODES:  # pragma: no cover - caller error
        raise ValueError(f"unknown guard mode {mode!r}; expected one of {GUARD_MODES}")
    if mode != "block" or verdict.ok:
        return answer, False
    if not refusal:
        from filing.eval.runner import REFUSAL as refusal_token

        refusal = refusal_token
    return refusal, True


# --------------------------------------------------------------------------
# the failure taxonomy
# --------------------------------------------------------------------------

#: The four tags M6 names, plus what each one is actually observable from. The
#: honesty that matters here is in `router_wrong`: the agent has no gold labels
#: at answer time, so it cannot know the router was wrong. What it *can* see is
#: that the router's choice did not survive contact with the store -- the SQL
#: branch found nothing and the repair loop moved the question to text. That is
#: a weaker claim than "wrong", and the tag is documented as the weaker claim
#: rather than named for the stronger one.
TAXONOMY = {
    "retrieval_miss": "the chosen store returned nothing, or nothing relevant",
    "router_wrong": "the router's store was abandoned by a repair",
    "grader_false_positive": "the grader passed evidence the verifier then failed",
    "synthesis_drift": "a figure in the answer is not in the evidence",
}


def classify(
    *,
    route: str,
    grade: Any,
    evidence: list[Any],
    numbers: tuple[NumberClaim, ...],
    dangling: tuple[int, ...],
    repairs: int,
    repair_log: list[str],
    unrechecked: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Tag what went wrong, so the failure gallery is a filter and not a read.

    Tags are a set rather than a choice: a question whose SQL branch missed, got
    moved to text, and then produced a figure from nowhere has committed three
    of these and reporting only the first would lose the two that matter more.
    """
    tags: list[str] = []
    ok = bool(getattr(grade, "ok", False))

    # No grade means the graph did not get far enough to have an opinion, which
    # is not the same as the grader having a bad one. Only an explicit failure,
    # or an empty evidence list, counts as a miss -- a tag that fires whenever a
    # field happens to be absent is a tag nobody can filter on.
    if route != "refuse" and (not evidence or (grade is not None and not ok)):
        tags.append("retrieval_miss")
    if repairs and any("fall back" in note or "text retriever" in note for note in repair_log):
        tags.append("router_wrong")
    drift = bool(unrechecked) or any(n.status == "unsupported" for n in numbers)
    if drift:
        tags.append("synthesis_drift")
    if ok and (drift or dangling):
        tags.append("grader_false_positive")
    return tuple(tags)


@dataclass
class GuardReport:
    """Aggregate over a run, for the three numbers the gate asks for."""

    answers: int = 0
    verified: int = 0
    blocked: int = 0
    figures: int = 0
    supported: int = 0
    by_digits: int = 0
    derived: int = 0
    unsupported: int = 0
    citations: int = 0
    dangling: int = 0
    unlocatable: int = 0
    uncited: int = 0
    tags: dict[str, int] = field(default_factory=dict)

    def add(self, verdict: Verdict, *, blocked: bool = False) -> None:
        self.answers += 1
        self.verified += int(verdict.ok)
        self.blocked += int(blocked)
        for n in verdict.checked:
            self.figures += 1
            if n.status == "supported":
                self.supported += 1
                self.by_digits += int(n.how == "digits")
            elif n.status == "derived":
                self.derived += 1
            else:
                self.unsupported += 1
        self.citations += len(verdict.markers)
        self.dangling += len(verdict.dangling)
        self.unlocatable += len(verdict.unlocatable)
        self.uncited += len(verdict.uncited)
        for tag in verdict.taxonomy:
            self.tags[tag] = self.tags.get(tag, 0) + 1

    @property
    def hallucination_rate(self) -> float:
        """Unsupported figures as a share of the figures actually checked."""
        return self.unsupported / self.figures if self.figures else 0.0

    @property
    def locator_rate(self) -> float:
        """Share of citations that resolve to evidence with something to look up."""
        total = self.citations + self.dangling + self.unlocatable
        return self.citations / total if total else 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["hallucination_rate"] = self.hallucination_rate
        d["locator_rate"] = self.locator_rate
        return d
