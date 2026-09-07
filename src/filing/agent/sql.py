"""Text to SQL, constrained so hard that it cannot go wrong in the usual ways.

A language model writing SQL against a warehouse is the demo everyone has seen
and the thing nobody runs unattended, because the failure mode is not a syntax
error -- it is a query that runs, returns a number, and is about the wrong
concept. This layer removes the two decisions that produce that failure and
leaves the model only the one it is good at.

**The model never writes SQL.** It fills in three slots: a ticker, a phrase
naming a financial concept, and a period. The SQL text is written here, once,
parameterised, against a read-only connection. There is no injection surface
because there is no string to inject into.

**The concept slot resolves against the registry, not against the model's
vocabulary.** M2 built a registry recording which XBRL tags this corpus actually
uses for each concept; that registry's tag union is the allow-list, and a phrase
that does not resolve inside it returns nothing rather than a guess. "That is
not a concept here" and "this company reports no such fact" are different
answers, and both beat a number lifted from an adjacent tag.

**Resolution reads the corpus's own labels.** ``NetIncomeLoss`` is published by
the SEC under the standard label "Net Income (Loss) Attributable to Parent",
which no amount of CamelCase-splitting reproduces, so the ``concepts`` table --
the label dictionary the filings themselves shipped with -- is the lookup.
Exact normalised match first, token overlap as the fallback, so "net income"
and "total assets" resolve the way a person would expect.

**Queries run at fact grain, by resolved tag.** Not against the wide ``annual``
view, and that is a deliberate reversal of what M2 built. The registry collapses
synonyms on purpose: ``Revenues`` and ``RevenueFromContractWithCustomer
ExcludingAssessedTax`` are both metric ``revenue``, which is right for a ratio
and wrong for a question -- someone asking about one of those tags is asking
about that tag, and the two do not always carry the same number in the same
filing. So the registry decides *what may be asked* and the fact table decides
*what is answered*.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from filing.stores.facts import FactsStore
from filing.stores.metrics import METRICS

# A fiscal year end moves. Retailers on a 52/53-week calendar land on a
# different date every year, so a question naming last year's date is still
# asking about this year's year end. Wider than a week starts matching quarters.
PERIOD_SLACK_DAYS = 6

# Below this a token-overlap match is a coincidence rather than a resolution.
# Two-token phrases are the hard case -- "gross profit" against "Gross Profit"
# scores 1.0, "total assets" against "Assets" scores 0.5 -- so the floor sits
# under a half match and the tie-break does the rest.
MIN_OVERLAP = 0.34

#: Every tag any registered metric names: the allow-list, in one expression.
ALLOWED_TAGS: frozenset[str] = frozenset(tag for metric in METRICS for tag in metric.tags)

#: Tag -> the metric that registered it. Where two metrics share a tag the first
#: registration wins, which is the registry's own priority order.
TAG_METRIC: dict[str, str] = {}
for _metric in METRICS:
    for _tag in _metric.tags:
        TAG_METRIC.setdefault(_tag, _metric.name)

_WORD = re.compile(r"[a-z0-9]+")

# Words carrying no discriminating power in a concept phrase. Kept short on
# purpose: dropping "net" or "total" would merge concepts that differ by
# exactly that word.
_STOP = frozenset({"the", "a", "an", "of", "for", "and", "at", "in", "on", "to", "s"})


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOP)


def _normalise(text: str) -> str:
    """Fold a label or a question phrase onto a comparable key."""
    return " ".join(_WORD.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class Concept:
    """A resolved concept: what the SQL filters on, and why it was chosen."""

    taxonomy: str
    tag: str
    label: str
    metric: str
    score: float
    how: str  # exact-label | exact-tag | metric-label | overlap


@dataclass(frozen=True, slots=True)
class FactRow:
    """One fact, carrying everything a citation needs."""

    ticker: str
    tag: str
    label: str
    unit: str
    span: str
    period_start: date | None
    period_end: date
    val: float
    accn: str
    form: str

    @property
    def citation(self) -> str:
        """Shaped like a text hit's citation, so a mixed answer reads evenly."""
        return f"{self.ticker} {self.form} {self.period_end} {self.tag} [{self.accn}]"


@dataclass(frozen=True, slots=True)
class SqlAnswer:
    """What the SQL branch returns -- including when it returns nothing.

    Empty ``rows`` with ``concept`` set means "we know that concept and this
    company reports no such fact"; ``concept`` unset means "that is not
    something this store can be asked about". A caller collapsing those two into
    "no answer" throws away the only part a user can act on.
    """

    query: str
    params: tuple[object, ...]
    rows: tuple[FactRow, ...]
    concept: Concept | None
    reason: str = ""

    @property
    def found(self) -> bool:
        return bool(self.rows)

    @property
    def value(self) -> float | None:
        return self.rows[0].val if self.rows else None


# One statement, one shape, written here rather than generated. `facts_current`
# is already restated-to-latest, so this never has to think about amendments;
# `filings` supplies the form the citation needs.
_LOOKUP = """
SELECT f.ticker, f.tag, coalesce(c.label, f.tag), f.unit, f.span,
       f.period_start, f.period_end, f.val, f.accn, g.form
FROM facts_current f
JOIN filings g ON g.accn = f.accn
LEFT JOIN concepts c ON c.taxonomy = f.taxonomy AND c.tag = f.tag
WHERE f.ticker = ?
  AND f.taxonomy = ?
  AND f.tag = ?
  AND f.unit = ?
  AND f.span IN ('FY', 'instant')
"""

# The same read without the taxonomy clause, because a piece of evidence
# carries a tag but not the taxonomy it came from, and inventing "us-gaap" here
# would make the verifier's recheck silently miss anything filed under another.
_RECHECK = """
SELECT f.ticker, f.tag, coalesce(c.label, f.tag), f.unit, f.span,
       f.period_start, f.period_end, f.val, f.accn, g.form
FROM facts_current f
JOIN filings g ON g.accn = f.accn
LEFT JOIN concepts c ON c.taxonomy = f.taxonomy AND c.tag = f.tag
WHERE f.ticker = ?
  AND f.tag = ?
  AND f.unit = ?
  AND f.span IN ('FY', 'instant')
"""

_ORDER_BY_PROXIMITY = " ORDER BY abs(date_diff('day', f.period_end, ?)) ASC, f.period_end DESC"
_ORDER_BY_RECENCY = " ORDER BY f.period_end DESC"


class UnknownConcept(LookupError):
    """The phrase did not resolve to a tag inside the registry's allow-list."""


class ConceptResolver:
    """Phrase -> tag, via the corpus's own label dictionary and the registry.

    Built once per store and held: it is a few hundred rows, and resolving sits
    on the hot path of every numeric question.
    """

    def __init__(self, store: FactsStore) -> None:
        self._by_label: dict[str, Concept] = {}
        self._by_tag: dict[str, Concept] = {}
        self._all: list[Concept] = []
        tags = sorted(ALLOWED_TAGS)
        holes = ",".join("?" * len(tags))
        rows = store.sql(
            f"SELECT taxonomy, tag, label FROM concepts WHERE tag IN ({holes})",
            list(tags),
        )
        for taxonomy, tag, label in rows:
            text = (label or tag).rstrip(".")
            concept = Concept(taxonomy, tag, text, TAG_METRIC[tag], 1.0, "exact-label")
            self._by_label.setdefault(_normalise(text), concept)
            self._by_tag.setdefault(tag.lower(), concept)
            self._all.append(concept)
        # The registry's own names are aliases: "Total current assets" is what
        # M2 calls `assets_current`, and a user may type either instead of the
        # SEC's "Assets, Current".
        self._by_metric: dict[str, Concept] = {}
        for metric in METRICS:
            for tag in metric.tags:
                found = self._by_tag.get(tag.lower())
                if found is not None:
                    self._by_metric.setdefault(_normalise(metric.label), found)
                    self._by_metric.setdefault(_normalise(metric.name), found)
                    break

    def __len__(self) -> int:
        return len(self._all)

    @property
    def tags(self) -> frozenset[str]:
        """The tags this store can be asked about: allow-list intersect corpus."""
        return frozenset(c.tag for c in self._all)

    def resolve(self, phrase: str) -> Concept | None:
        """Best concept for a phrase, or ``None`` rather than a plausible guess."""
        found = self.candidates(phrase, limit=1)
        return found[0] if found else None

    def require(self, phrase: str) -> Concept:
        concept = self.resolve(phrase)
        if concept is None:
            raise UnknownConcept(
                f"{phrase!r} is not a concept in the metric registry "
                f"({len(self._all)} tags available)"
            )
        return concept

    def candidates(self, phrase: str, *, limit: int = 5) -> list[Concept]:
        """Ranked resolutions. Exact routes short-circuit; overlap is the fallback."""
        key = _normalise(phrase)
        if not key:
            return []
        exact = self._by_label.get(key)
        if exact is not None:
            return [exact]
        by_tag = self._by_tag.get(key.replace(" ", ""))
        if by_tag is not None:
            return [_rescore(by_tag, 1.0, "exact-tag")]
        by_metric = self._by_metric.get(key)
        if by_metric is not None:
            return [_rescore(by_metric, 1.0, "metric-label")]
        return _rank_overlap(phrase, self._all, limit=limit)


def _rescore(concept: Concept, score: float, how: str) -> Concept:
    return Concept(concept.taxonomy, concept.tag, concept.label, concept.metric, score, how)


def _rank_overlap(phrase: str, concepts: Iterable[Concept], *, limit: int) -> list[Concept]:
    """Overlap scoring, biased toward the shorter label.

    Plain Jaccard punishes a long SEC label for being long: "net income" against
    "Net Income (Loss) Attributable to Parent" scores 2/6. Dividing by the
    query's token count instead asks "did the label contain what was asked?",
    and the length penalty returns as a small tie-break, so a phrase matching
    two labels equally picks the more specific one.
    """
    want = _tokens(phrase)
    if not want:
        return []
    scored: list[Concept] = []
    for concept in concepts:
        have = _tokens(concept.label)
        shared = want & have
        if not shared:
            continue
        score = len(shared) / len(want) - 0.01 * len(have - want)
        if score >= MIN_OVERLAP:
            scored.append(_rescore(concept, round(score, 4), "overlap"))
    scored.sort(key=lambda c: (-c.score, len(c.label), c.tag))
    return scored[:limit]


class SqlTool:
    """The numeric branch of the agent, and the only thing here that touches DuckDB.

    Deliberately not a general query interface. It answers exactly one question
    -- what did this company report for this concept in this period -- because
    that is the question the corpus answers exactly, and an agent that can only
    ask answerable questions is never caught guessing.
    """

    def __init__(self, store: FactsStore) -> None:
        self.store = store
        self.resolver = ConceptResolver(store)

    def schema(self) -> dict[str, object]:
        """What the planning call is told this tool accepts.

        Emitted from the code rather than written out beside it, so a metric
        added to the registry is a metric the model hears about in the same
        commit.
        """
        return {
            "name": "lookup_fact",
            "description": (
                "Look up one reported figure for one company in one period, from "
                "XBRL facts as filed. Returns the number, the tag it was reported "
                "under, and the accession number to cite."
            ),
            "parameters": {
                "ticker": "Ticker symbol, e.g. AAPL.",
                "concept": (
                    "The financial concept in plain words -- 'net income', "
                    "'total current assets' -- or the SEC standard label if the "
                    "question used one."
                ),
                "period_end": "Period end date as YYYY-MM-DD, if the question names one.",
            },
            "concepts": sorted({m.label for m in METRICS}),
        }

    def lookup(
        self,
        ticker: str,
        concept: str,
        *,
        period_end: str | date | None = None,
        unit: str = "USD",
        limit: int = 1,
    ) -> SqlAnswer:
        resolved = self.resolver.resolve(concept)
        if resolved is None:
            return SqlAnswer(
                query="",
                params=(),
                rows=(),
                concept=None,
                reason=f"no registry concept matches {concept!r}",
            )
        if resolved.tag not in ALLOWED_TAGS:  # pragma: no cover - resolver builds from it
            raise UnknownConcept(f"{resolved.tag} is outside the allow-list")

        params: list[object] = [ticker.upper(), resolved.taxonomy, resolved.tag, unit]
        query = _LOOKUP
        target = _as_date(period_end)
        if target is not None:
            query += _ORDER_BY_PROXIMITY
            params.append(target)
        else:
            query += _ORDER_BY_RECENCY
        query += f" LIMIT {int(limit)}"

        rows = tuple(FactRow(*row) for row in self.store.sql(query, params))
        if target is not None:
            rows = tuple(r for r in rows if _within(r.period_end, target))
        reason = ""
        if not rows:
            near = f" near {target}" if target else ""
            reason = f"{ticker.upper()} reports no {resolved.tag} in {unit}{near}"
        return SqlAnswer(
            query=query.strip(),
            params=tuple(params),
            rows=rows,
            concept=resolved,
            reason=reason,
        )

    def recheck(
        self,
        ticker: str,
        tag: str,
        *,
        period_end: str | date | None = None,
        unit: str = "USD",
        accn: str = "",
    ) -> FactRow | None:
        """Go back to the store for one fact, keyed by its identity.

        M6's verifier calls this on every figure an answer states that came from
        the fact branch. It deliberately does *not* go through the concept
        resolver: the resolver is one of the components being checked, and a
        recheck that asks the same phrase-matching layer the same question is a
        recheck that agrees with itself. Given a ticker, a tag and a period, this
        is a primary-key read.

        Two honest limits. It cannot tell you the tag was the right one to have
        picked -- only that the store holds this value under it. And for a fact
        the SQL branch itself retrieved, the value comes back from the same table
        it went into, so what the recheck actually rules out is the answer's
        number having drifted from the store's between retrieval and prose. That
        is the failure it is aimed at, and the only one it should be credited
        with catching.
        """
        params: list[object] = [ticker.upper(), tag, unit]
        query = _RECHECK
        if accn:
            query += " AND f.accn = ?"
            params.append(accn)
        target = _as_date(period_end)
        if target is not None:
            query += _ORDER_BY_PROXIMITY
            params.append(target)
        else:
            query += _ORDER_BY_RECENCY
        rows = tuple(FactRow(*row) for row in self.store.sql(query + " LIMIT 1", params))
        if not rows:
            return None
        if target is not None and not _within(rows[0].period_end, target):
            return None
        return rows[0]

    def periods(self, ticker: str) -> Sequence[date]:
        """Fiscal year ends this company actually reports. For the repair node."""
        rows = self.store.sql(
            "SELECT fiscal_end FROM fiscal_years WHERE ticker = ? ORDER BY fiscal_end DESC",
            [ticker.upper()],
        )
        return [row[0] for row in rows]


def _as_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _within(got: date, target: date) -> bool:
    return abs((got - target).days) <= PERIOD_SLACK_DAYS
