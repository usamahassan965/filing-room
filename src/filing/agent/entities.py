"""The graph branch: relationships between named organisations.

M3 built an entity graph -- supplier, customer, competitor, acquisition and
regulator edges, each carrying the sentence it was extracted from and the
character offsets of that sentence in the filing. This is the read side, shaped
so its output is :class:`~filing.agent.state.Evidence` like everything else.

**It is worth being blunt about what this branch is not tested by.** The frozen
v1.0 eval set has no relationship questions: all 150 are single-company,
single-period, and the numeric 80 are point lookups. So the graph branch ships
with unit tests and no eval-set evidence, and any accuracy claim in this project
is a claim about the SQL and text branches. Building it anyway is a judgement
call -- the gate asks for three retrieval stores and the graph is the one that
answers a question the other two structurally cannot -- but "we built it" and
"we measured it" are different sentences and only one of them is true here.

Matching is by entity name against the edge endpoints, scored by how much of the
name matched and tie-broken by recency. No embedding: the endpoints are proper
nouns that M3 already normalised, and fuzzy-matching proper nouns is how
"Micron" becomes "Microsoft".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from filing.agent.state import Evidence

_WORD = re.compile(r"[A-Za-z0-9&]+")

# Words that appear in company names and in ordinary questions alike. Matching
# on one of these alone would return every edge in the store.
_NOISE = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "company",
        "co",
        "ltd",
        "llc",
        "plc",
        "holdings",
        "group",
        "the",
        "and",
        "of",
        "for",
        "with",
        "who",
        "what",
        "which",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "s",
    }
)


def _terms(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _NOISE and len(w) > 1]


class GraphTool:
    """Question -> edges, as evidence.

    Loads the edge list once. Twenty companies produce a few thousand edges, so
    this is a list scan and a scoring loop; a graph database here would be
    ceremony, which is the same call M3 made when it chose networkx.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self._edges: list[Any] | None = None

    @property
    def available(self) -> bool:
        return bool(getattr(self.store, "exists", False))

    @property
    def edges(self) -> Sequence[Any]:
        if self._edges is None:
            self._edges = list(self.store.edges()) if self.available else []
        return self._edges

    def search(self, question: str, *, ticker: str = "", limit: int = 5) -> list[Evidence]:
        """Edges whose endpoints the question names, best first.

        The score is the share of the question's content words found in the two
        endpoints plus the relation kind, which means a question naming both
        ends of an edge outranks one naming only the filer. Ties go to the more
        recent filing, because a supplier relationship from 2019 and one from
        2024 are both true and only one of them is current.
        """
        want = set(_terms(question))
        if ticker:
            want.add(ticker.lower())
        if not want:
            return []
        scored: list[tuple[float, str, Any]] = []
        for edge in self.edges:
            have = set(_terms(f"{edge.source} {edge.target} {edge.kind}"))
            have.add(str(edge.ticker).lower())
            shared = want & have
            if not shared:
                continue
            score = len(shared) / len(want)
            if ticker and str(edge.ticker).upper() == ticker.upper():
                # The filer is the company whose filing asserted the
                # relationship, so an edge from the asked-about company's own
                # 10-K is better evidence than the same claim in a rival's.
                score += 0.25
            scored.append((score, str(edge.period_end), edge))
        scored.sort(key=lambda row: (-row[0], _neg_date(row[1])))
        return [
            Evidence.from_edge(edge, score=round(score, 4)) for score, _, edge in scored[:limit]
        ]


def _neg_date(period_end: str) -> str:
    """Sort dates descending inside an ascending sort, without parsing them.

    ISO dates sort lexically, so inverting each character's ordinal would work
    and be unreadable. Negating the sort key by complementing the digits is the
    same trick with one fewer surprise: '2024-12-31' -> '7975-87-68'.
    """
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in period_end)
