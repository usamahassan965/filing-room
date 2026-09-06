"""How deep you would have to search before the evidence appears.

``hit@10`` says a system failed. It does not say *how* it failed, and the two
ways are not the same problem:

**The evidence ranked 24th.** The representation is working -- the right passage
is in the neighbourhood and something merely put nine other passages in front of
it. Retrieving deeper and reranking recovers it, and the ceiling on that repair
is measurable in advance.

**The evidence is not in the top 500.** Nothing downstream can fix that. No
reranker reorders a list the passage is absent from, and no prompt makes a model
cite what it never saw. The only repairs are a different index or a different
tool entirely.

A results table cannot tell those apart, because ``hit@10`` is 0 in both cases.
So this module reports the whole curve -- ``hit@1`` through ``hit@500`` -- and,
beside it, whether the search at least reached the right *document*. The gap
between "found the filing" and "found the passage" is the single most useful
number for deciding what to build next: when it is wide, the fix is chunking and
ranking; when the filing itself is missing, the fix is upstream of both.

Costs nothing but CPU, calls no model, and is reproducible on any machine with
the index built -- which is what makes it safe to quote in a write-up.
"""

from __future__ import annotations

import json
import statistics as st
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from filing.config import Settings
from filing.eval import dataset
from filing.eval.dataset import EvalQuestion
from filing.eval.naive import NaiveRetriever

# Deep enough that "not here" means the dense representation genuinely cannot
# separate this passage from 48,000 others, rather than that we did not look.
DEEP = 500

# The curve is reported at these depths: the ones a system might actually use
# (1, 5, 10), the ones a fuse-then-rerank pipeline uses (25, 50, 100), and the
# floor (500), which is not a design point but a diagnosis.
DEPTHS = (1, 5, 10, 25, 50, 100, 500)


@dataclass
class SliceDepth:
    name: str
    n: int = 0
    found: int = 0
    median_rank: float | None = None
    hit_at: dict[int, float] = field(default_factory=dict)
    filing_found: int = 0
    filing_median_rank: float | None = None
    filing_hit_at_10: float = 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["hit_at"] = {str(k): v for k, v in d["hit_at"].items()}
        return d


def _gold_chunk_ids(question: EvalQuestion, by_accn: dict[str, list]) -> set[str]:
    return {
        c.chunk_id
        for s in question.spans
        for c in by_accn.get(s.accn, [])
        if c.char_start < s.char_end and c.char_end > s.char_start
    }


def measure(
    cfg: Settings,
    *,
    retriever: NaiveRetriever | None = None,
    depth: int = DEEP,
    version: str = dataset.DATASET_VERSION,
) -> dict[str, SliceDepth]:
    """Rank the gold chunk, and the gold filing, for every answerable question."""
    questions = [
        q
        for q in dataset.read(dataset.dataset_path(Path(cfg.data_dir), version))
        if q.spans  # the unanswerable slice has no evidence to rank
    ]
    ret = retriever
    if ret is None:
        ret = NaiveRetriever(cfg)
        ret.require()

    by_accn: dict[str, list] = {}
    for c in ret.chunks.values():
        by_accn.setdefault(c.accn, []).append(c)

    out: dict[str, SliceDepth] = {}
    for name in sorted({q.slice for q in questions}):
        subset = [q for q in questions if q.slice == name]
        chunk_ranks: list[int] = []
        filing_ranks: list[int] = []
        for q in subset:
            gold = _gold_chunk_ids(q, by_accn)
            accns = {s.accn for s in q.spans}
            hits = ret.search(q.question, k=depth)
            r = next((i for i, h in enumerate(hits, 1) if h.chunk_id in gold), None)
            f = next((i for i, h in enumerate(hits, 1) if h.accn in accns), None)
            if r:
                chunk_ranks.append(r)
            if f:
                filing_ranks.append(f)
        s = SliceDepth(name=name, n=len(subset), found=len(chunk_ranks))
        if chunk_ranks:
            s.median_rank = st.median(chunk_ranks)
        s.hit_at = {
            k: sum(1 for r in chunk_ranks if r <= k) / len(subset) for k in DEPTHS if k <= depth
        }
        s.filing_found = len(filing_ranks)
        if filing_ranks:
            s.filing_median_rank = st.median(filing_ranks)
            s.filing_hit_at_10 = sum(1 for r in filing_ranks if r <= 10) / len(subset)
        out[name] = s
    return out


def to_markdown(slices: dict[str, SliceDepth], *, title: str = "") -> str:
    ks = sorted({k for s in slices.values() for k in s.hit_at})
    lines = [f"### {title}", ""] if title else []
    head = ["slice", "n", *[f"hit@{k}" for k in ks], "median rank", "filing@10", "filing rank"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for s in slices.values():
        cells = [s.name, str(s.n)]
        cells += [f"{100 * s.hit_at[k]:.1f}%" if k in s.hit_at else "--" for k in ks]
        cells += [
            f"{s.median_rank:.0f}" if s.median_rank else "--",
            f"{100 * s.filing_hit_at_10:.1f}%",
            f"{s.filing_median_rank:.0f}" if s.filing_median_rank else "--",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write(cfg: Settings, slices: dict[str, SliceDepth], *, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "retrieval-depth.md").write_text(
        to_markdown(slices, title=f"how deep the evidence sits (dense only, top {DEEP})"),
        encoding="utf-8",
    )
    path = root / "retrieval-depth.json"
    path.write_text(
        json.dumps({k: v.to_json() for k, v in slices.items()}, indent=1), encoding="utf-8"
    )
    return path
