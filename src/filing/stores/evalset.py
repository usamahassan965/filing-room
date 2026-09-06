"""The retrieval smoke set: thirty questions with checkable gold evidence.

The hard part of a retrieval eval is not the metric, it is the ground truth. The
options are to hand-label passages -- thirty questions times fifty candidates is
1,500 judgements, and they go stale the moment the chunker changes -- or to
define relevance by a rule the corpus can be asked to evaluate. This module does
the second: a question carries a *predicate*, and the gold set is every indexed
chunk that satisfies it.

    "What did Chevron say about the arbitration over its Hess acquisition?"
        ticker CVX, narrative items, and the text contains "Hess"

That is a weaker notion of relevance than a human judgement, and it is stated
plainly rather than dressed up: it says a chunk is *on topic*, not that it
answers the question. Two properties make it worth having anyway. It is exactly
reproducible -- a re-chunk re-derives the gold set instead of invalidating it --
and it cannot be gamed by the thing under test, because the predicate is
evaluated against the corpus and never against what retrieval returned.

Two metrics, both defined here rather than assumed:

* **recall@50** -- the fraction of questions with at least one gold chunk in the
  fused top 50. With a gold *set* rather than a single gold passage, "did the
  evidence make the candidate window" is the question that matters: everything
  downstream, reranking included, can only reorder what recall let through.
* **precision@5** -- the mean fraction of the top five that are gold. Measured
  twice per question, once on the RRF order and once after the cross-encoder,
  over the identical candidate list. That difference is the only evidence that
  reranking is worth its latency.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from filing.stores.chunks import Chunk

# Item keys, by form, that a narrative question can land in. Kept as literals
# rather than derived, so a question that names an item nobody indexes fails at
# validation instead of scoring zero forever.
_RISK = ("1A", "II.1A", "I.1A", "FULL")
_MDNA = ("7", "I.2", "FULL")
_BUSINESS = ("1", "FULL")
_LEGAL = ("3", "II.1", "FULL")
_NARRATIVE = ("1", "1A", "1C", "3", "7", "7A", "I.2", "II.1", "II.1A", "I.1A", "FULL")


@dataclass(frozen=True, slots=True)
class TextQuestion:
    """A question, and the rule that decides which chunks count as evidence."""

    id: str
    question: str
    must_match: str  # regex, case-insensitive, against the chunk text
    tickers: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    period_from: str = ""  # inclusive, compared as ISO strings
    period_to: str = ""
    note: str = ""

    def is_gold(self, c: Chunk) -> bool:
        if self.tickers and c.ticker not in self.tickers:
            return False
        if self.items and c.item_key not in self.items:
            return False
        if self.forms and c.form.upper() not in self.forms:
            return False
        if self.period_from and (c.period_end or "") < self.period_from:
            return False
        if self.period_to and (c.period_end or "") > self.period_to:
            return False
        return re.search(self.must_match, c.text, re.I) is not None

    def gold(self, chunks: Sequence[Chunk]) -> set[str]:
        return {c.chunk_id for c in chunks if self.is_gold(c)}


# --------------------------------------------------------------------------
# the set
# --------------------------------------------------------------------------
#
# Thirty questions, spread deliberately rather than sampled: every sector in
# universe.yaml appears, every narrative item that carries answers appears, and
# five questions span more than one company because a single-company question
# can be answered by a retriever that has learned nothing but ticker matching.

SMOKE_QUESTIONS: tuple[TextQuestion, ...] = (
    # --- semiconductors ---
    TextQuestion(
        id="nvda-export-controls",
        question=(
            "What licensing requirements did the U.S. government impose on NVIDIA's "
            "exports of data center GPUs to China?"
        ),
        must_match=r"export licen|licensing requirement",
        tickers=("NVDA",),
        items=_RISK + _MDNA,
    ),
    TextQuestion(
        id="nvda-hopper",
        question="Which NVIDIA GPU architecture followed Ampere in the data center business?",
        must_match=r"Hopper",
        tickers=("NVDA",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="amd-xilinx",
        question="How did the Xilinx acquisition affect AMD's operating expenses and segments?",
        must_match=r"Xilinx",
        tickers=("AMD",),
        items=_MDNA,
    ),
    TextQuestion(
        id="intc-idm",
        question="What is Intel's IDM 2.0 strategy and what does it commit the company to build?",
        must_match=r"IDM 2\.0",
        tickers=("INTC",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="avgo-vmware",
        question=("What did Broadcom say about moving VMware's customers to a subscription model?"),
        must_match=r"VMware",
        tickers=("AVGO",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="qcom-apple",
        question="What did Qualcomm disclose about its modem supply agreement with Apple?",
        must_match=r"Apple",
        tickers=("QCOM",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="qcom-huawei",
        question="How did Qualcomm describe the effect of export restrictions on Huawei?",
        must_match=r"Huawei",
        tickers=("QCOM",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="semis-cyber",
        question=(
            "How do the semiconductor companies describe their cybersecurity risk "
            "management and governance?"
        ),
        must_match=r"cybersecurity",
        tickers=("NVDA", "AMD", "INTC", "AVGO", "QCOM"),
        items=("1C", "1A", "II.1A", "FULL"),
        note="cross-company; Item 1C only exists from fiscal 2023 onward",
    ),
    # --- retail ---
    TextQuestion(
        id="wmt-opioid",
        question="What did Walmart disclose about opioid litigation and its settlement?",
        must_match=r"opioid",
        tickers=("WMT",),
        items=_LEGAL + _RISK,
    ),
    TextQuestion(
        id="wmt-ecommerce",
        question="How did Walmart explain the growth of its eCommerce business?",
        must_match=r"eCommerce|e-commerce",
        tickers=("WMT",),
        items=_MDNA,
    ),
    TextQuestion(
        id="tgt-shrink",
        question="How did Target explain the increase in inventory shrink and its margin impact?",
        must_match=r"shrink",
        tickers=("TGT",),
        items=_MDNA + _RISK,
    ),
    TextQuestion(
        id="cost-membership",
        question="What does Costco say about membership fee revenue and renewal rates?",
        must_match=r"membership fee",
        tickers=("COST",),
        items=_MDNA + _BUSINESS,
    ),
    TextQuestion(
        id="hd-pro",
        question="What is Home Depot's strategy for its professional contractor customers?",
        must_match=r"[Pp]ro customer|professional customer|Pro and DIY",
        tickers=("HD",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="low-total-home",
        question="What is Lowe's Total Home strategy?",
        must_match=r"Total Home",
        tickers=("LOW",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="retail-covid",
        question=(
            "How did the large retailers describe COVID-19 disruption to their supply "
            "chains and stores in 2020?"
        ),
        must_match=r"COVID-19",
        tickers=("WMT", "TGT", "COST", "HD", "LOW"),
        items=_MDNA + _RISK,
        period_to="2021-06-30",
        note="cross-company, time-boxed",
    ),
    # --- pharmaceuticals ---
    TextQuestion(
        id="jnj-talc",
        question=(
            "What did Johnson & Johnson disclose about talc-related personal injury litigation?"
        ),
        must_match=r"talc",
        tickers=("JNJ",),
        items=_LEGAL + _RISK,
    ),
    TextQuestion(
        id="jnj-kenvue",
        question="Why did Johnson & Johnson separate its consumer health business as Kenvue?",
        must_match=r"Kenvue",
        tickers=("JNJ",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="pfe-comirnaty",
        question="Why did Pfizer's Comirnaty revenues decline after 2022?",
        must_match=r"Comirnaty",
        tickers=("PFE",),
        items=_MDNA,
    ),
    TextQuestion(
        id="pfe-paxlovid",
        question="What did Pfizer say about Paxlovid revenue and returns of government inventory?",
        must_match=r"Paxlovid",
        tickers=("PFE",),
        items=_MDNA + _RISK,
    ),
    TextQuestion(
        id="mrk-keytruda-loe",
        question="When does Merck expect Keytruda to lose market exclusivity, and what follows?",
        must_match=r"Keytruda",
        tickers=("MRK",),
        items=_RISK + _MDNA,
    ),
    TextQuestion(
        id="abbv-humira",
        question="How did AbbVie describe the effect of U.S. Humira biosimilar competition?",
        must_match=r"biosimilar",
        tickers=("ABBV",),
        items=_MDNA + _RISK,
    ),
    TextQuestion(
        id="lly-incretin",
        question="What did Eli Lilly say about manufacturing capacity for tirzepatide?",
        must_match=r"tirzepatide|Mounjaro|Zepbound",
        tickers=("LLY",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="pharma-pricing",
        question=(
            "How do the pharmaceutical companies describe the Inflation Reduction Act's "
            "drug price negotiation as a risk?"
        ),
        must_match=r"Inflation Reduction Act",
        tickers=("JNJ", "PFE", "MRK", "ABBV", "LLY"),
        items=_RISK + _MDNA,
        note="cross-company; the IRA appears from 2022 onward",
    ),
    # --- oil and gas ---
    TextQuestion(
        id="xom-permian",
        question="What did Exxon Mobil say about production growth in the Permian Basin?",
        must_match=r"Permian",
        tickers=("XOM",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="xom-pioneer",
        question="What did Exxon Mobil say about acquiring Pioneer Natural Resources?",
        must_match=r"Pioneer",
        tickers=("XOM",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="cvx-hess",
        question="What did Chevron disclose about the arbitration affecting its Hess acquisition?",
        must_match=r"Hess",
        tickers=("CVX",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="cop-willow",
        question="What did ConocoPhillips disclose about the Willow project in Alaska?",
        must_match=r"Willow",
        tickers=("COP",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="slb-digital",
        question="How does SLB describe its Digital & Integration division?",
        must_match=r"Digital & Integration|Digital and Integration",
        tickers=("SLB",),
        items=_NARRATIVE,
    ),
    TextQuestion(
        id="energy-climate",
        question=(
            "How do the oil and gas companies describe climate-related regulation and "
            "the energy transition as risks?"
        ),
        must_match=r"climate",
        tickers=("XOM", "CVX", "COP", "SLB", "PSX"),
        items=_RISK,
        note="cross-company",
    ),
    # --- across the whole corpus ---
    TextQuestion(
        id="corpus-inflation-2022",
        question=("How did companies describe inflationary pressure on their costs during 2022?"),
        must_match=r"inflation",
        items=_MDNA,
        period_from="2022-01-01",
        period_to="2022-12-31",
        note="no ticker filter: the retriever has to find the year, not the company",
    ),
)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: TextQuestion
    gold: int
    hit_at_50: bool
    first_gold_rank: int | None
    precision_at_5_fused: float
    precision_at_5_reranked: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    results: tuple[QuestionResult, ...] = ()
    seconds: float = 0.0
    missing_gold: tuple[str, ...] = ()  # questions with no evidence in the index at all

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def recall_at_50(self) -> float:
        return _mean([r.hit_at_50 for r in self.results])

    @property
    def precision_at_5_fused(self) -> float:
        return _mean([r.precision_at_5_fused for r in self.results])

    @property
    def precision_at_5_reranked(self) -> float:
        return _mean([r.precision_at_5_reranked for r in self.results])

    @property
    def rerank_delta(self) -> float:
        return self.precision_at_5_reranked - self.precision_at_5_fused

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        return tuple(r for r in self.results if not r.hit_at_50)


def _mean(xs: Sequence[float | bool]) -> float:
    return sum(float(x) for x in xs) / len(xs) if xs else 0.0


def evaluate(retriever, questions: Sequence[TextQuestion] = SMOKE_QUESTIONS) -> EvalReport:  # noqa: ANN001
    """Run every question once and score both orders off the same candidates.

    One retrieval per question, not two: the reranked top five is a reordering
    of the fused fifty, so scoring the fused order and the reranked order over
    the same list is the controlled comparison. Running the pipeline twice would
    measure the pipeline's variance as well as the reranker's effect.
    """
    import time

    from filing.stores.retrieve import FUSED_K, rrf

    started = time.monotonic()
    chunks = list(retriever.chunks.values())
    results: list[QuestionResult] = []
    missing: list[str] = []

    for q in questions:
        gold = q.gold(chunks)
        if not gold:
            missing.append(q.id)
            continue
        runs = {
            "dense": retriever.dense_run(q.question),
            "sparse": retriever.sparse_run(q.question),
        }
        fused = [f.chunk_id for f in rrf(runs)][:FUSED_K]
        first = next((i for i, c in enumerate(fused, 1) if c in gold), None)

        hits = retriever.resolve(fused)
        rankings = retriever.backend.rerank(q.question, [c.text for c in hits], top_n=5)
        reranked = [hits[r.index].chunk_id for r in rankings[:5]]

        results.append(
            QuestionResult(
                question=q,
                gold=len(gold),
                hit_at_50=first is not None,
                first_gold_rank=first,
                precision_at_5_fused=_mean([c in gold for c in fused[:5]]),
                precision_at_5_reranked=_mean([c in gold for c in reranked]),
            )
        )

    return EvalReport(
        results=tuple(results),
        seconds=time.monotonic() - started,
        missing_gold=tuple(missing),
    )
