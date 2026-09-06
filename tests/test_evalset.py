"""The evaluation's own correctness -- the part nobody else checks.

An eval is the one piece of code with no oracle above it. If ``is_gold`` is
wrong the number it produces is still a number, still plausible, and still
reported to three decimal places. So the predicate gets tested clause by clause,
and the question set gets the invariants that keep a typo from quietly scoring
zero forever: an item key nobody indexes, a regex that never compiles, a ticker
that is not in the universe.

The metrics are arithmetic over a list of results, so they are tested against
hand-built results rather than a real sweep. What the real sweep proves is a
gate, not a unit test.
"""

from __future__ import annotations

import re

import pytest

from conftest import make_chunk
from filing.stores.evalset import (
    SMOKE_QUESTIONS,
    EvalReport,
    QuestionResult,
    TextQuestion,
)

# --------------------------------------------------------------------------
# the predicate
# --------------------------------------------------------------------------


def question(**kw) -> TextQuestion:
    fields = dict(id="q", question="?", must_match=r"Hopper")
    fields.update(kw)
    return TextQuestion(**fields)


def test_the_text_has_to_match_and_the_match_is_case_insensitive():
    q = question()
    assert q.is_gold(make_chunk("The hopper architecture shipped in 2023."))
    assert not q.is_gold(make_chunk("The Ampere architecture shipped in 2020."))


def test_a_ticker_clause_excludes_the_other_nineteen_companies():
    q = question(tickers=("NVDA",))
    assert q.is_gold(make_chunk("Hopper", ticker="NVDA"))
    assert not q.is_gold(make_chunk("Hopper", ticker="AMD"))


def test_an_item_clause_is_matched_on_the_part_qualified_key():
    """A 10-Q's Part II Item 1A is not the same section as Part I's."""
    q = question(items=("II.1A",))
    assert q.is_gold(make_chunk("Hopper", form="10-Q", part="II", item="1A"))
    assert not q.is_gold(make_chunk("Hopper", form="10-Q", part="I", item="1A"))


def test_a_form_clause_ignores_the_case_the_filing_used():
    q = question(forms=("10-K",))
    assert q.is_gold(make_chunk("Hopper", form="10-k"))
    assert not q.is_gold(make_chunk("Hopper", form="10-Q"))


def test_a_period_window_is_inclusive_at_both_ends():
    q = question(period_from="2023-01-01", period_to="2023-12-31")
    assert q.is_gold(make_chunk("Hopper", period_end="2023-01-01"))
    assert q.is_gold(make_chunk("Hopper", period_end="2023-12-31"))
    assert not q.is_gold(make_chunk("Hopper", period_end="2022-12-31"))
    assert not q.is_gold(make_chunk("Hopper", period_end="2024-01-01"))


def test_a_chunk_with_no_period_is_outside_every_window():
    """Comparing "" as an ISO string is deliberate: unknown is not "close enough"."""
    assert not question(period_from="2020-01-01").is_gold(make_chunk("Hopper", period_end=""))


def test_every_clause_has_to_hold_at_once():
    q = question(tickers=("NVDA",), items=("1A",), forms=("10-K",))
    assert q.is_gold(make_chunk("Hopper"))
    assert not q.is_gold(make_chunk("Hopper", item="7"))


def test_gold_is_the_set_of_ids_and_not_a_count():
    """The metric needs ids: recall asks whether *these* chunks came back."""
    q = question(tickers=("NVDA",))
    hit = make_chunk("Hopper is here", ticker="NVDA")
    miss = make_chunk("Hopper is here", ticker="AMD")
    assert q.gold([hit, miss]) == {hit.chunk_id}


# --------------------------------------------------------------------------
# the question set
# --------------------------------------------------------------------------


def test_the_set_is_the_thirty_questions_the_gate_names():
    assert len(SMOKE_QUESTIONS) == 30


def test_question_ids_are_unique_so_a_result_can_be_traced_back():
    assert len({q.id for q in SMOKE_QUESTIONS}) == len(SMOKE_QUESTIONS)


@pytest.mark.parametrize("q", SMOKE_QUESTIONS, ids=lambda q: q.id)
def test_every_predicate_is_a_regex_that_compiles(q):
    assert re.compile(q.must_match, re.I)


@pytest.mark.parametrize("q", SMOKE_QUESTIONS, ids=lambda q: q.id)
def test_every_question_names_an_item_the_index_actually_holds(q):
    """An item key nobody indexes scores zero forever and looks like a retrieval bug."""
    from filing.stores.index import NARRATIVE_ITEMS

    indexed = set().union(*NARRATIVE_ITEMS.values())
    for item in q.items:
        assert item.split(".")[-1] in indexed, f"{q.id} wants {item}"


def test_every_ticker_a_question_names_is_in_the_universe():
    from filing.config import settings
    from filing.ingest.universe import load_universe

    known = {c.ticker for c in load_universe(cfg=settings()).companies}
    named = {t for q in SMOKE_QUESTIONS for t in q.tickers}
    assert named <= known
    # And the reverse, because a company with no question is a company the
    # smoke set never looks at.
    assert named == known


def test_five_questions_reach_past_a_single_company():
    """A single-company question can be answered by a retriever that only matches tickers."""
    spanning = [q for q in SMOKE_QUESTIONS if len(q.tickers) != 1]
    assert len(spanning) >= 5


# --------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------


def result(*, hit=True, fused=0.0, reranked=0.0) -> QuestionResult:
    return QuestionResult(
        question=question(),
        gold=3,
        hit_at_50=hit,
        first_gold_rank=1 if hit else None,
        precision_at_5_fused=fused,
        precision_at_5_reranked=reranked,
    )


def test_recall_is_the_fraction_of_questions_with_any_gold_in_the_window():
    report = EvalReport(results=(result(hit=True), result(hit=True), result(hit=False)))
    assert report.recall_at_50 == pytest.approx(2 / 3)
    assert report.n == 3


def test_the_rerank_delta_is_the_only_evidence_the_reranker_earns_its_latency():
    report = EvalReport(results=(result(fused=0.2, reranked=0.6), result(fused=0.4, reranked=0.4)))
    assert report.precision_at_5_fused == pytest.approx(0.3)
    assert report.precision_at_5_reranked == pytest.approx(0.5)
    assert report.rerank_delta == pytest.approx(0.2)


def test_a_negative_delta_is_reported_and_not_clamped():
    """Reranking is allowed to fail the gate. Hiding that would be the point of failure."""
    report = EvalReport(results=(result(fused=0.6, reranked=0.2),))
    assert report.rerank_delta == pytest.approx(-0.4)


def test_the_misses_are_kept_so_a_failure_names_its_questions():
    report = EvalReport(results=(result(hit=True), result(hit=False)))
    assert len(report.misses) == 1
    assert report.misses[0].hit_at_50 is False


def test_an_empty_report_scores_zero_rather_than_dividing_by_zero():
    empty = EvalReport()
    assert (empty.n, empty.recall_at_50, empty.precision_at_5_fused) == (0, 0.0, 0.0)


def test_questions_with_no_evidence_are_named_rather_than_scored_as_misses():
    """A question whose gold set is empty says the corpus is wrong, not that retrieval is."""
    report = EvalReport(results=(result(hit=True),), missing_gold=("amd-xilinx",))
    assert report.recall_at_50 == 1.0
    assert report.missing_gold == ("amd-xilinx",)
