"""Scoring. Every number in the results table starts as one of these functions.

The tests worth having here are the ones that pin down what a metric *refuses*
to reward: a number scored right at the wrong scale, a retrieval scored as a hit
because it landed in the right document, an abstention counted for a system that
answered. Those are the ways an eval flatters the thing it measures.
"""

from __future__ import annotations

import pytest

from filing.eval import metrics
from filing.eval.dataset import EvalQuestion, Span
from filing.eval.metrics import Outcome, RetrievedChunk


def chunk(accn="A", start=0, end=100, cid=None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid or f"{accn}:{start}", accn=accn, char_start=start, char_end=end
    )


def q_narrative(spans, qid="nar-001") -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        slice="narrative",
        question="why?",
        spans=tuple(spans),
        origin="test",
        gold_source="test",
    )


ONE_SPAN = (Span("A", 0, 50),)


def q_numeric(value=1000.0, spans=ONE_SPAN, qid="num-001") -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        slice="numeric",
        question="how much?",
        value=value,
        unit="USD",
        tag="Revenues",
        spans=tuple(spans),
        origin="test",
        gold_source="test",
    )


def q_unanswerable(qid="una-001") -> EvalQuestion:
    return EvalQuestion(
        id=qid, slice="unanswerable", question="what?", origin="test", gold_source="none"
    )


# ------------------------------------------------------------ number parsing


@pytest.mark.parametrize(
    ("answer", "gold"),
    [
        ("Revenue was $16,434 million.", 16_434_000_000.0),
        ("Revenue was $16.4 billion.", 16_434_000_000.0),
        ("16434", 16_434_000_000.0),
        ("Net loss of (1,234) million", -1_234_000_000.0),
        ("The figure is 60,922.", 60_922_000_000.0),
        ("gross margin was 77.7 percent", 77.7),
    ],
)
def test_numeric_match_accepts_every_honest_spelling(answer, gold):
    assert metrics.numeric_match(answer, gold)


@pytest.mark.parametrize(
    ("answer", "gold"),
    [
        ("Revenue was $16,000 million.", 16_434_000_000.0),
        ("INSUFFICIENT EVIDENCE", 16_434_000_000.0),
        ("Revenue rose sharply.", 16_434_000_000.0),
        ("Revenue was $1,434 million.", 16_434_000_000.0),
    ],
)
def test_numeric_match_rejects_the_wrong_number(answer, gold):
    assert not metrics.numeric_match(answer, gold)


def test_tolerance_is_relative_and_tight():
    assert metrics.numeric_match("1000.4", 1000.0)
    assert not metrics.numeric_match("1006", 1000.0)


def test_parse_numbers_offers_every_scale():
    got = metrics.parse_numbers("2.4")
    assert 2.4 in got and 2.4e6 in got and 2.4e9 in got


def test_a_named_scale_binds_to_its_own_number():
    assert 2.4e9 in metrics.parse_numbers("$2.4 billion of cash")


# ---------------------------------------------------------------- retrieval


def test_hit_rate_is_position_sensitive():
    q = q_narrative([Span("A", 500, 600)])
    got = (chunk(start=0, end=100), chunk(start=100, end=200), chunk(start=500, end=600))
    assert metrics.hit_rate(q, got, 1) == 0.0
    assert metrics.hit_rate(q, got, 5) == 1.0


def test_the_right_document_is_not_a_hit():
    """The M3 failure this whole gate exists to repair."""
    q = q_narrative([Span("A", 5000, 5100)])
    assert metrics.hit_rate(q, (chunk(accn="A", start=0, end=100),), 5) == 0.0


def test_recall_counts_alternatives_not_chunks():
    q = q_narrative([Span("A", 0, 100), Span("B", 0, 100), Span("C", 0, 100)])
    got = (chunk(accn="A"), chunk(accn="B"))
    assert metrics.recall(q, got, 5) == pytest.approx(2 / 3)
    assert metrics.hit_rate(q, got, 5) == 1.0


def test_two_chunks_over_one_span_do_not_double_count():
    q = q_narrative([Span("A", 0, 200)])
    got = (chunk(accn="A", start=0, end=100), chunk(accn="A", start=100, end=200))
    assert metrics.recall(q, got, 5) == 1.0


def test_ndcg_prefers_the_higher_rank():
    q = q_narrative([Span("A", 0, 100)])
    first = (chunk(accn="A"), chunk(accn="B"), chunk(accn="C"))
    third = (chunk(accn="B"), chunk(accn="C"), chunk(accn="A"))
    assert metrics.ndcg(q, first, 5) == 1.0
    assert 0.0 < metrics.ndcg(q, third, 5) < 1.0


def test_ndcg_ideal_is_capped_by_k():
    q = q_narrative([Span(a, 0, 100) for a in "ABCDEFG"])
    got = tuple(chunk(accn=a) for a in "AB")
    assert metrics.ndcg(q, got, 2) == 1.0


def test_no_gold_scores_zero_rather_than_dividing_by_nothing():
    assert metrics.ndcg(q_unanswerable(), (chunk(),), 5) == 0.0
    assert metrics.recall(q_unanswerable(), (chunk(),), 5) == 0.0


# --------------------------------------------------------------- scorecard


def test_score_separates_the_slices():
    questions = [q_numeric(), q_narrative([Span("A", 0, 50)]), q_unanswerable()]
    outcomes = [
        Outcome(qid="num-001", route="text", answer="$1,000", retrieved=(chunk(end=50),)),
        Outcome(qid="nar-001", route="text", answer="because", retrieved=(chunk(end=50),)),
        Outcome(qid="una-001", route="refuse", answer="INSUFFICIENT EVIDENCE", refused=True),
    ]
    card = metrics.score(questions, outcomes)
    assert card.slices["numeric"].exact_match == 1.0
    assert card.slices["numeric"].router_accuracy == 0.0  # answered from text, not SQL
    assert card.slices["narrative"].router_accuracy == 1.0
    assert card.slices["unanswerable"].abstention == 1.0
    assert card.slices["unanswerable"].over_answered == 0.0
    assert card.overall.n == 3


def test_a_system_that_never_abstains_is_visible():
    card = metrics.score(
        [q_unanswerable()], [Outcome(qid="una-001", route="text", answer="$5 billion")]
    )
    assert card.slices["unanswerable"].abstention == 0.0
    assert card.slices["unanswerable"].over_answered == 1.0


def test_a_refusal_never_counts_as_a_correct_number():
    card = metrics.score(
        [q_numeric()],
        [Outcome(qid="num-001", answer="INSUFFICIENT EVIDENCE (1000)", refused=True)],
    )
    assert card.slices["numeric"].exact_match == 0.0


def test_an_unattempted_question_is_scored_as_an_error_not_skipped():
    card = metrics.score([q_numeric()], [])
    assert card.slices["numeric"].n == 1
    assert card.slices["numeric"].errors == 1
    assert card.slices["numeric"].exact_match == 0.0


def test_citations_separate_resolvable_from_supported():
    q = q_narrative([Span("A", 0, 50)])
    good, bad = chunk(accn="A", end=50, cid="good"), chunk(accn="B", cid="bad")
    card = metrics.score(
        [q],
        [Outcome(qid="nar-001", route="text", retrieved=(good, bad), citations=("good", "bad"))],
    )
    s = card.slices["narrative"]
    assert s.citations_made == 2
    assert s.citations_resolvable == 1.0
    assert s.citations_supported == 0.5


def test_a_cited_id_that_was_never_retrieved_is_unresolvable():
    card = metrics.score(
        [q_narrative([Span("A", 0, 50)])],
        [Outcome(qid="nar-001", citations=("invented",))],
    )
    assert card.slices["narrative"].citations_resolvable == 0.0


def test_known_chunks_rescues_a_real_but_unretrieved_id():
    card = metrics.score(
        [q_narrative([Span("A", 0, 50)])],
        [Outcome(qid="nar-001", citations=("elsewhere",))],
        known_chunks={"elsewhere"},
    )
    assert card.slices["narrative"].citations_resolvable == 1.0
    assert card.slices["narrative"].citations_supported == 0.0


# --------------------------------------------------------------- rendering


def test_markdown_has_a_row_per_slice_plus_overall():
    card = metrics.score([q_numeric(), q_unanswerable()], [])
    table = metrics.to_markdown(card, title="baseline")
    assert "baseline" in table
    for name in ("numeric", "narrative", "unanswerable", "overall"):
        assert f"| {name} |" in table
    assert table.count("\n") == 8  # title + blank + header + rule + 3 slices + overall


def test_markdown_prints_missing_metrics_as_dashes():
    card = metrics.score([q_narrative([Span("A", 0, 1)])], [])
    assert "--" in metrics.to_markdown(card)


# --------------------------------------------------------------- round trip


def test_outcome_round_trips_through_json():
    o = Outcome(
        qid="num-001",
        route="sql",
        answer="a",
        retrieved=(chunk(),),
        citations=("x",),
        llm_calls=1,
    )
    assert Outcome.from_json(o.to_json()) == o


# ------------------------------------------- the metric's edge, held honestly


def test_a_question_answered_from_duckdb_is_excluded_not_scored_zero():
    """The tempting false zero.

    A supported citation is one whose chunk overlaps a gold character span. An
    XBRL fact has no span -- the value in DuckDB and the number printed in the
    filing are one fact reached two ways, and only one way carries offsets. So
    a question the agent answered from `sql` retrieved no text chunks, and
    scoring that as hit@5 = 0 would report a retrieval failure where there was
    no retrieval attempt. It reads, in a results table, as "the retriever
    missed" next to an exact_match of 97.5%.
    """
    card = metrics.score([q_numeric()], [Outcome(qid="num-001", route="sql", answer="$1,000")])
    num = card.slices["numeric"]
    assert num.retrieval_n == 0
    assert num.hit_rate == {}  # unmeasured, and the table renders it "--"
    assert num.exact_match == 1.0  # the claim that does survive


def test_a_text_route_that_retrieved_nothing_still_scores_zero():
    """The other half, or the exclusion would be an excuse.

    Only a route to a spanless store is outside the metric. A question that
    went to the text retriever and came back empty-handed failed at exactly the
    thing hit@k measures.
    """
    card = metrics.score([q_numeric()], [Outcome(qid="num-001", route="text", answer="$1,000")])
    num = card.slices["numeric"]
    assert num.retrieval_n == 1
    assert num.hit_rate[5] == 0.0


def test_the_retrieval_denominator_travels_with_the_numbers():
    """A hit@5 over a subset the system selected for itself needs its `n` in
    view: the questions that reached the text retriever are precisely the ones
    the SQL branch could not answer, so the subset is biased by construction."""
    qs = [q_numeric(qid="num-001"), q_numeric(qid="num-002")]
    outs = [
        Outcome(qid="num-001", route="sql", answer="$1,000"),
        Outcome(qid="num-002", route="text", answer="$1,000", retrieved=(chunk(end=50),)),
    ]
    card = metrics.score(qs, outs)
    assert card.slices["numeric"].n == 2
    assert card.slices["numeric"].retrieval_n == 1
    assert "ret n" in metrics.to_markdown(card)


def test_the_baseline_is_untouched_by_the_exclusion():
    """It has no route to a spanless store, so its published numbers cannot
    move -- which is what makes the two systems still comparable."""
    outs = [Outcome(qid="num-001", route="text", answer="$1,000", retrieved=(chunk(end=50),))]
    card = metrics.score([q_numeric()], outs)
    assert card.slices["numeric"].retrieval_n == 1
    assert card.slices["numeric"].hit_rate[5] == 1.0


# ------------------------------------------------- the M6 verification columns


def _verdict(ok=True, numbers=(), markers=(), dangling=()):
    """A verdict as it arrives on an outcome: JSON, not the dataclass.

    The results file is what the scorer reads, and by then the verdict has been
    through `to_json`. Building the dict directly is the honest fixture.
    """
    return {
        "ok": ok,
        "numbers": list(numbers),
        "markers": list(markers),
        "dangling": list(dangling),
        "unlocatable": [],
        "uncited": [],
        "unrechecked": [],
        "taxonomy": [],
        "reasons": [],
    }


def _num(status, value=1.0):
    return {"text": str(value), "value": value, "status": status, "how": "", "source": ""}


def test_a_run_without_verification_reports_no_verification_columns():
    """Not a clean zero. A check that never ran found nothing because it never ran."""
    card = metrics.score([q_numeric()], [Outcome(qid="num-001", route="sql", answer="$1,000")])
    row = card.slices["numeric"]
    assert row.verified_n == 0
    assert row.hallucinated is None and row.flag_rate is None
    assert "hallucinated" not in metrics.to_markdown(card)


def test_the_hallucination_rate_is_over_figures_not_over_answers():
    """Two answers, four figures, one of them invented -- 25%, not 50%."""
    outs = [
        Outcome(
            qid="num-001",
            route="sql",
            answer="$1,000",
            verdict=_verdict(numbers=[_num("supported"), _num("derived")]),
        ),
        Outcome(
            qid="num-002",
            route="sql",
            answer="$1,000",
            verdict=_verdict(ok=False, numbers=[_num("supported"), _num("unsupported")]),
        ),
    ]
    card = metrics.score([q_numeric(), q_numeric(qid="num-002")], outs)
    row = card.slices["numeric"]
    assert row.verified_n == 2
    assert row.figures_checked == 4
    assert row.hallucinated == 0.25
    assert row.flag_rate == 0.5


def test_a_year_is_not_in_the_hallucination_denominator():
    """`context` figures are the question's own numbers echoed back, and a
    denominator that counted them would flatter every rate computed from it."""
    outs = [
        Outcome(
            qid="num-001",
            route="sql",
            answer="$1,000",
            verdict=_verdict(numbers=[_num("context"), _num("supported")]),
        )
    ]
    card = metrics.score([q_numeric()], outs)
    assert card.slices["numeric"].figures_checked == 1


def test_the_locator_rate_counts_markers_not_answers():
    outs = [
        Outcome(
            qid="num-001",
            route="sql",
            answer="a [1][2][9]",
            verdict=_verdict(ok=False, markers=[1, 2], dangling=[9]),
        )
    ]
    card = metrics.score([q_numeric()], outs)
    assert card.slices["numeric"].locator_rate == pytest.approx(2 / 3)


def test_flagged_and_blocked_are_reported_apart():
    """The bail-out clause needs both numbers: how many the guard would have
    stopped, and how many it did."""
    outs = [
        Outcome(qid="num-001", route="sql", answer="a", verdict=_verdict(ok=False), blocked=True),
        Outcome(qid="num-002", route="sql", answer="a", verdict=_verdict(ok=False)),
    ]
    card = metrics.score([q_numeric(), q_numeric(qid="num-002")], outs)
    row = card.slices["numeric"]
    assert row.flag_rate == 1.0
    assert row.block_rate == 0.5
    table = metrics.to_markdown(card)
    assert "hallucinated" in table and "blocked" in table


def test_an_outcome_with_a_verdict_round_trips():
    o = Outcome(qid="num-001", answer="a", verdict=_verdict(), blocked=True)
    assert Outcome.from_json(o.to_json()) == o


def test_a_pre_m6_outcome_keeps_its_exact_shape():
    """Results files written before the verifier existed must not gain keys."""
    d = Outcome(qid="num-001", answer="a").to_json()
    assert "verdict" not in d and "blocked" not in d
