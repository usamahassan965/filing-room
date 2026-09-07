"""The verifier, tested against the ways it can be wrong in both directions.

A guard has two failure modes and only one of them is loud. Letting a
hallucinated figure through is the one everybody tests for. Blocking a correct
answer is the one that gets the guard switched off a week later, so roughly half
of what follows is the *negative* case: prose that is fine and must pass.
"""

from __future__ import annotations

import pytest

from filing.agent.state import Evidence, Grade
from filing.agent.verify import (
    GUARD_MODES,
    GuardReport,
    Verdict,
    classify,
    guard_answer,
    has_locator,
    resolve_citations,
    strip_citations,
    uncited_claims,
    verify_answer,
    verify_numbers,
)
from filing.eval.runner import REFUSAL

ACCN = "0001045810-24-000029"


def fact(
    value: float,
    *,
    tag: str = "Revenues",
    period: str = "2024-01-28",
    ticker: str = "NVDA",
    body: str = "",
) -> Evidence:
    return Evidence(
        kind="fact",
        body=body or f"{ticker} reported {tag} of {value:,.0f} USD for the period ending {period}.",
        citation=f"{ticker} 10-K {period} {tag} [{ACCN}]",
        accn=ACCN,
        value=value,
        unit="USD",
        tag=tag,
        period_end=period,
        ticker=ticker,
    )


def chunk(body: str, *, start: int = 100, end: int = 900) -> Evidence:
    return Evidence(
        kind="text",
        body=body,
        citation=f"NVDA 10-K [{ACCN}] {start}-{end}",
        accn=ACCN,
        score=0.8,
        chunk_id=f"{ACCN}:{start}",
        char_start=start,
        char_end=end,
        ticker="NVDA",
    )


# --------------------------------------------------------------------------
# reading numbers out of prose
# --------------------------------------------------------------------------


def test_citation_markers_are_not_read_as_numbers():
    """The bug this check exists to prevent: `[1]` scored as the figure one,
    giving every properly cited answer a hallucinated number."""
    ev = [chunk("Revenue rose to 60,922 million.")]
    claims = verify_numbers("Revenue was 60,922 [1].", ev)
    assert [c.text for c in claims] == ["60,922"]
    assert claims[0].status == "supported"


def test_strip_citations_preserves_length():
    text = "a [1] b 【2】 c"
    assert len(strip_citations(text)) == len(text)
    assert "1" not in strip_citations(text)


def test_fullwidth_markers_are_stripped_too():
    """M4.5's finding: gpt-oss-120b cites with U+3010/U+3011."""
    ev = [chunk("Revenue rose to 60,922 million.")]
    assert verify_numbers("Revenue was 60,922 【1】.", ev)[0].status == "supported"


# --------------------------------------------------------------------------
# the two tiers of support
# --------------------------------------------------------------------------


def test_digits_tier_is_recorded_separately_from_the_scaled_tier():
    """The distinction is the point: one match reproduced the printed digits,
    the other needed rescaling and rounding to land. A report that prints one
    number for both is hiding the weaker half."""
    ev = [chunk("Revenue for fiscal 2024 was 60,922 million dollars.")]
    exact = verify_numbers("Revenue was 60,922 [1].", ev)[0]
    rounded = verify_numbers("Revenue was about $60.9 billion [1].", ev)[0]
    assert (exact.status, exact.how) == ("supported", "digits")
    assert (rounded.status, rounded.how) == ("supported", "scaled")


def test_scaled_match_accepts_the_unit_the_filing_actually_used():
    """A fact stored in dollars, quoted in billions. Rejecting this would make
    the guard fire on nearly every well-formed answer."""
    claim = verify_numbers("Revenue was $60.9 billion [1].", [fact(60_922_000_000)])[0]
    assert claim.status == "supported"


def test_a_wrong_figure_is_unsupported():
    claim = verify_numbers("Revenue was $69.0 billion [1].", [fact(60_922_000_000)])[0]
    assert claim.status == "unsupported"
    assert claim.how == ""
    assert claim.source == ""


def test_the_supporting_evidence_is_named():
    """Not just that it matched -- which citation carried it, so a reviewer can
    follow the verifier's own reasoning back to a filing."""
    claim = verify_numbers("Revenue was $60.9 billion [1].", [fact(60_922_000_000)])[0]
    assert ACCN in claim.source


# --------------------------------------------------------------------------
# what is not a claim
# --------------------------------------------------------------------------


def test_a_fiscal_year_is_context_not_a_figure():
    claims = verify_numbers("Revenue was $60.9 billion in fiscal 2024 [1].", [fact(60_922_000_000)])
    year = next(c for c in claims if c.text == "2024")
    assert (year.status, year.how) == ("context", "period")


def test_a_year_only_in_the_question_is_still_context():
    ev = [chunk("Revenue grew substantially.")]
    year = verify_numbers("In 2023 revenue grew [1].", ev, question="What happened in 2023?")[0]
    assert year.status == "context"


def test_a_money_figure_shaped_like_a_year_is_not_waved_through():
    """The exemption is narrow on purpose. `$2,024 million` is a figure that
    happens to fall in the year range, and it must still be checked."""
    claim = verify_numbers("Revenue was $2,024 million [1].", [fact(60_922_000_000)])[0]
    assert claim.status == "unsupported"


def test_a_scaled_number_in_the_year_range_is_not_waved_through():
    claim = verify_numbers("Revenue was 1999 million [1].", [fact(60_922_000_000)])[0]
    assert claim.status == "unsupported"


def test_a_year_nobody_mentioned_is_not_exempt():
    """A comparison year the answer invented is a claim, not context."""
    claim = verify_numbers("Revenue fell from 2019 levels [1].", [fact(60_922_000_000)])[0]
    assert claim.status == "unsupported"


def test_a_number_echoed_from_the_question_is_context():
    ev = [chunk("The company operates worldwide.")]
    claim = verify_numbers(
        "The 500 stores are spread worldwide [1].",
        ev,
        question="Where are the 500 stores?",
    )[0]
    assert (claim.status, claim.how) == ("context", "question")


def test_context_figures_stay_out_of_the_denominator():
    """The hallucination rate is over figures the system is claiming. Padding it
    with years the question supplied would make the rate look better without
    the system doing anything."""
    v = verify_answer(
        "Revenue was $60.9 billion in fiscal 2024 [1].",
        [fact(60_922_000_000)],
        question="Revenue in fiscal 2024?",
    )
    assert len(v.numbers) == 2
    assert len(v.checked) == 1


# --------------------------------------------------------------------------
# recomputation
# --------------------------------------------------------------------------


def test_a_difference_of_two_evidence_values_is_recomputed():
    ev = [fact(60_922_000_000), fact(26_974_000_000, period="2023-01-29")]
    claim = next(
        c
        for c in verify_numbers("Revenue grew by $33,948 million [1][2].", ev)
        if c.text.startswith("$33,948")
    )
    assert (claim.status, claim.how) == ("derived", "recomputed")
    assert "difference" in claim.source


def test_a_growth_rate_is_recomputed():
    ev = [fact(60_922_000_000), fact(26_974_000_000, period="2023-01-29")]
    claim = next(
        c for c in verify_numbers("Revenue grew 125.9% [1][2].", ev) if c.text.startswith("125")
    )
    assert claim.status == "derived"


def test_derived_is_not_counted_as_supported():
    """Two verified inputs and an inferred operation is weaker evidence than a
    figure printed in the filing, and the report says so rather than folding
    the two together."""
    ev = [fact(60_922_000_000), fact(26_974_000_000, period="2023-01-29")]
    report = GuardReport()
    report.add(verify_answer("Revenue grew by $33,948 million [1][2].", ev))
    assert report.derived == 1
    assert report.supported == 0
    assert report.hallucination_rate == 0.0


def test_arithmetic_that_does_not_check_out_stays_unsupported():
    """The recomputation must not become a search that keeps trying operations
    until one fits."""
    ev = [fact(60_922_000_000), fact(26_974_000_000, period="2023-01-29")]
    claim = next(
        c
        for c in verify_numbers("Revenue grew by $41,000 million [1][2].", ev)
        if c.text.startswith("$41,000")
    )
    assert claim.status == "unsupported"


# --------------------------------------------------------------------------
# the citation resolver
# --------------------------------------------------------------------------


def test_a_fact_locates_by_tag_and_period():
    assert has_locator(fact(1.0))


def test_a_chunk_locates_by_character_span():
    assert has_locator(chunk("text"))


def test_an_accession_alone_is_not_a_locator():
    """ "It is somewhere in this 300-page 10-K" is the citation this project
    exists to not accept."""
    assert not has_locator(Evidence(kind="text", body="x", citation="c", accn=ACCN))


def test_no_accession_is_not_a_locator():
    assert not has_locator(Evidence(kind="text", body="x", citation="c", accn=""))


def test_a_marker_past_the_evidence_list_is_dangling():
    made, dangling, unlocatable = resolve_citations("a [1] b [4]", [fact(1.0)])
    assert made == (1,) and dangling == (4,) and unlocatable == ()


def test_a_marker_onto_unlocatable_evidence_is_worse_than_dangling():
    """It looks like a citation right up until someone tries to follow it."""
    ev = [Evidence(kind="text", body="x", citation="c", accn=ACCN)]
    made, dangling, unlocatable = resolve_citations("a [1]", ev)
    assert made == () and unlocatable == (1,)


def test_markers_are_deduplicated():
    made, _, _ = resolve_citations("a [1] b [1] c [1]", [fact(1.0)])
    assert made == (1,)


def test_zero_is_out_of_range():
    _, dangling, _ = resolve_citations("a [0]", [fact(1.0)])
    assert dangling == (0,)


# --------------------------------------------------------------------------
# uncited claims -- and the prose that must not trip it
# --------------------------------------------------------------------------


def test_a_figure_with_no_marker_anywhere_is_an_uncited_claim():
    assert uncited_claims("Revenue was $60.9 billion.") == ("Revenue was $60.9 billion.",)


def test_a_decimal_does_not_split_the_sentence():
    """The splitter's first bug: cutting `$60.9 billion` in half left a fragment
    carrying a figure and no marker, so the check fired on good answers."""
    assert uncited_claims("Revenue was $60.9 billion [1].") == ()


def test_a_sentence_with_no_figure_needs_no_citation():
    """Requiring a marker on every sentence flags prose that makes no checkable
    claim, and a guard that fires on good writing gets switched off."""
    assert uncited_claims("Revenue rose [1]. This was driven by data centre demand.") == ()


def test_the_uncited_sentence_is_named_not_just_counted():
    got = uncited_claims("Revenue rose [1]. Margins hit 78.4%.")
    assert got == ("Margins hit 78.4%.",)


def test_bullet_lines_are_separate_claims():
    answer = "Findings:\n- Revenue was $60.9 billion [1]\n- Margin was 72.7%"
    assert uncited_claims(answer) == ("- Margin was 72.7%",)


# --------------------------------------------------------------------------
# the whole verdict
# --------------------------------------------------------------------------


def test_a_clean_answer_verifies():
    v = verify_answer(
        "NVIDIA reported revenue of $60.9 billion for fiscal 2024 [1].",
        [fact(60_922_000_000)],
        question="What was NVDA revenue in fiscal 2024?",
        route="sql",
        grade=Grade(ok=True, reason="fact found"),
    )
    assert v.ok
    assert v.reasons == ()
    assert v.taxonomy == ()


def test_a_refusal_verifies_trivially():
    """An abstention makes no claim, so there is nothing in it to be
    unsupported. The opposite convention would make the guard's own output look
    like the thing it guards against."""
    v = verify_answer(REFUSAL, [], route="refuse", grade=Grade(ok=True, reason="declined"))
    assert v.ok


def test_the_reasons_name_the_offending_figures():
    v = verify_answer("Revenue was $69.0 billion [1].", [fact(60_922_000_000)])
    assert "$69.0" in v.reasons[0]


def test_verdict_round_trips_through_json():
    v = verify_answer("Revenue was $69.0 billion [1] and [7].", [fact(60_922_000_000)])
    assert Verdict.from_json(v.to_json()) == v


def test_json_carries_the_verdict_itself():
    """`ok` is a property, so it has to be written out explicitly or a reader of
    the results file has to recompute it."""
    assert verify_answer("Revenue was $69 billion [1].", [fact(1.0)]).to_json()["ok"] is False


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def test_block_replaces_a_failing_answer_with_the_refusal_token():
    v = verify_answer("Revenue was $69.0 billion [1].", [fact(60_922_000_000)])
    answer, blocked = guard_answer("Revenue was $69.0 billion [1].", v, mode="block")
    assert blocked and answer == REFUSAL


def test_block_leaves_a_passing_answer_alone():
    text = "Revenue was $60.9 billion [1]."
    v = verify_answer(text, [fact(60_922_000_000)])
    assert guard_answer(text, v, mode="block") == (text, False)


def test_flag_computes_the_same_verdict_and_ships_anyway():
    """The bail-out clause, written down in advance so switching to it is a
    reported decision rather than a quiet loosening of the check."""
    text = "Revenue was $69.0 billion [1]."
    v = verify_answer(text, [fact(60_922_000_000)])
    assert not v.ok
    assert guard_answer(text, v, mode="flag") == (text, False)


def test_off_ships_everything():
    text = "Revenue was $69.0 billion [1]."
    v = verify_answer(text, [fact(60_922_000_000)])
    assert guard_answer(text, v, mode="off") == (text, False)


def test_an_unknown_mode_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="unknown guard mode"):
        guard_answer("x", Verdict(), mode="lenient")


def test_the_three_modes_are_the_documented_ones():
    assert GUARD_MODES == ("off", "flag", "block")


# --------------------------------------------------------------------------
# the failure taxonomy
# --------------------------------------------------------------------------


def test_no_evidence_is_a_retrieval_miss():
    assert "retrieval_miss" in classify(
        route="text",
        grade=Grade(ok=False, reason="no evidence", missing="anything"),
        evidence=[],
        numbers=(),
        dangling=(),
        repairs=0,
        repair_log=[],
    )


def test_a_refusal_route_is_not_a_retrieval_miss():
    """The router declining is the system working, not the store failing."""
    assert (
        classify(
            route="refuse",
            grade=Grade(ok=True, reason="declined"),
            evidence=[],
            numbers=(),
            dangling=(),
            repairs=0,
            repair_log=[],
        )
        == ()
    )


def test_a_missing_grade_is_not_a_miss():
    """No grade means the graph never formed an opinion, which is different from
    the grader having a bad one."""
    assert (
        classify(
            route="text",
            grade=None,
            evidence=[chunk("x")],
            numbers=(),
            dangling=(),
            repairs=0,
            repair_log=[],
        )
        == ()
    )


def test_a_fallback_repair_tags_the_router():
    """Named for what is observable. The agent has no gold labels at answer
    time, so it cannot know the router was wrong -- only that its store was
    abandoned by a repair."""
    tags = classify(
        route="text",
        grade=Grade(ok=True, reason="ok"),
        evidence=[chunk("x")],
        numbers=(),
        dangling=(),
        repairs=1,
        repair_log=["repair 1: sql found nothing; fall back to the text retriever"],
    )
    assert "router_wrong" in tags


def test_a_repair_that_kept_the_store_does_not_tag_the_router():
    tags = classify(
        route="sql",
        grade=Grade(ok=True, reason="ok"),
        evidence=[fact(1.0)],
        numbers=(),
        dangling=(),
        repairs=1,
        repair_log=["repair 1: drop the period and take the most recent filing"],
    )
    assert "router_wrong" not in tags


def test_an_unsupported_figure_is_synthesis_drift():
    numbers = verify_numbers("Revenue was $69 billion [1].", [fact(60_922_000_000)])
    tags = classify(
        route="sql",
        grade=Grade(ok=True, reason="ok"),
        evidence=[fact(60_922_000_000)],
        numbers=numbers,
        dangling=(),
        repairs=0,
        repair_log=[],
    )
    assert "synthesis_drift" in tags


def test_drift_past_a_passing_grade_is_also_a_grader_false_positive():
    """Both tags, not one. A question that missed, got moved, and then made a
    figure up has committed several of these, and reporting only the first
    would lose the two that matter more."""
    numbers = verify_numbers("Revenue was $69 billion [1].", [fact(60_922_000_000)])
    tags = classify(
        route="sql",
        grade=Grade(ok=True, reason="ok"),
        evidence=[fact(60_922_000_000)],
        numbers=numbers,
        dangling=(),
        repairs=0,
        repair_log=[],
    )
    assert set(tags) == {"synthesis_drift", "grader_false_positive"}


def test_every_tag_is_documented():
    from filing.agent.verify import TAXONOMY

    assert set(TAXONOMY) == {
        "retrieval_miss",
        "router_wrong",
        "grader_false_positive",
        "synthesis_drift",
    }


# --------------------------------------------------------------------------
# the aggregate
# --------------------------------------------------------------------------


def test_report_counts_the_rates_the_gate_asks_for():
    ev = [fact(60_922_000_000)]
    report = GuardReport()
    report.add(verify_answer("Revenue was $60.9 billion [1].", ev))
    report.add(verify_answer("Revenue was $69.0 billion [1].", ev), blocked=True)
    assert report.answers == 2
    assert report.verified == 1
    assert report.blocked == 1
    assert report.figures == 2
    assert report.unsupported == 1
    assert report.hallucination_rate == 0.5
    assert report.locator_rate == 1.0


def test_locator_rate_counts_the_bad_markers_in_its_denominator():
    report = GuardReport()
    report.add(verify_answer("Revenue was $60.9 billion [1] and [7].", [fact(60_922_000_000)]))
    assert report.locator_rate == 0.5


def test_report_tallies_the_taxonomy():
    ev = [fact(60_922_000_000)]
    report = GuardReport()
    for _ in range(3):
        report.add(
            verify_answer("Revenue was $69 billion [1].", ev, grade=Grade(ok=True, reason="ok"))
        )
    assert report.tags["synthesis_drift"] == 3


def test_empty_report_does_not_divide_by_zero():
    report = GuardReport()
    assert report.hallucination_rate == 0.0
    assert report.locator_rate == 0.0


# --- what the negative control found (M6) ---
#
# Every case below is a bug the unit tests did not catch and a measurement did:
# 427 mutations of the agent's 116 real answers, run through the verifier to
# see what it would let past. The first four were false *positives* -- the
# verifier failing good prose -- and between them they produced all six flags
# in the first M6 run. The last is the false negative that survived them.


def test_a_marker_naming_several_pieces_of_evidence_is_one_citation_not_three():
    ev = [fact(1.0), fact(2.0), fact(3.0), fact(4.0), fact(5.0)]
    made, dangling, unlocatable = resolve_citations("Lilly hedges the euro [2, 3, 5].", ev)
    assert (made, dangling, unlocatable) == ((2, 3, 5), (), ())


def test_the_digits_of_a_multi_marker_are_not_read_as_figures():
    # `[2, 3, 5]` survived the strip, so its digits were reported as three
    # hallucinated numbers and the sentence looked uncited at the same time.
    verdict = verify_answer("Lilly hedges the euro [2, 3, 5].", [fact(1.0)] * 5)
    assert verdict.unsupported == ()
    assert verdict.uncited == ()
    assert verdict.ok


def test_the_date_a_period_ends_is_not_a_pair_of_figures():
    ev = [fact(7_392_000_000, period="2022-08-28", ticker="COST")]
    verdict = verify_answer(
        "Costco's operations provided 7,392,000,000 USD for the period ending 2022-08-28 [1].", ev
    )
    assert verdict.unsupported == ()
    assert verdict.ok


def test_a_loss_written_with_a_minus_sign_matches_the_stored_negative():
    # num-015. The store held -2,701,000,000, the answer said -2,701,000,000,
    # and the reader dropped the sign -- so a loss was compared against a
    # profit and the answer was reported as contradicting the store.
    ev = [fact(-2_701_000_000, tag="NetIncomeLoss", period="2020-12-31", ticker="COP")]
    verdict = verify_answer(
        "COP reported -2,701,000,000 USD for Net Income (Loss) for 2020-12-31 [1].", ev
    )
    assert [n.status for n in verdict.checked] == ["supported"]
    assert verdict.ok


def test_a_narrative_answer_that_cites_nothing_at_all_does_not_ship():
    # The false negative: the per-sentence rule only looks at sentences
    # carrying a figure, so an answer making a purely qualitative claim with
    # every marker removed had nothing to be checked and passed.
    ev = [
        Evidence(
            kind="text",
            body="Walmart discusses ROI.",
            citation="c",
            accn=ACCN,
            chunk_id="c1",
            char_start=0,
            char_end=20,
        )
    ]
    verdict = verify_answer("Walmart shares ROI because management believes it is useful.", ev)
    assert verdict.uncited
    assert not verdict.ok


def test_but_a_refusal_still_passes_without_citing_anything():
    ev = [fact(60_922_000_000)]
    verdict = verify_answer(REFUSAL, ev, refused=True)
    assert verdict.uncited == ()
    assert verdict.ok
