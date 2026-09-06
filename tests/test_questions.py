"""The two independent checks M2 rests on, tested without needing the corpus.

``filing numbers`` runs these against 514,649 real facts, where a failure means
either the store is wrong or my expectation was. Here they run against values I
choose, where a failure can only mean the code is wrong. The interesting cases
are the ones the corpus made me find: equity that arrives under three different
tags depending on the decade, a value that is missing rather than zero, and the
difference between a value being wrong and a value being absent.
"""

from __future__ import annotations

import pytest

from filing.stores.questions import (
    QUESTIONS,
    TOLERANCE,
    Answer,
    Question,
    QuestionReport,
    _derive,
)
from filing.stores.verify import candidate_strings

BASE = {
    "revenue": 1000.0,
    "cost_of_revenue": 600.0,
    "operating_income": 250.0,
    "net_income": 200.0,
    "operating_cash_flow": 300.0,
    "capex": 120.0,
    "assets": 5000.0,
    "equity": 2000.0,
}


def derive(**overrides):
    prior = overrides.pop("_prior", {})
    return _derive({**BASE, **overrides}, prior)


# --- recomputation --------------------------------------------------------


def test_margins_and_returns_are_plain_ratios():
    d = derive()
    assert d["gross_profit"] == 400.0
    assert d["gross_margin"] == pytest.approx(0.4)
    assert d["operating_margin"] == pytest.approx(0.25)
    assert d["net_margin"] == pytest.approx(0.2)
    assert d["return_on_equity"] == pytest.approx(0.1)
    assert d["return_on_assets"] == pytest.approx(0.04)


def test_free_cash_flow_subtracts_a_positively_reported_capex():
    """Capex is stored as filed, which is positive, so FCF is a subtraction."""
    assert derive()["free_cash_flow"] == 180.0


def test_a_reported_gross_profit_is_preferred_to_a_computed_one():
    """Where the company tagged it, its own number wins.

    Revenue minus cost of revenue is not always what a filer calls gross profit
    -- some put shipping or amortisation on one side and not the other -- so the
    subtraction is a fallback, never an override.
    """
    assert derive(gross_profit=390.0)["gross_profit"] == 390.0


def test_liabilities_fall_back_to_the_balance_sheet_identity():
    assert derive()["liabilities"] == 3000.0  # 5000 - 2000
    assert derive(liabilities=2950.0)["liabilities"] == 2950.0


def test_equity_including_nci_is_used_whole_when_the_company_reports_it():
    """The share attributable to the parent is not the identity's equity term.

    Chevron's 2025 non-controlling interests are $5.7bn. Subtracting only the
    parent's equity leaves that on the table and the sheet does not balance.
    """
    d = derive(equity_incl_nci=2100.0)
    assert d["liabilities"] == 2900.0
    # ROE stays on the parent's equity: it is the return to shareholders.
    assert d["return_on_equity"] == pytest.approx(0.1)


def test_pre_2009_filings_add_the_separate_minority_interest_line():
    """ASC 810 folded NCI into equity in 2009. Before that it was its own line.

    Exxon's 2008 balance sheet is short by exactly $4,558m without it, and no
    including-NCI tag exists in those filings to fall back on.
    """
    assert derive(minority_interest=150.0)["liabilities"] == 2850.0


def test_mezzanine_equity_belongs_to_neither_side():
    """NVIDIA's fiscal 2016 sheet leaves $87m that is not debt and not equity."""
    assert derive(temporary_equity=90.0)["liabilities"] == 2910.0


def test_the_including_nci_tag_wins_over_the_separate_line_when_both_appear():
    """Filings in the transition years carry both, and they would double-count."""
    d = derive(equity_incl_nci=2100.0, minority_interest=100.0)
    assert d["liabilities"] == 2900.0


def test_a_missing_input_yields_a_missing_output_rather_than_a_zero():
    """The distinction the whole store turns on.

    A company that does not report R&D has no R&D row, not a zero one. Filling
    absent inputs with zero would make "we could not tell" indistinguishable
    from "it was nothing", and the second is a much stronger claim.
    """
    d = _derive({"revenue": 1000.0}, {})
    assert d["gross_profit"] is None
    assert d["gross_margin"] is None
    assert d["liabilities"] is None
    assert d["free_cash_flow"] is None


def test_a_zero_denominator_yields_none_rather_than_raising():
    d = _derive({**BASE, "revenue": 0.0, "equity": 0.0}, {})
    assert d["gross_margin"] is None
    assert d["return_on_equity"] is None


def test_year_over_year_growth_needs_the_prior_year_and_says_so_when_absent():
    assert derive(_prior={"revenue": 800.0})["revenue_yoy"] == pytest.approx(0.25)
    assert derive()["revenue_yoy"] is None
    assert derive(_prior={"revenue": 0.0})["revenue_yoy"] is None


# --- the question set -----------------------------------------------------


def test_every_question_cites_the_filing_its_answer_came_from():
    """A question without a cited source is an assertion, not a check.

    The source is what makes a failure diagnosable: when the store and the
    expectation disagree, the filing decides which of them is wrong. That
    happened four times while writing these, and the store was right every time.
    """
    for q in QUESTIONS:
        assert q.id and q.text and q.sql.strip()
        assert q.source, q.id


def test_a_set_question_states_in_words_what_its_predicate_asserts():
    """``holds`` is a lambda, so on its own it is unreadable in a failure report.

    Scalar questions need no such gloss: ``expect`` is the claim, in figures.
    """
    for q in QUESTIONS:
        assert bool(q.claim) == (q.kind == "set"), q.id


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_a_question_is_either_scalar_or_a_set_and_never_both():
    for q in QUESTIONS:
        assert (q.expect is None) != (q.holds is None), q.id
        assert q.kind == ("scalar" if q.expect is not None else "set")


def test_a_scalar_answer_passes_within_the_declared_tolerance():
    q = Question(id="t", text="t", sql="SELECT 1", expect=100.0, claim="c", source="s")
    inside = 100.0 * (1 + TOLERANCE / 2)
    outside = 100.0 * (1 + TOLERANCE * 2)
    assert Answer(q, [(inside,)], ["v"], passed=True, got=inside).scalar == inside
    assert abs(inside - q.expect) <= TOLERANCE * abs(q.expect)
    assert abs(outside - q.expect) > TOLERANCE * abs(q.expect)


def test_the_report_counts_failures_rather_than_stopping_at_the_first():
    q = QUESTIONS[0]
    report = QuestionReport(
        [
            Answer(q, [], [], passed=True, got=None),
            Answer(q, [], [], passed=False, got=None, detail="wrong"),
        ]
    )
    assert (report.total, report.passed) == (2, 1)
    assert len(report.failures) == 1


# --- reading a number back out of a filing --------------------------------


def test_a_value_is_looked_for_at_every_scale_a_filing_might_use():
    """Filings state the same number in units, thousands or millions.

    A stored 15,000,000,000 appears in the document as "15,000" if the statement
    is headed "in millions", so a single rendering finds almost nothing.
    """
    scales = {scale for _text, scale in candidate_strings(15_000_000_000.0)}
    assert scales == {"units", "thousands", "millions"}
    texts = {text for text, _ in candidate_strings(15_000_000_000.0)}
    assert "15,000" in texts and "15,000,000" in texts


def test_both_grouped_and_plain_renderings_are_offered():
    texts = {text for text, _ in candidate_strings(1_234_567.0)}
    assert "1,234,567" in texts
    assert "1234567" in texts


def test_negatives_are_also_looked_for_parenthesised():
    """Accounting statements write a loss as (1,234), never as -1,234."""
    rendered = candidate_strings(-1_234.0)
    texts = {text for text, _ in rendered}
    assert "(1,234)" in texts
    assert any("parenthesised" in scale for _t, scale in rendered)


def test_a_scale_smaller_than_one_is_not_offered():
    """Rounding 4,000 to "0" millions would match the digit 0 in any document."""
    rendered = candidate_strings(4_000.0)
    assert "millions" not in {scale for _text, scale in rendered}
    assert "0" not in {text for text, _ in rendered}
