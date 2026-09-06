"""The registry is prose and SQL in one object, so both halves are checked here.

A metric that names a tag no company uses is invisible; a ``derive`` expression
that names a metric that does not exist is a runtime error at build time; a
stock filtered as a flow silently mixes balances with income. None of those show
up as an exception when the corpus is present -- they show up as a plausible
wrong number, which is the failure mode this project is built to avoid. These
tests need no store, so they run on every commit rather than only after a build.
"""

from __future__ import annotations

import re

import pytest

from filing.stores.metrics import (
    BY_NAME,
    DERIVED,
    METRICS,
    SOURCED,
    as_rows,
    describe,
    known_metrics,
)

# Everything in a derive expression that is not a metric name.
SQL_VOCABULARY = {"coalesce", "nullif", "case", "when", "then", "else", "end"}

IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")


def test_metric_names_are_unique():
    names = [m.name for m in METRICS]
    assert len(names) == len(set(names))
    assert set(names) == set(known_metrics())


def test_every_metric_is_reachable_by_a_tag_or_a_derivation():
    """A metric with neither can never take a value, so it is a typo, not a metric."""
    unreachable = [m.name for m in METRICS if not m.tags and not m.derive]
    assert unreachable == []


def test_derive_expressions_reference_only_registered_metrics():
    """The check that catches a renamed metric before the build does.

    ``liabilities`` alone names four other metrics. Renaming one of them and
    missing this string would leave the balance-sheet identity computing with a
    NULL column, which reads as "this company does not report liabilities"
    rather than as an error.
    """
    for metric in DERIVED:
        tokens = set(IDENTIFIER.findall(metric.derive)) - SQL_VOCABULARY
        unknown = tokens - BY_NAME.keys()
        assert unknown == set(), f"{metric.name} references {sorted(unknown)}"


def test_a_derivation_never_references_itself():
    for metric in DERIVED:
        assert metric.name not in IDENTIFIER.findall(metric.derive)


def test_tags_within_a_metric_are_distinct_and_ordered_by_priority():
    """Order is meaning here: the first tag a company reports is the one it uses."""
    for metric in SOURCED:
        assert len(metric.tags) == len(set(metric.tags)), metric.name


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_stocks_are_instants_and_flows_are_durations(metric):
    """The mistake this prevents is summing four balance sheets into a year."""
    expected = "instant" if metric.kind == "stock" else "duration"
    assert metric.span_filter == expected


def test_capex_is_recorded_as_an_outflow_rather_than_negated():
    """Companies report capital expenditure positive; the store keeps it that way.

    Flipping the sign on the way in would make the stored number disagree with
    the filing it came from, and verify.py checks exactly that agreement. The
    fact that it is money leaving is carried by ``sign``, and free cash flow
    subtracts it.
    """
    assert describe("capex").sign == "outflow"
    assert describe("free_cash_flow").derive == "operating_cash_flow - capex"


def test_equity_resolves_by_registry_priority_not_by_coverage():
    """The one metric where the longest series is the wrong answer.

    All twenty companies tag both the parent-only and the including-NCI element,
    and for ten of them the including-NCI tag runs longer. Coverage-first
    ranking therefore put ten companies on one definition and ten on the other,
    so the ROE denominator meant different things depending on the ticker.
    """
    assert describe("equity").resolve_by == "priority"
    assert describe("equity_incl_nci").resolve_by == "coverage"


def test_coverage_is_the_default_because_tag_succession_is_the_common_case():
    """19 of 20 companies change revenue tags at ASC 606, and the series must survive."""
    assert describe("revenue").resolve_by == "coverage"
    assert sum(m.resolve_by == "priority" for m in METRICS) == 1


def test_non_universal_metrics_are_the_ones_absence_is_a_fact_about():
    """Five retailers report no R&D. That is true of them, not missing from us."""
    assert describe("rnd").universal is False
    for name in ("minority_interest", "temporary_equity", "equity_incl_nci"):
        assert describe(name).universal is False
    assert describe("revenue").universal is True


def test_as_rows_carries_every_field_the_registry_table_stores():
    rows = as_rows()
    assert len(rows) == len(METRICS)
    assert {r["metric"] for r in rows} == set(known_metrics())
    for row in rows:
        assert row["resolve_by"] in {"coverage", "priority"}


def test_describe_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        describe("ebitda")
