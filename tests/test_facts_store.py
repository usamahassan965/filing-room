"""The transcription layer, tested on the shapes the real payloads actually have.

These tests build a store from hand-written payloads rather than from the corpus,
so they run in a second and fail for one reason at a time. The corpus itself is
checked by ``filing numbers``, which is a gate rather than a test: it needs 1.1 GB
of downloaded filings and it asserts things about the world, not about the code.

What is worth testing here is the handful of decisions that are easy to get wrong
and silent when wrong: span classification on a 4-4-5 retail calendar, instants
and durations never being mixed, the fiscal-year label being kept away from the
period keys, and restatements resolving to the latest filing rather than the
largest value or the last one parsed.
"""

from __future__ import annotations

import json

import pytest

from filing.ingest.manifest import Manifest
from filing.stores.facts import (
    CANONICAL_SPANS,
    SKIP_DATE,
    SKIP_VALUE,
    SPAN_TOLERANCE_DAYS,
    FactsStore,
    build_facts,
    classify_span,
    duplicate_keys,
    iter_facts,
)


def fact(
    *,
    start: str | None = None,
    end: str,
    val: float,
    accn: str = "0000000000-00-000000",
    form: str = "10-K",
    filed: str = "2025-01-01",
    fy: int = 2025,
    fp: str = "FY",
) -> dict:
    row = {"end": end, "val": val, "accn": accn, "form": form, "filed": filed, "fy": fy, "fp": fp}
    if start is not None:
        row["start"] = start
    return row


def payload(units: dict[str, list[dict]], *, tag: str = "Revenues", unit: str = "USD") -> dict:
    return {
        "cik": 1,
        "entityName": "Test Co",
        "facts": {
            "us-gaap": {
                tag: {"label": tag, "description": "d", "units": {unit: units[unit]}},
            }
        },
    }


# --- span classification -------------------------------------------------


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (90, "Q"),
        (91, "Q"),
        (92, "Q"),
        (98, "Q"),  # a 14-week quarter, which 4-4-5 retailers report
        (182, "H"),
        (168, "H"),  # 24 weeks -- a half-year on a retail calendar
        (273, "9M"),
        (252, "9M"),  # 36 weeks
        (365, "FY"),
        (371, "FY"),  # the 53-week year every retailer has periodically
        (None, "instant"),  # no start date at all -- a balance-sheet point
        (450, "other"),
        (140, "other"),  # genuinely between a quarter and a half
    ],
)
def test_span_is_nearest_canonical_length(days, expected):
    assert classify_span(days) == expected


def test_span_boundaries_are_the_declared_tolerance():
    """The tolerance is a constant, so the edges must be exactly where it says.

    Fixed ranges were the first thing tried and they discarded real retail
    periods: a 4-4-5 half-year is 168 days and a 4-4-5 nine months is 252, both
    of which sit outside any range built around 182 and 273 by a round number.
    """
    for span, length in CANONICAL_SPANS.items():
        assert classify_span(length + SPAN_TOLERANCE_DAYS) == span
        assert classify_span(length - SPAN_TOLERANCE_DAYS) == span


# --- parsing -------------------------------------------------------------


def test_instant_and_duration_are_distinguished_by_start():
    """195,414 of the corpus's facts have no start. They are points, not spans."""
    rows = list(
        iter_facts(
            payload({"USD": [fact(end="2024-12-31", val=1.0)]}, tag="Assets"),
            "0000000001",
            "TST",
        )
    )
    assert len(rows) == 1
    row = rows[0]
    assert row[5] is None  # period_start
    assert row[6] is not None  # period_end
    assert row[7] == "instant"
    assert row[9] == "instant"


def test_duration_keeps_both_endpoints_and_its_length():
    rows = list(
        iter_facts(
            payload({"USD": [fact(start="2024-01-01", end="2024-12-31", val=1.0)]}),
            "0000000001",
            "TST",
        )
    )
    (row,) = rows
    assert row[5] is not None
    assert row[8] == 366  # period_days, counted inclusively
    assert row[9] == "FY"


def test_fiscal_year_label_is_stored_but_is_not_a_period():
    """``fy``/``fp`` describe the filing, not the fact.

    They disagree with the year of ``period_end`` for 54.9% of the corpus, which
    is why they are recorded as ``filed_fy``/``filed_fp`` and never used as a key.
    A fact for calendar 2023 reported in a filing labelled FY2025 must keep 2023
    as its period and 2025 as its label.
    """
    rows = list(
        iter_facts(
            payload({"USD": [fact(start="2023-01-01", end="2023-12-31", val=1.0, fy=2025)]}),
            "0000000001",
            "TST",
        )
    )
    (row,) = rows
    assert row[6].year == 2023  # period_end
    assert row[15] == 2025  # filed_fy


def test_facts_without_a_date_or_a_value_are_skipped_not_guessed():
    rows = list(
        iter_facts(
            {
                "cik": 1,
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    # no end date
                                    {"val": 1.0, "accn": "a", "filed": "2025-01-01"},
                                    # no value
                                    {"end": "2024-12-31", "accn": "a", "filed": "2025-01-01"},
                                    # a string where a number belongs
                                    {
                                        "end": "2024-12-31",
                                        "val": "Test Co",
                                        "accn": "a",
                                        "filed": "2025-01-01",
                                    },
                                ]
                            }
                        }
                    }
                },
            },
            "0000000001",
            "TST",
        )
    )
    assert [r[0] for r in rows] == [SKIP_DATE, SKIP_VALUE, SKIP_VALUE]


# --- the store ------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """A one-company store built exactly the way the real one is."""
    raw = tmp_path / "raw" / "companyfacts"
    raw.mkdir(parents=True)
    entries = [
        # The same period reported twice, the later filing revising the value.
        fact(start="2023-01-01", end="2023-12-31", val=100.0, accn="a-1", filed="2024-02-01"),
        fact(start="2023-01-01", end="2023-12-31", val=110.0, accn="a-2", filed="2025-02-01"),
        fact(start="2024-01-01", end="2024-12-31", val=200.0, accn="a-2", filed="2025-02-01"),
        # A quarter inside a year that is also reported whole -- the span trap.
        fact(start="2024-01-01", end="2024-03-31", val=45.0, accn="a-2", filed="2025-02-01"),
    ]
    (raw / "CIK0000000001.json").write_text(json.dumps(payload({"USD": entries})))

    with Manifest(tmp_path / "manifest.duckdb", tmp_path) as m:
        m.upsert_company(cik="0000000001", ticker="TST", name="Test Co")
        m.upsert_facts(cik="0000000001", ticker="TST", path="raw/companyfacts/CIK0000000001.json")

    class Cfg:
        data_dir = tmp_path
        manifest_path = tmp_path / "manifest.duckdb"
        facts_path = tmp_path / "facts.duckdb"

    report = build_facts(Cfg())
    with FactsStore(Cfg.facts_path) as handle:
        yield handle.con, report


def test_build_is_driven_by_the_manifest_not_by_the_directory(tmp_path):
    """A directory glob would half-duplicate Exxon.

    ``data/raw/companyfacts/`` holds 21 payloads for a 20-company universe: the
    extra is Exxon's pre-reorganisation holding-company CIK, whose file the
    manifest does not list. Listing the directory instead of the manifest would
    load both and produce two partial Exxons that each look complete. Here the
    orphan is on disk before the build runs, and must not appear in the store.
    """
    raw = tmp_path / "raw" / "companyfacts"
    raw.mkdir(parents=True)
    known = [fact(start="2024-01-01", end="2024-12-31", val=1.0)]
    orphan = [fact(start="2024-01-01", end="2024-12-31", val=999.0)]
    (raw / "CIK0000000001.json").write_text(json.dumps(payload({"USD": known})))
    (raw / "CIK0009999999.json").write_text(json.dumps(payload({"USD": orphan})))

    with Manifest(tmp_path / "manifest.duckdb", tmp_path) as m:
        m.upsert_company(cik="0000000001", ticker="TST", name="Test Co")
        m.upsert_facts(cik="0000000001", ticker="TST", path="raw/companyfacts/CIK0000000001.json")

    class Cfg:
        data_dir = tmp_path
        manifest_path = tmp_path / "manifest.duckdb"
        facts_path = tmp_path / "facts.duckdb"

    report = build_facts(Cfg())
    assert report.companies == 1
    with FactsStore(Cfg.facts_path) as handle:
        assert handle.sql("SELECT DISTINCT cik FROM facts") == [("0000000001",)]


def test_identity_tuple_has_no_duplicates(store):
    con, _ = store
    assert duplicate_keys(con) == 0


def test_restatement_is_visible_and_resolves_to_the_latest_filing(store):
    """Both values are kept; ``facts_current`` picks by ``filed``, not by value."""
    con, report = store
    assert report.restatements == 1
    both = con.execute(
        "SELECT count(*) FROM facts WHERE period_end = DATE '2023-12-31' AND span = 'FY'"
    ).fetchone()[0]
    assert both == 2
    current = con.execute(
        "SELECT val FROM facts_current WHERE period_end = DATE '2023-12-31' AND span = 'FY'"
    ).fetchall()
    assert current == [(110.0,)]


def test_restatement_view_reports_the_movement(store):
    con, _ = store
    (first, last, delta) = con.execute(
        "SELECT first_val, last_val, delta FROM restatements"
    ).fetchone()
    assert (first, last, delta) == (100.0, 110.0, 10.0)


def test_quarters_and_years_coexist_without_being_summed(store):
    """The single largest trap in this data.

    ``NetIncomeLoss`` carries 2,499 quarters, 941 full years, 581 halves and 548
    nine-month stubs under one tag. Summing the tag over a year triple-counts.
    The span column is what lets a query say which it wants.
    """
    con, _ = store
    spans = dict(
        con.execute(
            "SELECT span, count(*) FROM facts WHERE period_end <= DATE '2024-12-31' "
            "AND period_start >= DATE '2024-01-01' GROUP BY span"
        ).fetchall()
    )
    assert spans == {"FY": 1, "Q": 1}
