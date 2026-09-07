"""The constrained SQL layer: what it resolves, what it refuses, and the grain.

Three things are worth holding in place here, and none of them is "SQL runs".

* A phrase resolves to the tag a person meant, including when the SEC's own
  label for that tag is unguessable from the tag name.
* A phrase that is not a registry concept resolves to nothing at all. The
  tempting failure is a near-miss -- returning ``Assets`` for a question about
  goodwill -- and it is worse than an empty answer because it is quotable.
* Queries are at fact grain by tag, not at metric grain. ``Revenues`` and
  ``RevenueFromContractWithCustomerExcludingAssessedTax`` are one metric and two
  questions, and collapsing them is a wrong answer that looks right.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from filing.agent.sql import (
    ALLOWED_TAGS,
    PERIOD_SLACK_DAYS,
    ConceptResolver,
    SqlTool,
    UnknownConcept,
)
from filing.stores.facts import FactsStore

# Two tickers, both revenue tags, a moving fiscal year end, and one tag that is
# real in the corpus but outside the registry -- the four cases the layer has to
# tell apart.
FACTS = [
    # ticker, taxonomy, tag, unit, start, end, span, val, accn, form
    ("TST", "us-gaap", "Revenues", "USD", "2023-01-01", "2023-12-31", "FY", 100.0, "a-1", "10-K"),
    ("TST", "us-gaap", "Revenues", "USD", "2024-01-01", "2024-12-31", "FY", 200.0, "a-2", "10-K"),
    (
        "TST",
        "us-gaap",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "USD",
        "2024-01-01",
        "2024-12-31",
        "FY",
        190.0,
        "a-2",
        "10-K",
    ),
    ("TST", "us-gaap", "Assets", "USD", None, "2024-12-31", "instant", 900.0, "a-2", "10-K"),
    ("TST", "us-gaap", "AssetsCurrent", "USD", None, "2024-12-31", "instant", 300.0, "a-2", "10-K"),
    ("TST", "us-gaap", "Goodwill", "USD", None, "2024-12-31", "instant", 50.0, "a-2", "10-K"),
    # A 52/53-week retailer whose year end drifts by two days.
    ("RTL", "us-gaap", "Revenues", "USD", "2023-01-29", "2024-02-03", "FY", 500.0, "b-1", "10-K"),
]

CONCEPTS = [
    ("us-gaap", "Revenues", "Revenues"),
    (
        "us-gaap",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenue from Contract with Customer, Excluding Assessed Tax",
    ),
    ("us-gaap", "Assets", "Assets"),
    ("us-gaap", "AssetsCurrent", "Assets, Current"),
    ("us-gaap", "NetIncomeLoss", "Net Income (Loss) Attributable to Parent"),
    ("us-gaap", "Goodwill", "Goodwill"),
]


@pytest.fixture
def store(tmp_path):
    """The four tables the layer reads, and nothing else it might lean on."""
    path = tmp_path / "facts.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE facts_current (ticker VARCHAR, taxonomy VARCHAR, tag VARCHAR, "
        "unit VARCHAR, period_start DATE, period_end DATE, span VARCHAR, val DOUBLE, "
        "accn VARCHAR, form VARCHAR)"
    )
    con.executemany("INSERT INTO facts_current VALUES (?,?,?,?,?,?,?,?,?,?)", FACTS)
    con.execute("CREATE TABLE filings (accn VARCHAR, form VARCHAR)")
    con.execute("INSERT INTO filings SELECT DISTINCT accn, form FROM facts_current")
    con.execute("CREATE TABLE concepts (taxonomy VARCHAR, tag VARCHAR, label VARCHAR)")
    con.executemany("INSERT INTO concepts VALUES (?,?,?)", CONCEPTS)
    con.execute(
        "CREATE VIEW fiscal_years AS SELECT DISTINCT ticker, period_end AS fiscal_end "
        "FROM facts_current WHERE span = 'FY'"
    )
    con.close()
    with FactsStore(path) as handle:
        yield handle


@pytest.fixture
def tool(store):
    return SqlTool(store)


# --- what the allow-list is -----------------------------------------------


def test_the_allow_list_is_the_registry_and_nothing_else():
    """Not a hand-kept list. A tag enters by being registered, and only that way."""
    from filing.stores.metrics import METRICS

    assert ALLOWED_TAGS == frozenset(t for m in METRICS for t in m.tags)
    assert "Revenues" in ALLOWED_TAGS
    assert "Goodwill" not in ALLOWED_TAGS


def test_a_tag_in_the_corpus_but_outside_the_registry_is_not_askable(tool):
    """Goodwill is in the fixture's facts and in its concepts. It is still not askable.

    This is the constraint doing its job: the store knowing a number is not the
    same as the agent being allowed to name it, because a tag nobody registered
    is a tag nobody checked the meaning of.
    """
    assert "Goodwill" not in tool.resolver.tags
    answer = tool.lookup("TST", "goodwill")
    assert answer.concept is None
    assert not answer.found
    assert "no registry concept" in answer.reason


def test_the_eval_sets_current_ratio_concepts_are_reachable(tool):
    """The M5 registry addition, asserted where a reader will look for it.

    Ten of the eighty numeric questions name these two. Before M5 they were
    unaskable -- not badly answered, structurally unanswerable -- and a router
    scored on them was being marked against a question the system could not
    have got right.
    """
    assert tool.resolver.resolve("Assets, Current").tag == "AssetsCurrent"
    assert tool.resolver.resolve("total current assets").tag == "AssetsCurrent"


# --- resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "tag", "how"),
    [
        ("Revenues", "Revenues", "exact-label"),
        ("Assets, Current", "AssetsCurrent", "exact-label"),
        # The label the SEC actually publishes, which no CamelCase split reaches.
        ("Net Income (Loss) Attributable to Parent", "NetIncomeLoss", "exact-label"),
        ("net income", "NetIncomeLoss", "metric-label"),
        ("total assets", "Assets", "metric-label"),
        # Normalisation strips the underscore and the comma alike, so the
        # registry's own name collides with the SEC's label. Same tag either
        # way; the route it took is an implementation detail, not a promise.
        ("assets_current", "AssetsCurrent", "exact-label"),
        ("NetIncomeLoss", "NetIncomeLoss", "exact-tag"),
    ],
)
def test_phrases_resolve_the_way_a_person_would_read_them(tool, phrase, tag, how):
    concept = tool.resolver.resolve(phrase)
    assert (concept.tag, concept.how) == (tag, how)


def test_a_label_the_store_does_not_have_resolves_to_nothing(tool):
    """The near-miss is the dangerous answer, so there is no near-miss."""
    assert tool.resolver.resolve("the mood of the chief financial officer") is None
    assert tool.resolver.resolve("") is None


def test_require_raises_rather_than_returning_a_default(tool):
    with pytest.raises(UnknownConcept):
        tool.resolver.require("earnings before hubris")


def test_overlap_prefers_the_more_specific_label(store):
    """A longer phrase must not settle for `Revenues` just because it is shorter."""
    resolver = ConceptResolver(store)
    best = resolver.resolve("revenue from contracts with customers")
    assert best.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert best.how == "overlap"


# --- the grain -------------------------------------------------------------


def test_two_revenue_tags_are_two_answers_not_one_metric(tool):
    """The reversal of M2's collapse, and the reason the query is at fact grain.

    Both tags are metric ``revenue``. Asked separately they carry 200 and 190 in
    the same filing, and a question that named one of them wants that one.
    """
    a = tool.lookup("TST", "Revenues", period_end="2024-12-31")
    b = tool.lookup(
        "TST",
        "Revenue from Contract with Customer, Excluding Assessed Tax",
        period_end="2024-12-31",
    )
    assert a.value == 200.0
    assert b.value == 190.0
    assert a.concept.metric == b.concept.metric == "revenue"


def test_the_period_picks_the_year_that_was_asked_for(tool):
    assert tool.lookup("TST", "Revenues", period_end="2023-12-31").value == 100.0
    assert tool.lookup("TST", "Revenues", period_end="2024-12-31").value == 200.0


def test_no_period_means_the_most_recent(tool):
    assert tool.lookup("TST", "Revenues").value == 200.0


def test_a_moving_fiscal_year_end_still_matches(tool):
    """A 52/53-week retailer lands on a different date every year.

    Asked about 2024-02-01 the answer is the year that ended 2024-02-03, because
    that is the year the question is about. Exact-date matching would report a
    gap that is a calendar artefact.
    """
    assert tool.lookup("RTL", "Revenues", period_end="2024-02-01").value == 500.0


def test_a_period_outside_the_slack_is_a_miss_not_the_nearest_row(tool):
    """Proximity ordering must not degrade into "whatever is closest"."""
    answer = tool.lookup("TST", "Revenues", period_end="2019-12-31")
    assert not answer.found
    assert answer.concept is not None  # the concept is known; the fact is not there
    assert "no Revenues" in answer.reason
    assert PERIOD_SLACK_DAYS < 30


def test_a_known_concept_a_company_does_not_report_is_a_different_answer(tool):
    """Empty-with-concept and empty-without-concept must stay distinguishable."""
    answer = tool.lookup("RTL", "total current assets")
    assert answer.concept is not None
    assert not answer.found
    assert answer.value is None


# --- the shape the rest of the agent depends on ----------------------------


def test_the_query_is_parameterised_and_never_interpolated(tool):
    """No user string reaches the SQL text -- only the placeholders and a LIMIT."""
    answer = tool.lookup("TST'; DROP TABLE facts_current; --", "Revenues")
    assert not answer.found
    assert "DROP" not in answer.query
    assert answer.query.count("?") == len(answer.params)
    assert tool.store.sql("SELECT count(*) FROM facts_current")[0][0] == len(FACTS)


def test_a_row_carries_its_own_citation(tool):
    row = tool.lookup("TST", "Revenues", period_end="2024-12-31").rows[0]
    assert row.citation == "TST 10-K 2024-12-31 Revenues [a-2]"


def test_periods_lists_what_the_company_actually_reports(tool):
    assert tool.periods("TST") == [date(2024, 12, 31), date(2023, 12, 31)]


def test_the_tool_schema_is_generated_from_the_registry(tool):
    """A metric added to the registry is one the model is told about, same commit."""
    schema = tool.schema()
    assert schema["name"] == "lookup_fact"
    assert "Total current assets" in schema["concepts"]
    assert set(schema["parameters"]) == {"ticker", "concept", "period_end"}


# --- the recheck path (M6) --------------------------------------------------


def test_a_recheck_reads_the_fact_by_its_identity_not_by_a_phrase(tool):
    """The verifier's read has to bypass the resolver it is checking."""
    row = tool.recheck("TST", "Revenues", period_end="2024-12-31")
    assert row is not None
    assert row.val == 200.0
    assert row.tag == "Revenues"


def test_a_recheck_is_case_insensitive_in_the_ticker_only(tool):
    assert tool.recheck("tst", "Revenues", period_end="2024-12-31") is not None
    # The tag is an identifier, not a phrase. A near-miss must miss.
    assert tool.recheck("TST", "revenues", period_end="2024-12-31") is None


def test_a_recheck_can_reach_a_tag_the_resolver_would_refuse(tool):
    """Goodwill is outside the registry, so `lookup` will not ask for it.

    The recheck still finds it, and that asymmetry is the point: the allow-list
    governs what a *question* may reach, and the verifier is not asking a
    question -- it is confirming a number the system already produced.
    """
    with pytest.raises(UnknownConcept):
        tool.resolver.require("goodwill")
    row = tool.recheck("TST", "Goodwill", period_end="2024-12-31")
    assert row is not None and row.val == 50.0


def test_a_recheck_outside_the_period_slack_returns_nothing(tool):
    assert tool.recheck("TST", "Revenues", period_end="2020-12-31") is None


def test_a_recheck_without_a_period_takes_the_most_recent(tool):
    row = tool.recheck("TST", "Revenues")
    assert row is not None and row.val == 200.0


def test_a_recheck_can_be_pinned_to_one_filing(tool):
    assert tool.recheck("TST", "Revenues", accn="a-1").val == 100.0
    assert tool.recheck("TST", "Revenues", accn="nope") is None
