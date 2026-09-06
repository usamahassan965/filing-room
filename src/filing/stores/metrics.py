"""From tags to metrics: the layer that knows what the numbers mean.

``facts`` is a transcription. This is the interpretation, and it is kept in one
readable table because every judgement call in it is arguable and an interviewer
is entitled to see them all in one place.

The reason this layer has to exist at all is that companies do not agree on
which tag to use. Measured across the twenty companies in this corpus:

    Revenues                20/20 with a fallback list, 0/20 with any single tag
    GrossProfit             10/20 -- half the corpus never tags it
    OperatingIncomeLoss     13/20 -- the seven oil and pharma names omit it
    ResearchAndDevelopment  15/20 -- and the missing five are all retailers

Three different problems hide in those four lines, and they need three different
answers, not one:

* **Different tag, same concept.** Revenue is ``RevenueFromContractWith
  CustomerExcludingAssessedTax`` for most and ``Revenues`` for others. A
  priority-ordered candidate list resolves it, per company, once.
* **Not tagged but computable.** The ten companies without ``GrossProfit`` all
  report revenue and cost of revenue, so gross profit is derived rather than
  missing. Derivation is declared in the registry and computed in SQL.
* **Genuinely absent.** Walmart has no R&D line because Walmart does no R&D.
  That is not a data gap and must not be reported as one -- "no data" and "this
  company does not report that" are different answers to a user's question.

``OperatingIncomeLoss`` is the case where the tempting shortcut is wrong.
``IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems
NoncontrollingInterest`` is present for 19/20 and would paper over the gap --
but it is *pretax income*, which is operating income plus interest, plus other
income, minus non-operating charges. Substituting it would produce an operating
margin that is wrong by a plausible-looking margin, which is the worst kind of
wrong. So operating income stays a 13/20 metric, ``pretax_income`` is registered
separately under its own name, and operating margin is simply unavailable for
seven companies. An honest gap beats a confident fabrication.

There is a fourth problem, and it is the one that produces wrong answers rather
than missing ones: **near-synonyms that are not synonyms.** ``StockholdersEquity``
is the parent's share; ``StockholdersEquityIncludingPortionAttributableTo
NoncontrollingInterest`` is the whole. Either is the right answer to a different
question -- the first is the denominator return on equity wants, the second is
the term in ``assets = liabilities + equity`` -- and using one where the other
belongs leaves the balance sheet short by exactly the noncontrolling interest,
which for Chevron in 2025 is $5.7 billion. The registry keeps them as separate
metrics under separate names so that choosing between them is something a query
does on purpose. ``MinorityInterest`` (the pre-2009 spelling, outside equity)
and the mezzanine line, which belongs to neither side, are registered for the
same reason. With all four the identity closes on 1,717 balance sheets and fails
on none.

Sign is documented, never mutated. ``PaymentsToAcquirePropertyPlantAndEquipment``
is reported positive even though it is cash leaving the business; the registry
records that as ``sign="outflow"`` and the free-cash-flow derivation subtracts
it. The stored fact still matches the filing it came from, which is what makes
the M2 spot-check against source documents possible at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import duckdb

Kind = Literal["flow", "stock", "per_share", "count"]


@dataclass(frozen=True, slots=True)
class Metric:
    """One financial concept, and every tag any company in the corpus uses for it."""

    name: str
    label: str
    kind: Kind
    unit: str
    tags: tuple[str, ...] = ()
    # SQL over other metric names, evaluated per (company, period) when no tag
    # resolves. Written against the wide annual/quarterly views.
    derive: str | None = None
    # False when absence is a fact about the business rather than a hole in the
    # data. The gate reports coverage for these but does not fail on them.
    universal: bool = True
    sign: Literal["as_reported", "outflow"] = "as_reported"
    # How to choose when a company reports more than one candidate tag.
    # "coverage" keeps the company on whichever tag runs longest, which is what
    # a tag succession needs. "priority" obeys the registry order regardless of
    # coverage, which is what near-synonyms need -- see ``equity``.
    resolve_by: Literal["coverage", "priority"] = "coverage"
    note: str = ""

    @property
    def span_filter(self) -> str:
        """Durations and instants never mix. A flow is a span; a stock is a date."""
        return "instant" if self.kind == "stock" else "duration"


#: The registry. Order within ``tags`` is priority order: the first tag a company
#: actually reports, in the metric's own unit, is the one that company uses.
METRICS: tuple[Metric, ...] = (
    Metric(
        "revenue",
        "Revenue",
        "flow",
        "USD",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ),
        note="ASC 606 renamed this in 2018; the older tags are why the pre-2019 "
        "years resolve at all.",
    ),
    Metric(
        "cost_of_revenue",
        "Cost of revenue",
        "flow",
        "USD",
        (
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",
            "CostOfServices",
        ),
        universal=False,
        note="19/20. XOM reports only CostsAndExpenses, which is total operating "
        "cost, not cost of revenue -- so it stays unresolved rather than wrong, "
        "and Exxon simply has no gross margin in this store.",
    ),
    Metric(
        "gross_profit",
        "Gross profit",
        "flow",
        "USD",
        ("GrossProfit",),
        derive="revenue - cost_of_revenue",
        note="Tagged by 10/20. Derived for the rest, which is exact: gross "
        "profit is defined as the difference, not estimated from it.",
    ),
    Metric(
        "operating_income",
        "Operating income",
        "flow",
        "USD",
        ("OperatingIncomeLoss",),
        universal=False,
        note="13/20. The energy and pharma names report a different income "
        "statement shape. Not substitutable with pretax income.",
    ),
    Metric(
        "pretax_income",
        "Pretax income",
        "flow",
        "USD",
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
        note="A different concept from operating income, registered separately "
        "so nothing is tempted to use it as a stand-in.",
    ),
    Metric("net_income", "Net income", "flow", "USD", ("NetIncomeLoss", "ProfitLoss")),
    Metric("income_tax", "Income tax expense", "flow", "USD", ("IncomeTaxExpenseBenefit",)),
    Metric(
        "rnd",
        "Research and development",
        "flow",
        "USD",
        (
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ),
        universal=False,
        note="15/20, and the five without it are the retailers. Absent because "
        "they do no R&D, not because the data is missing. The second tag is "
        "not a synonym anyone would guess: AbbVie and Pfizer report only the "
        "excluding-IPR&D variant, so with the obvious tag alone the two "
        "largest R&D spenders in the corpus looked like they spent nothing. "
        "IPR&D acquired in a deal is a one-off purchase price, not a research "
        "budget, so the narrower tag is the better answer anyway.",
    ),
    Metric(
        "sga",
        "Selling, general and administrative",
        "flow",
        "USD",
        ("SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"),
    ),
    Metric(
        "interest_expense",
        "Interest expense",
        "flow",
        "USD",
        ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"),
        universal=False,
    ),
    Metric(
        "operating_cash_flow",
        "Cash from operations",
        "flow",
        "USD",
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    Metric(
        "capex",
        "Capital expenditure",
        "flow",
        "USD",
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        universal=False,
        sign="outflow",
        note="Reported positive though it is cash leaving. Stored as filed; "
        "free_cash_flow subtracts it.",
    ),
    Metric(
        "free_cash_flow",
        "Free cash flow",
        "flow",
        "USD",
        (),
        derive="operating_cash_flow - capex",
        universal=False,
        note="Never a tag. Always the derivation, which is why capex's sign "
        "convention had to be written down.",
    ),
    Metric(
        "dividends_paid",
        "Dividends paid",
        "flow",
        "USD",
        ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
        universal=False,
        sign="outflow",
    ),
    Metric("assets", "Total assets", "stock", "USD", ("Assets",)),
    Metric(
        "liabilities",
        "Total liabilities",
        "stock",
        "USD",
        ("Liabilities",),
        derive="assets - coalesce(equity_incl_nci, equity + coalesce(minority_interest, 0))"
        "  - coalesce(temporary_equity, 0)",
        note="Only 13/20 tag it -- the rest tag current and noncurrent "
        "separately and let the balance sheet balance. The derivation is the "
        "accounting identity, so it is exact, not an estimate -- but only if "
        "the equity term is the whole right-hand side, which is why it "
        "subtracts total equity and mezzanine rather than the parent's share.",
    ),
    Metric(
        "equity",
        "Shareholders' equity",
        "stock",
        "USD",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        resolve_by="priority",
        note="The parent's share. Resolved by registry priority rather than by "
        "coverage, which is the exception to this store's usual rule and the "
        "reason `resolve_by` exists. All twenty companies tag both, and the "
        "including-NCI tag runs longer for ten of them, so coverage-first "
        "ranking made this the parent's share for half the corpus and the "
        "total for the other half -- an ROE denominator that meant two "
        "different things depending on the ticker, with nothing to show for it "
        "in the output. Tag successions want continuity; near-synonyms want "
        "the registry's judgement. This is the denominator return on equity "
        "wants: the earnings attributable to shareholders divided by what "
        "those shareholders own.",
    ),
    Metric(
        "equity_incl_nci",
        "Total equity including noncontrolling interests",
        "stock",
        "USD",
        ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",),
        universal=False,
        note="Registered separately rather than folded into `equity` because "
        "the two answer different questions and swapping them silently is a "
        "real error. Parent equity is the ROE denominator; this is the term in "
        "assets = liabilities + equity. Using the parent's share in the "
        "identity leaves the balance sheet short by exactly the "
        "noncontrolling interest -- $5.7B for Chevron in 2025, which is not a "
        "rounding difference and not a bug in the data.",
    ),
    Metric(
        "minority_interest",
        "Noncontrolling interests",
        "stock",
        "USD",
        ("MinorityInterest",),
        universal=False,
        note="The pre-2009 spelling. ASC 810 renamed minority interest to "
        "noncontrolling interests and moved it inside equity, so filings from "
        "2009 and earlier carry it as its own line outside `StockholdersEquity` "
        "and never tag the combined total at all. Exxon's 2008 balance sheet is "
        "short by exactly this $4,558M without it.",
    ),
    Metric(
        "temporary_equity",
        "Temporary (mezzanine) equity",
        "stock",
        "USD",
        (
            "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
            "TemporaryEquityCarryingAmountAttributableToParent",
            "TemporaryEquityValueExcludingAdditionalPaidInCapital",
            "RedeemableNoncontrollingInterestEquityCarryingAmount",
        ),
        universal=False,
        note="The mezzanine line, which sits between liabilities and equity "
        "and belongs to neither -- redeemable interests the issuer may be "
        "obliged to buy back. Rare, and small when present, but it is the "
        "last $87M of NVIDIA's fiscal 2016 balance sheet and without it the "
        "identity does not close.",
    ),
    Metric(
        "cash",
        "Cash and equivalents",
        "stock",
        "USD",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
    ),
    Metric("inventory", "Inventory", "stock", "USD", ("InventoryNet",), universal=False),
    Metric(
        "long_term_debt",
        "Long-term debt",
        "stock",
        "USD",
        ("LongTermDebtNoncurrent", "LongTermDebt"),
    ),
    Metric(
        "eps_diluted",
        "Diluted EPS",
        "per_share",
        "USD/shares",
        ("EarningsPerShareDiluted",),
    ),
    Metric(
        "shares_diluted",
        "Diluted shares outstanding",
        "count",
        "shares",
        ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    ),
)

BY_NAME: dict[str, Metric] = {m.name: m for m in METRICS}
SOURCED: tuple[Metric, ...] = tuple(m for m in METRICS if m.tags)
DERIVED: tuple[Metric, ...] = tuple(m for m in METRICS if m.derive)


# --------------------------------------------------------------------- build


def _resolution_rows(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Choose one tag per (company, metric): the first candidate they report.

    Done once, into a table, rather than as a CASE expression inside every
    query. Two reasons. It makes "which tag is AAPL's revenue?" a SELECT that a
    reviewer can run, and it makes a company that switched tags mid-window
    visible as a row count rather than as a hole in a chart.
    """
    rows: list[tuple] = []
    for metric in SOURCED:
        for rank, tag in enumerate(metric.tags):
            rows.extend(
                (cik, ticker, metric.name, tag, rank, n, first, last)
                for cik, ticker, n, first, last in con.execute(
                    """
                    SELECT cik, ticker, count(*), min(period_end), max(period_end)
                    FROM facts_current
                    WHERE taxonomy = 'us-gaap' AND tag = ? AND unit = ?
                      AND period_type = ? AND span <> 'other'
                    GROUP BY cik, ticker
                    """,
                    [tag, metric.unit, metric.span_filter],
                ).fetchall()
            )
    return rows


METRIC_TABLES = """
CREATE OR REPLACE TABLE metric_registry (
    metric     VARCHAR PRIMARY KEY,
    label      VARCHAR,
    kind       VARCHAR,
    unit       VARCHAR,
    tags       VARCHAR,   -- the candidate list, in priority order
    derive     VARCHAR,
    universal  BOOLEAN,
    sign       VARCHAR,
    resolve_by VARCHAR,
    note       VARCHAR
);

CREATE OR REPLACE TABLE metric_candidates (
    cik        VARCHAR, ticker VARCHAR, metric VARCHAR, tag VARCHAR,
    priority   INTEGER, n_facts INTEGER, first_period DATE, last_period DATE
);
"""

# Resolution is per company *and per period*, not one tag per company.
#
# The naive rule -- take the highest-priority candidate a company reports -- was
# measurably wrong here. 19 of the 20 companies report revenue under more than
# one tag, because ASC 606 replaced the revenue element in 2018 and the older
# facts were never retagged. Exxon is the clean example: the ASC 606 tag covers
# FY2017 to FY2021 and nothing else, while ``Revenues`` covers 2009 to 2025.
# Choosing by registry priority alone picked the newer tag and silently dropped
# thirteen years of Exxon revenue -- and it produced NULL, not an error.
#
# So the rank is coverage first, registry priority only as the tie-break. That
# keeps each company on one tag for as long as that tag runs, which matters more
# than the taxonomy's preference: a time series assembled from two tags with
# different scopes has a step change in it that is an artefact, not a business
# event. Lower-ranked tags then fill only the periods the primary never covered.
METRIC_VIEWS = """
CREATE OR REPLACE TABLE metric_tags AS
SELECT c.* EXCLUDE (resolve_by), row_number() OVER (
           PARTITION BY c.cik, c.metric
           ORDER BY CASE WHEN c.resolve_by = 'priority' THEN c.priority END,
                    c.n_facts DESC, c.priority, c.tag) AS resolution_rank
FROM (SELECT c.*, r.resolve_by FROM metric_candidates c
      JOIN metric_registry r ON r.metric = c.metric) c;

-- The tag a reviewer should look at first, per company and metric.
CREATE OR REPLACE VIEW metric_primary_tag AS
SELECT cik, ticker, metric, tag, n_facts, first_period, last_period
FROM metric_tags WHERE resolution_rank = 1;

-- Every fact a resolved tag claims, relabelled with the metric name, with one
-- winner per period. Joined on unit and period type as well as tag, so a metric
-- can never pick up a value reported in shares when it expects dollars.
CREATE OR REPLACE VIEW metric_facts AS
SELECT * EXCLUDE (_rn) FROM (
    SELECT f.cik, f.ticker, t.metric, r.label, f.tag, f.unit, f.span,
           f.period_start, f.period_end, f.period_days, f.val, f.accn, f.form,
           f.filed, t.resolution_rank,
           row_number() OVER (
               PARTITION BY f.cik, t.metric, f.span, f.period_start, f.period_end
               ORDER BY t.resolution_rank) AS _rn
    FROM facts_current f
    JOIN metric_tags     t ON t.cik = f.cik AND t.tag = f.tag
    JOIN metric_registry r ON r.metric = t.metric
    WHERE f.taxonomy = 'us-gaap' AND f.unit = r.unit AND f.span <> 'other'
) WHERE _rn = 1;

-- A company's fiscal year ends are the dates on which it reports a full-year
-- duration. Derived from the data rather than from the fiscal_year_end field,
-- because retailers move their year end by a few days annually and a 52/53-week
-- calendar does not land on the same date twice.
CREATE OR REPLACE VIEW fiscal_years AS
SELECT DISTINCT cik, ticker, period_end AS fiscal_end, period_start AS fiscal_start
FROM metric_facts WHERE span = 'FY';
"""


def _wide_view(name: str, span: str) -> str:
    """A column per metric, a row per company-period.

    Flows are filtered to the requested span and stocks to the instant on the
    same date -- an income statement measured over the year, a balance sheet
    measured at the end of it. Joining them on ``period_end`` is what makes a
    ratio like return on equity expressible at all.
    """
    cols = []
    for metric in METRICS:
        if not metric.tags:
            continue
        want = "instant" if metric.kind == "stock" else span
        cols.append(
            f"max(val) FILTER (WHERE metric = '{metric.name}' AND span = '{want}') AS {metric.name}"
        )
    return f"""
CREATE OR REPLACE VIEW {name}_raw AS
SELECT m.cik, m.ticker, y.fiscal_end AS period_end, y.fiscal_start AS period_start,
       {", ".join(cols)}
FROM metric_facts m
JOIN fiscal_years y ON y.cik = m.cik AND y.fiscal_end = m.period_end
GROUP BY m.cik, m.ticker, y.fiscal_end, y.fiscal_start;
"""


# Derived columns sit in a second layer over the raw pivot so that a derivation
# can reference a metric the same way a query would, and so the fallback is
# explicit: use the tagged value when there is one, compute it when there is not.
DERIVED_VIEW = """
CREATE OR REPLACE VIEW annual AS
SELECT * EXCLUDE (gross_profit, liabilities),
       -- Reported value where there is one, the identity where there is not.
       -- Both columns survive so a reviewer can see which came from the filing.
       gross_profit                                             AS gross_profit_tagged,
       liabilities                                              AS liabilities_tagged,
       coalesce(gross_profit, revenue - cost_of_revenue)        AS gross_profit,
       coalesce(liabilities, assets - coalesce(equity_incl_nci,
                                                     equity + coalesce(minority_interest, 0))
                                    - coalesce(temporary_equity, 0))    AS liabilities,
       operating_cash_flow - capex                              AS free_cash_flow,
       coalesce(gross_profit, revenue - cost_of_revenue) / nullif(revenue, 0)
                                                                AS gross_margin,
       operating_income / nullif(revenue, 0)                    AS operating_margin,
       net_income / nullif(revenue, 0)                          AS net_margin,
       net_income / nullif(equity, 0)                           AS return_on_equity,
       net_income / nullif(assets, 0)                           AS return_on_assets,
       coalesce(liabilities, assets - coalesce(equity_incl_nci,
                                                     equity + coalesce(minority_interest, 0))
                                    - coalesce(temporary_equity, 0))
           / nullif(equity, 0)                                  AS debt_to_equity,
       -- Growth is a window over the company's own fiscal years, ordered by
       -- period_end. Never by filed_fy: the label a filing carries is not the
       -- year the money was earned in, for 54.9% of this corpus.
       revenue / nullif(lag(revenue) OVER w, 0) - 1             AS revenue_yoy,
       net_income / nullif(lag(net_income) OVER w, 0) - 1       AS net_income_yoy,
       lag(period_end) OVER w                                   AS prior_period_end
FROM annual_raw
WINDOW w AS (PARTITION BY cik ORDER BY period_end);

-- Compound annual growth over whatever window the corpus actually holds for
-- each company, with the year count taken from the dates rather than assumed.
CREATE OR REPLACE VIEW growth AS
SELECT cik, ticker,
       min(period_end)                                       AS from_period,
       max(period_end)                                       AS to_period,
       count(*)                                              AS years,
       datediff('day', min(period_end), max(period_end)) / 365.25 AS span_years,
       arg_min(revenue, period_end)                          AS revenue_first,
       arg_max(revenue, period_end)                          AS revenue_last,
       pow(arg_max(revenue, period_end) / nullif(arg_min(revenue, period_end), 0),
           1.0 / nullif(datediff('day', min(period_end), max(period_end)) / 365.25, 0)) - 1
                                                             AS revenue_cagr,
       pow(arg_max(net_income, period_end) / nullif(arg_min(net_income, period_end), 0),
           1.0 / nullif(datediff('day', min(period_end), max(period_end)) / 365.25, 0)) - 1
                                                             AS net_income_cagr
FROM annual_raw
WHERE revenue IS NOT NULL
GROUP BY cik, ticker;
"""


@dataclass
class MetricReport:
    metrics: int = 0
    resolved: int = 0  # (company, metric) pairs with a tag
    expected: int = 0
    annual_rows: int = 0
    coverage: list[tuple] = field(default_factory=list)  # metric, n_companies, universal
    gaps: list[tuple] = field(default_factory=list)  # metric, tickers


def build_metrics(con: duckdb.DuckDBPyConnection) -> MetricReport:
    """Resolve tags per company and create the metric views. Idempotent."""
    con.execute(METRIC_TABLES)
    con.executemany(
        "INSERT INTO metric_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                m.name,
                m.label,
                m.kind,
                m.unit,
                ", ".join(m.tags),
                m.derive,
                m.universal,
                m.sign,
                m.resolve_by,
                m.note,
            )
            for m in METRICS
        ],
    )
    rows = _resolution_rows(con)
    con.executemany("INSERT INTO metric_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute(METRIC_VIEWS)
    con.execute(_wide_view("annual", "FY"))
    con.execute(DERIVED_VIEW)

    n_companies = con.execute("SELECT count(*) FROM companies").fetchone()[0]
    report = MetricReport(metrics=len(METRICS), expected=len(SOURCED) * n_companies)
    report.resolved = con.execute("SELECT count(*) FROM metric_primary_tag").fetchone()[0]
    report.annual_rows = con.execute("SELECT count(*) FROM annual").fetchone()[0]
    report.coverage = con.execute(
        """
        SELECT r.metric, count(t.cik) AS companies, r.universal,
               r.derive IS NOT NULL AS derivable
        FROM metric_registry r
        LEFT JOIN metric_primary_tag t ON t.metric = r.metric
        WHERE r.tags <> ''
        GROUP BY r.metric, r.universal, r.derive
        ORDER BY companies, r.metric
        """
    ).fetchall()
    report.gaps = con.execute(
        """
        SELECT r.metric, string_agg(c.ticker, ', ' ORDER BY c.ticker)
        FROM metric_registry r
        CROSS JOIN companies c
        LEFT JOIN metric_primary_tag t ON t.metric = r.metric AND t.cik = c.cik
        WHERE r.tags <> '' AND t.cik IS NULL
        GROUP BY r.metric
        ORDER BY r.metric
        """
    ).fetchall()
    return report


def describe(name: str) -> Metric:
    """Look a metric up by name, for the tools M4 will expose to the agent."""
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; known: {', '.join(sorted(BY_NAME))}") from None


def known_metrics() -> Sequence[str]:
    return tuple(BY_NAME)


def as_rows() -> list[dict[str, Any]]:
    """The registry as plain dicts, for docs and for the agent's tool schema."""
    return [
        {
            "metric": m.name,
            "label": m.label,
            "kind": m.kind,
            "unit": m.unit,
            "tags": list(m.tags),
            "derive": m.derive,
            "universal": m.universal,
            "sign": m.sign,
            "resolve_by": m.resolve_by,
            "note": m.note,
        }
        for m in METRICS
    ]
