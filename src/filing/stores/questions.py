"""Ten questions the store must answer without a language model.

This is the M2 gate's honesty check. Everything downstream -- retrieval,
routing, synthesis -- is easier to make *look* right than to make right, because
a fluent model will produce a confident paragraph from a bad number as readily
as from a good one. So before any model is allowed near this data, the numeric
questions are answered by SQL alone, and the answers are checked against values
read by hand out of the filings.

The ten are chosen to exercise different machinery, not to be ten of the same
question. Between them they cover a point lookup, a ratio, a cross-company
ranking, a time series, a compound growth rate, a derived metric, a window
function over consecutive years, an accounting identity, the restatement view,
and -- the one most systems get wrong -- a question whose correct answer is
"this company does not report that."

``expect`` is a number obtained from outside this pipeline: read off the face of
the filing, or computed with a calculator from two numbers that were. Where a
question returns a set rather than a scalar there is no ``expect`` and the
assertion is on the shape of the result, which is stated in ``holds``. A
question with neither is not a check and does not belong here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb

# Filings round to the cent and the store carries full precision, so an exact
# comparison would fail on values that are in fact identical. A relative
# tolerance is the honest test; it is tight enough that a scale error, a wrong
# period, or a wrong tag cannot pass through it.
TOLERANCE = 0.005


@dataclass(frozen=True)
class Question:
    """One question, its SQL, and what makes the answer right."""

    id: str
    text: str
    sql: str
    # A scalar read from the filing itself, in the store's own units.
    expect: float | None = None
    unit: str = ""
    # For set-valued answers: a predicate over the returned rows, plus prose
    # saying what it asserts, so a reader knows what passing means.
    holds: Callable[[list[tuple]], bool] | None = None
    claim: str = ""
    source: str = ""

    @property
    def kind(self) -> str:
        return "scalar" if self.expect is not None else "set"


@dataclass
class Answer:
    question: Question
    rows: list[tuple]
    columns: list[str]
    passed: bool
    got: float | None = None
    detail: str = ""

    @property
    def scalar(self) -> Any:
        return self.rows[0][0] if self.rows and self.rows[0] else None


@dataclass
class QuestionReport:
    answers: list[Answer] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.answers)

    @property
    def passed(self) -> int:
        return sum(1 for a in self.answers if a.passed)

    @property
    def failures(self) -> list[Answer]:
        return [a for a in self.answers if not a.passed]


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="q01-point-lookup",
        text="What was NVIDIA's revenue in fiscal 2026?",
        sql="""
        SELECT revenue FROM annual
        WHERE ticker = 'NVDA' AND period_end = DATE '2026-01-25'
        """,
        expect=215_938_000_000,
        unit="USD",
        source="NVDA FY2026 10-K, consolidated statements of income",
    ),
    Question(
        id="q02-ratio",
        text="What was NVIDIA's gross margin in fiscal 2026?",
        sql="""
        SELECT round(gross_margin * 100, 1) FROM annual
        WHERE ticker = 'NVDA' AND period_end = DATE '2026-01-25'
        """,
        # 215,938 - 62,401 = 153,537; 153,537 / 215,938 = 71.10%. Computed from
        # two numbers on the face of the income statement, not from this store.
        expect=71.1,
        unit="percent",
        source="NVDA FY2026 10-K: revenue 215,938 less cost of revenue 62,401",
    ),
    Question(
        id="q03-ranking",
        text="Which five companies had the largest revenue in fiscal 2024?",
        sql="""
        SELECT ticker, revenue FROM annual
        WHERE period_end BETWEEN DATE '2024-06-01' AND DATE '2025-05-31'
          AND revenue IS NOT NULL
        ORDER BY revenue DESC LIMIT 5
        """,
        holds=lambda rows: (
            len(rows) == 5
            and [r[0] for r in rows][:3] == ["WMT", "XOM", "COST"]
            and all(rows[i][1] > rows[i + 1][1] for i in range(4))
        ),
        claim="Walmart, Exxon and Costco lead, in that order, strictly descending",
        source="WMT $681.0B, XOM $349.6B, COST $254.5B. Costco outsells Chevron "
        "($202.8B) by fifty billion dollars, which is not what a sector-shaped "
        "guess predicts -- the ranking is answered from the data, not from priors.",
    ),
    Question(
        id="q04-time-series",
        text="What was Eli Lilly's revenue in each of the last five fiscal years?",
        sql="""
        SELECT extract(year FROM period_end) AS fy, revenue FROM annual
        WHERE ticker = 'LLY' AND revenue IS NOT NULL
        ORDER BY period_end DESC LIMIT 5
        """,
        holds=lambda rows: (
            len(rows) == 5
            # Lilly grew in every one of these years -- the series must be
            # strictly decreasing as it walks backwards in time.
            and all(rows[i][1] > rows[i + 1][1] for i in range(4))
            and rows[-1][1] > 20e9
        ),
        claim="five consecutive years, revenue rising in every one of them",
        source="LLY 10-Ks: growth in each year of the window, from >$20B",
    ),
    Question(
        id="q05-cagr",
        text="What compound annual revenue growth did AMD achieve over the corpus window?",
        sql="""
        SELECT round(revenue_cagr * 100, 1), span_years FROM growth WHERE ticker = 'AMD'
        """,
        holds=lambda rows: len(rows) == 1 and rows[0][1] >= 15 and 9 <= rows[0][0] <= 13,
        claim="fifteen years or more of history, compounding at 9-13% a year",
        source="$5.808B in fiscal 2008 to $34.639B in fiscal 2025 is 5.96x over "
        "17.0 years; 5.96^(1/17) = 1.111. The span is measured from the dates the "
        "store holds, not assumed to be a round number of years.",
    ),
    Question(
        id="q06-cross-company-ratio",
        text="What share of revenue did each semiconductor name spend on R&D in fiscal 2024?",
        sql="""
        SELECT ticker, round(rnd / revenue * 100, 1) AS rnd_pct FROM annual
        WHERE ticker IN ('NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM')
          AND period_end BETWEEN DATE '2024-06-01' AND DATE '2025-05-31'
          AND rnd IS NOT NULL AND revenue IS NOT NULL
        ORDER BY rnd_pct DESC
        """,
        holds=lambda rows: (
            len(rows) == 5
            # Intel spent more on R&D than it earned in margin that year; NVIDIA,
            # on far larger revenue, spent the least. Any ordering that does not
            # put those two at the ends is reading the wrong periods.
            and rows[0][0] == "INTC"
            and rows[-1][0] == "NVDA"
            and all(2 < r[1] < 60 for r in rows)
        ),
        claim="all five report R&D, Intel highest and NVIDIA lowest as a share of revenue",
        source="INTC FY2024 R&D $16.5B on $53.1B revenue; NVDA FY2025 $12.9B on $130.5B",
    ),
    Question(
        id="q07-derived",
        text="What was Exxon Mobil's free cash flow in fiscal 2024?",
        sql="""
        SELECT round(free_cash_flow / 1e9, 1), round(operating_cash_flow / 1e9, 1),
               round(capex / 1e9, 1)
        FROM annual WHERE ticker = 'XOM' AND period_end = DATE '2024-12-31'
        """,
        holds=lambda rows: (
            len(rows) == 1
            # The derivation must be exactly the subtraction it claims to be,
            # and capex must have been subtracted rather than added -- the sign
            # trap this metric exists to document.
            and abs(rows[0][0] - (rows[0][1] - rows[0][2])) < 0.15
            and rows[0][0] < rows[0][1]
            and rows[0][2] > 0
        ),
        claim="free cash flow equals operating cash flow less a positive capex",
        source="XOM 2024 10-K statement of cash flows",
    ),
    Question(
        id="q08-window",
        text="In which fiscal years did Target's revenue fall against the prior year?",
        sql="""
        SELECT extract(year FROM period_end), round(revenue_yoy * 100, 1)
        FROM annual
        WHERE ticker = 'TGT' AND revenue_yoy IS NOT NULL AND revenue_yoy < 0
        ORDER BY period_end
        """,
        holds=lambda rows: (
            # Five, and the three consecutive ones at the end must be the last
            # three fiscal years in the series. A store that mis-joined
            # consecutive years would break the run, not just the count.
            [int(r[0]) for r in rows] == [2014, 2017, 2024, 2025, 2026]
            and all(-6 <= r[1] < 0 for r in rows)
        ),
        claim="five down years: 2014, 2017, and an unbroken run of 2024-2026",
        source="TGT 10-Ks -- 2014 is the Canadian expansion, 2017 the pharmacy "
        "divestiture, and 2024-2026 the unwind of the +19.8% pandemic year. The "
        "run of three is the point: a plausible guess is one post-pandemic dip.",
    ),
    Question(
        id="q09-identity",
        text="Does the balance sheet balance -- do assets equal liabilities plus equity?",
        # Grouped by accession, not by period, and read from `facts` rather than
        # `facts_current`. That is the whole substance of this question.
        #
        # A balance sheet is internally consistent within one filing. It is not
        # consistent across filings, because a restatement revises a set of
        # lines together and a company only re-tags the years it still presents.
        # Costco's fiscal 2016 10-K restated fiscal 2014 assets from $33,024M to
        # $32,662M and never re-tagged that year's liabilities, so taking the
        # newest value of each line -- correct per fact -- assembles a balance
        # sheet from two filings that is out by $362M and balances in neither.
        # "Latest wins" is right for a fact and wrong for a statement.
        #
        # The right-hand side is also the whole right-hand side: total equity
        # rather than the parent's share, the pre-2009 minority interest line,
        # and the mezzanine line that belongs to neither side. Drop any one of
        # those three and this returns violations that are the filings being
        # right and the query being wrong.
        sql="""
        WITH balance_sheet AS (
            SELECT ticker, accn, period_end,
                   max(val) FILTER (WHERE tag = 'Assets')      AS assets,
                   max(val) FILTER (WHERE tag = 'Liabilities') AS liabilities,
                   coalesce(
                       max(val) FILTER (WHERE tag =
                           'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
                       max(val) FILTER (WHERE tag = 'StockholdersEquity')
                         + coalesce(max(val) FILTER (WHERE tag = 'MinorityInterest'), 0)
                   ) AS equity,
                   coalesce(max(val) FILTER (WHERE tag LIKE 'TemporaryEquity%'), 0)
                       AS temporary_equity
            FROM facts
            WHERE taxonomy = 'us-gaap' AND unit = 'USD' AND span = 'instant'
            GROUP BY ALL
        )
        SELECT count(*) FROM balance_sheet
        WHERE assets IS NOT NULL AND liabilities IS NOT NULL AND equity IS NOT NULL
          AND abs(assets - (liabilities + equity + temporary_equity)) > 0.005 * assets
        """,
        expect=0,
        unit="balance sheets violating the identity, of 1,717 checked",
        source="the accounting identity itself, applied one filing at a time",
    ),
    Question(
        id="q10-absence",
        text="Which companies in the corpus report no research and development at all?",
        sql="""
        SELECT c.ticker FROM companies c
        WHERE NOT EXISTS (
            SELECT 1 FROM metric_facts m WHERE m.cik = c.cik AND m.metric = 'rnd'
        )
        ORDER BY c.ticker
        """,
        holds=lambda rows: {r[0] for r in rows} == {"WMT", "COST", "HD", "LOW", "TGT"},
        claim="exactly the five retailers, and no pharma or semiconductor name",
        source="Walmart, Costco, Home Depot, Lowe's and Target run no R&D "
        "function. Any pharma name appearing here is a missing tag rather than "
        "a missing budget -- which is how AbbVie and Pfizer were caught.",
    ),
)


#: Three companies with three different income-statement and balance-sheet
#: shapes, so the recomputation exercises every branch of the derivations rather
#: than the same branch three times. NVIDIA tags gross profit and has no
#: noncontrolling interests; Exxon tags neither gross profit nor cost of revenue
#: and carries NCI; Walmart tags no total liabilities and no R&D at all.
CHECK_COMPANIES = ("NVDA", "XOM", "WMT")

#: The base metrics every derivation is written in terms of. Pulled one level
#: below the views under test.
BASE_METRICS = (
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capex",
    "assets",
    "liabilities",
    "equity",
    "equity_incl_nci",
    "minority_interest",
    "temporary_equity",
)

BASE_SQL = """
SELECT ticker, period_end, metric, max(val) AS val
FROM metric_facts
WHERE ticker IN ({tickers})
  AND metric IN ({metrics})
  AND span = CASE WHEN metric IN ('assets', 'liabilities', 'equity',
                                  'equity_incl_nci', 'minority_interest',
                                  'temporary_equity')
                  THEN 'instant' ELSE 'FY' END
GROUP BY ALL
"""


def _derive(
    row: dict[str, float | None], prior: dict[str, float | None]
) -> dict[str, float | None]:
    """Every derived column, recomputed in Python from the base metrics.

    Written out longhand rather than by evaluating the registry's ``derive``
    strings, which would only prove the SQL agrees with itself. The point is
    that a second person, reading the definitions off the registry's prose,
    arrives at the same numbers -- so this is what that person would type.
    """

    def get(name: str) -> float | None:
        return row.get(name)

    def div(a: float | None, b: float | None) -> float | None:
        return None if a is None or not b else a / b

    def minus(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    revenue, assets, equity = get("revenue"), get("assets"), get("equity")
    net_income, cost = get("net_income"), get("cost_of_revenue")

    # Total equity is the parent's share plus whatever sits outside it. The
    # 2009 filings spell that MinorityInterest; later ones fold it into the
    # including-NCI tag, which is used whole when present.
    total_equity = get("equity_incl_nci")
    if total_equity is None and equity is not None:
        total_equity = equity + (get("minority_interest") or 0.0)

    gross_profit = get("gross_profit")
    if gross_profit is None:
        gross_profit = minus(revenue, cost)

    liabilities = get("liabilities")
    if liabilities is None and assets is not None and total_equity is not None:
        liabilities = assets - total_equity - (get("temporary_equity") or 0.0)

    return {
        "gross_profit": gross_profit,
        "liabilities": liabilities,
        "free_cash_flow": minus(get("operating_cash_flow"), get("capex")),
        "gross_margin": div(gross_profit, revenue),
        "operating_margin": div(get("operating_income"), revenue),
        "net_margin": div(net_income, revenue),
        "return_on_equity": div(net_income, equity),
        "return_on_assets": div(net_income, assets),
        "debt_to_equity": div(liabilities, equity),
        "revenue_yoy": (
            None if revenue is None or not prior.get("revenue") else revenue / prior["revenue"] - 1
        ),
    }


@dataclass
class Mismatch:
    ticker: str
    period_end: Any
    column: str
    stored: float | None
    recomputed: float | None


@dataclass
class DerivationReport:
    companies: tuple[str, ...]
    rows_checked: int = 0
    values_checked: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.rows_checked > 0


def check_derivations(
    con: duckdb.DuckDBPyConnection, companies: tuple[str, ...] = CHECK_COMPANIES
) -> DerivationReport:
    """Recompute every derived column in Python and compare with the view.

    The two paths share the fact table and nothing above it: this reads
    ``metric_facts``, the SQL reads ``annual_raw`` and ``annual``. A pivot that
    picked the wrong span, a coalesce that fired in the wrong order, or a window
    function partitioned by the wrong key shows up here as a disagreement.
    """
    tickers = ", ".join(f"'{t}'" for t in companies)
    metrics = ", ".join(f"'{m}'" for m in BASE_METRICS)
    base: dict[tuple[str, Any], dict[str, float | None]] = {}
    for ticker, period_end, metric, val in con.execute(
        BASE_SQL.format(tickers=tickers, metrics=metrics)
    ).fetchall():
        base.setdefault((ticker, period_end), {})[metric] = val

    stored = con.execute(
        f"SELECT * FROM annual WHERE ticker IN ({tickers}) ORDER BY ticker, period_end"
    )
    columns = [d[0] for d in stored.description or []]
    report = DerivationReport(companies)

    prior_by_ticker: dict[str, dict[str, float | None]] = {}
    for values in stored.fetchall():
        row = dict(zip(columns, values, strict=True))
        ticker, period_end = row["ticker"], row["period_end"]
        # Only the fiscal years the pivot actually built, so a company's first
        # year is checked for everything except year-over-year growth.
        recomputed = _derive(base.get((ticker, period_end), {}), prior_by_ticker.get(ticker, {}))
        prior_by_ticker[ticker] = base.get((ticker, period_end), {})
        report.rows_checked += 1
        for column, expected in recomputed.items():
            got = row[column]
            report.values_checked += 1
            if expected is None and got is None:
                continue
            if expected is None or got is None:
                report.mismatches.append(Mismatch(ticker, period_end, column, got, expected))
                continue
            limit = max(abs(expected) * 1e-9, 1e-9)
            if abs(got - expected) > limit:
                report.mismatches.append(Mismatch(ticker, period_end, column, got, expected))
    return report


def answer(con: duckdb.DuckDBPyConnection, question: Question) -> Answer:
    """Run one question and judge it."""
    cur = con.execute(question.sql)
    columns = [d[0] for d in cur.description or []]
    rows = cur.fetchall()

    if question.expect is not None:
        got = float(rows[0][0]) if rows and rows[0][0] is not None else None
        if got is None:
            return Answer(question, rows, columns, False, None, "no value returned")
        # Relative for magnitudes, absolute for counts and percentages, so a
        # question expecting zero is not asked to be within 0.5% of zero.
        limit = max(abs(question.expect) * TOLERANCE, 0.05)
        ok = abs(got - question.expect) <= limit
        detail = "" if ok else f"expected {question.expect:,.4g}, got {got:,.4g}"
        return Answer(question, rows, columns, ok, got, detail)

    if question.holds is None:  # pragma: no cover - guarded by test_questions
        raise ValueError(f"{question.id} asserts nothing")
    ok = bool(rows) and question.holds(rows)
    detail = "" if ok else f"{len(rows)} row(s) did not satisfy: {question.claim}"
    return Answer(question, rows, columns, ok, None, detail)


def run_questions(con: duckdb.DuckDBPyConnection) -> QuestionReport:
    """Answer all ten. No model is consulted and none is available here."""
    return QuestionReport([answer(con, q) for q in QUESTIONS])
