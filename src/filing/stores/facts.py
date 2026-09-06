"""XBRL company facts, transcribed into SQL without interpretation.

This table is deliberately dumb. Every row is a faithful copy of one fact as
EDGAR published it -- same value, same sign, same unit -- plus the few derived
columns needed to *find* it again. Nothing is rescaled, netted, or renamed here.
Interpretation lives one layer up in ``metrics.py``, because the moment a loader
starts deciding that capex "should" be negative, the store stops being checkable
against the filing it came from.

Four properties of the source data shaped this schema, all of them measured on
the real corpus rather than assumed:

1. **``fy`` belongs to the filing, not the fact.** A 10-K filed for FY2024
   restates three prior years, and every one of those facts carries ``fy=2024``.
   Across the corpus ``fy`` disagrees with the calendar year of ``period_end``
   in 54.9% of rows. So it is stored as ``filed_fy`` -- provenance, never a key
   -- and ``period_end`` is the only anchor anything is allowed to join on.

2. **Durations and instants are different kinds of number.** 319,509 facts carry
   a ``start`` and 195,414 do not. An instant (Assets on a date) cannot be
   summed; a duration (NetIncomeLoss over a span) cannot be compared to a point
   in time. ``period_start`` is therefore nullable and ``period_type`` says
   which kind a row is, so the distinction survives into every query.

3. **The same tag carries quarters, half-years, nine-month stubs and full years
   at once.** ``NetIncomeLoss`` appears 2,499 times as a quarter, 941 as a year,
   581 as a half and 548 as three quarters. Summing a tag over a calendar year
   therefore double- or triple-counts it, and nothing in the payload warns you.
   ``span`` exists so that every query must state which shape it wants.

4. **Facts repeat, and sometimes they disagree.** A period is re-reported by up
   to seven later filings. 11,359 period-keys out of 265,535 (4.3%) carry more
   than one value -- genuine restatements, not noise. All of them are kept;
   ``facts_current`` picks the latest and ``restatements`` lists the rest, so a
   changed number is a queryable event instead of a silent overwrite.

The source of rows is the manifest, never a directory listing. ``data/raw/
companyfacts/`` currently holds 21 payloads for a 20-company universe -- the
extra one is the holding-company CIK that XOM's ticker used to resolve to. A
loader that globbed the directory would ingest it and half-duplicate Exxon.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from filing.ingest.manifest import Manifest
from filing.stores.metrics import MetricReport, build_metrics

log = logging.getLogger(__name__)

# Canonical reporting spans, in days. A fact is labelled with the nearest one it
# lands within TOLERANCE of; anything else is 'other' and stays out of the
# metric views. Nearest-canonical rather than fixed ranges because retailers on
# 4-4-5 calendars land on 168 and 252 days -- a half and a nine-month stub that
# hard-coded 170-195 / 260-285 windows would both have thrown away.
CANONICAL_SPANS: dict[str, int] = {"Q": 91, "H": 182, "9M": 273, "FY": 365}
SPAN_TOLERANCE_DAYS = 25

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    cik           VARCHAR NOT NULL,
    ticker        VARCHAR NOT NULL,
    taxonomy      VARCHAR NOT NULL,   -- us-gaap, dei, srt, ecd, ffd, invest, rxp
    tag           VARCHAR NOT NULL,
    unit          VARCHAR NOT NULL,   -- USD, shares, USD/shares, pure, and junk
    period_start  DATE,               -- NULL for instants, and only for instants
    period_end    DATE NOT NULL,      -- the only unambiguous anchor in the payload
    period_type   VARCHAR NOT NULL,   -- 'duration' | 'instant'
    period_days   INTEGER,            -- inclusive; NULL for instants
    span          VARCHAR NOT NULL,   -- 'instant' | 'Q' | 'H' | '9M' | 'FY' | 'other'
    val           DOUBLE NOT NULL,    -- verbatim, sign included
    accn          VARCHAR NOT NULL,   -- the filing that reported it
    form          VARCHAR,
    filed         DATE NOT NULL,
    frame         VARCHAR,            -- SEC's own comparability key, when it assigns one
    filed_fy      INTEGER,            -- the FILING's fiscal year. Never a period key.
    filed_fp      VARCHAR             -- likewise: FY/Q1/Q2/Q3, of the filing
    -- No UNIQUE constraint on the identity tuple, deliberately. SQL treats
    -- NULLs as distinct and period_start is NULL for all 195,414 instants, so
    -- the declaration would silently exempt 38% of the table while reading as
    -- though it covered it. A constraint that holds for some rows is worse than
    -- none: it moves the check out of sight. duplicate_keys() below does the
    -- real assertion with GROUP BY, which does treat NULLs as equal, and the
    -- gate runs it on every build.
);

-- Tag labels run 60-200 characters and repeat across every fact using the tag.
-- Held once here instead of 515k times in facts.
CREATE TABLE IF NOT EXISTS concepts (
    taxonomy     VARCHAR NOT NULL,
    tag          VARCHAR NOT NULL,
    label        VARCHAR,
    description  VARCHAR,
    PRIMARY KEY (taxonomy, tag)
);
"""

# facts_current: the latest report of every period, which is what "what were
# revenues in 2023" almost always means. ORDER BY accn after filed only so the
# result is stable when two filings land the same day; 3 keys corpus-wide need it.
VIEWS = """
CREATE OR REPLACE VIEW facts_current AS
SELECT * EXCLUDE (_rn) FROM (
    SELECT *, row_number() OVER (
        PARTITION BY cik, taxonomy, tag, unit, period_start, period_end
        ORDER BY filed DESC, accn DESC) AS _rn
    FROM facts
) WHERE _rn = 1;

-- Periods a company reported more than one value for. A first-class view
-- because "this number changed after the fact" is a finding, not an error.
CREATE OR REPLACE VIEW restatements AS
SELECT cik, ticker, taxonomy, tag, unit, period_start, period_end, span,
       count(DISTINCT val)                       AS n_values,
       min(filed)                                AS first_filed,
       max(filed)                                AS last_filed,
       arg_min(val, filed)                       AS first_val,
       arg_max(val, filed)                       AS last_val,
       arg_max(val, filed) - arg_min(val, filed) AS delta
FROM facts
GROUP BY ALL
HAVING count(DISTINCT val) > 1;
"""


def classify_span(days: int | None) -> str:
    """Label a duration by the reporting period it is nearest to."""
    if days is None:
        return "instant"
    label, best = "other", SPAN_TOLERANCE_DAYS
    for name, canonical in CANONICAL_SPANS.items():
        gap = abs(days - canonical)
        if gap <= best:
            label, best = name, gap
    return label


def _csv_row(row: tuple) -> list[str]:
    """Serialise for the staging file, distinguishing NULL from empty string."""
    return [NULL_SENTINEL if v is None else str(v) for v in row]


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:  # EDGAR has shipped malformed dates before; skip, don't crash
        return None


@dataclass
class BuildReport:
    companies: int = 0
    facts: int = 0
    concepts: int = 0
    skipped_no_date: int = 0
    skipped_no_value: int = 0
    by_span: dict[str, int] = field(default_factory=dict)
    by_taxonomy: dict[str, int] = field(default_factory=dict)
    restatements: int = 0
    duplicate_keys: int = 0
    metrics: MetricReport | None = None
    elapsed_s: float = 0.0


SKIP_DATE = "__skip_date__"
SKIP_VALUE = "__skip_value__"


def iter_facts(payload: dict[str, Any], cik: str, ticker: str) -> Iterator[tuple]:
    """Flatten one companyfacts payload into rows, dropping only unusable ones."""
    for taxonomy, tags in payload.get("facts", {}).items():
        for tag, body in tags.items():
            for unit, entries in body.get("units", {}).items():
                for fact in entries:
                    end = _as_date(fact.get("end"))
                    filed = _as_date(fact.get("filed"))
                    accn = fact.get("accn")
                    val = fact.get("val")
                    if end is None or filed is None or accn is None:
                        yield (SKIP_DATE,)
                        continue
                    if val is None or isinstance(val, str):
                        # A few facts are text (entity names in dei, say).
                        # A DOUBLE column is the wrong home for those.
                        yield (SKIP_VALUE,)
                        continue
                    start = _as_date(fact.get("start"))
                    days = (end - start).days + 1 if start else None
                    yield (
                        cik,
                        ticker,
                        taxonomy,
                        tag,
                        unit,
                        start,
                        end,
                        "duration" if start else "instant",
                        days,
                        classify_span(days),
                        float(val),
                        accn,
                        fact.get("form"),
                        filed,
                        fact.get("frame"),
                        fact.get("fy"),
                        fact.get("fp"),
                    )


# Rows reach DuckDB through a CSV staging file rather than executemany, which
# is not a micro-optimisation: measured on this corpus, executemany inserts
# about 380 rows a second, so the 514,649-row load would take 22 minutes. The
# Python binding runs one prepared statement per row; DuckDB's CSV reader is
# vectorised C++. Same rows, same types, two orders of magnitude apart. \N is
# the null sentinel because an empty CSV field is ambiguous for VARCHAR columns.
NULL_SENTINEL = "\\N"

COPY_FACTS = """
COPY facts FROM '{path}' (FORMAT CSV, HEADER false, NULLSTR '{null}')
"""


def build_facts(
    cfg: Any,
    *,
    progress: Callable[[str, str], None] | None = None,
) -> BuildReport:
    """Rebuild ``data/facts.duckdb`` from the payloads the manifest knows about."""
    started = datetime.now()
    report = BuildReport()

    with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
        rows = manifest.con.execute(
            "SELECT cik, ticker, path FROM companyfacts ORDER BY ticker"
        ).fetchall()
        companies = manifest.con.execute("SELECT * FROM companies").fetchall()
        company_cols = [d[0] for d in manifest.con.description]
        # ``path`` comes along so the store can find the document a fact was
        # reported in. That is what makes verify.py able to check a stored
        # number against the filing it came from rather than against itself.
        filings = manifest.con.execute(
            "SELECT accn, cik, ticker, form, filed_date, period_end, path FROM filings"
        ).fetchall()
        paths = {cik: manifest.resolve(path) for cik, _ticker, path in rows}

    facts_path: Path = cfg.facts_path
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    # A rebuild is a rebuild. Reusing the file would leave rows for a company
    # the universe has since dropped -- the exact failure the manifest's prune()
    # exists to prevent one layer down.
    facts_path.unlink(missing_ok=True)
    con = duckdb.connect(str(facts_path))
    con.execute(SCHEMA)

    # Dimension tables are copied, not attached: the point of a separate
    # database is that it answers questions without the manifest present.
    con.execute(f"CREATE TABLE companies ({', '.join(c + ' VARCHAR' for c in company_cols)})")
    con.execute(
        "CREATE TABLE filings (accn VARCHAR PRIMARY KEY, cik VARCHAR, ticker VARCHAR, "
        "form VARCHAR, filed_date DATE, period_end DATE, path VARCHAR)"
    )
    # Guarded because executemany rejects an empty list rather than doing
    # nothing. Facts and filings are fetched independently, so a manifest that
    # has companyfacts and no filings yet is a real state, not a broken one:
    # the store still answers every numeric question, and only verify.py --
    # which needs a document to read a number back out of -- has nothing to do.
    if companies:
        con.executemany(
            f"INSERT INTO companies VALUES ({', '.join(['?'] * len(company_cols))})", companies
        )
    if filings:
        con.executemany("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?)", filings)

    concepts: dict[tuple[str, str], tuple] = {}
    with tempfile.TemporaryDirectory(prefix="filing-facts-") as tmp:
        facts_csv = Path(tmp) / "facts.csv"
        concepts_csv = Path(tmp) / "concepts.csv"

        with facts_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for cik, ticker, _path in rows:
                payload = json.loads(paths[cik].read_text(encoding="utf-8"))
                written = 0
                for row in iter_facts(payload, cik, ticker):
                    if row[0] == SKIP_DATE:
                        report.skipped_no_date += 1
                    elif row[0] == SKIP_VALUE:
                        report.skipped_no_value += 1
                    else:
                        writer.writerow(_csv_row(row))
                        written += 1
                for taxonomy, tags in payload.get("facts", {}).items():
                    for tag, body in tags.items():
                        concepts.setdefault(
                            (taxonomy, tag),
                            (taxonomy, tag, body.get("label"), body.get("description")),
                        )
                report.companies += 1
                report.facts += written
                if progress:
                    progress(ticker, f"{written:,} facts")

        with concepts_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for concept in concepts.values():
                writer.writerow(_csv_row(concept))

        con.execute(COPY_FACTS.format(path=facts_csv.as_posix(), null=NULL_SENTINEL))
        con.execute(
            f"COPY concepts FROM '{concepts_csv.as_posix()}' "
            f"(FORMAT CSV, HEADER false, NULLSTR '{NULL_SENTINEL}')"
        )

    report.concepts = len(concepts)
    con.execute(VIEWS)
    # The metric layer is part of the store, not a separate artefact: a facts
    # database without it answers "what is tagged Assets" and not "what were
    # total assets", and only the second is a question anyone asks.
    report.metrics = build_metrics(con)

    report.by_span = dict(
        con.execute("SELECT span, count(*) FROM facts GROUP BY 1 ORDER BY 2 DESC").fetchall()
    )
    report.by_taxonomy = dict(
        con.execute("SELECT taxonomy, count(*) FROM facts GROUP BY 1 ORDER BY 2 DESC").fetchall()
    )
    report.restatements = con.execute("SELECT count(*) FROM restatements").fetchone()[0]
    report.duplicate_keys = duplicate_keys(con)
    con.close()

    report.elapsed_s = (datetime.now() - started).total_seconds()
    return report


def duplicate_keys(con: duckdb.DuckDBPyConnection) -> int:
    """Rows sharing a full identity key, which must never happen.

    The table declares this tuple UNIQUE, but that declaration is not the check.
    SQL treats NULLs as distinct, and ``period_start`` is NULL for all 195,414
    instants -- so the constraint quietly stops applying to 38% of the table.
    ``GROUP BY`` does treat NULLs as equal, so this is the assertion that
    actually holds, and the gate runs it rather than trusting the DDL.
    """
    return con.execute(
        """
        SELECT count(*) FROM (
            SELECT 1 FROM facts
            GROUP BY cik, taxonomy, tag, unit, period_start, period_end, accn
            HAVING count(*) > 1)
        """
    ).fetchone()[0]


class FactsStore:
    """Read-only handle on the built store."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"no facts store at {path} -- run `filing facts`")
        self.path = path
        self.con = duckdb.connect(str(path), read_only=True)

    def __enter__(self) -> FactsStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def sql(self, query: str, params: list[Any] | None = None) -> list[tuple]:
        return self.con.execute(query, params or []).fetchall()
