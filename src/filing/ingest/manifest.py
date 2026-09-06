"""The manifest: what the corpus contains, in SQL.

The manifest is the source of truth about the corpus -- not the filesystem.
That distinction is the whole idempotency story. A directory listing cannot
tell a finished download from one that died halfway through and left a
truncated file; a manifest row written only after the bytes are on disk can.
So the resume logic asks this table what exists, and the disk is checked only
to catch the case where a file has since been deleted underneath it.

Paths are stored relative to ``data_dir``. An absolute path would make the
manifest break the moment the project is moved or cloned somewhere else, which
for a portfolio repo is the common case, not the exotic one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik              VARCHAR PRIMARY KEY,
    ticker           VARCHAR NOT NULL,
    name             VARCHAR,
    sector           VARCHAR,
    registrant       VARCHAR,   -- EDGAR's own name for the entity
    fiscal_year_end  VARCHAR,   -- MMDD, straight from submissions
    sic              VARCHAR,
    sic_description  VARCHAR
);

CREATE TABLE IF NOT EXISTS filings (
    accn         VARCHAR PRIMARY KEY,   -- unique per submission, so a natural key
    cik          VARCHAR NOT NULL,
    ticker       VARCHAR NOT NULL,
    form         VARCHAR NOT NULL,
    fy           INTEGER,               -- see note in upsert_filing: a filter aid, not a label
    filed_date   DATE,
    period_end   DATE,
    primary_doc  VARCHAR,
    path         VARCHAR NOT NULL,      -- relative to data_dir
    bytes        BIGINT,
    sha256       VARCHAR,
    fetched_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS companyfacts (
    cik         VARCHAR PRIMARY KEY,
    ticker      VARCHAR NOT NULL,
    path        VARCHAR NOT NULL,
    bytes       BIGINT,
    sha256      VARCHAR,
    n_concepts  INTEGER,                -- distinct us-gaap/dei tags in the payload
    fetched_at  TIMESTAMP
);
"""


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class Stats:
    companies: int
    filings: int
    by_form: dict[str, int]
    companies_with_filings: int
    companyfacts: int
    filing_bytes: int
    facts_bytes: int


class Manifest:
    def __init__(self, path: Path, data_dir: Path) -> None:
        self.path = path
        self.data_dir = data_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def __enter__(self) -> Manifest:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------ paths

    def resolve(self, rel: str) -> Path:
        return self.data_dir / rel

    def relative(self, absolute: Path) -> str:
        return absolute.relative_to(self.data_dir).as_posix()

    # ----------------------------------------------------------------- writes

    def upsert_company(self, **row: Any) -> None:
        self.con.execute(
            """
            INSERT INTO companies
                (cik, ticker, name, sector, registrant, fiscal_year_end, sic, sic_description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cik) DO UPDATE SET
                ticker = excluded.ticker, name = excluded.name, sector = excluded.sector,
                registrant = excluded.registrant, fiscal_year_end = excluded.fiscal_year_end,
                sic = excluded.sic, sic_description = excluded.sic_description
            """,
            [
                row["cik"],
                row["ticker"],
                row.get("name"),
                row.get("sector"),
                row.get("registrant"),
                row.get("fiscal_year_end"),
                row.get("sic"),
                row.get("sic_description"),
            ],
        )

    def upsert_filing(self, **row: Any) -> None:
        """Record a filing that is already on disk.

        ``fy`` is the calendar year of the period end and nothing more. It is
        deliberately *not* the company's own fiscal-year label, because the
        companies disagree about that: the year ending 2024-01-31 is Walmart's
        FY2024 and the year ending 2024-02-03 is Target's FY2023. Treating a
        derived label as authoritative here would push that ambiguity into
        every downstream comparison. The authoritative ``fy``/``fp`` pair
        arrives with the XBRL facts in M2; this column is a filter convenience
        until then.
        """
        self.con.execute(
            """
            INSERT INTO filings
                (accn, cik, ticker, form, fy, filed_date, period_end,
                 primary_doc, path, bytes, sha256, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (accn) DO UPDATE SET
                path = excluded.path, bytes = excluded.bytes,
                sha256 = excluded.sha256, fetched_at = excluded.fetched_at
            """,
            [
                row["accn"],
                row["cik"],
                row["ticker"],
                row["form"],
                row.get("fy"),
                row.get("filed_date") or None,
                row.get("period_end") or None,
                row.get("primary_doc"),
                row["path"],
                row.get("bytes"),
                row.get("sha256"),
                datetime.now(UTC),
            ],
        )

    def upsert_facts(self, **row: Any) -> None:
        self.con.execute(
            """
            INSERT INTO companyfacts
                (cik, ticker, path, bytes, sha256, n_concepts, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cik) DO UPDATE SET
                path = excluded.path, bytes = excluded.bytes, sha256 = excluded.sha256,
                n_concepts = excluded.n_concepts, fetched_at = excluded.fetched_at
            """,
            [
                row["cik"],
                row["ticker"],
                row["path"],
                row.get("bytes"),
                row.get("sha256"),
                row.get("n_concepts"),
                datetime.now(UTC),
            ],
        )

    # ------------------------------------------------------------------ reads

    def has_filing(self, accn: str) -> bool:
        """True only if the manifest knows it *and* the bytes are still there.

        Two conditions, because they fail in different ways and both are real:
        a missing row means the download never finished, a missing file means
        someone cleaned out data/. Either way the answer is the same -- fetch
        it again -- and the caller does not need to care which happened.
        """
        row = self.con.execute("SELECT path FROM filings WHERE accn = ?", [accn]).fetchone()
        return bool(row) and self.resolve(row[0]).exists()

    def has_facts(self, cik: str) -> bool:
        row = self.con.execute("SELECT path FROM companyfacts WHERE cik = ?", [cik]).fetchone()
        return bool(row) and self.resolve(row[0]).exists()

    def prune(self, keep_ciks: set[str]) -> int:
        """Drop rows for CIKs the universe no longer declares.

        Needed because a company can be *repointed*, not just added or removed:
        XOM's rows were written under the CIK the ticker map returned, and
        pinning the real filing entity in universe.yaml gives it a different
        one. Without this the manifest would carry both and report 21 companies
        for a 20-company universe.

        Rows only. The files stay on disk -- deleting bytes on the strength of
        an edit to a YAML file is not a trade this makes, and an orphaned file
        costs disk space where a wrongly deleted one costs a re-download.
        """
        marks = ", ".join("?" for _ in keep_ciks) or "NULL"
        removed = 0
        for table in ("filings", "companyfacts", "companies"):
            before = self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            self.con.execute(f"DELETE FROM {table} WHERE cik NOT IN ({marks})", list(keep_ciks))
            removed += before - self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        return removed

    def missing_files(self) -> list[tuple[str, str]]:
        """Manifest rows whose file is no longer on disk. The gate checks this."""
        rows = self.con.execute("SELECT accn, path FROM filings").fetchall()
        gone = [(accn, path) for accn, path in rows if not self.resolve(path).exists()]
        facts = self.con.execute("SELECT cik, path FROM companyfacts").fetchall()
        gone += [(cik, path) for cik, path in facts if not self.resolve(path).exists()]
        return gone

    def stats(self) -> Stats:
        one = lambda sql: self.con.execute(sql).fetchone()[0]  # noqa: E731
        by_form = dict(
            self.con.execute(
                "SELECT form, count(*) FROM filings GROUP BY form ORDER BY form"
            ).fetchall()
        )
        return Stats(
            companies=one("SELECT count(*) FROM companies"),
            filings=one("SELECT count(*) FROM filings"),
            by_form=by_form,
            companies_with_filings=one("SELECT count(DISTINCT cik) FROM filings"),
            companyfacts=one("SELECT count(*) FROM companyfacts"),
            filing_bytes=one("SELECT coalesce(sum(bytes), 0) FROM filings"),
            facts_bytes=one("SELECT coalesce(sum(bytes), 0) FROM companyfacts"),
        )

    def per_company(self) -> list[tuple]:
        return self.con.execute(
            """
            SELECT c.sector, c.ticker, c.cik,
                   count(f.accn) FILTER (WHERE f.form = '10-K') AS annual,
                   count(f.accn) FILTER (WHERE f.form = '10-Q') AS quarterly,
                   coalesce(sum(f.bytes), 0)                    AS bytes,
                   min(f.period_end)                            AS earliest,
                   max(f.period_end)                            AS latest,
                   count(DISTINCT cf.cik)                       AS facts
            FROM companies c
            LEFT JOIN filings f      ON f.cik = c.cik
            LEFT JOIN companyfacts cf ON cf.cik = c.cik
            GROUP BY c.sector, c.ticker, c.cik
            ORDER BY c.sector, c.ticker
            """
        ).fetchall()

    def largest(self, n: int = 5) -> list[tuple]:
        return self.con.execute(
            "SELECT ticker, form, period_end, bytes FROM filings ORDER BY bytes DESC LIMIT ?", [n]
        ).fetchall()
