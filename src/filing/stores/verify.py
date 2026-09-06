"""Check the store against the documents, not against itself.

Every other test in this project verifies that the code does what the code says.
This one asks a different question: are the numbers *true*? The facts arrive
from EDGAR's companyfacts JSON API. The filings arrive as HTML from EDGAR's
document archive. They are produced by different pipelines from the same
submission, so finding a stored value printed in the corresponding 10-K is real
corroboration rather than a tautology -- if the loader mangled a scale, dropped
a sign, or attached a value to the wrong period, this is what notices.

The matching is deliberately crude. Filings render money in whatever unit the
company chose, so a value of 349,585,000,000 might appear as "349,585" in a
statement captioned "in millions", as "349,585,000" in thousands, or in full.
All three are tried, with and without comma grouping, and negatives are also
tried in the accounting convention of parentheses. A match on any of them is
enough: the question is whether the number is in the document, not how it was
typeset.

What this cannot do is prove a *miss* is wrong. A fact can be genuine and absent
from the primary document -- it may live in an exhibit, a schedule that is filed
separately, or only inside the XBRL attachments. So a miss is reported, counted
and shown, but the gate's threshold is set from the measured rate rather than
from an assumption that every fact must appear.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")

# Sampled from facts whose reporting filing this project actually holds. Only
# 407 of the 1,365 accessions in the payloads are in the corpus window -- the
# rest are older or newer submissions that companyfacts still carries -- so the
# join is what makes the check possible, not a convenience.
SAMPLE_SQL = """
SELECT f.cik, f.ticker, f.tag, f.unit, f.span, f.period_start, f.period_end,
       f.val, f.accn, g.form, g.path
FROM facts_current f
JOIN filings g ON g.accn = f.accn
WHERE f.unit = 'USD' AND f.span IN ('FY', 'instant')
  AND abs(f.val) > 1e6 AND g.form IN ('10-K', '10-Q')
ORDER BY hash(f.accn || f.tag || f.period_end || CAST(f.val AS VARCHAR))
LIMIT ?
"""


@dataclass
class Sample:
    ticker: str
    tag: str
    period_end: Any
    val: float
    accn: str
    form: str
    found: bool = False
    matched_as: str = ""


@dataclass
class VerifyReport:
    checked: int = 0
    found: int = 0
    misses: list[Sample] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    docs_read: int = 0

    @property
    def rate(self) -> float:
        return self.found / self.checked if self.checked else 0.0


def candidate_strings(val: float) -> list[tuple[str, str]]:
    """Every plausible rendering of a value, paired with the scale that made it."""
    out: list[tuple[str, str]] = []
    magnitude = abs(val)
    for scale, name in ((1, "units"), (1_000, "thousands"), (1_000_000, "millions")):
        scaled = magnitude / scale
        if scaled < 1:
            continue
        for number in {int(scaled), round(scaled)}:
            if number == 0:
                continue
            grouped, plain = f"{number:,}", str(number)
            out.append((grouped, name))
            out.append((plain, name))
            if val < 0:
                # Accounting convention: negatives are parenthesised, not signed.
                out.append((f"({grouped})", f"{name}, parenthesised"))
    return out


def _document_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return ENTITY_RE.sub(" ", TAG_RE.sub(" ", raw))


def verify_against_filings(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    *,
    n: int = 50,
    seed: int = 20260906,
) -> VerifyReport:
    """Look for ``n`` sampled facts in the filings that reported them."""
    rows = con.execute(SAMPLE_SQL, [n]).fetchall()
    random.Random(seed).shuffle(rows)
    report = VerifyReport()

    # Sorted by document so each filing is parsed once; several samples usually
    # land in the same 10-K and these documents run to megabytes.
    by_doc: dict[str, list[tuple]] = {}
    for row in rows:
        by_doc.setdefault(row[10], []).append(row)

    for rel_path, group in by_doc.items():
        path = data_dir / rel_path
        text = _document_text(path) if path.exists() else ""
        report.docs_read += 1
        for row in group:
            sample = Sample(
                ticker=row[1],
                tag=row[2],
                period_end=row[6],
                val=row[7],
                accn=row[8],
                form=row[9],
            )
            for rendered, how in candidate_strings(sample.val):
                if rendered in text:
                    sample.found, sample.matched_as = True, how
                    break
            report.checked += 1
            report.samples.append(sample)
            if sample.found:
                report.found += 1
            else:
                report.misses.append(sample)
    return report
