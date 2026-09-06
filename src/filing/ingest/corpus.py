"""Acquiring the corpus, and reporting on what was acquired.

Two properties matter more than speed here, and both come from the manifest:

*Idempotent.* A second run downloads nothing. Not "downloads little" -- nothing,
and the M1 gate asserts it. That is what makes the corpus safe to extend: adding
a company to universe.yaml fetches one company, not twenty-one.

*Resumable.* One company failing does not end the run. Errors are collected and
reported, the other nineteen finish, and the next run picks up exactly what is
missing. EDGAR is a public service on a fair-access policy -- a run that has to
start from scratch after a transient 503 is a run that will get the IP blocked.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from filing.config import Settings, settings
from filing.ingest.edgar import EdgarClient, FilingRef, select_filings
from filing.ingest.manifest import Manifest, sha256_bytes
from filing.ingest.universe import Company, Universe, load_universe

log = logging.getLogger(__name__)

# What M3 will spend parsing this corpus, per megabyte of filing HTML, on one
# CPU. An assumption, not a measurement -- docling is not installed until M3 --
# and it is a single named constant precisely so that M3 replaces one number
# with a real rate instead of re-deriving the arithmetic. Flagged as an
# assumption everywhere it is reported.
ASSUMED_PARSE_MB_PER_MIN = 1.5


@dataclass
class IngestReport:
    companies: int = 0
    filings_downloaded: int = 0
    filings_skipped: int = 0
    facts_downloaded: int = 0
    facts_skipped: int = 0
    pruned: int = 0
    requests: int = 0
    bytes_downloaded: int = 0
    elapsed_s: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def downloaded_nothing(self) -> bool:
        return self.filings_downloaded == 0 and self.facts_downloaded == 0


Progress = Callable[[str, str], None]  # (ticker, message)


def _noop(ticker: str, message: str) -> None:  # pragma: no cover - default sink
    log.info("%s %s", ticker, message)


def ingest(
    cfg: Settings | None = None,
    *,
    universe: Universe | None = None,
    forms: set[str] | None = None,
    limit: int | None = None,
    do_filings: bool = True,
    do_facts: bool = True,
    progress: Progress = _noop,
) -> IngestReport:
    cfg = cfg or settings()
    started = time.monotonic()
    report = IngestReport()

    client = EdgarClient(cfg)
    try:
        uni = (universe or load_universe(cfg=cfg)).resolved(client.ticker_map())
        wanted_forms = forms or set(uni.forms)
        companies = uni.companies[:limit] if limit else uni.companies
        report.companies = len(companies)

        with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
            # Only when the whole universe is in play: a --limit run would read
            # as "the other nineteen are no longer declared" and delete them.
            if limit is None:
                report.pruned = manifest.prune({c.cik for c in uni.companies})

            for company in companies:
                try:
                    _one_company(
                        client,
                        manifest,
                        cfg,
                        company,
                        uni,
                        wanted_forms,
                        report,
                        do_filings=do_filings,
                        do_facts=do_facts,
                        progress=progress,
                    )
                except Exception as exc:  # noqa: BLE001 - one company must not end the run
                    log.exception("ingest failed for %s", company.ticker)
                    report.errors.append((company.ticker, f"{type(exc).__name__}: {exc}"))
                    progress(company.ticker, f"[red]failed: {type(exc).__name__}[/red]")
    finally:
        client.close()
        report.requests = client.requests
        report.bytes_downloaded = client.bytes_downloaded
        report.elapsed_s = time.monotonic() - started
    return report


def _one_company(
    client: EdgarClient,
    manifest: Manifest,
    cfg: Settings,
    company: Company,
    uni: Universe,
    forms: set[str],
    report: IngestReport,
    *,
    do_filings: bool,
    do_facts: bool,
    progress: Progress,
) -> None:
    subs = client.submissions(company.cik, since=uni.period_end_from)
    manifest.upsert_company(
        cik=company.cik,
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        registrant=subs.get("name"),
        fiscal_year_end=subs.get("fiscalYearEnd"),
        sic=subs.get("sic"),
        sic_description=subs.get("sicDescription"),
    )

    if do_filings:
        refs = select_filings(
            subs["_filings"],
            forms=forms,
            period_end_from=uni.period_end_from,
            period_end_to=uni.period_end_to,
        )
        fetched = skipped = 0
        for ref in refs:
            if manifest.has_filing(ref.accn):
                skipped += 1
                continue
            _fetch_filing(client, manifest, cfg, company, ref)
            fetched += 1
        report.filings_downloaded += fetched
        report.filings_skipped += skipped
        progress(company.ticker, f"{len(refs)} filings ({fetched} new, {skipped} cached)")

    if do_facts:
        if manifest.has_facts(company.cik):
            report.facts_skipped += 1
        else:
            _fetch_facts(client, manifest, cfg, company)
            report.facts_downloaded += 1


def _fetch_filing(
    client: EdgarClient,
    manifest: Manifest,
    cfg: Settings,
    company: Company,
    ref: FilingRef,
) -> None:
    blob = client.document(company.cik, ref)
    dest = cfg.raw_dir / "filings" / company.cik / ref.accn_nodash / ref.primary_doc
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    # The manifest row goes in only once the bytes are down. A crash between
    # the two costs one re-download on the next run; the reverse ordering would
    # cost a silently truncated document that every later gate trusts.
    manifest.upsert_filing(
        accn=ref.accn,
        cik=company.cik,
        ticker=company.ticker,
        form=ref.form,
        fy=int(ref.period_end[:4]) if ref.period_end else None,
        filed_date=ref.filed_date,
        period_end=ref.period_end,
        primary_doc=ref.primary_doc,
        path=manifest.relative(dest),
        bytes=len(blob),
        sha256=sha256_bytes(blob),
    )


def _fetch_facts(client: EdgarClient, manifest: Manifest, cfg: Settings, company: Company) -> None:
    blob = client.companyfacts(company.cik)
    dest = cfg.raw_dir / "companyfacts" / f"CIK{company.cik}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    manifest.upsert_facts(
        cik=company.cik,
        ticker=company.ticker,
        path=manifest.relative(dest),
        bytes=len(blob),
        sha256=sha256_bytes(blob),
        n_concepts=count_concepts(blob),
    )


def count_concepts(blob: bytes) -> int:
    """How many distinct XBRL tags this company has ever reported.

    Recorded now because it is the cheapest possible early warning for M2: a
    company whose companyfacts payload carries a handful of concepts instead of
    hundreds has not filed the XBRL the structured store is about to assume.
    """
    try:
        facts = json.loads(blob).get("facts", {})
    except (ValueError, AttributeError):  # pragma: no cover - malformed payload
        return 0
    return sum(len(tags) for tags in facts.values() if isinstance(tags, dict))


# ---------------------------------------------------------------- reporting


def write_corpus_doc(cfg: Settings | None = None, path: Path | None = None) -> Path:
    """Write docs/corpus.md from the manifest.

    Regenerated, never hand-edited -- the M1 gate asks for corpus size and an
    estimated parse time on record, and a number typed by hand is a number that
    drifts away from the corpus the first time the corpus changes.
    """
    cfg = cfg or settings()
    path = path or (cfg.data_dir.parent / "docs" / "corpus.md")
    with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
        stats = manifest.stats()
        rows = manifest.per_company()
        largest = manifest.largest(5)
        uni = load_universe(cfg=cfg)

    mb = stats.filing_bytes / 1e6
    parse_min = mb / ASSUMED_PARSE_MB_PER_MIN

    lines: list[str] = []
    add = lines.append
    add("# Corpus")
    add("")
    add(f"Generated by `filing corpus` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.")
    add("Do not hand-edit: regenerate it.")
    add("")
    add(
        f"Scope is declared in [`universe.yaml`](../universe.yaml): **{len(uni.companies)} "
        f"companies** across **{len(uni.sectors)} sectors**, forms "
        f"{', '.join(sorted(uni.forms))}, period ends from `{uni.period_end_from}` "
        f"to `{uni.period_end_to}`."
    )
    add("")
    add("## Totals")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Companies | {stats.companies} |")
    add(f"| Companies with filings | {stats.companies_with_filings} |")
    add(f"| Filings on disk | **{stats.filings}** |")
    for form, n in sorted(stats.by_form.items()):
        add(f"| &nbsp;&nbsp;{form} | {n} |")
    add(f"| companyfacts payloads | {stats.companyfacts} |")
    add(f"| Filing bytes | {_human(stats.filing_bytes)} |")
    add(f"| companyfacts bytes | {_human(stats.facts_bytes)} |")
    add(f"| **Total on disk** | **{_human(stats.filing_bytes + stats.facts_bytes)}** |")
    add("")
    add("## Estimated parse time (M3)")
    add("")
    add(f"At an assumed **{ASSUMED_PARSE_MB_PER_MIN} MB/min** of filing HTML on one CPU:")
    add("")
    add(f"- {mb:,.0f} MB of filings -> **~{parse_min / 60:,.1f} hours** for a full pass.")
    add("")
    add("That rate is an *assumption*, not a measurement -- docling is not installed until M3.")
    add("It lives in one named constant (`ASSUMED_PARSE_MB_PER_MIN` in")
    add("`src/filing/ingest/corpus.py`); M3 replaces it with a measured rate and this file")
    add("is regenerated. Parsing is cached to parquet, so the cost is paid once.")
    add("")
    add("## By company")
    add("")
    add("| Sector | Ticker | CIK | 10-K | 10-Q | Size | Earliest period | Latest period | Facts |")
    add("|---|---|---|---:|---:|---:|---|---|:-:|")
    for sector, ticker, cik, annual, quarterly, nbytes, earliest, latest, facts in rows:
        add(
            f"| {sector} | {ticker} | {cik} | {annual} | {quarterly} | {_human(nbytes)} "
            f"| {earliest or '--'} | {latest or '--'} | {'yes' if facts else 'NO'} |"
        )
    add("")
    add("## Two things this table will look wrong about")
    add("")
    add("**The retail period ends are in January and February, and that is correct.**")
    add("A fiscal-year *label* is a company convention, and the companies disagree:")
    add("the year ending 2024-01-31 is Walmart's FY2024, while the year ending")
    add("2024-02-03 is Target's FY2023. So the corpus window filters on period end,")
    add("which is unambiguous, and the manifest's `fy` column is only the calendar")
    add("year of that period end -- a filter convenience, never a label. The")
    add("authoritative `fy`/`fp` pair arrives with the XBRL facts in M2. The four")
    add("retailers are in the universe precisely to keep this honest.")
    add("")
    add("**XOM's CIK is pinned in `universe.yaml`, and every other company's is not.**")
    add("SEC's `company_tickers.json` maps a ticker to the registrant trading under")
    add("it *today*. For XOM that is `0002115436`, an entity created in a holding-")
    add("company reorganisation whose first filing postdates this entire window;")
    add("the history sits under `0000034088`, which now carries no ticker at all.")
    add("Resolution by ticker succeeds either way -- it just returns a company with")
    add("no filings, and only the per-company coverage check in `filing corpus`")
    add("notices. A corpus-wide total never would.")
    add("")
    add("## Largest filings")
    add("")
    add("The M3 parse budget is set by these, not by the average.")
    add("")
    add("| Ticker | Form | Period end | Size |")
    add("|---|---|---|---:|")
    for ticker, form, period, nbytes in largest:
        add(f"| {ticker} | {form} | {period} | {_human(nbytes)} |")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _human(n: int | float) -> str:
    """Decimal units, deliberately.

    1 MB = 1e6 bytes here, not 2^20, so these figures agree with the parse-time
    estimate below rather than sitting ~5% apart from it for no reason a reader
    could see.
    """
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1000
    return f"{n:,.2f} GB"  # pragma: no cover - unreachable, loop returns first
