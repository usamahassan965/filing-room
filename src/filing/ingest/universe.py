"""Reading universe.yaml, and turning tickers into CIKs.

The corpus is declared in one committed file so that "which companies, which
forms, which years?" is answerable from the repository rather than from
whatever happens to be on disk. Everything downstream -- the eval set, every
number in the README -- is scoped by this declaration, so it is validated
strictly: a typo in a ticker should stop the run, not silently shrink the
corpus by one company and shift every peer comparison that mentions it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from filing.config import Settings, settings
from filing.ingest.edgar import cik10


class UniverseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    name: str  # as declared; the registrant name from EDGAR is stored separately
    sector: str
    cik: str = ""  # filled in by resolve(); empty until then


@dataclass(frozen=True, slots=True)
class Universe:
    companies: tuple[Company, ...]
    forms: frozenset[str]
    period_end_from: str
    period_end_to: str

    @property
    def sectors(self) -> dict[str, list[Company]]:
        out: dict[str, list[Company]] = {}
        for c in self.companies:
            out.setdefault(c.sector, []).append(c)
        return out

    def resolved(self, ticker_map: dict[str, tuple[str, str]]) -> Universe:
        """Attach a CIK to every company, or say exactly which ones failed.

        Fail loudly and all at once. Resolving nineteen of twenty and carrying
        on would produce a corpus that looks complete in every count except the
        one nobody checks, and a peer-comparison question whose peer set is
        quietly wrong is worse than a crash.

        A company that declares its own ``cik`` keeps it and is never looked up.
        That escape hatch exists because SEC's ticker map points at the entity
        trading under the ticker *today*, which is not always the entity that
        filed the history: after a holding-company reorganisation the ticker
        moves to a brand-new CIK with no filings behind it, and the old
        registrant keeps the ten years of 10-Ks with no ticker at all. Nothing
        errors in that case -- resolution succeeds, and the corpus is short one
        company's entire history. See XOM in universe.yaml.
        """
        resolved: list[Company] = []
        missing: list[str] = []
        for c in self.companies:
            if c.cik:
                resolved.append(c)
                continue
            hit = ticker_map.get(c.ticker.upper())
            if hit is None:
                missing.append(c.ticker)
                continue
            resolved.append(Company(ticker=c.ticker, name=c.name, sector=c.sector, cik=hit[0]))
        if missing:
            raise UniverseError(
                f"{len(missing)} ticker(s) in universe.yaml did not resolve to a CIK: "
                f"{', '.join(missing)}. SEC's company_tickers.json lists currently "
                "listed registrants only -- a delisted or renamed ticker will not be "
                "there. Check the ticker, or replace the company."
            )
        return Universe(
            companies=tuple(resolved),
            forms=self.forms,
            period_end_from=self.period_end_from,
            period_end_to=self.period_end_to,
        )


def load_universe(path: Path | None = None, cfg: Settings | None = None) -> Universe:
    cfg = cfg or settings()
    path = path or cfg.universe_path
    if not path.exists():
        raise UniverseError(f"no universe file at {path}. It is committed; restore it from git.")

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    window = doc.get("window") or {}
    forms = doc.get("forms") or []
    sectors = doc.get("sectors") or {}

    for key, value in (("window", window), ("forms", forms), ("sectors", sectors)):
        if not value:
            raise UniverseError(f"universe.yaml has no {key!r} section")

    companies: list[Company] = []
    seen: set[str] = set()
    for sector, members in sectors.items():
        for entry in members or []:
            ticker = str(entry["ticker"]).upper()
            if ticker in seen:
                # A duplicate would double every count it appears in and
                # silently over-weight one company in the eval set.
                raise UniverseError(f"{ticker} appears twice in universe.yaml")
            seen.add(ticker)
            companies.append(
                Company(
                    ticker=ticker,
                    name=str(entry.get("name") or ticker),
                    sector=sector,
                    # Optional, and zero-padded here so a hand-typed CIK matches
                    # the ten-digit form every EDGAR JSON path expects.
                    cik=cik10(entry["cik"]) if entry.get("cik") else "",
                )
            )

    return Universe(
        companies=tuple(companies),
        forms=frozenset(str(f) for f in forms),
        period_end_from=str(window["period_end_from"]),
        period_end_to=str(window["period_end_to"]),
    )
