"""The EDGAR client: four endpoints, one rate limit, one User-Agent.

SEC's fair-access policy is why this is a class and not four functions. Two
rules, and breaking either gets the IP *blocked* rather than throttled:

  * every request carries a User-Agent naming a real contact address
  * no more than 10 requests per second, sustained

The limiter is the same sliding-window class the model tier uses, with a
one-second window instead of a sixty-second one. "No window ever contains more
than N marks" is the same invariant at either scale, so there is no second
implementation to keep honest -- see ``filing.llm.limiter``.

The retry policy, by contrast, is deliberately *not* shared with the model tier.
A 429 from a model provider means wait a moment; a 429 from EDGAR means you are
already over the line and the next step is a timed block. So this backs off
harder, gives up sooner, and treats 403 as fatal rather than transient -- a 403
here is a missing User-Agent, and retrying a misconfiguration only spends the
allowance faster.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from filing.config import Settings, settings
from filing.llm.limiter import RateLimiter

log = logging.getLogger(__name__)

# 403 is absent on purpose: it is a configuration error, not congestion.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EdgarError(RuntimeError):
    pass


class EdgarForbidden(EdgarError):
    """EDGAR refused the client outright.

    Almost always the User-Agent. SEC requires one naming a real contact
    address and answers anything else with a bare 403, so this error supplies
    the explanation EDGAR does not.
    """

    def __init__(self, url: str, user_agent: str) -> None:
        shown = user_agent or "(empty)"
        super().__init__(
            f"EDGAR returned 403 for {url}\n"
            f"Current SEC_USER_AGENT: {shown}\n"
            "SEC requires a User-Agent naming a real contact address, e.g.\n"
            '  SEC_USER_AGENT="Jane Doe jane@example.com"\n'
            "Set it in .env. A missing or fake contact address is the cause "
            "roughly every time, and retrying will not fix it."
        )


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def _log_retry(state) -> None:  # noqa: ANN001 - tenacity's callback signature
    exc = state.outcome.exception() if state.outcome else None
    log.warning("edgar retry %s: %s", state.attempt_number, exc)


_RETRY = dict(
    retry=retry_if_exception(_retryable),
    # Longer initial wait than the model tier: EDGAR's own guidance when it
    # pushes back is to slow down, and hammering a 429 is what escalates it.
    wait=wait_exponential_jitter(initial=5, max=120),
    stop=stop_after_attempt(4),
    before_sleep=_log_retry,
    reraise=True,
)


def cik10(cik: int | str) -> str:
    """EDGAR's zero-padded form: ``320193`` becomes ``0000320193``.

    Both forms appear in SEC's own APIs -- the padded one in submissions and
    companyfacts URLs, the bare integer in Archives paths -- so the conversion
    lives here instead of being re-derived at each call site.
    """
    return str(int(str(cik).lstrip("CIK") or 0)).zfill(10)


@dataclass(frozen=True, slots=True)
class FilingRef:
    """One filing as EDGAR describes it, before anything is downloaded."""

    accn: str  # accession number, dashed form -- unique per submission
    form: str
    filed_date: str  # ISO
    period_end: str  # ISO; EDGAR calls it reportDate
    primary_doc: str
    size: int  # the whole submission, not the primary document alone

    @property
    def accn_nodash(self) -> str:
        return self.accn.replace("-", "")


class EdgarClient:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        self.limiter = RateLimiter(self.cfg.sec_rps, window_s=1.0)
        self._http = httpx.Client(
            timeout=self.cfg.request_timeout_s,
            follow_redirects=True,
            headers={
                "User-Agent": self.cfg.sec_user_agent,
                # SEC serves gzip and filings are large text. Without this the
                # corpus download moves several times the bytes it needs to.
                "Accept-Encoding": "gzip, deflate",
            },
        )
        self.requests = 0
        self.bytes_downloaded = 0

    # ------------------------------------------------------------------ http

    @retry(**_RETRY)
    def _get(self, url: str) -> bytes:
        self.limiter.acquire()
        self.requests += 1
        resp = self._http.get(url)
        if resp.status_code == 403:
            raise EdgarForbidden(url, self.cfg.sec_user_agent)
        resp.raise_for_status()
        self.bytes_downloaded += len(resp.content)
        return resp.content

    def _get_json(self, url: str) -> Any:
        return json.loads(self._get(url))

    # ------------------------------------------------------------- endpoints

    def ticker_map(self) -> dict[str, tuple[str, str]]:
        """``{TICKER: (cik10, registrant name)}`` for every listed company.

        One request covers all ~10,000 of them, which is why resolution here is
        a dictionary lookup rather than a search per ticker.
        """
        raw = self._get_json(f"{self.cfg.sec_base_url}/files/company_tickers.json")
        return {
            row["ticker"].upper(): (cik10(row["cik_str"]), row["title"]) for row in raw.values()
        }

    def submissions(self, cik: str, *, since: str = "") -> dict[str, Any]:
        """Company metadata plus its filing index, overflow chunks included.

        ``filings.recent`` holds only the last ~1000 submissions. A company that
        files a lot of Form 4s can burn through 1000 in under three years, so
        five years of 10-Ks may well sit in an overflow chunk. Following them is
        the difference between a full corpus and a quietly truncated one.

        ``since`` skips chunks that cannot matter, and the reasoning is exact
        rather than heuristic: a filing is always filed after the period it
        reports, so a chunk whose newest *filing* date precedes the oldest
        *period end* we want contains nothing we want.
        """
        doc = self._get_json(f"{self.cfg.sec_data_url}/submissions/CIK{cik10(cik)}.json")
        rows = _index_rows(doc["filings"]["recent"])
        for chunk in doc["filings"].get("files", []):
            if since and chunk.get("filingTo", "") < since:
                continue
            extra = self._get_json(f"{self.cfg.sec_data_url}/submissions/{chunk['name']}")
            rows.extend(_index_rows(extra))
        doc["_filings"] = rows
        return doc

    def companyfacts(self, cik: str) -> bytes:
        """Every XBRL fact the company has ever reported, as served.

        Bytes rather than parsed JSON: the bytes are what gets hashed and
        stored, and re-serialising a parsed document would change the hash
        without changing the content.
        """
        return self._get(f"{self.cfg.sec_data_url}/api/xbrl/companyfacts/CIK{cik10(cik)}.json")

    def document(self, cik: str, ref: FilingRef) -> bytes:
        """The filing's primary document.

        Note the un-padded CIK: Archives paths use the bare integer while the
        JSON APIs use the zero-padded form. Getting this wrong returns a 404,
        not a redirect.
        """
        bare = int(cik10(cik))
        return self._get(
            f"{self.cfg.sec_base_url}/Archives/edgar/data/"
            f"{bare}/{ref.accn_nodash}/{ref.primary_doc}"
        )

    def close(self) -> None:
        self._http.close()


def _index_rows(block: dict[str, list]) -> list[dict[str, Any]]:
    """EDGAR ships the filing index as parallel arrays; zip them into rows.

    ``strict=True`` on purpose. The arrays are parallel by contract, and if that
    contract ever breaks the loose zip would silently truncate the index to the
    shortest column -- losing filings without losing a single count anyone
    checks. A raised exception costs one company and gets reported; a short
    index costs a corpus and does not.
    """
    if not block or not block.get("accessionNumber"):
        return []
    keys = list(block.keys())
    columns = [block[k] for k in keys]
    return [dict(zip(keys, values, strict=True)) for values in zip(*columns, strict=True)]


def select_filings(
    rows: list[dict[str, Any]],
    *,
    forms: set[str],
    period_end_from: str,
    period_end_to: str,
) -> list[FilingRef]:
    """Filter the index down to the corpus, newest period first.

    Filtering on period end rather than filing date is what makes the window
    mean the same thing for every company. Filing dates drift by weeks and vary
    with how late a company files; a period end is the quarter or year the
    filing is actually *about*.
    """
    out: list[FilingRef] = []
    for row in rows:
        if row.get("form") not in forms:
            continue
        period = row.get("reportDate") or ""
        if not (period_end_from <= period <= period_end_to):
            continue
        if not row.get("primaryDocument"):
            continue  # nothing to download: paper filings and stray index rows
        out.append(
            FilingRef(
                accn=row["accessionNumber"],
                form=row["form"],
                filed_date=row.get("filingDate") or "",
                period_end=period,
                primary_doc=row["primaryDocument"],
                size=int(row.get("size") or 0),
            )
        )
    out.sort(key=lambda f: (f.period_end, f.form), reverse=True)
    return out
