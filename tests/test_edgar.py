"""EDGAR's contract, asserted offline.

Every test here is a failure that would otherwise be silent. A truncated filing
index, a filing filtered on the wrong date field, a padded CIK in a path that
wants a bare one -- none of these raise. They just produce a smaller corpus than
the one the README claims, and nothing downstream notices.
"""

from __future__ import annotations

import json

import httpx
import pytest

from filing.config import Settings
from filing.ingest.edgar import (
    RETRYABLE_STATUS,
    EdgarClient,
    EdgarForbidden,
    FilingRef,
    _index_rows,
    _retryable,
    cik10,
    select_filings,
)


@pytest.fixture
def cfg(tmp_path):
    return Settings(
        sec_user_agent="Test Runner test@example.com",
        sec_rps=1000,  # the limiter has its own test; don't sleep in this one
        data_dir=tmp_path,
    )


def client_with(cfg, handler) -> EdgarClient:
    """An EdgarClient whose transport is a function, not a socket."""
    client = EdgarClient(cfg)
    client._http = httpx.Client(
        transport=httpx.MockTransport(handler), headers=dict(client._http.headers)
    )
    return client


# --------------------------------------------------------------------- CIKs


@pytest.mark.parametrize("given", [320193, "320193", "0000320193", "CIK0000320193"])
def test_cik10_pads_whatever_form_it_is_given(given):
    assert cik10(given) == "0000320193"


def test_archives_paths_use_the_bare_cik(cfg):
    """The JSON APIs want 0000320193; Archives wants 320193. Wrong one 404s."""
    seen = []
    ref = FilingRef(
        accn="0000320193-24-000123",
        form="10-K",
        filed_date="2024-11-01",
        period_end="2024-09-28",
        primary_doc="aapl-20240928.htm",
        size=0,
    )

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=b"<html/>")

    client_with(cfg, handler).document("0000320193", ref)
    assert seen == [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    ]


# ------------------------------------------------------------------- errors


def test_403_names_the_user_agent_as_the_cause(cfg):
    """EDGAR answers a missing User-Agent with a bare 403 and no explanation."""
    client = client_with(cfg, lambda r: httpx.Response(403))
    with pytest.raises(EdgarForbidden) as exc:
        client.ticker_map()
    assert "SEC_USER_AGENT" in str(exc.value)


def test_403_is_not_retried():
    """A misconfiguration retried four times just spends the allowance faster."""
    assert 403 not in RETRYABLE_STATUS
    response = httpx.Response(403, request=httpx.Request("GET", "https://x"))
    assert not _retryable(httpx.HTTPStatusError("", request=response.request, response=response))


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
def test_congestion_is_retried(status):
    response = httpx.Response(status, request=httpx.Request("GET", "https://x"))
    assert _retryable(httpx.HTTPStatusError("", request=response.request, response=response))


def test_transport_failures_are_retried():
    assert _retryable(httpx.ConnectError("boom"))
    assert not _retryable(ValueError("not a network problem"))


# -------------------------------------------------------------- the index


def test_index_rows_zips_parallel_arrays():
    rows = _index_rows(
        {
            "accessionNumber": ["a-1", "a-2"],
            "form": ["10-K", "10-Q"],
            "reportDate": ["2024-12-31", "2025-03-31"],
        }
    )
    assert rows == [
        {"accessionNumber": "a-1", "form": "10-K", "reportDate": "2024-12-31"},
        {"accessionNumber": "a-2", "form": "10-Q", "reportDate": "2025-03-31"},
    ]


def test_index_rows_is_empty_when_the_block_is():
    assert _index_rows({}) == []
    assert _index_rows({"accessionNumber": []}) == []


def test_ragged_arrays_raise_rather_than_truncate():
    """A loose zip would drop the trailing filings and report no error at all."""
    with pytest.raises(ValueError):
        _index_rows({"accessionNumber": ["a-1", "a-2"], "form": ["10-K"]})


# ----------------------------------------------------------------- overflow


def submissions_payload(recent, files=()):
    return {
        "name": "NVIDIA CORP",
        "fiscalYearEnd": "0126",
        "filings": {"recent": recent, "files": list(files)},
    }


def test_submissions_follows_overflow_chunks(cfg):
    """``filings.recent`` caps at ~1000, and a heavy filer's 10-Ks live past it."""
    recent = {"accessionNumber": ["new"], "form": ["4"], "reportDate": ["2025-01-01"]}
    older = {"accessionNumber": ["old"], "form": ["10-K"], "reportDate": ["2020-01-26"]}
    routes = {
        "/submissions/CIK0001045810.json": submissions_payload(
            recent,
            [{"name": "chunk-01.json", "filingFrom": "2019-01-01", "filingTo": "2021-06-01"}],
        ),
        "/submissions/chunk-01.json": older,
    }

    def handler(request):
        return httpx.Response(200, content=json.dumps(routes[request.url.path]).encode())

    rows = client_with(cfg, handler).submissions("1045810", since="2020-01-01")["_filings"]
    assert [r["accessionNumber"] for r in rows] == ["new", "old"]


def test_submissions_skips_chunks_that_provably_cannot_matter(cfg):
    """A filing is always filed *after* the period it reports, so a chunk whose
    newest filing date precedes the oldest wanted period end holds nothing."""
    fetched = []

    def handler(request):
        fetched.append(request.url.path)
        if request.url.path.endswith("CIK0001045810.json"):
            body = submissions_payload(
                {"accessionNumber": [], "form": [], "reportDate": []},
                [
                    {"name": "stale.json", "filingTo": "2015-12-31"},
                    {"name": "live.json", "filingTo": "2021-06-01"},
                ],
            )
        else:
            body = {"accessionNumber": ["x"], "form": ["10-K"], "reportDate": ["2020-06-30"]}
        return httpx.Response(200, content=json.dumps(body).encode())

    client_with(cfg, handler).submissions("1045810", since="2020-01-01")
    assert "/submissions/stale.json" not in fetched
    assert "/submissions/live.json" in fetched


def test_no_since_means_every_chunk_is_read(cfg):
    def handler(request):
        if request.url.path.endswith("CIK0001045810.json"):
            body = submissions_payload(
                {"accessionNumber": [], "form": [], "reportDate": []},
                [{"name": "stale.json", "filingTo": "2015-12-31"}],
            )
        else:
            body = {"accessionNumber": ["x"], "form": ["10-K"], "reportDate": ["2014-06-30"]}
        return httpx.Response(200, content=json.dumps(body).encode())

    rows = client_with(cfg, handler).submissions("1045810")["_filings"]
    assert [r["accessionNumber"] for r in rows] == ["x"]


# ---------------------------------------------------------------- selection


def row(accn, form, period, filed, doc="x.htm"):
    return {
        "accessionNumber": accn,
        "form": form,
        "reportDate": period,
        "filingDate": filed,
        "primaryDocument": doc,
        "size": 10,
    }


INDEX = [
    row("a", "10-K", "2024-01-31", "2024-03-15"),
    row("b", "10-Q", "2024-04-30", "2024-06-01"),
    row("c", "8-K", "2024-05-01", "2024-05-02"),  # wrong form
    row("d", "10-K", "2019-01-31", "2019-03-15"),  # period ends before the window
    row("e", "10-K", "2026-01-31", "2026-03-15"),  # and after it
    row("f", "10-K", "2023-01-31", "2023-03-15", doc=""),  # nothing to download
]

WINDOW = {"period_end_from": "2020-01-01", "period_end_to": "2025-02-28"}


def test_select_filters_on_form_window_and_downloadability():
    picked = select_filings(INDEX, forms={"10-K", "10-Q"}, **WINDOW)
    assert [f.accn for f in picked] == ["b", "a"]  # newest period first


def test_select_uses_the_period_end_not_the_filing_date():
    """Filing dates drift by weeks; a period end is the quarter it is *about*."""
    late = [row("g", "10-K", "2019-12-31", "2020-03-01")]
    assert select_filings(late, forms={"10-K"}, **WINDOW) == []


def test_select_honours_a_narrower_form_set():
    picked = select_filings(INDEX, forms={"10-K"}, **WINDOW)
    assert [f.accn for f in picked] == ["a"]


def test_filing_ref_strips_dashes_for_the_archive_path():
    ref = FilingRef("0001045810-25-000023", "10-K", "2025-02-26", "2025-01-26", "x.htm", 0)
    assert ref.accn_nodash == "000104581025000023"


# ------------------------------------------------------------------ counters


def test_client_counts_requests_and_bytes(cfg):
    """These are what the ingest report and the budget conversation rest on."""
    client = client_with(cfg, lambda r: httpx.Response(200, content=b"0123456789"))
    client._get("https://www.sec.gov/x")
    client._get("https://www.sec.gov/y")
    assert client.requests == 2
    assert client.bytes_downloaded == 20
