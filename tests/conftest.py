"""Fixtures shared by the M3 store tests.

A ``Chunk`` needs sixteen fields to exist and three of them to be interesting,
so the builder here fills the boring thirteen. Every test that makes chunks uses
it, which means a field added to ``Chunk`` breaks one line rather than forty.
"""

from __future__ import annotations

import hashlib

import pytest

from filing.stores.chunks import Chunk


def make_chunk(
    text: str = "body text",
    *,
    chunk_id: str | None = None,
    accn: str = "0000000000-00-000000",
    ticker: str = "NVDA",
    form: str = "10-K",
    item: str = "1A",
    part: str = "",
    title: str = "Risk Factors",
    fy: int | None = 2024,
    period_end: str = "2024-01-28",
    filed_date: str = "2024-02-21",
    char_start: int = 0,
    cik: str = "0001045810",
) -> Chunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Chunk(
        chunk_id=chunk_id or digest[:32],
        accn=accn,
        cik=cik,
        ticker=ticker,
        form=form,
        fy=fy,
        filed_date=filed_date,
        period_end=period_end,
        part=part,
        item=item,
        title=title,
        char_start=char_start,
        char_end=char_start + len(text),
        n_chars=len(text),
        text_sha256=digest,
        text=text,
    )


@pytest.fixture
def chunk():
    return make_chunk


# ---------------------------------------------------------------------------
# no test reaches the internet
# ---------------------------------------------------------------------------

_LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


@pytest.fixture(autouse=True)
def _no_outbound_network(monkeypatch):
    """Fail any test that opens a socket to something that is not this machine.

    Written after a test spent live API quota. It passed ``None`` for the chat
    backend and asserted that the run would serve everything from cache -- so on
    a cache hit nothing happened, and on a cache *miss* the runner did exactly
    what it is supposed to do and built the real Gemini client. An unrelated
    bug emptied the cache, and a hermetic-looking test quietly burned a day's
    requests and reported the 429s as ordinary failures.

    The lesson is not "fix that test" (it is fixed) but that a suite's
    hermeticity should be enforced rather than assumed: every network-using test
    here already injects an ``httpx.MockTransport``, so a real connection is
    always a mistake, and it should be a red test rather than an invoice.
    Localhost stays open -- Qdrant and Ollama are services, not the internet.
    """
    import socket

    real = socket.socket.connect

    def guarded(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOCAL:
            raise AssertionError(
                f"a test tried to connect to {host!r}. Tests are offline by design: "
                "inject an httpx.MockTransport or a fake backend."
            )
        return real(self, address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", guarded)
