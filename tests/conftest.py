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
