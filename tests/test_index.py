"""The index's decisions, tested without a network or a running Qdrant.

Three of them matter enough to pin down. What gets indexed at all, because the
financial statements are 36% of the corpus and belong in the numeric store.
What the collection is called, because that name is the guard that turns a
backend swap into an error rather than a wrong neighbour. And how the token
pacer spends a per-minute budget, because getting that wrong is what made the
first index build take twice as long as the free tier actually required.

The BM25 half is exercised end to end -- it is on-disk and local, so there is no
reason to fake it, and its one real failure mode (an index whose id list has
drifted from its corpus) only shows up when you round-trip it.
"""

from __future__ import annotations

import time

import pytest

from conftest import make_chunk
from filing.stores.index import (
    NARRATIVE_ITEMS,
    SparseIndex,
    TokenPacer,
    collection_name,
    estimate_tokens,
    is_narrative,
    payload_of,
    select,
)

# --------------------------------------------------------------------------
# what gets indexed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("form", "item", "part", "expected"),
    [
        ("10-K", "1A", "", True),  # risk factors
        ("10-K", "7", "", True),  # MD&A
        ("10-K", "8", "", False),  # financial statements -> facts.duckdb
        ("10-K", "15", "", False),  # exhibit index -> a list of file names
        ("10-Q", "1", "I", False),  # 10-Q financial statements
        ("10-Q", "2", "I", True),  # 10-Q MD&A
        ("10-Q", "1A", "II", True),  # risk factors where most filers put them
        ("10-Q", "1A", "I", True),  # ... and where a few others do
    ],
)
def test_the_narrative_filter_keeps_prose_and_drops_statements(form, item, part, expected):
    assert is_narrative(make_chunk(form=form, item=item, part=part)) is expected


def test_a_degraded_filing_is_indexed_whole():
    """All twenty whole-document filings are Intel. Dropping them drops a company."""
    assert is_narrative(make_chunk(form="10-K", item="FULL")) is True


def test_an_unknown_form_is_indexed_rather_than_silently_dropped():
    """A form the table has never seen is a gap in the table, not a decision.

    Failing open is the safe direction: an extra chunk in the index costs
    embedding time, a missing one costs an answer and says nothing about why.
    """
    assert is_narrative(make_chunk(form="8-K", item="8.01")) is True


def test_all_items_indexes_everything():
    chunks = [make_chunk(item="1A"), make_chunk(item="8", text="balance sheet")]
    assert len(select(chunks)) == 1
    assert len(select(chunks, all_items=True)) == 2


def test_the_two_forms_agree_on_full():
    assert all("FULL" in items for items in NARRATIVE_ITEMS.values())


# --------------------------------------------------------------------------
# the collection guard
# --------------------------------------------------------------------------


def test_the_collection_name_carries_backend_model_and_width():
    assert (
        collection_name("local", "BAAI/bge-small-en-v1.5", 384)
        == "filing__local__BAAI_bge-small-en-v1-5__384"
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # a different backend, same width
        (("local", "m", 384), ("gemini", "m", 384)),
        # the same model at two Matryoshka widths
        (("gemini", "gemini-embedding-001", 1536), ("gemini", "gemini-embedding-001", 768)),
        # two models on one backend
        (("local", "BAAI/bge-small-en-v1.5", 384), ("local", "intfloat/e5-small-v2", 384)),
    ],
)
def test_vector_spaces_that_differ_get_different_collections(a, b):
    """Every one of these pairs would otherwise return plausible, wrong neighbours."""
    assert collection_name(*a) != collection_name(*b)


def test_the_payload_carries_metadata_but_never_the_text():
    payload = payload_of(make_chunk(text="a passage a citation will quote"))
    assert payload["ticker"] == "NVDA"
    assert payload["item_key"] == "1A"
    assert "text" not in payload
    assert payload["char_start"] == 0 and payload["char_end"] == 31


def test_a_ten_q_payload_carries_the_part_qualified_item():
    """``1A`` means different sections in Part I and Part II. The filter needs both."""
    assert payload_of(make_chunk(form="10-Q", part="II", item="1A"))["item_key"] == "II.1A"


# --------------------------------------------------------------------------
# pacing
# --------------------------------------------------------------------------


def test_the_pacer_lets_a_minutes_worth_through_without_waiting():
    pacer = TokenPacer(1000)
    assert pacer.wait(400) == 0.0
    assert pacer.wait(400) == 0.0


def test_the_pacer_holds_a_batch_that_would_breach_the_budget(monkeypatch):
    """The wait is measured, not slept: a real one would take a minute."""
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    pacer = TokenPacer(1000)
    pacer.wait(600)
    pacer.wait(300)
    waited = pacer.wait(600)  # 1500 in one window; must wait the first spend out
    assert waited >= 55.0
    assert clock[0] - 1000.0 >= 55.0


def test_the_pacer_forgets_spending_older_than_a_minute(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    pacer = TokenPacer(1000)
    pacer.wait(900)
    clock[0] += 61.0
    assert pacer.wait(900) == 0.0


def test_a_batch_larger_than_the_whole_budget_is_let_through(monkeypatch):
    """Otherwise one oversized chunk deadlocks the build against a budget it can never meet."""
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    assert TokenPacer(100).wait(5000) == 0.0


def test_token_estimate_is_four_characters_and_never_zero():
    assert estimate_tokens("x" * 4000) == 1000
    assert estimate_tokens("") == 1


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


def test_bm25_round_trips_through_disk_with_its_id_mapping(tmp_path):
    """Saved and reloaded, the index must still name the right chunks.

    ``bm25s`` retrieves by corpus position. If the ids are rebuilt separately
    from the index, the scores stay real and the documents they point at are
    wrong -- the failure that looks exactly like a working system.
    """
    chunks = [
        make_chunk("nvidia hopper architecture data center gpu", chunk_id="a"),
        make_chunk("walmart grocery pickup and delivery", chunk_id="b"),
        make_chunk("pfizer comirnaty vaccine revenues", chunk_id="c"),
    ]
    index = SparseIndex(tmp_path)
    assert index.build(chunks) == 3
    assert index.exists

    reloaded = SparseIndex(tmp_path)
    reloaded.load()
    assert reloaded.search("hopper gpu", limit=1)[0][0] == "a"
    assert reloaded.search("comirnaty", limit=1)[0][0] == "c"


def test_bm25_asked_for_more_than_it_holds_returns_what_it_has(tmp_path):
    index = SparseIndex(tmp_path)
    index.build([make_chunk("only one document here", chunk_id="a")])
    assert len(index.search("document", limit=50)) == 1


def test_bm25_indexes_the_header_the_dense_half_embeds(tmp_path):
    """A ticker is in ``embed_text``, not in ``text``. Both halves see it or neither does."""
    index = SparseIndex(tmp_path)
    index.build([make_chunk("component shortages affected results", chunk_id="a")])
    assert index.search("NVDA", limit=1)[0][0] == "a"
