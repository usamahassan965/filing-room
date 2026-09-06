"""The baseline chunker and retriever.

What is worth asserting about a deliberately naive system is that it is naive in
the specific ways the comparison claims -- fixed stride, no overlap, no item, no
header on the embedded text. Each of those is a line in the table at the top of
``naive.py``, and a baseline that quietly acquired one of the real system's
advantages would make every M5 number an overstatement in the same direction.
"""

from __future__ import annotations

import hashlib

import pytest

from conftest import make_chunk
from filing.eval import naive
from filing.eval.naive import NAIVE_CHARS, NAIVE_CHUNKER, NaiveRetriever, naive_chunks, naive_cut
from filing.stores.chunks import chunk_id
from filing.stores.parse import WHOLE_DOC_ITEM

META = dict(
    cik="0001045810",
    ticker="NVDA",
    form="10-K",
    fy=2024,
    filed_date="2024-02-21",
    period_end="2024-01-28",
)


# ------------------------------------------------------------------- cutting


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 0),
        (1, 1),
        (NAIVE_CHARS, 1),
        (NAIVE_CHARS + 1, 2),
        (3 * NAIVE_CHARS, 3),
    ],
)
def test_the_stride_is_fixed_and_the_tail_is_kept(n, expected):
    assert len(naive_cut("x" * n)) == expected


def test_the_cuts_tile_the_document_with_no_overlap_and_no_gap():
    cuts = naive_cut("x" * (5 * NAIVE_CHARS + 17))
    assert cuts[0][0] == 0
    assert all(b[0] == a[1] for a, b in zip(cuts, cuts[1:], strict=False))
    assert cuts[-1][1] == 5 * NAIVE_CHARS + 17


def test_a_short_tail_is_emitted_rather_than_merged():
    """A 40-character final chunk is a real thing a fixed-stride chunker makes."""
    cuts = naive_cut("x" * (NAIVE_CHARS + 40))
    assert cuts[-1] == (NAIVE_CHARS, NAIVE_CHARS + 40)


# -------------------------------------------------------------------- chunks


def test_chunks_reconstruct_the_document_exactly():
    text = "".join(f"sentence {i}. " for i in range(600))
    assert "".join(c.text for c in naive_chunks(text, "A", **META)) == text


def test_no_chunk_claims_to_know_its_item():
    text = "y" * (3 * NAIVE_CHARS)
    made = naive_chunks(text, "A", **META)
    assert {c.item for c in made} == {WHOLE_DOC_ITEM}
    assert {c.part for c in made} == {""}
    assert {c.title for c in made} == {""}


def test_a_naive_chunk_is_never_the_same_point_as_a_real_one():
    """Same filing, same offsets, different id -- or the two collections collide."""
    made = naive_chunks("z" * 100, "A", **META)[0]
    assert made.chunk_id == chunk_id("naive:A", 0, 100)
    assert made.chunk_id != chunk_id("A", 0, 100)


def test_offsets_and_hashes_describe_the_text_they_carry():
    for c in naive_chunks("w" * (2 * NAIVE_CHARS + 5), "A", **META):
        assert c.n_chars == c.char_end - c.char_start == len(c.text)
        assert c.text_sha256 == hashlib.sha256(c.text.encode("utf-8")).hexdigest()


def test_metadata_rides_along_unchanged():
    c = naive_chunks("body", "0001045810-24-000029", **META)[0]
    assert (c.accn, c.ticker, c.form, c.period_end) == (
        "0001045810-24-000029",
        "NVDA",
        "10-K",
        "2024-01-28",
    )


def test_an_empty_filing_makes_no_chunks():
    assert naive_chunks("", "A", **META) == []


# ------------------------------------------------------------- embedded text


def test_the_embedder_sees_the_raw_slice_and_no_header():
    c = make_chunk("our results were affected by component shortages")
    assert naive.naive_embed_text(c) == c.text
    assert "NVDA" in c.embed_text and "NVDA" not in naive.naive_embed_text(c)


# ---------------------------------------------------------------- retrieval


class FakeEmbed:
    name = "local"

    def __init__(self) -> None:
        self.seen: list[tuple[list[str], str]] = []

    def embed(self, texts, *, input_type: str = "passage"):  # noqa: ANN001
        self.seen.append((list(texts), input_type))
        return [[0.0, 1.0] for _ in texts]


class FakeDense:
    def __init__(self, hits) -> None:  # noqa: ANN001
        self.hits = hits
        self.limits: list[int] = []

    def search(self, vector, *, limit: int):  # noqa: ANN001
        self.limits.append(limit)
        return self.hits[:limit]


@pytest.fixture
def retriever(tmp_path):
    from filing.config import Settings

    cfg = Settings(data_dir=tmp_path / "data", embed_backend="local")
    return NaiveRetriever(cfg, backend=FakeEmbed())


def test_the_baseline_collection_is_not_the_real_one(retriever):
    assert retriever.dense.name.endswith(f"__{NAIVE_CHUNKER}")


def test_search_asks_the_query_side_of_the_model(retriever):
    retriever._chunks = {}
    retriever.dense = FakeDense([])
    retriever.search("what happened?")
    assert retriever.backend.seen == [(["what happened?"], "query")]


def test_search_returns_chunks_in_the_index_order(retriever):
    made = [make_chunk(f"body {i}", chunk_id=f"c{i}") for i in range(3)]
    retriever._chunks = {c.chunk_id: c for c in made}
    retriever.dense = FakeDense([("c2", 0.9), ("c0", 0.8), ("c1", 0.7)])
    assert [c.chunk_id for c in retriever.search("q", k=3)] == ["c2", "c0", "c1"]


def test_k_reaches_the_index(retriever):
    retriever._chunks = {}
    retriever.dense = FakeDense([])
    retriever.search("q", k=7)
    assert retriever.dense.limits == [7]


def test_an_id_the_store_does_not_have_is_dropped_not_faked(retriever):
    """A point left over from an older build must not become a citation."""
    kept = make_chunk("kept", chunk_id="c1")
    retriever._chunks = {"c1": kept}
    retriever.dense = FakeDense([("stale", 0.9), ("c1", 0.5)])
    assert [c.chunk_id for c in retriever.search("q")] == ["c1"]
