"""The chunker's arithmetic, and the promise the M3 gate checks on 200 samples.

Every chunk in this system is a *contiguous slice of the filing named by
offsets*, and everything downstream leans on that: the citation quotes it, the
gate re-resolves it, the graph puts absolute character positions on every edge.
So the property tested hardest here is the boring one -- ``text[start:end]`` is
the chunk's text, byte for byte, for every chunk the chunker emits.

The packing rules are tested against constructed documents rather than real
filings, because each rule exists to fix a shape that a real filing produced
once: a subheading followed by a block too big to join it, a paragraph small
enough to repeat as overlap, a fifty-character tail left over at the end of a
section. The shape is the test; the filing that revealed it is in the comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_chunk
from filing.stores.chunks import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    Chunk,
    ChunkStore,
    FilingOutcome,
    chunk_filing,
    chunk_id,
)
from filing.stores.parse import WHOLE_DOC_ITEM, ParsedFiling, Section

META = {
    "cik": "0001045810",
    "ticker": "NVDA",
    "form": "10-K",
    "fy": 2024,
    "filed_date": "2024-02-21",
    "period_end": "2024-01-28",
}


def document(*paragraphs: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Text plus block offsets, laid out the way the flattener lays a filing out."""
    text = ""
    blocks: list[tuple[int, int]] = []
    for p in paragraphs:
        start = len(text)
        text += p
        blocks.append((start, len(text)))
        text += "\n\n"
    return text.rstrip("\n"), tuple(blocks)


def cut(*paragraphs: str, item: str = "1A", title: str = "Risk Factors") -> list[Chunk]:
    text, blocks = document(*paragraphs)
    pf = ParsedFiling(
        accn="0001045810-24-000029",
        path=Path("nvda.htm"),
        text=text,
        blocks=blocks,
        sections=(Section("", item, title, 0, 0, len(text)),),
    )
    chunks, _ = chunk_filing(pf, **META)
    return chunks


def para(n: int, word: str = "risk") -> str:
    """A paragraph of exactly ``n`` characters, with spaces to break at."""
    out = (word + " ") * (n // (len(word) + 1) + 2)
    return out[:n].rstrip() + "." * (n - len(out[:n].rstrip()))


# --------------------------------------------------------------------------
# the promise
# --------------------------------------------------------------------------


def test_every_chunk_is_the_slice_its_offsets_name():
    """The M3 gate checks this on 200 random chunks of the real corpus."""
    text, blocks = document(para(900), para(1500), para(60), para(2400))
    pf = ParsedFiling(
        accn="a",
        path=Path("x"),
        text=text,
        blocks=blocks,
        sections=(Section("", "1A", "Risk Factors", 0, 0, len(text)),),
    )
    chunks, _ = chunk_filing(pf, **META)
    assert chunks
    for c in chunks:
        assert text[c.char_start : c.char_end] == c.text
        assert c.n_chars == c.char_end - c.char_start == len(c.text)


def test_a_chunk_id_is_a_pure_function_of_the_filing_and_the_offsets():
    """Re-indexing an unchanged corpus has to produce the same ids or it re-embeds."""
    assert chunk_id("a", 0, 100) == chunk_id("a", 0, 100)
    assert chunk_id("a", 0, 100) != chunk_id("a", 0, 101)
    assert chunk_id("a", 0, 100) != chunk_id("b", 0, 100)


def test_the_hash_is_of_the_text_the_chunk_carries():
    import hashlib

    for c in cut(para(600), para(600)):
        assert c.text_sha256 == hashlib.sha256(c.text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def test_short_blocks_are_packed_together_up_to_the_budget():
    chunks = cut(para(600), para(600), para(600), para(600))
    assert len(chunks) == 2
    assert all(c.n_chars <= CHUNK_CHARS for c in chunks)


def test_a_block_larger_than_the_budget_is_cut_at_whitespace():
    body = para(2400)
    chunks = cut(body)
    assert len(chunks) >= 2
    assert all(c.n_chars <= CHUNK_CHARS for c in chunks)
    # Cut at a space, so no word is sliced in half and the space itself belongs
    # to neither side.
    assert body[chunks[0].char_end] == " "
    assert body[chunks[1].char_start] != " "


def test_a_small_tail_block_is_repeated_as_overlap():
    """Two chunks that share a paragraph, so a sentence on the seam is retrievable."""
    chunks = cut(para(1500), para(150), para(900))
    assert len(chunks) == 2
    assert chunks[1].char_start < chunks[0].char_end
    assert chunks[0].char_end - chunks[1].char_start <= CHUNK_OVERLAP + 2


def test_a_block_bigger_than_the_overlap_window_is_never_repeated():
    """Embedding a whole paragraph twice costs twice and retrieves the same passage twice."""
    chunks = cut(para(900), para(900), para(900), para(900))
    assert all(a.char_end <= b.char_start for a, b in zip(chunks, chunks[1:], strict=False))


def test_a_subheading_is_kept_with_the_thing_it_heads():
    """Pfizer's Item 1A is a run of these. Alone, "INFORMATION TECHNOLOGY" is a
    35-character vector with no content under it, so the packer overshoots instead."""
    chunks = cut("INFORMATION TECHNOLOGY AND SECURITY", para(1790))
    assert len(chunks) == 1
    assert chunks[0].n_chars > CHUNK_CHARS
    assert chunks[0].text.startswith("INFORMATION TECHNOLOGY")


def test_a_trailing_sliver_is_given_to_the_chunk_it_came_off():
    chunks = cut(para(1000), para(1790), para(50))
    assert len(chunks) == 2
    assert chunks[-1].text.endswith(para(50)[-20:])
    assert all(c.n_chars >= MIN_CHUNK_CHARS for c in chunks)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def test_a_section_is_chunked_from_its_heading():
    """ "Item 1A. Risk Factors" is both what a citation quotes and the strongest
    lexical signal BM25 has for the section."""
    chunks = cut("Item 1A. Risk Factors", para(1400))
    assert chunks[0].text.startswith("Item 1A. Risk Factors")


def test_a_stub_section_is_counted_rather_than_chunked():
    """ "Item 6. [Reserved]" is a real section and not a passage."""
    text, blocks = document("Item 6. [Reserved]")
    pf = ParsedFiling(
        accn="a",
        path=Path("x"),
        text=text,
        blocks=blocks,
        sections=(Section("", "6", "Reserved", 0, 0, len(text)),),
    )
    chunks, stubs = chunk_filing(pf, **META)
    assert (chunks, stubs) == ([], 1)


def test_each_chunk_carries_the_item_that_produced_it():
    for c in cut(para(600), para(600), item="7", title="MD&A"):
        assert (c.item, c.title) == ("7", "MD&A")


# --------------------------------------------------------------------------
# what the embedder sees
# --------------------------------------------------------------------------


def test_the_embedded_text_names_the_company_and_the_period():
    """Filing prose is anonymous from the inside: two hundred filings mention
    component shortages and none of them say whose."""
    chunk = make_chunk("Our results were affected by component shortages.")
    assert chunk.embed_text.startswith("NVDA 10-K 2024-01-28 -- Item 1A. Risk Factors")
    assert chunk.text in chunk.embed_text


def test_a_whole_document_chunk_gets_no_item_header():
    """ "Item FULL" would be a header naming a section that does not exist."""
    header = make_chunk("body", item=WHOLE_DOC_ITEM, title="").embed_text.split("\n")[0]
    assert "Item" not in header


@pytest.mark.parametrize(
    ("part", "item", "key"),
    [("", "7", "7"), ("II", "1A", "II.1A"), ("I", "2", "I.2")],
)
def test_the_item_key_carries_the_part_only_when_a_part_was_announced(part, item, key):
    assert make_chunk(part=part, item=item).item_key == key


# --------------------------------------------------------------------------
# the parquet cache
# --------------------------------------------------------------------------


def outcome(**kw) -> FilingOutcome:
    fields = dict(
        accn="0001045810-24-000029",
        ticker="NVDA",
        form="10-K",
        source_sha256="deadbeef",
        chunker="v1-1800-200",
        outcome="split",
        reason="",
        n_sections=4,
        n_stubs=1,
        n_chunks=2,
        n_chars=3600,
    )
    fields.update(kw)
    return FilingOutcome(**fields)


def test_chunks_and_outcomes_round_trip_through_parquet(tmp_path):
    store = ChunkStore(tmp_path)
    assert store.exists is False
    chunks = [make_chunk("first", chunk_id="a"), make_chunk("second", chunk_id="b")]
    store.write(chunks, [outcome()])

    reopened = ChunkStore(tmp_path)
    assert reopened.exists
    assert reopened.chunks() == chunks
    assert reopened.outcomes() == {"0001045810-24-000029": outcome()}


def test_chunks_can_be_read_back_for_named_filings_only(tmp_path):
    """A rebuild re-chunks the filings that changed and reuses the rest."""
    store = ChunkStore(tmp_path)
    store.write(
        [make_chunk("x", chunk_id="a", accn="A"), make_chunk("y", chunk_id="b", accn="B")],
        [outcome(accn="A"), outcome(accn="B")],
    )
    assert [c.accn for c in ChunkStore(tmp_path).chunks({"B"})] == ["B"]


def test_an_absent_store_reads_as_empty_rather_than_an_error(tmp_path):
    assert ChunkStore(tmp_path).chunks() == []
    assert ChunkStore(tmp_path).outcomes() == {}
