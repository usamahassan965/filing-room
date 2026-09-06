"""Fusion, filtering and citation -- the parts of retrieval with no model in them.

The pipeline as a whole is measured by the M3 gate, on the real index, against
the smoke set. What is left for a unit test is the arithmetic and the
bookkeeping: that RRF really uses ranks and not scores, that a filter means the
same thing on both sides of the fusion, and that a citation names offsets a
reader could check.

``Retriever`` itself is exercised with stub rankers. Standing up Qdrant and an
encoder to prove that a list gets sorted would test the fixture, not the code.
"""

from __future__ import annotations

from conftest import make_chunk
from filing.stores.retrieve import RRF_K, Hit, matches, quote, rrf

# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def test_a_document_both_rankers_return_beats_one_either_ranker_loves():
    """The whole reason to fuse. Agreement outranks a single confident vote.

    ``b`` is second on both runs; ``a`` is first on one and absent from the
    other. 2/62 > 1/61, so ``b`` wins -- and it wins by a small margin, which is
    what k=60 is for.
    """
    runs = {
        "dense": [("a", 0.99), ("b", 0.98)],
        "sparse": [("c", 31.0), ("b", 12.0)],
    }
    assert [f.chunk_id for f in rrf(runs)] == ["b", "a", "c"]


def test_fusion_ignores_the_scores_entirely():
    """BM25's 31.0 and cosine's 0.99 are not on one scale, so neither is used."""
    ranked = [("a", 1e9), ("b", 1e-9)]
    unranked = [("a", 0.1), ("b", 0.0)]
    assert [f.score for f in rrf({"x": ranked})] == [f.score for f in rrf({"x": unranked})]


def test_the_fused_score_is_the_sum_of_reciprocal_ranks():
    [only] = rrf({"dense": [("a", 0.5)], "sparse": [("a", 3.0)]})
    assert only.score == 2 / (RRF_K + 1)


def test_fusion_records_where_each_ranker_put_a_document():
    """The provenance a trace needs: which half of the hybrid found this, and at what rank."""
    runs = {"dense": [("a", 1.0)], "sparse": [("b", 1.0), ("a", 0.5)]}
    fused = {f.chunk_id: f for f in rrf(runs)}
    assert fused["a"].ranks == {"dense": 1, "sparse": 2}
    assert fused["b"].ranks == {"sparse": 1}


def test_ties_break_on_the_id_so_the_order_is_reproducible():
    """Two documents at rank 1 of different runs score identically. An eval needs one answer."""
    order = [f.chunk_id for f in rrf({"dense": [("z", 1.0)], "sparse": [("a", 1.0)]})]
    assert order == ["a", "z"]


def test_fusing_nothing_returns_nothing():
    assert rrf({"dense": [], "sparse": []}) == []


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------


def test_no_filter_matches_everything():
    assert matches(make_chunk(), None) is True
    assert matches(make_chunk(), {}) is True


def test_a_scalar_filter_is_equality():
    chunk = make_chunk(ticker="NVDA")
    assert matches(chunk, {"ticker": "NVDA"})
    assert not matches(chunk, {"ticker": "AMD"})


def test_a_sequence_filter_is_membership():
    chunk = make_chunk(ticker="NVDA")
    assert matches(chunk, {"ticker": ["NVDA", "AMD"]})
    assert not matches(chunk, {"ticker": ["INTC", "AMD"]})


def test_every_clause_must_hold():
    chunk = make_chunk(ticker="NVDA", form="10-K")
    assert not matches(chunk, {"ticker": "NVDA", "form": "10-Q"})


def test_item_key_filters_on_the_part_qualified_item():
    """``item_key`` is a property, not a column. Getattr on ``item`` would silently
    match Part I's Item 1A against Part II's."""
    chunk = make_chunk(form="10-Q", part="II", item="1A")
    assert matches(chunk, {"item_key": "II.1A"})
    assert not matches(chunk, {"item_key": "1A"})


def test_an_unknown_field_never_matches_rather_than_being_ignored():
    """A typo in a filter should return nothing, not quietly return everything."""
    assert not matches(make_chunk(), {"no_such_field": "x"})


# --------------------------------------------------------------------------
# citation and quoting
# --------------------------------------------------------------------------


def test_a_citation_names_the_filing_the_item_and_the_offsets():
    hit = Hit(chunk=make_chunk(text="x" * 500, char_start=1000), fused_score=0.1)
    assert hit.citation == "NVDA 10-K 2024-01-28 Item 1A [0000000000-00-000000 1000:1500]"


def test_a_whole_document_citation_omits_the_item():
    """ "Item FULL" would be a citation to a section that does not exist."""
    hit = Hit(chunk=make_chunk(item="FULL", title=""), fused_score=0.1)
    assert " Item " not in hit.citation


def test_quote_returns_the_matching_sentence_in_context():
    text = "Preamble. " * 30 + "Export controls limited our sales. " + "Afterword. " * 30
    got = quote(make_chunk(text), r"export controls")
    assert got is not None
    assert "Export controls limited our sales." in got
    assert len(got) < len(text)


def test_quote_flattens_newlines_so_a_citation_stays_one_line():
    assert "\n" not in (quote(make_chunk("alpha\nbeta gamma"), "beta") or "")


def test_quote_returns_none_when_the_pattern_is_not_there():
    assert quote(make_chunk("nothing relevant"), r"export controls") is None


# --------------------------------------------------------------------------
# what the retriever knows about
# --------------------------------------------------------------------------


class _StubBackend:
    """Enough of a backend for ``Retriever`` to name a collection."""

    name = "local"


def test_the_retriever_knows_only_the_chunks_the_index_holds(tmp_path):
    """Item 8 is 36% of the corpus and none of it is indexed.

    Leaving it in the map costs more than memory: the eval computes its gold
    set from exactly this dictionary, so an unindexed chunk counted as gold
    scores retrieval down for a selection decision M3 made on purpose.
    """
    from filing.config import Settings
    from filing.stores.chunks import ChunkStore, FilingOutcome
    from filing.stores.retrieve import Retriever

    prose = make_chunk("risk factors", chunk_id="prose", item="1A")
    statements = make_chunk("consolidated balance sheets", chunk_id="statements", item="8")
    cfg = Settings(data_dir=tmp_path)
    ChunkStore(cfg.chunks_dir).write(
        [prose, statements],
        [
            FilingOutcome(
                accn=prose.accn,
                ticker="NVDA",
                form="10-K",
                source_sha256="deadbeef",
                chunker="v1",
                outcome="split",
                reason="",
                n_sections=2,
                n_stubs=0,
                n_chunks=2,
                n_chars=40,
            )
        ],
    )

    known = Retriever(cfg, backend=_StubBackend()).chunks
    assert set(known) == {"prose"}
