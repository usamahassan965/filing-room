"""The asymmetry, tested without loading a model.

A bi-encoder trained asymmetrically has one silent failure mode: prefix the
passages instead of the queries and everything still runs, every vector is still
384 numbers, and retrieval is merely worse. Nothing raises. So the prefix table
gets a test, and the test is cheap because ``prefixed`` is a pure function --
loading 133MB of weights to check a string concatenation would be the slowest
possible way to learn nothing.
"""

from __future__ import annotations

import pytest

from filing.config import model_for
from filing.llm.embed_local import BGE_QUERY_PREFIX, LocalEmbedder, _family


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("BAAI/bge-small-en-v1.5", "bge"),
        ("BAAI/bge-base-en-v1.5", "bge"),
        ("intfloat/e5-small-v2", "e5"),
        ("thenlper/gte-small", "gte"),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_the_family_comes_from_the_model_name(model_id, family):
    assert _family(model_id) == family


def test_a_bge_query_carries_the_instruction_it_was_trained_on():
    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    assert embedder.prefixed("export controls", "query") == BGE_QUERY_PREFIX + "export controls"


def test_a_bge_passage_carries_nothing():
    """Prefixing the passage side is the mistake this table exists to prevent."""
    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    assert embedder.prefixed("Our export licences", "passage") == "Our export licences"


def test_e5_prefixes_both_sides_because_that_is_how_e5_was_trained():
    embedder = LocalEmbedder("intfloat/e5-small-v2")
    assert embedder.prefixed("x", "query") == "query: x"
    assert embedder.prefixed("x", "passage") == "passage: x"


def test_an_unlisted_model_is_treated_as_symmetric():
    embedder = LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert embedder.prefixed("x", "query") == embedder.prefixed("x", "passage") == "x"


def test_the_configured_embedding_model_is_one_the_prefix_table_knows():
    """A model swapped in without a prefix entry would silently lose the asymmetry."""
    spec = model_for("embed", "local")
    assert spec.asymmetric is True
    assert _family(spec.id) == "bge"


def test_the_registered_width_is_the_one_the_collection_name_promises():
    """``collection_name`` bakes this number in, so it has to be the model's real width."""
    assert model_for("embed", "local").dim == 384


def test_encoding_nothing_loads_nothing():
    """The index build hands over empty batches at the end of a filing."""
    assert LocalEmbedder("a-model-that-does-not-exist").encode([]) == []
