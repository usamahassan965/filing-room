"""The local cross-encoder: that it ranks, and that it stays out of the way.

The second half matters more than it looks. Importing sentence-transformers
costs ~25s on this machine -- almost all of it torch and transformers, not the
80MB of weights, which load in 0.4s. If that import ever creeps up to module
scope, every `filing` command pays it, including the ones that never rerank.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from filing.llm.rerank_local import LocalReranker

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QUERY = "What did the company say about supply chain risk?"
PASSAGES = [
    "Net revenues for fiscal 2024 increased 8% to $394.3 billion compared with fiscal 2023.",
    "Our results could be harmed if our suppliers cannot obtain components, or if "
    "manufacturing is disrupted at facilities concentrated in a single region.",
    "The board declared a quarterly dividend of $0.25 per share payable in November.",
]
RELEVANT = 1


def test_importing_the_module_is_cheap():
    """No torch at import time, or the CLI's startup budget is gone."""
    probe = (
        "import sys; import filing.llm.rerank_local as m; "
        "assert 'sentence_transformers' not in sys.modules, 'imported eagerly'; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_no_passages_needs_no_model():
    """The empty case must not pay 25 seconds to return an empty list."""
    assert LocalReranker(MODEL).rank("anything", []) == []


@pytest.fixture(scope="module")
def reranker():
    r = LocalReranker(MODEL)
    try:
        r.warm()
    except Exception as exc:  # noqa: BLE001 - offline or model not cached
        pytest.skip(f"cross-encoder unavailable ({type(exc).__name__}); run `filing probe` online")
    return r


def test_it_ranks_the_relevant_passage_first(reranker):
    """A real assertion, not a smoke formality.

    Cosine over embeddings gets this wrong often enough to be worth the test:
    the revenue passage shares more surface vocabulary with a filing than the
    supply-chain one shares with the question.
    """
    ranked = reranker.rank(QUERY, PASSAGES)

    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True), "not in descending score order"
    assert ranked[0].index == RELEVANT
    # And it should not be a close call.
    assert ranked[0].score - ranked[1].score > 1.0


def test_scores_are_returned_for_every_passage(reranker):
    ranked = reranker.rank(QUERY, PASSAGES)
    assert sorted(r.index for r in ranked) == list(range(len(PASSAGES)))
