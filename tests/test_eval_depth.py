"""The depth diagnostic: distinguishing "ranked 24th" from "absent".

Driven entirely by a fake retriever with a scripted ranking, because what is
being tested is the arithmetic of the curve -- not whether bge-small is any
good, which is a fact about the world and belongs in a results file.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_chunk
from filing.config import Settings
from filing.eval import dataset, depth
from filing.eval.dataset import EvalQuestion, Span


def question(qid: str, slice_: str, accn: str, start: int, end: int) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        slice=slice_,
        question=f"question {qid}",
        value=1.0 if slice_ == "numeric" else None,
        unit="USD" if slice_ == "numeric" else "",
        tag="Revenues" if slice_ == "numeric" else "",
        spans=(Span(accn, start, end, "gold"),),
        origin="test",
        gold_source="test",
    )


class ScriptedRetriever:
    """Returns a fixed ranking per query, and knows a fixed set of chunks."""

    def __init__(self, ranking: dict[str, list], chunks: list) -> None:
        self.ranking = ranking
        self.chunks = {c.chunk_id: c for c in chunks}

    def require(self) -> None:  # pragma: no cover - the real one talks to Qdrant
        pass

    def search(self, query: str, *, k: int = 5):
        return self.ranking[query][:k]


@pytest.fixture
def env(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data")
    qs = [
        question("nar-001", "narrative", "A", 0, 100),
        question("nar-002", "narrative", "B", 0, 100),
        question("num-001", "numeric", "C", 0, 10),
    ]
    dataset.write(qs, dataset.dataset_path(cfg.data_dir))
    return cfg


def chunks_for(accn: str, n: int, *, width: int = 100):
    return [
        make_chunk(f"{accn} body {i}", accn=accn, char_start=i * width, chunk_id=f"{accn}{i}")
        for i in range(n)
    ]


def test_a_gold_chunk_ranked_deep_is_found_but_not_hit_early(env):
    """The whole point: rank 3 is a miss at 1 and a hit at 5."""
    gold, filler = chunks_for("A", 1)[0], chunks_for("Z", 5)
    ret = ScriptedRetriever(
        {
            "question nar-001": [filler[0], filler[1], gold],
            "question nar-002": [filler[0]],
            "question num-001": [filler[0]],
        },
        [gold, *filler],
    )
    s = depth.measure(env, retriever=ret, depth=10)["narrative"]
    assert s.n == 2 and s.found == 1
    assert s.hit_at[1] == 0.0
    assert s.hit_at[5] == 0.5  # one of the two narrative questions
    assert s.median_rank == 3


def test_the_right_filing_is_counted_even_when_the_right_chunk_is_not(env):
    """The gap between these two is the finding the module exists to expose."""
    right_filing_wrong_place = make_chunk("elsewhere", accn="A", char_start=5_000, chunk_id="A99")
    gold = chunks_for("A", 1)[0]
    ret = ScriptedRetriever(
        {
            "question nar-001": [right_filing_wrong_place],
            "question nar-002": [],
            "question num-001": [],
        },
        [gold, right_filing_wrong_place],
    )
    s = depth.measure(env, retriever=ret, depth=10)["narrative"]
    assert s.found == 0, "the gold chunk was never returned"
    assert s.filing_found == 1 and s.filing_median_rank == 1
    assert s.filing_hit_at_10 == 0.5


def test_the_unanswerable_slice_is_excluded(env, tmp_path):
    """It has no evidence, so a rank for it would be a category error."""
    qs = list(dataset.read(dataset.dataset_path(env.data_dir)))
    una = EvalQuestion(
        id="una-001",
        slice="unanswerable",
        question="question una-001",
        value=None,
        unit="",
        tag="",
        spans=(),
        origin="test",
        gold_source="test",
    )
    dataset.write([*qs, una], dataset.dataset_path(env.data_dir))
    ret = ScriptedRetriever({f"question {q.id}": [] for q in [*qs, una]}, [])
    assert "unanswerable" not in depth.measure(env, retriever=ret, depth=10)


def test_depths_beyond_the_search_are_not_reported(env):
    """A curve should not carry a hit@500 when only 10 were retrieved."""
    ret = ScriptedRetriever({f"question {q}": [] for q in ("nar-001", "nar-002", "num-001")}, [])
    s = depth.measure(env, retriever=ret, depth=25)["narrative"]
    assert set(s.hit_at) == {1, 5, 10, 25}


def test_it_writes_a_table_and_a_json_file(env, tmp_path):
    ret = ScriptedRetriever({f"question {q}": [] for q in ("nar-001", "nar-002", "num-001")}, [])
    slices = depth.measure(env, retriever=ret, depth=10)
    root = tmp_path / "results"
    path = depth.write(env, slices, root=root)
    assert json.loads(path.read_text("utf-8"))["numeric"]["n"] == 1
    table = (root / "retrieval-depth.md").read_text("utf-8")
    assert "| narrative |" in table and "hit@10" in table
