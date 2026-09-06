"""Hybrid retrieval: two rankers, one fusion rule, one cross-encoder.

The pipeline is dense -> sparse -> RRF -> rerank, and every stage is a separate,
inspectable object because the M3 gate has to show that the last one *earns its
place*. "We added a reranker" is not a claim this project is allowed to make
without a precision@5 measured with it and without it.

**Why Reciprocal Rank Fusion and not score normalisation.** Cosine similarity
lives in [-1, 1] and is comparable across queries; BM25 is unbounded and its
scale moves with corpus statistics and query length. Putting them on one scale
means picking a normalisation -- min-max over the returned window, z-scores, a
weight -- and every one of those is a tuning knob with no principled value and a
strong tendency to be fitted to the eval set. RRF throws the scores away and
keeps only the ranks::

    score(d) = sum over rankers of 1 / (k + rank(d))

It has one constant, and the constant is not sensitive: k=60 is what the
original paper used, and it exists to stop rank 1 from dominating rank 2 by a
factor of two.

**Why a cross-encoder afterwards.** RRF fuses two rankers that never read the
query and the passage together -- a bi-encoder compares two vectors made
independently, and BM25 counts terms. The cross-encoder reads the pair. It is
far too slow to run over 32,000 chunks and exactly right over 50.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from filing.config import Settings
from filing.stores.chunks import Chunk, ChunkStore
from filing.stores.index import CollectionMismatch, SparseIndex, VectorIndex, select

# From the paper (Cormack, Clarke & Buettcher 2009). Left at 60 rather than
# tuned: a constant fitted on this project's own 30-question smoke set would
# make the smoke set a training set and the gate a tautology.
RRF_K = 60

# How many candidates each ranker contributes, and how many survive fusion into
# the cross-encoder. 50 in / 5 out is the shape the DoD measures.
DENSE_K = 50
SPARSE_K = 50
FUSED_K = 50


@dataclass(frozen=True, slots=True)
class Fused:
    chunk_id: str
    score: float
    ranks: Mapping[str, int]  # ranker name -> 1-based rank, only where it hit


def rrf(runs: Mapping[str, Sequence[tuple[str, float]]], *, k: int = RRF_K) -> list[Fused]:
    """Fuse ranked runs by reciprocal rank. Scores in, ranks used, order out."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for ranker, run in runs.items():
        for i, (doc_id, _score) in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i)
            ranks.setdefault(doc_id, {})[ranker] = i
    return [
        Fused(chunk_id=d, score=s, ranks=ranks[d])
        for d, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk and the evidence for why it is here."""

    chunk: Chunk
    fused_score: float
    ranks: Mapping[str, int] = field(default_factory=dict)
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        """What M6 will print. Offsets included, because they are checkable."""
        item = "" if self.chunk.item == "FULL" else f" Item {self.chunk.item_key}"
        return (
            f"{self.chunk.ticker} {self.chunk.form} {self.chunk.period_end}{item} "
            f"[{self.chunk.accn} {self.chunk.char_start}:{self.chunk.char_end}]"
        )


def matches(chunk: Chunk, where: Mapping[str, object] | None) -> bool:
    """The same filter the dense side pushes into Qdrant, applied in Python.

    BM25 has no notion of a payload filter, so its half of a filtered search is
    filtered here. Both halves must agree or a filter would quietly mean "dense
    only", which is a retrieval quality change disguised as a scoping option.
    """
    if not where:
        return True
    for key, want in where.items():
        got = chunk.item_key if key == "item_key" else getattr(chunk, key, None)
        if isinstance(want, (list, tuple, set)):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


class Retriever:
    """Dense + sparse + RRF + cross-encoder, over the chunks on disk."""

    def __init__(self, cfg: Settings, *, backend=None) -> None:  # noqa: ANN001
        from filing.llm.factory import build_embed_backend

        self.cfg = cfg
        self.backend = backend or build_embed_backend(cfg)
        self.dense = VectorIndex(cfg, backend=self.backend.name)
        self.sparse = SparseIndex(cfg.index_dir)
        self._by_id: dict[str, Chunk] | None = None

    def require(self) -> None:
        self.dense.require()
        if not self.sparse.exists:
            raise CollectionMismatch(f"no BM25 index at {self.sparse.path} -- run `filing index`")

    @property
    def chunks(self) -> dict[str, Chunk]:
        """The chunks the index holds -- not every chunk on disk.

        ``select`` is applied here for the same reason it is applied at build
        time: a retriever that knows about Item 8 can never return it, and an
        evaluation that counts an Item 8 chunk as gold scores retrieval down
        for a selection decision M3 made on purpose. The set the metric is
        computed over has to be the set the index could return.
        """
        if self._by_id is None:
            chunks = select(ChunkStore(self.cfg.chunks_dir).chunks())
            self._by_id = {c.chunk_id: c for c in chunks}
        return self._by_id

    def resolve(self, ids: Iterable[str]) -> list[Chunk]:
        by_id = self.chunks
        return [by_id[i] for i in ids if i in by_id]

    # ------------------------------------------------------------- rankers

    def dense_run(
        self, query: str, *, limit: int = DENSE_K, where: Mapping[str, object] | None = None
    ) -> list[tuple[str, float]]:
        # input_type="query" is not cosmetic. Every embedding model this
        # project uses is asymmetric -- bge takes an instruction prefix on the
        # query side, gemini takes a taskType -- and embedding a question as
        # though it were a passage puts it in the wrong region of the space.
        vector = self.backend.embed([query], input_type="query")[0]
        return self.dense.search(vector, limit=limit, where=dict(where) if where else None)

    def sparse_run(
        self, query: str, *, limit: int = SPARSE_K, where: Mapping[str, object] | None = None
    ) -> list[tuple[str, float]]:
        # Over-fetch when filtering, since the filter is applied afterwards.
        raw = self.sparse.search(query, limit=limit * (8 if where else 1))
        by_id = self.chunks
        out = [(i, s) for i, s in raw if i in by_id and matches(by_id[i], where)]
        return out[:limit]

    # ------------------------------------------------------------- pipeline

    def search(
        self,
        query: str,
        *,
        k: int = FUSED_K,
        where: Mapping[str, object] | None = None,
        rerank: bool = True,
        top_n: int = 5,
    ) -> list[Hit]:
        runs = {
            "dense": self.dense_run(query, limit=DENSE_K, where=where),
            "sparse": self.sparse_run(query, limit=SPARSE_K, where=where),
        }
        fused = rrf(runs)[:k]
        by_id = self.chunks
        hits = [
            Hit(chunk=by_id[f.chunk_id], fused_score=f.score, ranks=f.ranks)
            for f in fused
            if f.chunk_id in by_id
        ]
        if not rerank or not hits:
            return hits[:top_n] if rerank else hits

        rankings = self.backend.rerank(query, [h.chunk.text for h in hits], top_n=top_n)
        return [
            Hit(
                chunk=hits[r.index].chunk,
                fused_score=hits[r.index].fused_score,
                ranks=hits[r.index].ranks,
                rerank_score=r.score,
            )
            for r in rankings[:top_n]
        ]


def quote(chunk: Chunk, pattern: str, *, window: int = 160) -> str | None:
    """The sentence in a chunk that matched, for showing why it was retrieved."""
    mo = re.search(pattern, chunk.text, re.I)
    if not mo:
        return None
    lo = max(0, mo.start() - window // 2)
    return chunk.text[lo : mo.end() + window // 2].replace("\n", " ")
