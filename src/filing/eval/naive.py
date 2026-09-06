"""The baseline: RAG the way a tutorial writes it.

Every claim this project makes is a comparison, and a comparison needs the other
side built in good faith. So this is not a hobbled version of the real system --
it is the system a competent person builds in an afternoon, with each of the
decisions M3 agonised over replaced by the obvious one:

===========================  ====================  ==========================
what                         the real system       here
===========================  ====================  ==========================
what gets indexed            narrative items only  the whole document
where a chunk may end        a block boundary      every 2,048 characters
overlap                      200 chars, by block   none
what the embedder sees       ticker, form, period  the raw slice
                             and item, then text
retrieval                    dense + BM25 + RRF    dense only
depth                        50 fused, reranked    top 5
                             to 5
===========================  ====================  ==========================

Two of those are the interesting ones. **Cutting on a fixed stride** is what
makes a naive index cheap to build and expensive to trust: a chunk begins
mid-sentence, ends mid-table, and belongs to whichever item happened to be under
the knife -- so a citation from it cannot name a section, which is the first
thing M6 will need. **Embedding the raw slice** matters more than it looks:
filing prose is anonymous from the inside, and "our results were affected by
component shortages" is a sentence four hundred documents in this corpus could
have written. The naive index has no way to tell them apart, and the eval set's
cross-company questions are where that shows up.

The baseline gets one genuine advantage in exchange, and it is not a courtesy:
it indexes the financial statements, which the real system's text index leaves
to SQL. On the numeric questions the naive system is therefore the only one of
the two doing retrieval at all. If it wins any of them, that is a finding.

Both systems share the parser and the embedding model. That is deliberate --
what is being compared is chunking and retrieval, not HTML extraction, and a
baseline that used a different encoder would confound the two.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from filing.config import Settings
from filing.ingest.manifest import Manifest
from filing.stores.chunks import Chunk, ChunkStore, FilingOutcome, chunk_id
from filing.stores.index import VectorIndex
from filing.stores.parse import WHOLE_DOC_ITEM, parse_filing

# 512 tokens of English is about 2,000 characters, and 512 is both the number
# every tutorial uses and bge-small's trained context. Rounded to 2,048 so the
# constant says "a fixed stride" rather than "a tuned one" -- tuning it would
# make this a second real system instead of a baseline.
NAIVE_CHARS = 2_048

# Names the parameters, the same way CHUNKER does, and goes into the Qdrant
# collection name so the baseline's points can never be mistaken for the real
# index's. Bump it if NAIVE_CHARS changes.
NAIVE_CHUNKER = "naive-2048"

# Top-5, dense-only. The real system fuses fifty candidates and reranks; this
# takes the first five vectors it is given, which is the whole difference
# between a retrieval pipeline and a similarity search.
NAIVE_K = 5


@dataclass(frozen=True, slots=True)
class NaiveReport:
    filings: int = 0
    chunks: int = 0
    chars: int = 0
    quarantined: tuple[tuple[str, str], ...] = ()
    collection: str = ""
    already_indexed: int = 0
    upserted: int = 0
    embed_http_calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    by_form: dict[str, int] = field(default_factory=dict)


def naive_dir(cfg: Settings) -> Path:
    """Beside the real chunk cache, never inside it."""
    return cfg.data_dir / "chunks_naive"


# --------------------------------------------------------------------------
# cutting
# --------------------------------------------------------------------------


def naive_cut(text: str) -> list[tuple[int, int]]:
    """Every ``NAIVE_CHARS`` characters, from the top, ignoring the content.

    No block boundaries, no section awareness, no overlap, no minimum size. The
    tail is whatever is left, however short -- a 40-character final chunk is a
    real thing a fixed-stride chunker emits, and hiding it here would be tuning
    the baseline.
    """
    return [(i, min(i + NAIVE_CHARS, len(text))) for i in range(0, len(text), NAIVE_CHARS)]


def naive_chunks(text: str, accn: str, **meta: object) -> list[Chunk]:
    """Chunks over the whole flattened filing, with no item attached.

    ``item`` is ``WHOLE_DOC_ITEM`` for every one of them, which is the truth: a
    fixed-stride cut lands wherever it lands, so this chunker genuinely does not
    know which section a chunk came from. It is stored as unknown rather than
    guessed, because a citation that names the wrong item is worse than one that
    names none.

    The chunk id is namespaced by ``naive:`` so that a baseline chunk and a real
    chunk with coincidentally identical offsets are still different points.
    """
    out: list[Chunk] = []
    for start, end in naive_cut(text):
        body = text[start:end]
        out.append(
            Chunk(
                chunk_id=chunk_id(f"naive:{accn}", start, end),
                accn=accn,
                part="",
                item=WHOLE_DOC_ITEM,
                title="",
                char_start=start,
                char_end=end,
                n_chars=end - start,
                text_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                text=body,
                **meta,  # type: ignore[arg-type]
            )
        )
    return out


def naive_embed_text(chunk: Chunk) -> str:
    """The raw slice. No header, which is the point.

    ``Chunk.embed_text`` prefixes the ticker, form and period; that was a
    considered decision in M3 and it is one of the things being measured here,
    so the baseline must not inherit it.
    """
    return chunk.text


_FILING_SQL = """
SELECT accn, cik, ticker, form, fy,
       CAST(filed_date AS VARCHAR), CAST(period_end AS VARCHAR),
       path, sha256
FROM filings
ORDER BY ticker, period_end, accn
"""


def build_naive_chunks(cfg: Settings, *, rebuild: bool = False) -> NaiveReport:
    """Cut every filing on the fixed stride and cache the result.

    Cached on the same terms as the real chunker -- the filing's sha256 plus the
    chunker name -- so a re-run costs no embedding calls either. The baseline
    has to be reproducible for the same reason the system does: an ablation
    table whose rows were measured against different corpora is not a table.
    """
    started = time.monotonic()
    store = ChunkStore(naive_dir(cfg))
    cached = {} if rebuild else store.outcomes()

    with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
        rows = manifest.con.execute(_FILING_SQL).fetchall()
        paths = {r[0]: manifest.resolve(r[7]) for r in rows}

    def fresh(accn: str, sha: str) -> bool:
        hit = cached.get(accn)
        return hit is not None and hit.source_sha256 == sha and hit.chunker == NAIVE_CHUNKER

    reusable = {r[0] for r in rows if fresh(r[0], r[8])}
    chunks: list[Chunk] = store.chunks(reusable) if reusable else []
    outcomes: list[FilingOutcome] = [cached[a] for a in sorted(reusable)]
    quarantined: list[tuple[str, str]] = [
        (o.accn, o.reason) for o in outcomes if o.outcome == "quarantined"
    ]

    for accn, cik, ticker, form, fy, filed, period, _path, sha in rows:
        if accn in reusable:
            continue
        pf = parse_filing(paths[accn], accn=accn, form=form)
        made = (
            naive_chunks(
                pf.text,
                accn,
                cik=cik,
                ticker=ticker,
                form=form,
                fy=fy,
                filed_date=filed or "",
                period_end=period or "",
            )
            if pf.ok
            else []
        )
        if not pf.ok:
            quarantined.append((accn, pf.quarantine or ""))
        chunks.extend(made)
        outcomes.append(
            FilingOutcome(
                accn=accn,
                ticker=ticker,
                form=form,
                source_sha256=sha,
                chunker=NAIVE_CHUNKER,
                outcome="quarantined" if not pf.ok else "split",
                reason=pf.quarantine or "",
                n_sections=0,  # the baseline does not look for sections
                n_stubs=0,
                n_chunks=len(made),
                n_chars=sum(c.n_chars for c in made),
            )
        )

    chunks.sort(key=lambda c: (c.ticker, c.period_end, c.accn, c.char_start))
    store.write(chunks, outcomes)

    by_form: dict[str, int] = {}
    for c in chunks:
        by_form[c.form] = by_form.get(c.form, 0) + 1

    return NaiveReport(
        filings=len(rows),
        chunks=len(chunks),
        chars=sum(c.n_chars for c in chunks),
        quarantined=tuple(sorted(quarantined)),
        by_form=dict(sorted(by_form.items(), key=lambda kv: -kv[1])),
        seconds=time.monotonic() - started,
    )


def build_naive_index(
    cfg: Settings,
    *,
    rebuild: bool = False,
    limit: int | None = None,
    backend=None,  # noqa: ANN001 - an already-built LLMBackend, for tests
    on_batch=None,  # noqa: ANN001 - progress callback(done, total)
) -> NaiveReport:
    """Embed the naive chunks into their own collection. Dense only, no BM25.

    Resumable the same two ways the real build is -- Qdrant is asked what it
    already holds, and the embedding client caches by text -- because a
    three-hour build that cannot be resumed is a build that gets abandoned.
    """
    from filing.llm.factory import build_embed_backend

    started = time.monotonic()
    be = backend or build_embed_backend(cfg)
    store = ChunkStore(naive_dir(cfg))
    chunks = store.chunks()
    if limit:
        chunks = chunks[:limit]

    dense = VectorIndex(cfg, backend=be.name, variant=NAIVE_CHUNKER)
    if rebuild:
        dense.drop()
    dense.ensure()

    before = be.usage()
    already = upserted = 0
    batch = cfg.embed_batch_size
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        have = dense.present([c.chunk_id for c in window])
        todo = [c for c in window if c.chunk_id not in have]
        already += len(window) - len(todo)
        if todo:
            dense.upsert(todo, be.embed([naive_embed_text(c) for c in todo], input_type="passage"))
            upserted += len(todo)
        if on_batch:
            on_batch(start + len(window), len(chunks))

    after = be.usage()
    by_form: dict[str, int] = {}
    for c in chunks:
        by_form[c.form] = by_form.get(c.form, 0) + 1

    return NaiveReport(
        filings=len({c.accn for c in chunks}),
        chunks=len(chunks),
        chars=sum(c.n_chars for c in chunks),
        collection=dense.name,
        already_indexed=already,
        upserted=upserted,
        embed_http_calls=after.http_calls - before.http_calls,
        cache_hits=after.cache_hits - before.cache_hits,
        by_form=dict(sorted(by_form.items(), key=lambda kv: -kv[1])),
        seconds=time.monotonic() - started,
    )


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


class NaiveRetriever:
    """One embedding call, one vector search, five chunks. That is all of it."""

    def __init__(self, cfg: Settings, *, backend=None) -> None:  # noqa: ANN001
        from filing.llm.factory import build_embed_backend

        self.cfg = cfg
        self.backend = backend or build_embed_backend(cfg)
        self.dense = VectorIndex(cfg, backend=self.backend.name, variant=NAIVE_CHUNKER)
        self._chunks: dict[str, Chunk] | None = None

    def require(self) -> None:
        self.dense.require()

    @property
    def chunks(self) -> dict[str, Chunk]:
        if self._chunks is None:
            self._chunks = {c.chunk_id: c for c in ChunkStore(naive_dir(self.cfg)).chunks()}
        return self._chunks

    def search(self, question: str, *, k: int = NAIVE_K) -> list[Chunk]:
        return [c for c, _ in self.search_scored(question, k=k)]

    def search_scored(self, question: str, *, k: int = NAIVE_K) -> list[tuple[Chunk, float]]:
        """The same search, keeping the cosine similarity beside each chunk.

        The score changes no metric -- every retrieval number here is computed
        from rank and overlap, not from similarity -- but it is what tells a
        miss apart from a near miss when reading the outcomes back, and it costs
        nothing to carry.
        """
        vector = self.backend.embed([question], input_type="query")[0]
        known = self.chunks
        return [(known[i], score) for i, score in self.dense.search(vector, limit=k) if i in known]
