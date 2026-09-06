"""The text index: dense vectors in Qdrant, a sparse BM25 index beside it.

Two indexes because the two halves fail differently. A dense index will happily
return a passage about "component shortages" for a query about "supply chain
disruption" and miss the one filing that spells a part number out; BM25 does the
opposite. Fusing them (see ``filing.stores.retrieve``) is not belt-and-braces --
it is the cheapest way to stop a retrieval system from having one blind spot it
can never be told about.

**The collection name carries the vector's identity.** It is
``filing__{backend}__{model}__{dim}``, so switching ``embed_backend`` from
local to gemini does not query 384-dimension neighbours with a 1536-dimension
vector.
It looks for a collection that does not exist and says so. That is the whole
point: a backend swap has to be a startup error, because the alternative -- a
silently wrong nearest neighbour -- is indistinguishable from a working system
until someone reads the answers closely.

**What gets indexed, and what does not.** The financial statements (10-K Item 8,
10-Q Item 1) are 36% of the corpus by chunk count and they are the one part of a
filing this project already has in structured form: M2's ``facts.duckdb`` holds
every XBRL fact with its own units, periods and dimensions. Retrieving a
fragment of a rendered balance sheet to answer "what was revenue in FY24" is a
worse answer than the SQL row, and it costs the most embedding tokens of
anything in the corpus. The exhibit indexes, fee tables and signature pages go
for a duller reason: they are lists of file names.

So the index covers what a *narrative* question lands in, and the numbers come
from the numeric store. ``NARRATIVE_ITEMS`` is the list, by form.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from filing.config import Settings, model_for
from filing.stores.chunks import Chunk, ChunkStore

# 10-K keys are bare; 10-Q keys carry the part, because a 10-Q restarts its
# numbering (see filing.stores.parse.Section.key). "FULL" is the whole-document
# section a degraded filing gets -- all twenty of those are Intel, and dropping
# them would drop a company.
NARRATIVE_ITEMS: dict[str, frozenset[str]] = {
    "10-K": frozenset(
        {
            "1",  # Business
            "1A",  # Risk Factors
            "1C",  # Cybersecurity
            "2",  # Properties
            "3",  # Legal Proceedings
            "5",  # Market for common equity
            "7",  # MD&A
            "7A",  # Quantitative and qualitative disclosures about market risk
            "9A",  # Controls and procedures
            "FULL",
        }
    ),
    "10-Q": frozenset(
        {
            "I.2",  # MD&A
            "I.3",  # Market risk
            "I.4",  # Controls and procedures
            "I.1A",  # Risk factors, where a filer puts them in Part I
            "II.1",  # Legal proceedings
            "II.1A",  # Risk factors
            "II.2",  # Unregistered sales
            "II.5",  # Other information
            "FULL",
        }
    ),
}


def is_narrative(chunk: Chunk) -> bool:
    allowed = NARRATIVE_ITEMS.get(chunk.form.upper())
    return chunk.item_key in allowed if allowed else True


def collection_name(backend: str, model_id: str, dim: int, variant: str = "") -> str:
    """Backend, model and width, in the name. See the module docstring.

    ``variant`` names the *chunking* that produced the points. Two collections
    can share a backend, a model and a width and still be incomparable, because
    what was cut up differently is not the same corpus -- M4's naive baseline is
    exactly that, and it has to sit beside the real index rather than on top of
    it. Empty for the system's own index, so its name is unchanged.
    """
    slug = model_id.replace("/", "_").replace(":", "_").replace(".", "-")
    tail = f"__{variant}" if variant else ""
    return f"filing__{backend}__{slug}__{dim}{tail}"


class CollectionMismatch(RuntimeError):
    """The index on disk was not built by the backend that is now asking."""


@dataclass(frozen=True, slots=True)
class IndexReport:
    collection: str = ""
    model: str = ""
    dim: int = 0
    candidates: int = 0  # chunks in the store
    selected: int = 0  # chunks the narrative filter kept
    already_indexed: int = 0
    upserted: int = 0
    embed_http_calls: int = 0
    cache_hits: int = 0
    sparse_documents: int = 0
    paced_seconds: float = 0.0  # time spent under the token budget rather than in a 429
    seconds: float = 0.0
    by_item: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------

_PAYLOAD_FIELDS = (
    "accn",
    "ticker",
    "form",
    "fy",
    "period_end",
    "item_key",
    "title",
    "char_start",
    "char_end",
    "text_sha256",
)

# Indexed so a filter is a filter and not a full scan. Qdrant will filter on an
# unindexed payload key, slowly and without telling you.
_FILTER_FIELDS = ("ticker", "form", "item_key", "fy", "accn")


def payload_of(chunk: Chunk) -> dict[str, object]:
    """Metadata a filter needs -- not the text.

    The text lives in one place, ``chunks.parquet``, and is looked up by id at
    query time. Copying it here would give the system two versions of the string
    a citation quotes, which is one more than can be kept honest.
    """
    d = {f: getattr(chunk, f, None) for f in _PAYLOAD_FIELDS}
    d["item_key"] = chunk.item_key
    return d


class VectorIndex:
    """Qdrant, wrapped thinly enough that LanceDB could replace it."""

    def __init__(self, cfg: Settings, *, backend: str | None = None, variant: str = "") -> None:
        from qdrant_client import QdrantClient

        self.cfg = cfg
        self.backend = backend or cfg.embed_backend
        self.variant = variant
        spec = model_for("embed", self.backend)  # type: ignore[arg-type]
        self.model = spec.id
        self.dim = spec.dim or 0
        if not self.dim:  # pragma: no cover - registry error
            raise ValueError(f"no dim registered for {self.backend} embed model {self.model}")
        self.name = collection_name(self.backend, self.model, self.dim, variant)
        self.client = QdrantClient(url=cfg.qdrant_url, timeout=cfg.qdrant_timeout_s)

    def exists(self) -> bool:
        return self.client.collection_exists(self.name)

    def ensure(self) -> None:
        from qdrant_client import models

        if not self.exists():
            self.client.create_collection(
                self.name,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
        info = self.client.get_collection(self.name)
        size = info.config.params.vectors.size  # type: ignore[union-attr]
        if size != self.dim:  # pragma: no cover - the name makes this near-impossible
            raise CollectionMismatch(
                f"{self.name} holds {size}-dimension vectors, {self.model} makes {self.dim}"
            )
        for f in _FILTER_FIELDS:
            try:
                self.client.create_payload_index(self.name, f, field_schema="keyword")
            except Exception:  # already indexed; Qdrant has no create-if-missing
                pass

    def require(self) -> None:
        """Fail loudly rather than search an index this backend did not build."""
        if not self.exists() or self.count() == 0:
            raise CollectionMismatch(
                f"no index for backend {self.backend!r} / model {self.model} "
                f"({self.dim}d): collection {self.name} is missing or empty. "
                f"Run `filing index` with LLM_BACKEND={self.backend}."
            )

    def count(self) -> int:
        return self.client.count(self.name, exact=True).count

    def present(self, ids: list[str]) -> set[str]:
        got = self.client.retrieve(self.name, ids=ids, with_payload=False, with_vectors=False)
        return {str(p.id) for p in got}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        self.client.upsert(
            self.name,
            points=[
                models.PointStruct(id=c.chunk_id, vector=v, payload=payload_of(c))
                for c, v in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        *,
        limit: int,
        where: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        from qdrant_client import models

        flt = None
        if where:
            flt = models.Filter(
                must=[
                    models.FieldCondition(key=k, match=models.MatchAny(any=list(v)))
                    if isinstance(v, (list, tuple, set))
                    else models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in where.items()
                ]
            )
        res = self.client.query_points(
            self.name, query=vector, limit=limit, query_filter=flt, with_payload=False
        )
        return [(str(p.id), float(p.score)) for p in res.points]

    def drop(self) -> None:
        if self.exists():
            self.client.delete_collection(self.name)


# --------------------------------------------------------------------------
# sparse
# --------------------------------------------------------------------------


class SparseIndex:
    """BM25 over the same chunks, on disk, with the id mapping beside it.

    ``bm25s`` retrieves by corpus position, so the position-to-chunk-id map is
    part of the index and is saved with it. Rebuilding one without the other
    would return real scores attached to the wrong documents -- the failure mode
    that looks like a working system.
    """

    FILE = "bm25"
    IDS = "bm25_ids.json"

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.ids: list[str] = []
        self._retriever = None
        self._stemmer = None

    @property
    def path(self) -> Path:
        return self.dir / self.FILE

    @property
    def exists(self) -> bool:
        return self.path.exists() and (self.dir / self.IDS).exists()

    def _stem(self):  # noqa: ANN202
        if self._stemmer is None:
            import Stemmer

            self._stemmer = Stemmer.Stemmer("english")
        return self._stemmer

    def build(self, chunks: list[Chunk]) -> int:
        import bm25s

        self.dir.mkdir(parents=True, exist_ok=True)
        # The same string the dense half embeds, so a term in the header
        # ("NVDA", "10-K") is searchable in both halves or in neither.
        corpus = [c.embed_text for c in chunks]
        tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self._stem(), show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        retriever.save(str(self.path))
        (self.dir / self.IDS).write_text(json.dumps([c.chunk_id for c in chunks]), encoding="utf-8")
        self.ids = [c.chunk_id for c in chunks]
        self._retriever = retriever
        return len(corpus)

    def load(self) -> None:
        import bm25s

        if not self.exists:
            raise CollectionMismatch(f"no BM25 index at {self.path} -- run `filing index`")
        self._retriever = bm25s.BM25.load(str(self.path), mmap=False)
        self.ids = json.loads((self.dir / self.IDS).read_text(encoding="utf-8"))

    def search(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        import bm25s

        if self._retriever is None:
            self.load()
        tokens = bm25s.tokenize(query, stemmer=self._stem(), show_progress=False)
        k = min(limit, len(self.ids))
        if k == 0:
            return []
        docs, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return [(self.ids[int(i)], float(s)) for i, s in zip(docs[0], scores[0], strict=True)]


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


# The client's own retry policy is tuned for a call that should have worked.
# This one is for the call that was always going to be refused: the free tier
# serves roughly 40k embedding tokens a minute, the corpus needs about fifteen
# million of them, and a four-hour pass will meet the ceiling many times. Five
# exponential retries inside the client is the wrong shape for that -- what is
# needed is to stop asking for a while. Each wait is a minute, because that is
# the width of the window being waited on.
QUOTA_WAIT_S = 60.0
QUOTA_WAITS = 20


class TokenPacer:
    """Spend a tokens-per-minute budget on purpose instead of discovering it.

    The free tier's embedding limit is a token ceiling, not a request ceiling:
    single calls go through in a second while a steady stream of full batches
    is refused. Waiting for the 429 and backing off costs far more than the
    quota saved -- tenacity climbs to a minute, the caller waits another minute
    on top, and a batch that needs thirty seconds of patience takes three and a
    half minutes. Measured on this corpus: 40 chunks a minute reactively
    against a ceiling that permits about 60.

    So the budget is spent forwards. Every batch declares its size, the pacer
    holds it until the last sixty seconds have room, and the 429 handler below
    stays as the thing that catches a ceiling this does not know about.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        self.budget = tokens_per_minute
        self._spent: deque[tuple[float, int]] = deque()

    def wait(self, tokens: int) -> float:
        waited = 0.0
        while True:
            now = time.monotonic()
            while self._spent and now - self._spent[0][0] > 60.0:
                self._spent.popleft()
            used = sum(n for _, n in self._spent)
            if not self._spent or used + tokens <= self.budget:
                self._spent.append((now, tokens))
                return waited
            nap = min(5.0, 60.0 - (now - self._spent[0][0]) + 0.1)
            time.sleep(nap)
            waited += nap


def estimate_tokens(text: str) -> int:
    """Four characters to a token. Close enough to pace with, and free."""
    return max(1, len(text) // 4)


def _embed_waiting_out_the_quota(backend, texts: list[str]) -> list[list[float]]:  # noqa: ANN001
    last: Exception | None = None
    for attempt in range(QUOTA_WAITS):
        try:
            return backend.embed(texts, input_type="passage")
        except Exception as exc:  # noqa: BLE001 - re-raised below if it never clears
            last = exc
            if "429" not in str(exc) and "quota" not in str(exc).lower():
                raise
            time.sleep(QUOTA_WAIT_S * (1 + attempt // 5))
    raise RuntimeError(f"embedding quota did not clear after {QUOTA_WAITS} waits: {last}") from last


def select(chunks: list[Chunk], *, all_items: bool = False) -> list[Chunk]:
    return chunks if all_items else [c for c in chunks if is_narrative(c)]


def build_index(
    cfg: Settings,
    *,
    rebuild: bool = False,
    all_items: bool = False,
    limit: int | None = None,
    backend=None,  # noqa: ANN001 - an already-built LLMBackend, for tests
    on_batch=None,  # noqa: ANN001 - progress callback(done, total)
) -> IndexReport:
    """Embed the narrative chunks into Qdrant and build BM25 beside them.

    Resumable by construction, in two independent ways. Qdrant is asked which
    ids it already holds, so a run that died at 60% picks up at 60%; and the
    embedding client caches every vector by text, so even an id Qdrant has lost
    is re-upserted without a second HTTP call. That is what makes the gate's
    "re-indexing an unchanged corpus costs zero embedding calls" a property of
    the design rather than a thing to remember not to do.
    """
    from filing.llm.factory import build_embed_backend

    started = time.monotonic()
    be = backend or build_embed_backend(cfg)
    store = ChunkStore(cfg.chunks_dir)
    everything = store.chunks()
    chosen = select(everything, all_items=all_items)
    if limit:
        chosen = chosen[:limit]

    dense = VectorIndex(cfg, backend=be.name)
    if rebuild:
        dense.drop()
    dense.ensure()

    before = be.usage()
    already = upserted = 0
    paced = 0.0
    # A local encoder has no quota to pace against, and pacing it would cap a
    # forward pass at the hosted tier's speed for no reason at all.
    pacer = None if model_for("embed", be.name).local else TokenPacer(cfg.embed_tokens_per_minute)
    batch = cfg.embed_batch_size
    for start in range(0, len(chosen), batch):
        window = chosen[start : start + batch]
        have = dense.present([c.chunk_id for c in window])
        todo = [c for c in window if c.chunk_id not in have]
        already += len(window) - len(todo)
        if todo:
            texts = [c.embed_text for c in todo]
            if pacer is not None:
                paced += pacer.wait(sum(estimate_tokens(t) for t in texts))
            vectors = _embed_waiting_out_the_quota(be, texts)
            dense.upsert(todo, vectors)
            upserted += len(todo)
        if on_batch:
            on_batch(start + len(window), len(chosen))

    after = be.usage()
    sparse = SparseIndex(cfg.index_dir)
    n_sparse = sparse.build(chosen)

    by_item: dict[str, int] = {}
    for c in chosen:
        by_item[f"{c.form} {c.item_key}"] = by_item.get(f"{c.form} {c.item_key}", 0) + 1

    return IndexReport(
        collection=dense.name,
        model=dense.model,
        dim=dense.dim,
        candidates=len(everything),
        selected=len(chosen),
        already_indexed=already,
        upserted=upserted,
        embed_http_calls=after.http_calls - before.http_calls,
        cache_hits=after.cache_hits - before.cache_hits,
        sparse_documents=n_sparse,
        paced_seconds=paced,
        seconds=time.monotonic() - started,
        by_item=dict(sorted(by_item.items(), key=lambda kv: -kv[1])),
    )
