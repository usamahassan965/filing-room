"""Chunking: filing text cut into spans that can still be quoted.

A chunk here is a *slice*, not a rendering. ``text[char_start:char_end]`` taken
from a freshly re-parsed filing has to equal the chunk's stored text, character
for character -- that is the M3 gate check, and it is what lets M6 answer "where
in the filing does this sentence come from?" with an offset rather than with a
guess. Three rules follow from it and are worth stating, because each is a thing
a chunker normally does:

* **Nothing is reflowed, stripped, normalised or re-joined after the offsets are
  taken.** Whitespace collapsing happens once, inside the flattener, before any
  offset exists. A ``.strip()`` here would move every citation in the corpus by
  a character or two, silently, and only for the chunks that happened to begin
  on a space.
* **Context is added at embedding time, never to the stored text.** A chunk
  embeds better with "NVDA 10-K 2024-01-28 -- Item 1A. Risk Factors" in front of
  it, so ``Chunk.embed_text`` puts it there. Putting it in ``text`` would make
  the stored chunk a thing that appears nowhere in the filing.
* **A chunk never crosses a section boundary.** A span straddling the end of
  Item 7 and the start of Item 8 has to be cited as one of them, and is then
  wrong about the other half of itself.

Boundaries land on the document's own blocks -- paragraphs, table rows, list
items -- because those are the units the filer wrote, and a mid-sentence cut is
reserved for a block that is on its own bigger than the target. Consecutive
chunks overlap by ``CHUNK_OVERLAP`` characters so a sentence that lands on a
boundary survives whole in one of them.

The cache exists for idempotency, not for speed. Parsing the corpus takes about
two and a half minutes (see docs/parsing.md), so nothing here is waiting on the
parser; what re-parsing on every run would buy is a fresh chance to produce
different offsets from the same bytes. The key is the filing's sha256 from the
manifest -- content, not mtime -- plus ``CHUNKER``, which names the parameters
that produced the offsets. Change a parameter and the cache is stale by
definition, because offsets that do not match the settings that made them are
worse than no offsets at all.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from filing.config import Settings
from filing.ingest.manifest import Manifest
from filing.stores.parse import WHOLE_DOC_ITEM, ParsedFiling, parse_filing

# Roughly 450 tokens of prose, which is the size retrieval evaluations keep
# landing on: small enough that a hit is mostly the thing you asked for, large
# enough to carry the sentence that qualifies it. Filing tables run longer per
# character than prose, so this is a ceiling on characters and not on tokens.
CHUNK_CHARS = 1_800
CHUNK_OVERLAP = 200

# Below this a span is not a passage. "Item 6. [Reserved]" is 18 characters and
# a real section; embedding it spends a call to make a neighbour that can only
# ever be noise. Stub sections are counted in the report rather than dropped
# quietly.
MIN_CHUNK_CHARS = 120

# Names the parameters, because the cache key includes it. Bump it whenever a
# constant above changes.
CHUNKER = "v1-1800-200"

# Fixed so a chunk id is a pure function of (accn, offsets) on any machine and
# in any run. Qdrant will only take an unsigned integer or a UUID as a point id,
# and a UUID5 is both deterministic and legal.
CHUNK_NAMESPACE = uuid.UUID("1f7a0c26-3d51-5b8e-9a44-6e0c2f8d1b73")


def chunk_id(accn: str, start: int, end: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{accn}:{start}:{end}"))


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable span, and everything a citation needs to name it."""

    chunk_id: str
    accn: str
    cik: str
    ticker: str
    form: str
    fy: int | None
    filed_date: str
    period_end: str
    part: str
    item: str
    title: str
    char_start: int
    char_end: int
    n_chars: int
    text_sha256: str
    text: str

    @property
    def item_key(self) -> str:
        return f"{self.part}.{self.item}" if self.part else self.item

    @property
    def embed_text(self) -> str:
        """What goes to the embedding model -- the chunk, plus who and when.

        Filing prose is anonymous from the inside: "our results were affected by
        component shortages" names neither the company nor the year, and two
        hundred filings say something like it. The header disambiguates the
        vector without touching the text a citation quotes.
        """
        head = f"{self.ticker} {self.form} {self.period_end}"
        if self.item != WHOLE_DOC_ITEM:
            head += f" -- Item {self.item}"
            if self.title:
                head += f". {self.title}"
        return f"{head}\n\n{self.text}"


@dataclass(frozen=True, slots=True)
class FilingOutcome:
    """What the parser made of one filing, cached so it is decided once."""

    accn: str
    ticker: str
    form: str
    source_sha256: str
    chunker: str
    outcome: str  # split | degraded | quarantined
    reason: str
    n_sections: int
    n_stubs: int
    n_chunks: int
    n_chars: int


@dataclass(frozen=True, slots=True)
class ChunkReport:
    filings: int = 0
    parsed: int = 0
    reused: int = 0
    split: int = 0
    degraded: int = 0
    quarantined: tuple[tuple[str, str], ...] = ()
    chunks: int = 0
    chars: int = 0
    stub_sections: int = 0
    by_item: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def accounted_for(self) -> int:
        return self.split + self.degraded + len(self.quarantined)


# --------------------------------------------------------------------------
# cutting
# --------------------------------------------------------------------------


def _split_long(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Cut one oversized block at whitespace, as late as the target allows."""
    out: list[tuple[int, int]] = []
    while end - start > CHUNK_CHARS:
        cut = text.rfind(" ", start + CHUNK_CHARS // 2, start + CHUNK_CHARS)
        if cut <= start:
            cut = start + CHUNK_CHARS  # a run with no space in it; take the hit
            out.append((start, cut))
            start = cut
            continue
        out.append((start, cut))
        start = cut + 1  # the space itself belongs to neither side
    if end > start:
        out.append((start, end))
    return out


def _units(
    text: str, blocks: Sequence[tuple[int, int]], start: int, end: int
) -> list[tuple[int, int]]:
    """The indivisible spans inside one section: its blocks, oversized ones cut."""
    out: list[tuple[int, int]] = []
    for s, e in blocks[bisect_left(blocks, (start, -1)) :]:
        if s >= end:
            break
        e = min(e, end)
        if e > s:
            out.extend(_split_long(text, s, e))
    if not out and end > start:  # a section with no block boundaries in it
        out = _split_long(text, start, end)
    return out


def _pack(units: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedily fill to ``CHUNK_CHARS``, then step back for the overlap.

    The span of a chunk runs from the first unit's start to the last unit's end,
    which is to say it includes the separators between them. That is deliberate:
    the chunk is a contiguous slice of the filing, so what it measures and what
    it quotes are the same string.
    """
    out: list[tuple[int, int]] = []
    i, n = 0, len(units)
    while i < n:
        j = i + 1
        while j < n and units[j][1] - units[i][0] <= CHUNK_CHARS:
            j += 1
        # A subheading followed by a block too big to join it would otherwise be
        # emitted alone: Pfizer's Item 1A is a run of them, and "INFORMATION
        # TECHNOLOGY AND SECURITY" as a 35-character chunk is a vector with no
        # content under it. Overshoot the target instead -- a heading belongs
        # with the thing it heads.
        while j < n and units[j - 1][1] - units[i][0] < MIN_CHUNK_CHARS:
            j += 1
        out.append((units[i][0], units[j - 1][1]))
        if j >= n:
            break
        # Step back over the tail units that fit inside the overlap window. A
        # unit larger than the window is never repeated: embedding a whole
        # paragraph twice costs twice and retrieves the same passage twice.
        k = j
        while k > i + 1 and units[j - 1][1] - units[k - 1][0] <= CHUNK_OVERLAP:
            k -= 1
        i = k
    # A trailing sliver is a boundary artefact, not a passage; give it to the
    # chunk it came off.
    if len(out) > 1 and out[-1][1] - out[-1][0] < MIN_CHUNK_CHARS:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return out


def chunk_filing(pf: ParsedFiling, **meta: object) -> tuple[list[Chunk], int]:
    """Chunks for one parsed filing, and the number of stub sections skipped.

    Each section is chunked from its *heading*, not from the body, so the first
    chunk of Item 1A opens with "Item 1A. Risk Factors" -- which is both what a
    citation should quote and the strongest lexical signal in the section.
    """
    chunks: list[Chunk] = []
    stubs = 0
    for section in pf.sections:
        if section.end - section.start < MIN_CHUNK_CHARS:
            stubs += 1
            continue
        for cs, ce in _pack(_units(pf.text, pf.blocks, section.start, section.end)):
            body = pf.text[cs:ce]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(pf.accn, cs, ce),
                    accn=pf.accn,
                    part=section.part,
                    item=section.item,
                    title=section.title,
                    char_start=cs,
                    char_end=ce,
                    n_chars=ce - cs,
                    text_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    text=body,
                    **meta,  # type: ignore[arg-type]
                )
            )
    return chunks, stubs


# --------------------------------------------------------------------------
# the parquet cache
# --------------------------------------------------------------------------

_CHUNK_COLUMNS = (
    "chunk_id",
    "accn",
    "cik",
    "ticker",
    "form",
    "fy",
    "filed_date",
    "period_end",
    "part",
    "item",
    "title",
    "char_start",
    "char_end",
    "n_chars",
    "text_sha256",
    "text",
)

_CHUNKS_DDL = """
CREATE TABLE chunks (
    chunk_id     VARCHAR,
    accn         VARCHAR,
    cik          VARCHAR,
    ticker       VARCHAR,
    form         VARCHAR,
    fy           INTEGER,
    filed_date   VARCHAR,
    period_end   VARCHAR,
    part         VARCHAR,
    item         VARCHAR,
    title        VARCHAR,
    char_start   INTEGER,
    char_end     INTEGER,
    n_chars      INTEGER,
    text_sha256  VARCHAR,
    text         VARCHAR
)
"""

_FILINGS_DDL = """
CREATE TABLE filings (
    accn           VARCHAR,
    ticker         VARCHAR,
    form           VARCHAR,
    source_sha256  VARCHAR,
    chunker        VARCHAR,
    outcome        VARCHAR,
    reason         VARCHAR,
    n_sections     INTEGER,
    n_stubs        INTEGER,
    n_chunks       INTEGER,
    n_chars        INTEGER
)
"""


class ChunkStore:
    """Two parquet files: what the parser decided, and what the chunker cut.

    Kept apart from the manifest and from ``facts.duckdb`` for the reason M1
    gave: this is derived and disposable, and a rebuild should never need a
    write handle on the 1.12 GB that took a rate-limited host an hour to serve.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.chunks_path = directory / "chunks.parquet"
        self.filings_path = directory / "filings.parquet"

    @property
    def exists(self) -> bool:
        return self.chunks_path.exists() and self.filings_path.exists()

    def outcomes(self) -> dict[str, FilingOutcome]:
        if not self.filings_path.exists():
            return {}
        con = duckdb.connect()
        try:
            rows = con.execute(f"SELECT * FROM read_parquet({_lit(self.filings_path)})").fetchall()
        finally:
            con.close()
        return {r[0]: FilingOutcome(*r) for r in rows}

    def chunks(self, accns: set[str] | None = None) -> list[Chunk]:
        if not self.chunks_path.exists():
            return []
        con = duckdb.connect()
        try:
            rows = con.execute(f"SELECT * FROM read_parquet({_lit(self.chunks_path)})").fetchall()
        finally:
            con.close()
        out = [Chunk(*r) for r in rows]
        return [c for c in out if c.accn in accns] if accns is not None else out

    def write(self, chunks: Sequence[Chunk], outcomes: Sequence[FilingOutcome]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        try:
            con.execute(_CHUNKS_DDL)
            con.execute(_FILINGS_DDL)
            con.executemany(
                f"INSERT INTO chunks VALUES ({','.join('?' * len(_CHUNK_COLUMNS))})",
                [
                    (
                        c.chunk_id,
                        c.accn,
                        c.cik,
                        c.ticker,
                        c.form,
                        c.fy,
                        c.filed_date,
                        c.period_end,
                        c.part,
                        c.item,
                        c.title,
                        c.char_start,
                        c.char_end,
                        c.n_chars,
                        c.text_sha256,
                        c.text,
                    )
                    for c in chunks
                ],
            )
            con.executemany(
                "INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        o.accn,
                        o.ticker,
                        o.form,
                        o.source_sha256,
                        o.chunker,
                        o.outcome,
                        o.reason,
                        o.n_sections,
                        o.n_stubs,
                        o.n_chunks,
                        o.n_chars,
                    )
                    for o in outcomes
                ],
            )
            con.execute(f"COPY chunks TO {_lit(self.chunks_path)} (FORMAT PARQUET)")
            con.execute(f"COPY filings TO {_lit(self.filings_path)} (FORMAT PARQUET)")
        finally:
            con.close()


def _lit(path: Path) -> str:
    """A SQL string literal for a path. DuckDB will not bind one in COPY TO."""
    return "'" + str(path).replace("'", "''") + "'"


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

_FILING_SQL = """
SELECT accn, cik, ticker, form, fy,
       CAST(filed_date AS VARCHAR), CAST(period_end AS VARCHAR),
       path, sha256
FROM filings
ORDER BY ticker, period_end, accn
"""


def build_chunks(cfg: Settings, *, rebuild: bool = False) -> ChunkReport:
    """Chunk every filing in the manifest, parsing only what the cache lacks.

    ``rebuild`` ignores the cache. Without it a filing is re-parsed only when
    its bytes or the chunker's parameters have changed, which is what makes the
    second run of this cost zero embedding calls downstream: same bytes, same
    parameters, same offsets, same chunk ids, same content hashes.
    """
    started = time.monotonic()
    store = ChunkStore(cfg.chunks_dir)
    cached = {} if rebuild else store.outcomes()

    with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
        rows = manifest.con.execute(_FILING_SQL).fetchall()

    def fresh(accn: str, sha: str) -> bool:
        hit = cached.get(accn)
        return hit is not None and hit.source_sha256 == sha and hit.chunker == CHUNKER

    reusable = {r[0] for r in rows if fresh(r[0], r[8])}

    chunks: list[Chunk] = store.chunks(reusable) if reusable else []
    outcomes: list[FilingOutcome] = [cached[a] for a in sorted(reusable)]
    quarantined: list[tuple[str, str]] = [
        (o.accn, o.reason) for o in outcomes if o.outcome == "quarantined"
    ]

    for accn, cik, ticker, form, fy, filed, period, path, sha in rows:
        if accn in reusable:
            continue
        pf = parse_filing(cfg.data_dir / path, accn=accn, form=form)
        made, skipped = (
            chunk_filing(
                pf,
                cik=cik,
                ticker=ticker,
                form=form,
                fy=fy,
                filed_date=filed or "",
                period_end=period or "",
            )
            if pf.ok
            else ([], 0)
        )
        chunks.extend(made)
        outcome = "quarantined" if not pf.ok else ("split" if pf.split else "degraded")
        if not pf.ok:
            quarantined.append((accn, pf.quarantine or ""))
        outcomes.append(
            FilingOutcome(
                accn=accn,
                ticker=ticker,
                form=form,
                source_sha256=sha,
                chunker=CHUNKER,
                outcome=outcome,
                reason=pf.quarantine or pf.degraded or "",
                n_sections=len(pf.sections),
                n_stubs=skipped,
                n_chunks=len(made),
                n_chars=sum(c.n_chars for c in made),
            )
        )

    chunks.sort(key=lambda c: (c.ticker, c.period_end, c.accn, c.char_start))
    store.write(chunks, outcomes)

    by_item: dict[str, int] = {}
    for c in chunks:
        by_item[c.item_key] = by_item.get(c.item_key, 0) + 1

    return ChunkReport(
        filings=len(rows),
        parsed=len(rows) - len(reusable),
        reused=len(reusable),
        split=sum(o.outcome == "split" for o in outcomes),
        degraded=sum(o.outcome == "degraded" for o in outcomes),
        quarantined=tuple(sorted(quarantined)),
        chunks=len(chunks),
        chars=sum(c.n_chars for c in chunks),
        stub_sections=sum(o.n_stubs for o in outcomes),
        by_item=dict(sorted(by_item.items(), key=lambda kv: -kv[1])),
        seconds=time.monotonic() - started,
    )
