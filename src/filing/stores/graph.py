"""The entity graph: who acquired, sued, supplied and competed with whom.

Retrieval answers "what does this filing say about X". It cannot answer "which
of these five companies bought a foundry" without reading all five, because that
question is about the *shape* of the corpus rather than the content of any one
passage. That is what the graph is for, and it is why every edge here carries
the sentence that produced it: a graph you cannot audit back to prose is a
confident-looking way to be wrong.

**Why not an LLM extractor.** The obvious build is "prompt a model per chunk for
(subject, relation, object)". Over 32,000 narrative chunks that is 32,000 calls
on a free tier that serves about two embedding batches a minute -- days of
wall-clock, and the output would still need the same auditing. So extraction
here is rules over sentences, and the rules are visible in this file. The cost
is recall: a relation phrased in a way no cue matches is simply not found. The
benefit is that every edge is reproducible, offset-checkable, and free.

**How mentions are found -- two passes, because filings introduce themselves.**
A company is written in full the first time ("Xilinx, Inc.") and in short form
every time after ("Xilinx"). Pass one finds only the full forms, by looking for
a capitalised run ending in a corporate suffix, and canonicalises them. Pass two
then matches the short forms everywhere, because pass one has learned that
"Xilinx" names an organisation. A discovered name has to appear in its full form
in at least two filings before it becomes a node, so a one-off typo does not.

**The filer is a mention.** Filings do not write "AMD acquired Xilinx", they
write "we acquired Xilinx". A sentence with a first-person marker gets the
filing's own registrant inserted as the leading mention -- without that, most of
the real relations in a 10-K have no subject at all.

**Direction is a guess, and the sentence is the record.** "X acquired Y" and
"the acquisition of Y by X" mean the same thing in opposite orders; the passive
is detected, and beyond that the edge points from the first mention to the
second. Symmetric relations (competes with, litigates with, partners with) are
stored with their endpoints sorted so one fact is one edge. When direction
matters to an answer, read ``Edge.sentence``.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from filing.config import Settings
from filing.stores.chunks import Chunk, ChunkStore
from filing.stores.index import select

EXTRACTOR = "v1-rules"

# A discovered organisation needs its full, suffixed form in this many distinct
# filings before it becomes a node. One is a typo; two is a fact of the corpus.
MIN_ORG_FILINGS = 2

# Sentences longer than this are almost always a table row or a run-on heading
# that survived the parser. Relations extracted from them are noise.
MAX_SENTENCE_CHARS = 600
MIN_SENTENCE_CHARS = 40


# --------------------------------------------------------------------------
# sentences
# --------------------------------------------------------------------------

# Abbreviations that end in a period and do not end a sentence. Short and
# corpus-specific rather than exhaustive: these are the ones that actually
# appear mid-sentence in filings.
_ABBREVS = frozenset(
    """inc corp co ltd llc plc no nos u.s u.k e.g i.e dr mr mrs ms st jr sr vs
    approx fig al etc ph.d s.a n.v a.g dept est ref sec fda irs""".split()
)

_BOUNDARY = re.compile(r"[.!?](?=[\s\"')\]]|$)|\n+")
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans as ``(start, end)`` offsets into ``text``.

    Not a parser -- a scanner that knows filings. A period is a boundary unless
    the word before it is a known abbreviation or a single initial, and a run of
    newlines is always a boundary because the chunker's blocks are real ones.
    """
    out: list[tuple[int, int]] = []
    start = 0
    for mo in _BOUNDARY.finditer(text):
        end = mo.end()
        head = text[start:end]
        if mo.group().startswith("\n") is False:
            word = _TRAILING_WORD.search(head)
            if word and word.group(1).lower().rstrip(".") in _ABBREVS:
                continue
            # "J. Smith" and "U. S." -- one capital, standing alone.
            if len(head) >= 2 and head[-2].isupper() and (len(head) < 3 or not head[-3].isalpha()):
                continue
        if head.strip():
            out.append((start, start + len(head.rstrip())))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text) and text[start:].strip():
        out.append((start, len(text.rstrip())))
    return out


# --------------------------------------------------------------------------
# entity mentions
# --------------------------------------------------------------------------

_SUFFIXES = frozenset(
    """inc incorporated corp corporation company companies co ltd limited llc
    lp llp plc nv sa ag se gmbh holdings holding group technologies technology
    pharmaceuticals pharma laboratories labs systems semiconductor
    semiconductors industries partners therapeutics biosciences networks
    solutions energy petroleum resources motors airlines bank stores""".split()
)

# Capitalised runs, with the small connectors that appear inside real names.
# A run stops at a newline: filing text puts a heading on its own line, and
# "Strategy\nOur Company" is not a company called Strategy.
_TOKEN = r"[A-Z][A-Za-z0-9&.'’-]*"
_CAP_RUN = re.compile(rf"\b{_TOKEN}(?:[ \t]+(?:{_TOKEN}|of|and|for|the|de|&)){{0,5}}")

_CONNECTORS = frozenset("of and for the de &".split())

# Capitalised runs that are not organisations. Everything here appears in
# filings constantly and would otherwise dominate the graph.
_STOP_NAMES = frozenset(
    n.strip()
    for n in """
    the company | the companies | the registrant | the board | the group
    | our company | class a | class b | common stock | united states | america
    | north america | europe | asia | china | japan | india | canada | mexico
    | annual report | quarterly report | consolidated balance sheets
    | consolidated statements | new york stock exchange | nasdaq
    | first quarter | second quarter | third quarter | fourth quarter
    | fiscal year | management discussion and analysis | risk factors
    | legal proceedings | generally accepted accounting principles
    | international financial reporting standards
    | financial accounting standards board | table of contents
    | taiwan | russia | ukraine | israel | korea | south korea | singapore
    | brazil | germany | france | ireland | united kingdom | netherlands
    | switzerland | australia | vietnam | malaysia | saudi arabia | qatar
    """.split("|")
)
_STOP_WORDS = frozenset(
    """the a an and or of in on at we our us it its this that these those
    january february march april may june july august september october
    november december monday tuesday item note part exhibit""".split()
)

# Agencies and regulators. Not discovered -- named, because "FDA" has no
# corporate suffix to find it by and because "the FDA approved" is one of the
# most load-bearing relations a pharmaceutical filing contains.
AGENCIES: dict[str, str] = {
    "sec": "Securities and Exchange Commission",
    "securities and exchange commission": "Securities and Exchange Commission",
    "fda": "Food and Drug Administration",
    "food and drug administration": "Food and Drug Administration",
    "ema": "European Medicines Agency",
    "european medicines agency": "European Medicines Agency",
    "ftc": "Federal Trade Commission",
    "federal trade commission": "Federal Trade Commission",
    "doj": "Department of Justice",
    "department of justice": "Department of Justice",
    "bis": "Bureau of Industry and Security",
    "bureau of industry and security": "Bureau of Industry and Security",
    "epa": "Environmental Protection Agency",
    "environmental protection agency": "Environmental Protection Agency",
    "irs": "Internal Revenue Service",
    "internal revenue service": "Internal Revenue Service",
    "cms": "Centers for Medicare and Medicaid Services",
    "european commission": "European Commission",
    "department of commerce": "Department of Commerce",
    "federal reserve": "Federal Reserve",
}

_FIRST_PERSON = re.compile(r"\b(?:we|our|us|the Company|the Companies|the Registrant)\b", re.I)


@dataclass(frozen=True, slots=True)
class Mention:
    key: str  # canonical, lowercase -- the node id
    name: str  # display form
    kind: str  # company | organization | agency
    start: int  # offset within the sentence
    end: int


def canonical(name: str) -> tuple[str, str]:
    """``("Xilinx, Inc.")`` -> ``("xilinx", "Xilinx")``.

    Corporate suffixes are stripped for the key and for the display name, so
    "Xilinx", "Xilinx Inc" and "Xilinx, Inc." are one node. The suffix is not
    information -- every issuer has one -- and keeping it would split every
    company across however many ways its filings punctuate it.
    """
    clean = re.sub(r"[,.]+$", "", name.strip()).replace("’", "'")
    tokens = clean.split()
    while tokens and tokens[-1].strip(",.").lower() in _SUFFIXES:
        tokens.pop()
    while tokens and tokens[-1].lower() in _CONNECTORS:
        tokens.pop()
    while tokens and tokens[0].lower() in _CONNECTORS:
        tokens.pop(0)
    display = " ".join(tokens).strip(" ,.")
    return display.lower(), display


def _is_org_run(run: str) -> bool:
    """True when a capitalised run ends in a corporate suffix."""
    tokens = [t.strip(",.").lower() for t in run.split()]
    return len(tokens) >= 2 and tokens[-1] in _SUFFIXES


def _plausible(key: str) -> bool:
    if len(key) < 3 or key in _STOP_NAMES:
        return False
    words = key.split()
    if not words or all(w in _STOP_WORDS for w in words):
        return False
    # An initialism is not a name here. "U.S. Bank" ends in a suffix and
    # canonicalises to "u.s", which then matches every "U.S." in the corpus --
    # 867 edges to a node that means "America". Require one real word.
    return any(len(re.sub(r"[^a-z]", "", w)) >= 3 for w in words)


_LOWER_WORD = re.compile(r"\b[a-z][a-z'’-]{2,}\b")


def discover_organizations(chunks: Sequence[Chunk]) -> dict[str, str]:
    """Pass one: names written in full, in at least ``MIN_ORG_FILINGS`` filings.

    The threshold is per *filing*, not per mention: a name repeated forty times
    inside one document is still one company's word for something, while a name
    that two registrants both spell out is a fact about the corpus.

    Sentence-initial capitals and section headings are the hazard here. "While
    our suppliers..." reads as a capitalised run ending in a suffix, and so does
    "Liquidity and Capital Resources". The corpus settles both: a candidate
    whose every word this same text also writes in lower case is a phrase, not
    a name. "Microsoft", "GLOBALFOUNDRIES" and "Taiwan Semiconductor
    Manufacturing" survive that test; "While", "Strategy" and "Liquidity and
    Capital" do not. It costs any third party named entirely from ordinary
    words -- a "General Electric" would be dropped -- which is a real miss and
    the reason the twenty issuers come from universe.yaml instead of from here.
    """
    seen: dict[str, set[str]] = {}
    display: dict[str, str] = {}
    lowercased: set[str] = set()
    for c in chunks:
        lowercased.update(mo.group() for mo in _LOWER_WORD.finditer(c.text))
        for mo in _CAP_RUN.finditer(c.text):
            run = mo.group().strip()
            if not _is_org_run(run):
                continue
            key, name = canonical(run)
            if not _plausible(key):
                continue
            seen.setdefault(key, set()).add(c.accn)
            display.setdefault(key, name)
    return {
        k: display[k]
        for k, accns in seen.items()
        if len(accns) >= MIN_ORG_FILINGS and not all(w in lowercased for w in k.split())
    }


def build_gazetteer(cfg: Settings, chunks: Sequence[Chunk]) -> dict[str, tuple[str, str]]:
    """alias -> (node key, node kind). Issuers, agencies, discovered orgs."""
    from filing.ingest.universe import load_universe

    gaz: dict[str, tuple[str, str]] = {}
    for key in discover_organizations(chunks):
        gaz[key] = (key, "organization")
    for alias, agency in AGENCIES.items():
        gaz[alias] = (agency.lower(), "agency")
    # Issuers last, so a company that is also a discovered organisation is
    # typed as the issuer it is.
    for company in load_universe(cfg=cfg).companies:
        key, name = canonical(company.name)
        gaz[key] = (key, "company")
        gaz[company.ticker.lower()] = (key, "company")
        # "NVIDIA Corporation" -> also match "NVIDIA"; the first word alone is
        # only safe when it is distinctive, so require five characters.
        first = key.split()[0]
        if len(first) >= 5:
            gaz.setdefault(first, (key, "company"))
    return gaz


def mentions(sentence: str, gaz: dict[str, tuple[str, str]]) -> list[Mention]:
    """Pass two: every gazetteer hit in one sentence, left to right, no overlaps."""
    found: list[Mention] = []
    for mo in _CAP_RUN.finditer(sentence):
        run = mo.group()
        # Token spans, so a prefix match reports where it really ended -- runs
        # are split on \s+ and a filing's whitespace includes newlines.
        spans = [(t.start(), t.end()) for t in re.finditer(r"\S+", run)]
        # Longest-first: "Advanced Micro Devices" before "Advanced".
        for n in range(len(spans), 0, -1):
            key, name = canonical(run[: spans[n - 1][1]])
            hit = gaz.get(key)
            if hit is None:
                continue
            node, kind = hit
            found.append(
                Mention(
                    key=node,
                    name=name,
                    kind=kind,
                    start=mo.start() + spans[0][0],
                    end=mo.start() + spans[n - 1][1],
                )
            )
            break
    out: list[Mention] = []
    for m in found:
        if out and m.start < out[-1].end:
            continue
        if not out or m.key != out[-1].key:
            out.append(m)
    return out


# --------------------------------------------------------------------------
# relations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Relation:
    kind: str
    cue: re.Pattern[str]
    symmetric: bool = False
    agency: bool = False  # only fires when one endpoint is a regulator


# Order is priority: the first cue that matches the sentence wins. Specific
# before general, so "sued its competitor" is litigation and not competition.
RELATIONS: tuple[Relation, ...] = (
    Relation(
        "regulated_by",
        re.compile(
            r"\b(approv\w+|clearance|cleared|authoriz\w+|regulat\w+|investigat\w+"
            r"|subpoena\w*|consent decree|inquiry|inquiries|complian\w+|licen[cs]\w+"
            r"|restrict\w+|sanction\w*)\b",
            re.I,
        ),
        agency=True,
    ),
    Relation(
        "litigates_with",
        re.compile(
            r"\b(lawsuit\w*|litigation|sued|suing|complaint\w*|arbitration|infring\w+"
            r"|plaintiff\w*|defendant\w*|allege\w+|class action|settlement agreement"
            r"|filed suit|v\.)\b",
            re.I,
        ),
        symmetric=True,
    ),
    Relation(
        "acquired",
        re.compile(
            r"\b(acquir\w+|acquisition\w*|merged|merger|business combination"
            r"|tender offer|purchase of all)\b",
            re.I,
        ),
    ),
    Relation(
        "divested",
        re.compile(r"\b(divest\w+|spin-?off|spun off|separation of|disposition of)\b", re.I),
    ),
    Relation(
        "invests_in",
        re.compile(
            r"\b(equity (?:method )?(?:investment|interest)|ownership interest"
            r"|minority interest|stake in|shares of common stock of)\b",
            re.I,
        ),
    ),
    Relation(
        "partners_with",
        re.compile(
            r"\b(collaborat\w+|partnership|partnered|joint venture|alliance"
            r"|co-?develop\w+|co-?promot\w+|licen[cs]e agreement|agreement with)\b",
            re.I,
        ),
        symmetric=True,
    ),
    # Above supply and customer: "our primary competitor in the supply of
    # microprocessors is Intel" contains both cues, and the specific relational
    # noun -- competitor -- is the one the sentence is actually asserting.
    Relation(
        "competes_with",
        re.compile(r"\b(compet\w+)\b", re.I),
        symmetric=True,
    ),
    Relation(
        "supplies",
        re.compile(
            r"\b(suppl\w+|foundr\w+|fabricat\w+|contract manufactur\w+|wafer\w*"
            r"|assembl\w+|subcontract\w+|vendor\w*|outsourc\w+)\b",
            re.I,
        ),
    ),
    Relation(
        "customer_of",
        re.compile(r"\b(customer\w*|distributor\w*|reseller\w*|sales to|revenue from)\b", re.I),
    ),
)

_PASSIVE_BY = re.compile(r"\bby\b", re.I)

# How far past a mention a cue may sit and still be about that pair. Wide
# enough for "we rely on TSMC (TSMC) for the production of all wafers", narrow
# enough that the next clause's verb does not leak in.
CUE_WINDOW = 60


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    kind: str
    source_kind: str
    target_kind: str
    ticker: str
    accn: str
    form: str
    period_end: str
    item_key: str
    char_start: int  # absolute in the filing text, so the sentence is checkable
    char_end: int
    sentence: str


def cue_span(sentence: str, found: Sequence[Mention], i: int) -> str:
    """The stretch of text allowed to name the relation between two mentions.

    From just before the first to just after the second, and never across a
    third: in "with our acquisition of Xilinx, we now compete with Intel", the
    span for (Xilinx, Intel) starts after Xilinx, so the acquisition verb -- a
    fact about a different pair -- cannot claim it.
    """
    a, b = found[i], found[i + 1]
    left = found[i - 1].end if i > 0 else 0
    right = found[i + 2].start if i + 2 < len(found) else len(sentence)
    lo = 0 if a.start < 0 else max(left, a.start - CUE_WINDOW)
    hi = min(right, b.end + CUE_WINDOW)
    return sentence[max(0, lo) : max(0, hi)]


def _relation_for(sentence: str, a: Mention, b: Mention) -> Relation | None:
    is_agency = "agency" in (a.kind, b.kind)
    for rel in RELATIONS:
        if rel.agency and not is_agency:
            continue
        if not rel.agency and is_agency:
            # An agency pair that matched no regulatory cue is not a supplier
            # or a competitor; it is a sentence that mentioned a regulator.
            continue
        if rel.cue.search(sentence):
            return rel
    return None


def extract(chunk: Chunk, gaz: dict[str, tuple[str, str]], filer: Mention | None) -> list[Edge]:
    """Every relation this chunk's sentences support."""
    out: list[Edge] = []
    for s_start, s_end in sentences(chunk.text):
        raw = chunk.text[s_start:s_end]
        if not (MIN_SENTENCE_CHARS <= len(raw) <= MAX_SENTENCE_CHARS):
            continue
        # A chunk boundary can cut a sentence in half. Skip the half-sentence:
        # chunks overlap by 200 characters, so the whole of it is in the next
        # chunk and the relation is not lost -- only the truncated copy is.
        if s_end >= len(chunk.text) and raw[-1] not in ".!?":
            continue
        found = mentions(raw, gaz)
        if filer is not None and _FIRST_PERSON.search(raw):
            if all(m.key != filer.key for m in found):
                found.insert(0, filer)
        if len(found) < 2:
            continue
        sentence = " ".join(raw.split())
        # Adjacent pairs only. Two mentions with a third between them are not a
        # pair -- whatever verb sits between belongs to one of the two closer
        # pairs, and taking every combination turns one busy sentence into a
        # clique of relations nobody wrote.
        for i in range(len(found) - 1):
            a, b = found[i], found[i + 1]
            if a.key == b.key:
                continue
            rel = _relation_for(cue_span(raw, found, i), a, b)
            if rel is None:
                continue
            src, dst = a, b
            if rel.symmetric:
                if src.key > dst.key:
                    src, dst = dst, src
            elif a.start >= 0 and _PASSIVE_BY.search(raw[a.end : b.start]):
                src, dst = b, a
            out.append(
                Edge(
                    source=src.key,
                    target=dst.key,
                    kind=rel.kind,
                    source_kind=src.kind,
                    target_kind=dst.kind,
                    ticker=chunk.ticker,
                    accn=chunk.accn,
                    form=chunk.form,
                    period_end=chunk.period_end,
                    item_key=chunk.item_key,
                    char_start=chunk.char_start + s_start,
                    char_end=chunk.char_start + s_end,
                    sentence=sentence,
                )
            )
    return out


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

_EDGE_COLUMNS = (
    "source",
    "target",
    "kind",
    "source_kind",
    "target_kind",
    "ticker",
    "accn",
    "form",
    "period_end",
    "item_key",
    "char_start",
    "char_end",
    "sentence",
)

_EDGES_DDL = """
CREATE TABLE edges (
    source       VARCHAR,
    target       VARCHAR,
    kind         VARCHAR,
    source_kind  VARCHAR,
    target_kind  VARCHAR,
    ticker       VARCHAR,
    accn         VARCHAR,
    form         VARCHAR,
    period_end   VARCHAR,
    item_key     VARCHAR,
    char_start   INTEGER,
    char_end     INTEGER,
    sentence     VARCHAR
)
"""


def _lit(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


class GraphStore:
    """One parquet file of edges. The graph is a view over it, never the store."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.edges_path = directory / "edges.parquet"

    @property
    def exists(self) -> bool:
        return self.edges_path.exists()

    def edges(self) -> list[Edge]:
        if not self.exists:
            return []
        con = duckdb.connect()
        try:
            rows = con.execute(f"SELECT * FROM read_parquet({_lit(self.edges_path)})").fetchall()
        finally:
            con.close()
        return [Edge(*r) for r in rows]

    def write(self, edges: Sequence[Edge]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        try:
            con.execute(_EDGES_DDL)
            con.executemany(
                f"INSERT INTO edges VALUES ({','.join('?' * len(_EDGE_COLUMNS))})",
                [
                    (
                        e.source,
                        e.target,
                        e.kind,
                        e.source_kind,
                        e.target_kind,
                        e.ticker,
                        e.accn,
                        e.form,
                        e.period_end,
                        e.item_key,
                        e.char_start,
                        e.char_end,
                        e.sentence,
                    )
                    for e in edges
                ],
            )
            con.execute(f"COPY edges TO {_lit(self.edges_path)} (FORMAT PARQUET)")
        finally:
            con.close()


# --------------------------------------------------------------------------
# building and reading the graph
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphReport:
    chunks: int = 0
    sentences: int = 0
    edges: int = 0
    nodes: int = 0
    organizations: int = 0
    by_kind: dict[str, int] | None = None
    top_nodes: tuple[tuple[str, int], ...] = ()
    seconds: float = 0.0


def build_graph(cfg: Settings, *, all_items: bool = False) -> GraphReport:
    """Extract over the same chunks the index covers, and write the edge list.

    Same scope as the dense and sparse indexes on purpose: a graph that knows
    about text the retriever cannot cite would let an agent assert an edge it
    has no passage to show.
    """
    from filing.ingest.universe import load_universe

    started = time.monotonic()
    chunks = ChunkStore(cfg.chunks_dir).chunks()
    if not all_items:
        chunks = select(chunks)
    if not chunks:
        raise FileNotFoundError(
            f"no chunks at {cfg.chunks_dir}. Run `filing chunks` before `filing graph`."
        )

    gaz = build_gazetteer(cfg, chunks)
    filers: dict[str, Mention] = {}
    for company in load_universe(cfg=cfg).companies:
        key, name = canonical(company.name)
        filers[company.ticker] = Mention(key=key, name=name, kind="company", start=-1, end=-1)

    seen: set[tuple[str, str, str, str, int]] = set()
    edges: list[Edge] = []
    n_sentences = 0
    for c in chunks:
        n_sentences += len(sentences(c.text))
        for e in extract(c, gaz, filers.get(c.ticker)):
            # Chunks overlap by design, so one sentence can arrive twice.
            key = (e.source, e.target, e.kind, e.accn, e.char_start)
            if key in seen:
                continue
            seen.add(key)
            edges.append(e)

    GraphStore(cfg.graph_dir).write(edges)
    counts = Counter(e.kind for e in edges)
    degree = Counter[str]()
    for e in edges:
        degree[e.source] += 1
        degree[e.target] += 1
    return GraphReport(
        chunks=len(chunks),
        sentences=n_sentences,
        edges=len(edges),
        nodes=len(degree),
        organizations=sum(1 for v in gaz.values() if v[1] == "organization"),
        by_kind=dict(counts.most_common()),
        top_nodes=tuple(degree.most_common(10)),
        seconds=time.monotonic() - started,
    )


def load_graph(cfg: Settings):  # -> nx.MultiDiGraph
    """Rebuild the NetworkX graph from the edge list.

    A MultiDiGraph, not a DiGraph: two filings saying the same thing are two
    pieces of evidence, and collapsing them would throw away the count that
    makes one edge more trustworthy than another.
    """
    import networkx as nx

    g = nx.MultiDiGraph()
    for e in GraphStore(cfg.graph_dir).edges():
        g.add_node(e.source, kind=e.source_kind)
        g.add_node(e.target, kind=e.target_kind)
        g.add_edge(
            e.source,
            e.target,
            key=f"{e.kind}:{e.accn}:{e.char_start}",
            kind=e.kind,
            ticker=e.ticker,
            accn=e.accn,
            form=e.form,
            period_end=e.period_end,
            item_key=e.item_key,
            char_start=e.char_start,
            char_end=e.char_end,
            sentence=e.sentence,
        )
    return g


def neighbours(graph, node: str, *, kinds: Iterable[str] | None = None, limit: int = 20):  # noqa: ANN001
    """Everything one hop from ``node``, with the sentence for each.

    Undirected on purpose: "who did NVDA acquire" and "who acquired NVDA" are
    both answered by looking at the edges touching NVDA, and an agent that has
    to guess the direction before it can ask is an agent that will guess wrong.
    """
    want = set(kinds) if kinds else None
    out: list[dict[str, object]] = []
    touching = list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True))
    for u, v, data in touching:
        if want and data["kind"] not in want:
            continue
        other = v if u == node else u
        out.append({"other": other, "direction": "out" if u == node else "in", **data})
    out.sort(key=lambda d: (str(d["kind"]), str(d["period_end"])), reverse=True)
    return out[:limit]
