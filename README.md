# Filing Room

Agentic RAG over SEC filings. Numeric facts from XBRL, narrative from the filing
text, and relationships from an entity graph — routed by an agent, verified
against the source, and every answer traceable to the filing it came from.

Runs on free API tiers. Full build plan: [`docs/build-plan.html`](docs/build-plan.html).

**Status: M3 shipped — the filing text is retrievable, and the retrieval is
measured rather than asserted.**
M0 the rig, M1 the corpus, M2 the fact store, M3 the text index and the entity
graph. M4 (baseline + eval harness) is next.

| Gate | What it produced | Check |
|---|---|---|
| **M0** the rig | one model interface, three backends, limiter, cache, tracing | `filing smoke` — 5/5 |
| **M1** the corpus | 20 companies, 407 filings, 1,044 MB, every hash verified | `filing corpus --check` — 6/6 |
| **M2** the numbers | 514,649 facts, 25 metrics, 0 duplicate keys | `filing numbers` — 5/5 |
| **M3** the text | 58,844 chunks, 32,218 indexed, 2,986 graph edges | `filing text` — 6/6 |

419 tests, `ruff` clean.

---

## Quick start

```bash
conda create -p .conda python=3.12 -y
.conda/python.exe -m pip install -e ".[dev]"
cp .env.example .env      # then paste a key from https://aistudio.google.com/apikey
docker compose up -d      # phoenix (traces) + qdrant (vectors)
.conda/python.exe -m filing.cli smoke
```

On Windows, `pwsh -File make.ps1 <target>` stands in for `make <target>`.

**No Docker?** The trace collector can run as a plain process instead, which is
also the shape CI wants:

```bash
.conda/python.exe -m pip install -e ".[phoenix]"
.conda/Scripts/phoenix serve      # serves localhost:6006, same as the container
```

Everything except the vector store works this way.

## The commands

```bash
filing smoke      # M0 gate: chat + embed + rerank + cache dedup + tracing
filing probe      # which model IDs are still live
filing cache      # cache stats, and --clear
filing ingest     # download the corpus (SEC, rate-limited to 8 req/s)
filing corpus     # M1 gate: what was fetched, --check re-verifies every hash
filing facts      # build data/facts.duckdb from the downloaded XBRL
filing numbers    # M2 gate: 5 checks over the fact store
filing chunks     # parse + split + chunk the filing text, cached to parquet
filing index      # embed the narrative chunks into Qdrant, build BM25 beside
filing graph      # entity/relation extraction into a NetworkX graph
filing text       # M3 gate: 6 checks over the chunks, indexes and graph
```

`ingest` takes hours, and `index` takes two and a half on CPU. Everything else
is rebuildable in under a minute — and a re-run of `index` over an unchanged
corpus costs nothing at all, which is what the M3 gate's third check proves.

## What is in the repo, and what is not

`data/manifest.duckdb` **is committed** (3.4 MB). It is the only record of what
was fetched, when, and with which hash, and re-deriving it means pulling 1.1 GB
back through a host that rate-limits to 10 requests a second.

`data/facts.duckdb` **is not** (15 MB). It is a pure function of the JSON on
disk and `filing facts` rebuilds it in about 25 seconds.

The filings themselves are not, either. Clone, set `SEC_USER_AGENT`, and run
`filing ingest` — the manifest tells it exactly what to fetch.

---

## M0 — the rig

| Piece | Where | Why it exists |
|---|---|---|
| One model interface | `llm/base.py` | `chat` / `embed` / `rerank`. Nothing in this project calls a provider directly. |
| Model registry | `config.py` | Model IDs and rate limits live in one dict. Swapping providers is an entry, not a refactor. |
| Sliding-window limiter | `llm/limiter.py` | Per model, not per provider. Proven by unit test, not by hope. |
| Content-hash cache | `llm/cache.py` | Five weeks of eval re-runs on a finite free-tier budget. |
| Tracing | `tracing.py` | On from the first commit, because M5–M7 read these spans. |
| Three backends | `llm/gemini.py`, `llm/nvidia.py`, `llm/fallback_ollama.py` | One interface, three providers. `LLM_BACKEND` picks. |
| Local reranker | `llm/rerank_local.py` | A real cross-encoder, no key and no quota — the one verb Gemini does not serve. |

`smoke` runs five checks and exits non-zero if any fails: **chat**, **embed** (a
vector of the dimension the registry claims), **rerank** (the cross-encoder puts
the *relevant* passage first, not merely any passage), **cache dedup** (the same
prompt twice issues one HTTP request, asserted on a counter rather than inferred
from timing), and **tracing** (at least three spans reached the collector).

The offline half runs in `pytest` with the network stubbed, so CI needs no API
key. Only `chat` and `embed` genuinely need one.

### When a model ID dies

Expected, not exceptional. Run `filing probe`, find a candidate marked `live`,
and copy it into `MODEL_REGISTRY` in `src/filing/config.py`. That is the entire
fix — no other file names a model.

Not hypothetical: on the first live run every Gemini 2.x chat ID in the registry
answered 404 — *"no longer available to new users"* — while embeddings kept
working. The fix was two registry lines and no code.

One thing that run taught, which the code now encodes: **a model listing is not
a probe.** `ListModels` cheerfully returned `gemini-2.5-flash` for a key that
could not call it, so `filing probe` issues a real request per ID instead of
trusting the catalog. It also avoids `*-latest` aliases as primaries — those
float under you, and the one time it mattered `gemini-flash-latest` answered 503
while the pinned ID was fine.

The same lever handles a whole provider dying, which is also not hypothetical:
this project moved off NVIDIA NIM mid-M0 when its account verification proved
impassable. That cost one new backend file and one registry entry; the limiter,
cache, tracing, and every test carried over untouched. Containing that blast
radius is what M0 was for.

## M1 — the corpus

20 companies across four sectors, five fiscal years, 10-K and 10-Q, plus the
full XBRL company-facts payload per company. 407 filings, 1,044 MB.
[`docs/corpus.md`](docs/corpus.md) is generated, not written.

Two things this gate taught, both of which are now enforced in code:

- **A gate that only counts can pass while a company contributes nothing.**
  The first `corpus --check` was green while Exxon had zero filings: its
  submissions live under a *different CIK* than the one the ticker map gives,
  because the operating company was reorganised under a new holding company. The
  check now asserts per-company minimums, not a corpus-wide total.
- **The manifest is the source of truth, not the filesystem.** There are 21
  company-facts payloads on disk for a 20-company universe — the extra is
  Exxon's old CIK. A directory glob would load both and produce two half-Exxons
  that each look complete. `tests/test_facts_store.py` asserts the build reads
  the manifest.

## M2 — the numbers

`data/facts.duckdb`: 514,649 facts, 2,827 concepts, 25 named metrics, and views
that answer the questions the agent will ask (`annual`, `growth`,
`facts_current`, `restatements`). `filing numbers` runs five checks over it.

The load goes through CSV staging and `COPY`, not `executemany` — DuckDB's
Python `executemany` does roughly 380 rows/s, which is 22 minutes for this
table. `COPY` does it in 17 seconds.

What this gate taught, in the order it hurt:

- **A tag is not a period.** `NetIncomeLoss` carries quarters, halves,
  nine-month stubs and full years under one name. Summing the tag over a year
  triple-counts. Every fact gets a `span`, classified by nearest canonical
  length with a 25-day tolerance — because a 4-4-5 retail half-year is 168 days
  and a fixed range around 182 discards it.
- **`fy`/`fp` describe the filing, not the fact.** They disagree with the year
  of `period_end` in 54.9% of rows. Stored as `filed_fy`/`filed_fp`; never a key.
- **A declared UNIQUE constraint is not the check.** SQL treats NULLs as
  distinct, and `period_start` is NULL for the 195,325 instants, so the
  constraint silently exempts 38% of the table. The check is a `GROUP BY`, which
  treats NULLs as equal.
- **"Latest value wins" is right per fact and wrong per statement.** 11,359 keys
  are restated. Taking the newest value for each one independently assembles a
  balance sheet out of two different filings that has no reason to balance —
  Costco's FY2014 and FY2016 numbers, mixed. The identity check groups by `accn`.
- **The balance-sheet identity has four right-hand terms**, not two: total
  liabilities, equity including NCI, the pre-ASC-810 `MinorityInterest` line,
  and mezzanine equity. With all four it closes on 1,717 statements with zero
  violations.
- **Sign is documented, never mutated.** Capex is reported positive, so the
  metric carries `sign="outflow"` and `free_cash_flow` is a subtraction.

### Two resolution rules, because there are two kinds of ambiguity

A metric that arrives under several tags resolves either by coverage or by
priority, and which one is correct depends on *why* there are several tags.

- **Coverage** (`n_facts DESC`) for a **succession** — one tag replaced another.
  Revenue's tags are the pre- and post-ASC-606 names for the same line, so the
  one with the most facts is simply the one in force for most of the window.
- **Priority** (registry order) for **near-synonyms** — several tags coexist and
  mean subtly different things. Equity is the only such metric: the
  parent-only, including-NCI and total variants are all live, all common, and
  picking the most frequent picks whichever the bigger companies happen to use.

### How the numbers are checked

Not against a spreadsheet I typed. `filing numbers` asks 50 questions whose
answers were read out of the filings, and separately takes a sample of stored
values and **searches for each one in the filing document it claims to come
from** — at every scale a statement might use (units, thousands, millions) and
both grouped and parenthesised, because an accounting statement writes a loss as
`(1,234)`.

49 of 50 resolve. The one that does not is honest and worth stating: that check
can only sample facts whose accession number matches a filing we actually
downloaded, and **only 31% of the store does**. The manifest covers FY2020–2024;
company-facts reaches back to 2009 and forward into 2026. Facts outside the
window are real and correct, they simply have no local document to be read out
of. This constrains M6 — an answer resting on a pre-2020 fact cannot cite a
locator — and the retriever will be scoped accordingly.

## M3 — the filing text

407 filings parsed into 58,844 chunks that never lose their character offsets,
32,218 of them indexed into Qdrant and BM25, and 2,986 entity relations in a
NetworkX graph with the source sentence on every edge. `filing text` runs six
checks; the details are in [`docs/parsing.md`](docs/parsing.md) and
[`docs/retrieval.md`](docs/retrieval.md).

**The free tier's wall is documents per day, not requests per minute.**
`batchEmbedContents` bills *each text in the batch* as its own `embed_content`
request against a 1,000/day free-tier cap — so a 32-chunk batch spends 32 of
the day's thousand, and this corpus would take **33 days** to index once. Rate
limits you can pace around; a daily cap you cannot. The vector space moved onto
the CPU in this machine: `bge-small-en-v1.5`, **32,218 chunks in 140.6 minutes**
at 230 chunks/min, no key and no quota. Generation stays hosted, because that is
one call per *answer*; embedding is one call per *chunk*, and only one of those
scales with the corpus.

Four things this gate taught:

- **A bi-encoder's asymmetry is a prompt, not a parameter.** BGE wants an
  instruction on the query and nothing on the passage; E5 wants a prefix on
  both. Get it backwards and nothing raises — every vector is still 384
  numbers and retrieval is merely worse. The prefix table is unit-tested per
  family for exactly that reason.
- **Recall@50 of 100% is a weak result wearing a strong number.** Gold sets run
  21–318 chunks, so landing one of them in a 50-candidate window is easy. The
  informative statistic is where the *first* gold chunk lands: rank 1 for 19 of
  30 questions, inside the top 5 for 26, worst case 24.
- **The cross-encoder did not earn its latency here.** p@5 went 0.687 → 0.693,
  which passes a `> 0` gate and means nothing: 8 questions better, 7 worse, 15
  unchanged. The threshold stays where it was written — moving it after seeing
  the split would be picking a gate this run happens to pass — and M4's 150
  questions are where a delta that size becomes legible.
- **A gate command nobody has run is not a gate.** `filing text` had two
  crashing bugs on first execution: a missing constructor argument and a path
  joined against the project root instead of the data directory. Both were in
  code that reads perfectly well.

---

## Layout

```
src/filing/
├── config.py        settings + MODEL_REGISTRY — the only place model IDs live
├── tracing.py       phoenix / openinference wiring
├── cli.py           every command above
├── llm/
│   ├── base.py            the three-verb interface
│   ├── limiter.py         sliding-window rate limiter
│   ├── cache.py           content-addressed call cache
│   ├── gemini.py          default backend
│   ├── nvidia.py          NIM backend
│   ├── fallback_ollama.py local backend
│   ├── rerank_local.py    cross-encoder
│   └── factory.py         backend selection
├── ingest/
│   ├── universe.py        universe.yaml -> companies
│   ├── edgar.py           the SEC client: rate limit, retry, hash on write
│   ├── manifest.py        what was fetched, when, with which hash
│   └── corpus.py          the M1 gate and docs/corpus.md
└── stores/
    ├── facts.py           XBRL payloads -> facts.duckdb
    ├── metrics.py         the metric registry: tags, spans, derivations
    ├── questions.py       50 questions with answers read from filings
    ├── verify.py          finds a stored number in its own source document
    ├── parse.py           filing HTML -> flat text + blocks + item sections
    ├── chunks.py          sections -> chunks that keep their character offsets
    ├── index.py           Qdrant collection + bm25s index, and what to index
    ├── retrieve.py        RRF fusion, filtering, reranking, citation
    ├── graph.py           entity/relation extraction over sentences
    └── evalset.py         30 smoke questions, gold by predicate, the metrics
tests/               419 tests, no network and no API key
results/             one committed JSON per config, from M4 onward
docs/                build plan, corpus notes
```

## Backends

`LLM_BACKEND` picks one. All three satisfy the same `chat` / `embed` / `rerank`
interface, and `filing smoke` is the identical test against each. Exact model IDs
are deliberately not repeated here — `MODEL_REGISTRY` in `src/filing/config.py` is
the only place they live, and `filing probe` prints the live ones.

| | chat | embed | rerank |
|---|---|---|---|
| **`gemini`** (default) | Gemini Flash | `gemini-embedding-001` | **local cross-encoder** |
| `nvidia` | Llama 3.3 70B | `nv-embedqa-1b-v2` | `nv-rerankqa-1b-v2` (hosted) |
| `ollama` | Llama 3.1 8B, local | `nomic-embed-text` | cosine stand-in — **degraded** |

The Ollama row is honest about being worse: cosine "reranking" is the retriever's
own opinion asked twice, so it cannot correct the retriever's mistakes. It logs a
warning and marks its spans `filing.degraded`. Do not report retrieval numbers
produced that way.

### Provider quirks the code is shaped around

- **Gemini's embeddings are asymmetric, but not in OpenAI's vocabulary.**
  `taskType` (`RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`) has no OpenAI equivalent,
  and Google's compatibility shim drops fields it does not recognise *silently*.
  So chat goes through the shim and embeddings go through the native REST API.
- **A truncated Gemini embedding is not unit-norm.** `gemini-embedding-001` is
  normalised at its full 3072 dimensions only; ask for 1536 and you get a
  Matryoshka prefix, so the code renormalises. Skipping this makes cosine
  similarity quietly stop being cosine similarity — a mediocre retrieval score,
  never an error.
- **Rate limits are per model.** Gemini's free tier allows roughly 10 rpm for the
  chat model and 100 for embeddings. One shared limiter would drag indexing down
  to the speed of the slowest model in the registry.
- **NVIDIA's reranker is not OpenAI-shaped at all**: different host, different
  request body, different response. It gets its own code path and its own
  hand-written span, since the OpenAI instrumentor cannot see it.

### Installing torch

`sentence-transformers` pulls torch. On Windows the PyPI wheel is already
CPU-only. On Linux, install the CPU build first or pip drags in ~2.5GB of CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Notes on the tracing

Two things about Phoenix cost an afternoon each, so they are written down:

- `phoenix.otel`'s `TracerProvider.add_span_processor` **replaces** the default
  processor unless you pass `replace_default_processor=False`. Adding the span
  counter the obvious way shuts down the OTLP exporter it just installed. This
  fails silently in the worst possible way: spans are still created and still
  counted in-process, so everything looks healthy and nothing is exported.
- Spans are batched, and a CLI command exits long before the batch timer fires.
  Every entry point calls `flush_tracing()` on the way out; without it the
  export is attempted at interpreter shutdown and lands nowhere.

`tests/test_tracing.py` asserts both on a stub OTLP collector, so neither test
needs Docker or an API key.
