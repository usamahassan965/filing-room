# Filing Room

Agentic RAG over SEC filings. Numeric facts from XBRL, narrative from the filing
text, and relationships from an entity graph — routed by an agent, verified
against the source, and every answer traceable to the filing it came from.

Runs on free API tiers. Full build plan: [`docs/build-plan.html`](docs/build-plan.html).

**Status: M4 shipped — 150 frozen questions, and a naive baseline that scores
7.5% exact on numbers while retrieving the right evidence 7.9% of the time.
That gap is M5's whole brief.**
M0 the rig, M1 the corpus, M2 the fact store, M3 the text index and the entity
graph, M4 the frozen question set and the naive baseline.

| Gate | What it produced | Check |
|---|---|---|
| **M0** the rig | one model interface, three backends, limiter, cache, tracing | `filing smoke` — 5/5 |
| **M1** the corpus | 20 companies, 407 filings, 1,044 MB, every hash verified | `filing corpus --check` — 6/6 |
| **M2** the numbers | 514,649 facts, 25 metrics, 0 duplicate keys | `filing numbers` — 5/5 |
| **M3** the text | 58,844 chunks, 32,218 indexed, 2,986 graph edges | `filing text` — 6/6 |
| **M4** the yardstick | 150 frozen questions, 260 gold spans, naive index of 48,934 chunks | `filing.eval run --config baseline` — 150/150, 4 live calls |

590 tests, `ruff` clean, and no test may open a socket off this machine.

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

python -m filing.eval verify                        # the frozen 150 still land on their spans
python -m filing.eval run --config baseline-retrieval   # score the retriever, no LLM at all
python -m filing.eval depth --depth 500             # how far down the ranking the evidence sits
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
| Six backends | `llm/gemini.py`, `llm/openai_compat.py`, `llm/fallback_ollama.py`, `llm/local.py` | One interface; hosted, self-hosted, in-process. `LLM_BACKEND` picks. Three of the six share one file. |
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
radius is what M0 was for. The NIM backend was later deleted outright, once it
was clear the account could never be verified from here -- a provider nobody can
run is not a fallback, it is a claim the README cannot back.

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

## M4 — the yardstick

150 frozen questions — 80 numeric with gold read out of XBRL, 60 narrative with
hand-checked gold spans, 10 unanswerable — tagged `v1.0` with a datasheet, and
a deliberately naive baseline to measure against: 2,048-character chunks, one
dense search, one LLM call. The write-up is
[`docs/baseline.md`](docs/baseline.md).

**The baseline is reported as two runs.** A RAG answer fails two separable
ways: the retriever never found the evidence, or the generator fumbled evidence
it had. Only the second needs a model, and conflating them means every later
improvement gets argued about instead of attributed.

| slice | n | exact | router | cite ok | cite gold | abstain |
|---|---|---|---|---|---|---|
| numeric | 80 | 7.5% | — | 100% | 22.2% | — |
| narrative | 60 | — | 51.7% | 100% | 12.9% | — |
| unanswerable | 10 | — | 100% | — | — | **100%** |
| overall | 150 | 7.5% | 27.3% | 100% | 13.9% | 100% |

**It always cites, and it rarely cites right.** `cite ok` 100% against `cite
gold` 13.9%: every answer resolves to a real chunk, and seven times in eight it
is not the chunk holding the evidence. A citation that resolves is not a
citation that supports, and a reader checking a filing only cares about the
second. The one thing it does perfectly is refuse — all ten unanswerable
questions abstain, and not by hedging, since it answers the other 140.

One number needs chasing rather than celebrating: numeric `exact` (7.5%) is
*higher* than numeric `hit@5` (2.5%), so the model gets figures right more often
than the evidence for them is retrieved. Either the gold is under-annotated or
flash-lite is reciting Apple's revenue from pretraining — and if it is the
second, that 7.5% is contamination, not capability.

The generator is `gemini-3.5-flash-lite`, chosen by the gate rather than by
preference: the free tier caps `gemini-3.5-flash` at **20 requests per day**, so
150 questions is an eight-day baseline and a re-run is eight more, which fails
M4's own requirement that a full run fit the rate-limit budget. The results file
names the model, so the weaker generator is stated rather than hidden.

`--config baseline-retrieval` blanks every chat field before fingerprinting and
scores the retriever alone: 150 questions, 104 seconds, **zero API calls**, and
generation columns report `None` rather than `0.0%`, because "routed everything
wrong" and "does not route" are different claims.

| slice | n | hit@1 | hit@5 | hit@10 | nDCG@5 |
|---|---|---|---|---|---|
| numeric | 80 | 2.5% | 2.5% | 5.0% | 0.008 |
| narrative | 60 | 10.0% | 15.0% | 30.0% | 0.105 |
| overall | 150 | 5.7% | 7.9% | 15.7% | 0.050 |

That is the **ceiling** on everything the generating half can score, which is
why it was worth measuring first.

**hit@10 says the retriever failed; depth says how.** `filing.eval depth` walks
the ranking to 500 and asks where the gold chunk actually sits:

| slice | n | hit@10 | hit@50 | hit@100 | hit@500 | median rank | right filing @10 |
|---|---|---|---|---|---|---|---|
| narrative | 60 | 30.0% | 55.0% | 63.3% | 86.7% | 24 | 70.0% |
| numeric | 80 | 5.0% | 12.5% | 18.8% | 43.8% | 147 | 26.2% |

Two different failures wearing one low number. On narrative the evidence is
*present but mis-ranked* — median rank 24, and 30% → 55% between depth 10 and
50 is headroom a reranker or a better chunking can actually collect. On numeric
it is not close: median rank 147, and for 45 of 80 questions the gold chunk is
absent from the top 500 of 48,934 entirely. Nothing downstream repairs that; no
reranker reorders a list the passage is not in. This is the project's routing
thesis stated as a measurement rather than an assumption — **numeric questions
belong in SQL, and here is the 2.5% hit@5 that says so.**

Three things this gate taught:

- **A suspicious number gets a positive control before it gets published.** The
  first control — querying with the gold text — returned 2 of 12 and looked
  like a broken retriever. The clean one, a chunk's own text as its query,
  returned rank 1 six times out of six at cosine 0.95–0.98. The plumbing was
  fine; the control was confounded, because numeric gold spans have a **median
  length of 6 characters** — the digits of the figure — so the first control
  had queried with a six-character string. Narrative spans run 816.
- **Errors are never cached.** A 429 is a fact about the afternoon, not about
  the system. The asymmetry decides it: re-answering a question that would have
  succeeded costs one call, while remembering a 429 forever silently caps the
  score with no failing test anywhere.
- **Hermeticity is enforced, not assumed.** A test that passed `None` for its
  backend and trusted a warm cache turned into a *live* API call the moment a
  cache miss appeared — and quietly spent the day's real quota inside `pytest`.
  `tests/conftest.py` now fails any test that opens a socket off this machine,
  and the two affected files went from 121s to 1.9s, nearly all of it retry
  backoff against a wall.
- **A results file that misnames its model is worse than no results file.** A
  config declaring `chat_backend="ollama"` was fingerprinted, cached and written
  under `llama3.2:3b` while all 180 of its calls went to Gemini, because the
  runner built its client from the environment and used the config's field only
  as a label. It completed cleanly and the numbers looked fine, which is exactly
  what made it dangerous — nobody re-checks a plausible number. It surfaced by
  accident, through a 503 quoting *"this model is currently experiencing high
  demand"*, which is not a sentence a localhost server says. The fix is one
  argument; the test asserts the constructed backend equals the fingerprinted
  one, so the label and the call cannot drift apart again.

---

## M4.5 — the provider bake-off

M0 claimed that swapping providers is a registry entry, not a refactor. Nothing
had tested that claim: the project had run on one hosted provider since the
first commit, and a claim nobody has tried is a comment, not an interface.

Three free tiers went in — Cohere, Groq, OVHcloud — chosen for how differently
they meter rather than for how they benchmark, because the interesting question
is what the *shape* of a free tier does to a 150-question run. All three speak
OpenAI, so all three arrive as one file, [`llm/openai_compat.py`](src/filing/llm/openai_compat.py),
keyed by provider name. There is no `CohereBackend` and no `GroqBackend`. The
claim holds.

**Same retriever, different generator.** `baseline-cohere`, `baseline-groq` and
`baseline-ovh` reuse the naive baseline's index, top-5 and prompt unchanged, so
the generator is the only variable. That the ablation is clean is checked rather
than asserted: every retrieval metric below is *byte-identical* to `baseline`.

| config | generator | exact (num) | router | abstain | cite ok | cite gold | wall |
|---|---|---|---|---|---|---|---|
| `baseline` | `gemini-3.5-flash-lite` | 7.5% | 27.3% | 100% | 100% | **13.9%** | 0.9 min |
| `baseline-cohere` | `command-a-03-2025` | **8.8%** | **31.3%** | 100% | 100% | 9.5% | 8.9 min |

A bigger model is a better reader and a worse citer: `command-a` gains 1.3 points
of exact match and 4 of routing over flash-lite, and gives back 4.4 points of
citation groundedness. Both refuse all ten unanswerables. Neither changes the
picture M4 established — the retriever is the ceiling, and no generator argues
its way past a passage it was never handed.

**What each free tier actually meters, measured rather than read.**

| provider | the binding limit | what it costs a 150-question run |
|---|---|---|
| Cohere | **calls** — 20 rpm, 1,000 a *month* | 9 minutes, and 15% of the month |
| Groq | **tokens** — 8,000/min against 1,000 requests/day | ~1 hour; requests are never the problem |
| OVHcloud | **a shared anonymous pool** | unavailable — see below |

Groq's row is the one worth stating carefully. The documented headline is 1,000
requests a day, which at 150 questions sounds like a sixth of the budget and a
fast run. The live `x-ratelimit` headers say something else: 8,000 tokens per
minute, resetting in 607 ms. At ~3.2k tokens a question that is roughly two
questions a minute, so the registry's `rpm=2` is not caution — it is the actual
ceiling, and it comes from the response headers rather than from a docs page.

**Free with no account is not free.** OVHcloud was on the list for one property:
it answers unauthenticated requests, so no country list can take it away — which
mattered after `build.nvidia.com` turned out to be a wall this project could not
climb. The property is real and the availability is not. The anonymous pool is
shared and small, and across ~10 minutes of spaced probes it returned `429` on
every request and on every model in the registry entry — `Qwen3.5-397B`,
`Qwen3.5-9B`, `Llama-3.3-70B`, `gpt-oss-120b` alike, which is what makes it a
pool-level throttle rather than a wiring problem. The config stays in the runner
because the finding is the point: a tier with no credential also has no queue of
your own, and 150 sequential questions is more than an unowned queue will carry.

Three things this gate taught:

- **An absent header and an empty one are different requests.** The OpenAI SDK
  requires an `api_key` string, so a keyless provider gets a placeholder, and
  `Authorization: Bearer no-key-required` is *worse* than sending nothing:
  OVHcloud stops reading the request as anonymous and starts reading it as a
  **failed** credential. That is `403 authentication failed` on all 150
  questions of a run, which looks exactly like a key problem and is the precise
  opposite of one. The header is now stripped at the transport layer by an httpx
  request hook, and the proof it worked is that the 403 became a 429 — rejected
  became accepted-then-throttled.
- **A 403 is not always about the credential.** Groq sits behind Cloudflare bot
  protection, which fingerprints the client *before* the origin ever sees the
  key: a default Python `User-Agent` earns `403 Error 1010 — access denied based
  on your browser's signature`. Two providers, two 403s, two causes, neither of
  them the key. Both are now regression tests, because the cost of rediscovering
  either is an afternoon.
- **A metric that disagrees with every answer is the metric's fault.**
  `gpt-oss-120b` scored 0% on citations while citing correctly on every single
  answer, because it writes its brackets as U+3010/U+3011 — the CJK lenticular
  pair — and `parse_citations` matched ASCII only. The prompt asks for
  "bracketed numbers" and never promises a codepoint, so the model obeyed the
  contract and the regex did not. Published, that would have been a confident
  finding about a model, drawn entirely from a broken ruler.

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
│   ├── fallback_ollama.py local backend
│   ├── rerank_local.py    cross-encoder
│   └── factory.py         backend selection
├── ingest/
│   ├── universe.py        universe.yaml -> companies
│   ├── edgar.py           the SEC client: rate limit, retry, hash on write
│   ├── manifest.py        what was fetched, when, with which hash
│   └── corpus.py          the M1 gate and docs/corpus.md
├── stores/
│   ├── facts.py           XBRL payloads -> facts.duckdb
│   ├── metrics.py         the metric registry: tags, spans, derivations
│   ├── questions.py       50 questions with answers read from filings
│   ├── verify.py          finds a stored number in its own source document
│   ├── parse.py           filing HTML -> flat text + blocks + item sections
│   ├── chunks.py          sections -> chunks that keep their character offsets
│   ├── index.py           Qdrant collection + bm25s index, and what to index
│   ├── retrieve.py        RRF fusion, filtering, reranking, citation
│   ├── graph.py           entity/relation extraction over sentences
│   └── evalset.py         30 smoke questions, gold by predicate, the metrics
└── eval/
    ├── dataset.py         the frozen 150, their gold spans and their slices
    ├── authoring.py       how the set was built, and how it re-verifies
    ├── naive.py           the baseline: 2,048-char stride, dense top-k
    ├── runner.py          config fingerprint, outcome cache, the run
    ├── metrics.py         scoring — arithmetic only, nothing is asked a model
    └── depth.py           how far down the ranking the evidence actually sits

scripts/eval_v1_0/   the authoring record — rebuilds the frozen set byte for byte
tests/               590 tests, no network and no API key
results/             one committed JSON + markdown table per config
docs/                build plan, corpus notes, the baseline write-up
```

## Backends

`LLM_BACKEND` picks one. They all satisfy the same `chat` / `embed` / `rerank`
interface, and `filing smoke` is the identical test against each. Exact model IDs
are deliberately not repeated here — `MODEL_REGISTRY` in `src/filing/config.py` is
the only place they live, and `filing probe` prints the live ones.

| | chat | embed | rerank | what it meters |
|---|---|---|---|---|
| **`gemini`** (default) | Gemini Flash | `gemini-embedding-001` | **local cross-encoder** | requests/day, per model |
| `cohere` | `command-a` | `embed-v4.0` | local cross-encoder | calls — 1,000 a month |
| `groq` | `gpt-oss-120b` | — none served | local cross-encoder | tokens — 8,000 a minute |
| `ovh` | `Qwen3.5-397B` | `bge-m3` | local cross-encoder | a shared anonymous pool |
| `ollama` | Llama 3.1 8B, local | `nomic-embed-text` | cosine stand-in — **degraded** | nothing; it is your CPU |

The last four arrive through one file. `cohere`, `groq` and `ovh` are the same
`OpenAICompatBackend` under three registry entries — see
[M4.5](#m45--the-provider-bake-off) for what that bought and what it cost.

Groq's embed cell is a dash rather than a zero. It serves no embedding model, so
the registry omits the entry and `model_for("embed", "groq")` raises naming the
roles Groq does serve. A `dim=0` placeholder would have made the registry total
and made it lie, and a registry that lies is worse than one that raises.

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
- **Cloudflare answers before the origin does.** Groq fingerprints the client
  at the edge, so a default Python `User-Agent` gets `403 Error 1010` *before*
  the key is checked — an authentication-shaped failure with no authentication
  in it. `openai_compat.py` sends a browser UA, and a test asserts it reaches
  the client, because this one is invisible in every stack trace it causes.
- **A keyless provider must be sent no header, not a blank one.** OVHcloud reads
  `Authorization: Bearer <anything>` as a credential and fails it; it reads a
  *missing* header as anonymous and serves. The OpenAI SDK insists on an
  `api_key` string, so the header is removed on the way to the socket by an
  httpx request event hook.
- **Not every provider verb is OpenAI-shaped.** NVIDIA NIM's reranker used a
  different host, body and response, so it needed its own code path and a
  hand-written span the OpenAI instrumentor could not see. That backend is gone,
  but the lesson set the interface: `rerank` is a verb on the backend, not an
  OpenAI call the code assumes every provider serves.

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
