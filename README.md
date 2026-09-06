# Filing Room

Agentic RAG over SEC filings. Numeric facts from XBRL, narrative from the filing
text, and relationships from an entity graph — routed by an agent, verified
against the source, and every answer traceable to the filing it came from.

Runs on free API tiers. Full build plan: [`docs/build-plan.html`](docs/build-plan.html).

**Status: M0 complete — the rig.** No domain code yet, by design.

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

Everything except the vector store works this way, so M0 closes without Docker
at all.

## What M0 gives you

| Piece | Where | Why it exists |
|---|---|---|
| One model interface | `llm/base.py` | `chat` / `embed` / `rerank`. Nothing in this project calls a provider directly. |
| Model registry | `config.py` | Model IDs and rate limits live in one dict. Swapping providers is an entry, not a refactor. |
| Sliding-window limiter | `llm/limiter.py` | Per model, not per provider. Proven by unit test, not by hope. |
| Content-hash cache | `llm/cache.py` | Five weeks of eval re-runs on a finite free-tier budget. |
| Tracing | `tracing.py` | On from the first commit, because M5–M7 read these spans. |
| Three backends | `llm/gemini.py`, `llm/nvidia.py`, `llm/fallback_ollama.py` | One interface, three providers. `LLM_BACKEND` picks. |
| Local reranker | `llm/rerank_local.py` | A real cross-encoder, no key and no quota — the one verb Gemini does not serve. |

## Gate M0 — definition of done

Each of these is a command, not a claim.

```bash
.conda/python.exe -m filing.cli smoke     # chat + embed + rerank + cache + trace
.conda/python.exe -m pytest               # limiter, cache, registry invariants
.conda/python.exe -m filing.cli probe     # which model IDs are still live
```

`smoke` runs five checks and exits non-zero if any fails:

1. **chat** — a completion comes back
2. **embed** — a vector of the dimension the registry claims
3. **rerank** — the cross-encoder puts the *relevant* passage first, not merely
   any passage
4. **cache dedup** — the same prompt twice issues one HTTP request
   (asserted on a counter, not inferred from timing)
5. **tracing** — at least three spans reached the collector

The offline half of that runs in `pytest` with the network stubbed, so CI needs
no API key — 51 tests covering the limiter under a fake clock, cache dedup,
registry invariants, OTLP export against a stub collector, Gemini's taskType and
renormalisation, and the cross-encoder's actual ranking. Only `chat` and `embed`
genuinely need a key.

## When a model ID dies

Expected, not exceptional. Run `filing probe`, find a candidate marked `live`,
and copy it into `MODEL_REGISTRY` in `src/filing/config.py`. That is the entire
fix — no other file names a model.

This is not a hypothetical either. On the first live run every Gemini 2.x chat
ID in the registry answered 404 — *"no longer available to new users"* — while
embeddings kept working. The fix was two registry lines and no code.

One thing that run taught, which the code now encodes: **a model listing is not
a probe.** `ListModels` cheerfully returned `gemini-2.5-flash` for a key that
could not call it, so `filing probe` issues a real request per ID instead of
trusting the catalog. It also avoids `*-latest` aliases as primaries — those
float under you, and the one time it mattered `gemini-flash-latest` answered 503
while the pinned ID was fine.

The same lever handles a whole provider dying, which is not hypothetical: this
project moved off NVIDIA NIM mid-M0 when its account verification proved
impassable. That cost one new backend file and one registry entry; the limiter,
cache, tracing, and every test above carried over untouched. Containing that
blast radius is what M0 was for.

## Layout

```
src/filing/
├── config.py        settings + MODEL_REGISTRY — the only place model IDs live
├── tracing.py       phoenix / openinference wiring
├── cli.py           smoke, probe, cache
└── llm/
    ├── base.py            the three-verb interface
    ├── limiter.py         sliding-window rate limiter
    ├── cache.py           content-addressed call cache
    ├── nvidia.py          NIM backend
    ├── fallback_ollama.py local backend
    └── factory.py         backend selection
tests/               limiter proof, cache dedup, registry invariants
results/             one committed JSON per config, from M4 onward
docs/                build plan, corpus notes, trace samples
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

`tests/test_tracing.py` asserts both on a stub OTLP collector, so neither can
regress quietly, and neither test needs Docker or an API key.
