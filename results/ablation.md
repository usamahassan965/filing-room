## Ablation ladder

What each stage was worth, measured on the frozen 150-question set. Deltas in brackets are against the rung above, within the same lane.

### Retrieval lane -- no model, no key, no quota

Every rung here is the same 150 questions with the generator switched off, so the columns are the ceiling on anything downstream: a writer cannot cite evidence the search never returned. A dash in the answer columns means nothing generated an answer, not that it generated a bad one.

| rung | config | n | hit@5 | hit@10 | nDCG@10 | exact | cite->gold | halluc | calls/q | p50 s | p95 s | ready s |
|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|
| naive | `baseline-retrieval` | 140 | 7.9% | 15.7% | 7.4% | -- | -- | -- | 0.00 | 0.2 | 53.0 | 0.0 |
| + hybrid | `agent-retrieval-norerank` | 140 | 25.0% (+17.1) | 31.4% (+15.7) | 17.5% (+10.2) | -- | -- | -- | 0.00 | 0.2 | 3.8 | 0.0 |
| + rerank | `agent-retrieval` | 140 | 19.3% (-5.7) | 25.7% (-5.7) | 15.5% (-2.1) | -- | -- | -- | 0.00 | 12.0 | 23.6 | 0.0 |

- **naive** (`baseline-retrieval`) -- fixed 2,048-char chunks, dense top-10, no fusion and no reranker
- **+ hybrid** (`agent-retrieval-norerank`) -- semantic chunks, dense and BM25 fused by RRF, fused order kept
- **+ rerank** (`agent-retrieval`) -- the same fusion, reordered by the local MiniLM cross-encoder

### Answer lane -- one generator, held fixed

The same 150 questions end to end at `gemini-3.5-flash-lite`, held constant across all three rungs so the deltas are about the graph rather than about the model. Retrieval columns are scored over the questions that actually reached a retriever, which is why `n` differs between the naive rung and the agent's.

| rung | config | n | hit@5 | hit@10 | nDCG@10 | exact | cite->gold | halluc | calls/q | p50 s | p95 s | ready s |
|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|
| naive + generate | `baseline` | 150 | 7.9% | 7.9% | 5.0% | 7.5% | 13.9% | -- | 1.00 | 1.0 | 5.8 | 0.0 |
| + router, grader, repair | `agent` | 150 | 40.0% (+32.1) | 40.0% (+32.1) | 26.7% (+21.7) | 98.8% (+91.3) | 28.0% (+14.1) | -- | 2.00 | 14.8 | 17.9 | 0.0 |
| + verifier | `agent-guarded` | 150 | 40.0% (=) | 40.0% (=) | 26.7% (=) | 98.8% (=) | 28.0% (=) | 0.0% | 2.00 | 12.0 | 17.6 | 0.0 |

- **naive + generate** (`baseline`) -- dense top-5 into one prompt, one LLM call, no router and no grader
- **+ router, grader, repair** (`agent`) -- plan, route to sql|text|graph, rerank, grade, repair at most twice
- **+ verifier** (`agent-guarded`) -- every figure checked against evidence and every citation resolved

`calls/q`, `p50` and `p95` come from a separate pass with both caches off, over a small sample per rung -- the committed runs are served from the outcome cache and can only report what a replay cost. `ready s` is that rung's first question, held out of the percentiles: it is a model loading off disk, not a slow query, and leaving it in had the naive retriever reporting a 53-second p95 against a 0.2-second median.

