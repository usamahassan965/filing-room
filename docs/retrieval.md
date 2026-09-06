# Retrieval

How 32,218 passages of filing prose become a candidate list you can cite, why
the vector space runs on the CPU in this machine rather than on a free API tier,
and what the two numbers in the M3 gate actually measure.

Code: `src/filing/stores/index.py`, `retrieve.py`, `graph.py`,
`src/filing/llm/embed_local.py`, `local.py`.
Tests: `tests/test_index.py`, `test_retrieve.py`, `test_graph.py`,
`test_embed_local.py`, `test_evalset.py`.

## The measurement that moved the vector space

The first index build ran for a while and then started returning 429s on every
batch — while a single-text request to the same endpoint returned 200. That
combination rules out a throughput problem, and the quota string says what it
actually is:

```
Quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests,
limit: 1000, model: gemini-embedding-1.0
quotaId: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier
```

`batchEmbedContents` counts **each text in the batch as its own
`embed_content` request**. A 32-chunk batch spends 32 of the day's thousand. The
disk cache corroborated it exactly: 997 rows written before the wall.

The free tier is documented as 100 requests/minute and 30,000 tokens/minute, and
both of those are pacing problems with pacing solutions. **1,000 documents a day
is not.** At that rate this corpus takes 33 days to index once, which means the
chunker can never be changed — every retrieval experiment would cost a month.

So the vector space moved onto this machine.

| | Gemini free tier | local `bge-small-en-v1.5` |
|---|---|---|
| Cost of one full index build | 33 days | **2 h 21 min** |
| Throughput | 1,000 chunks/day | 230 chunks/min, 4 CPU threads |
| Dimensions | 1536 (Matryoshka) | 384 |
| Model load | — | 74 s, once per process |
| Marginal cost of a re-index | a month of quota | one afternoon |

Two and a half hours is a real cost. It is also a cost you can *choose to pay*,
which a daily cap is not, and that difference is the whole argument: the M8
evaluation sweep is only honest if re-embedding the corpus is something the
project is allowed to do.

## Two backends, split by what they buy

`LLM_BACKEND` and `EMBED_BACKEND` are separate settings with different defaults
(`gemini` and `local`), because generation and retrieval buy different things
per call.

* **Generation** is one hosted call per *answer*. A 33M-parameter local model
  would be a real downgrade at synthesis, and one request per answer is
  affordable on anybody's free tier.
* **Embedding** is one call per *chunk*. Hosted quality buys very little at
  chunk granularity and costs 32,218 requests.

`LocalBackend.chat` raises rather than falling back to something worse, and the
model registry has no `chat` role for the `local` backend at all — asking for
one is a `KeyError` naming the roles it does have. That is tested
(`tests/test_registry.py`), because a silent downgrade is exactly the failure
this project keeps trying to make impossible.

### BGE's asymmetry is a prompt, not a parameter

A passage is embedded bare. A query is embedded behind a fixed instruction:

```
Represent this sentence for searching relevant passages: <query>
```

Get that backwards and nothing raises — every vector is still 384 numbers,
retrieval is merely worse. So the prefix table lives beside the model id in
`embed_local.py`, is applied from `input_type` alone, and is unit-tested per
family (bge takes a query prefix only; e5 takes both `query:` and `passage:`; an
unlisted model is treated as symmetric).

Local embeddings are **not** disk-cached; local *queries* are. A passage's
vector already lives in Qdrant, so a second copy would be 32,000 duplicates of
data the index holds. A query is re-asked by every eval sweep.

## The collection name is the guard

```
filing__local__BAAI_bge-small-en-v1-5__384
filing__gemini__gemini-embedding-001__1536
```

Backend, model and width are all in the name. Swapping `EMBED_BACKEND` does not
return worse neighbours; it fails to find a collection and says which one it
wanted. The same applies to the two Matryoshka widths of one Gemini model, which
would otherwise be silently mixed.

## What is indexed, and what is not

| | chunks | share |
|---|---:|---:|
| Whole corpus (407 filings) | 58,844 | 100% |
| **Indexed — narrative items** | **32,218** | **54.8%** |
| Not indexed — financial statements (`8`, `I.1`) | 21,175 | 36.0% |
| Not indexed — exhibits, signatures, controls, other | 5,451 | 9.3% |

The financial statements are 36% of the corpus and they belong in
`facts.duckdb`, where a number can be summed and compared rather than
paraphrased. Embedding them would spend a third of the build to make
neighbours that are, at best, a worse route to a number M2 already has exactly.

Two deliberate exceptions:

* **`FULL`** — the twenty Intel filings the item splitter could not divide are
  indexed whole. Dropping them drops a company from the corpus.
* **An unknown form fails open.** A form the table has never seen is a gap in
  the table, not a decision. An extra chunk costs embedding time; a missing one
  costs an answer and says nothing about why.

Both are tested.

## Re-indexing an unchanged corpus costs zero embedding calls

Two independent mechanisms, deliberately not one:

1. **Qdrant `present()`** — chunk ids are UUID5 of `(accn, char_start,
   char_end)`, so an unchanged chunk keeps its id across machines and runs. The
   build asks the collection which ids it already holds and skips them.
2. **The per-text call cache** — keyed on backend, model, kind and payload, so
   even a build against an empty collection re-embeds nothing it has seen.

The first survives a deleted cache; the second survives a deleted collection.
The M3 gate checks the resulting claim directly: a second `filing index` over an
unchanged corpus makes zero embedding calls.

## Fusion: explicit RRF, not a tuned blend

BM25 returns 31.0 and cosine returns 0.99. Those are not on one scale, and no
amount of normalisation makes them comparable across queries. Reciprocal rank
fusion throws the scores away and uses only the ranks:

```
score(d) = Σ_rankers 1 / (k + rank_r(d)),  k = 60
```

`RRF_K = 60` is the value from the original paper and is **deliberately
untuned**. Tuning it on the same thirty questions the gate scores would be
fitting the constant to the test. What k buys is a flat-ish weighting where a
document both rankers return beats a document one ranker loves — which is the
entire reason to run a hybrid rather than either half.

Each fused hit records which ranker found it and at what rank, so a trace can
show whether an answer came from the dense half, the lexical half, or both.

The sparse half indexes `embed_text` — the chunk plus a
`NVDA 10-K 2024-01-28 — Item 1A` header — not the bare text. Filing prose is
anonymous from the inside: "our results were affected by component shortages"
names neither the company nor the year, and two hundred filings say something
like it. Both halves see the header or neither does.

## Reranking, and the number that justifies it

Fusion produces 50 candidates; a local cross-encoder
(`ms-marco-MiniLM-L-6-v2`) rescores them and the top 5 go to the reader. A
cross-encoder sees the query and the passage together, which is strictly more
information than two independent vectors — and it costs 50 forward passes per
question, which is why it runs on 50 candidates and not on 32,218 chunks.

Whether that is worth its latency is an empirical question, so the eval measures
it: **precision@5 is scored twice over the identical candidate list**, once on
the RRF order and once after the cross-encoder. Running the pipeline twice would
measure the pipeline's variance as well as the reranker's effect.

## The two metrics, defined rather than assumed

The hard part of a retrieval eval is not the metric, it is the ground truth.
Hand-labelling thirty questions × fifty candidates is 1,500 judgements that go
stale the moment the chunker changes. So relevance here is a **predicate the
corpus evaluates**:

> "What did Chevron say about the arbitration over its Hess acquisition?"
> → ticker `CVX`, narrative items, text matches `Hess`

* **recall@50** — the fraction of questions with at least one gold chunk in the
  fused top 50. Everything downstream, reranking included, can only reorder what
  recall let through.
* **precision@5** — the mean fraction of the top five that are gold, scored
  twice as above.

This is a weaker notion of relevance than a human judgement and it is stated
plainly rather than dressed up: it says a chunk is *on topic*, not that it
answers the question. Two properties make it worth having. It is exactly
reproducible — a re-chunk re-derives the gold set instead of invalidating it —
and it cannot be gamed by the thing under test, because the predicate is
evaluated against the corpus and never against what retrieval returned.

The set is thirty questions covering all twenty issuers, every narrative item
that carries answers, and five questions that reach past a single company —
because a single-company question can be scored well by a retriever that has
learned nothing but ticker matching. A question whose gold set is empty is
reported as `missing_gold` rather than scored as a miss: that is the corpus
being wrong, not retrieval.

## The entity graph

Retrieval answers "what does this filing say about X". It cannot answer "which
of these five companies bought a foundry" without reading all five, because that
question is about the *shape* of the corpus. Hence a NetworkX MultiDiGraph over
2,986 edges and 231 nodes, every edge carrying the sentence that produced it and
the absolute character offsets to check it against.

**Why not an LLM extractor.** 32,000 chunks is 32,000 calls, days of wall clock,
and the output would still need the same auditing. Extraction is rules over
sentences instead. The cost is recall: a relation phrased in a way no cue
matches is simply not found. The benefit is that every edge is reproducible,
offset-checkable and free.

| relation | edges | | relation | edges |
|---|---:|---|---|---:|
| `acquired` | 932 | | `customer_of` | 193 |
| `regulated_by` | 614 | | `competes_with` | 142 |
| `partners_with` | 329 | | `divested` | 118 |
| `litigates_with` | 305 | | `invests_in` | 99 |
| `supplies` | 254 | | | |

Five rules earn their place, and each was written against a specific piece of
bad output:

* **Two-pass mentions.** Filings introduce themselves once ("Xilinx, Inc.") and
  use the short form forever after. Pass one learns full, suffixed forms; pass
  two matches short forms everywhere.
* **The filer is a mention.** Filings write "we acquired Xilinx", never "AMD
  acquired Xilinx". Without inserting the registrant on first-person sentences,
  most real relations in a 10-K have no subject.
* **Initialisms are rejected.** "U.S. Bank" ends in a corporate suffix and
  canonicalises to `u.s`, which then matches every "U.S." in the corpus — 867
  edges to a node meaning "America".
* **A candidate whose every word the corpus also writes in lower case is a
  phrase, not a name.** "Liquidity and Capital Resources" ends in a suffix too.
  This costs any third party named entirely from ordinary words — a "General
  Electric" would be dropped — which is why the twenty issuers come from
  `universe.yaml` and not from discovery.
* **Adjacent pairs only, with a clipped cue window.** Taking every combination
  of mentions turns one busy sentence into a clique of relations nobody wrote,
  and a cue that sits past a third mention belongs to a different pair.

Direction is a guess: the passive voice is detected, symmetric relations are
stored with sorted endpoints so one fact is one edge, and beyond that the edge
points from the first mention to the second. When direction matters to an
answer, read `Edge.sentence` — which is the point of storing it.

## Results

`filing text`, on the built index — 6/6.

| check | result |
|---|---|
| every filing parsed or quarantined with a reason | 407/407, **0 quarantined** |
| chunk offsets resolve to identical source text | 200 chunks across 149 filings, **0 differ** |
| re-indexing an unchanged corpus costs zero embedding calls | 0 upserted, 32,218 already present, **0 HTTP calls** |
| recall@50 on the smoke set | **100%** over 30 questions (floor 85%) |
| reranking improves precision@5 | 0.687 → 0.693 (**+0.007**) |
| graph edges quote their source sentence | 200 of 2,986 checked, **0 wrong** |

The build itself: 32,218 chunks in **140.6 minutes**, 230 chunks/min, zero
network calls.

### Recall@50 of 100% is a weak result wearing a strong number

The gold sets here run from 21 chunks to 318. Landing *one* of 318 in a
fifty-candidate window is not a hard task, so 100% says the retriever is not
broken — it does not say retrieval is good. The number with signal in it is
where the first gold chunk lands: **rank 1 for 19 of 30 questions, inside the
top 5 for 26, worst case rank 24.** That is the distribution the floor should
have been written against, and M4's larger eval set is where that gets fixed
properly rather than by retuning this one.

### The reranker does not earn its latency on this set

+0.007 clears a `> 0` gate. It should not be reported as a win, because the
per-question breakdown is a coin flip:

| | questions |
|---|---:|
| reranking improved p@5 | 8 |
| reranking made it worse | 7 |
| no change | 15 |

Fifty forward passes per query bought a wash. The pattern underneath is not
random, though: **the reranker gains where fusion did badly and loses where
fusion did well.** Its biggest gains are all on questions RRF scored 0.4 or
below (`nvda-hopper` 0.0 → 0.6, `pfe-paxlovid` 0.4 → 0.8, `jnj-talc`
0.6 → 1.0); its biggest losses are all on questions RRF had already scored
0.6 or better, where the only available move was downward.

One caveat in the reranker's favour, which is a caveat and not a defence: gold
here is *"the chunk matches the predicate"*, and a cross-encoder is trained for
*"the passage answers the query"*. Those disagree at the margin, so some of the
seven regressions are the reranker preferring a passage this metric cannot
credit. That is a limit of the metric — stated above before the result came
in, and not a reason to discount the number now that it is disappointing.

The threshold stays at `> 0` regardless. Moving it after seeing 8–7–15
would be choosing a gate that this run happens to pass, which is the one thing a
pre-registered criterion exists to prevent. What changes instead is M4: 150
questions with hand-checked spans, where a delta this size can be told apart
from noise instead of merely surviving a sign test.
