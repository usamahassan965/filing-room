# The baseline

What the naive system scores over all 150 questions of eval set v1.0 — and why
it is reported as two runs rather than one, since a RAG answer fails in two
separable ways and only one of them needs a model.

Code: `src/filing/eval/naive.py`, `runner.py`, `depth.py`, `metrics.py`.
Tests: `tests/test_eval_runner.py`, `test_eval_depth.py`, `test_eval_metrics.py`.
Results: `results/baseline-retrieval.json`, `results/retrieval-depth.json`.

```bash
python -m filing.eval run --config baseline             # 150 questions, 54s, generates
python -m filing.eval run --config baseline-retrieval   # 150 questions, 104s, 0 API calls
python -m filing.eval depth                             # where the evidence actually ranks
```

## Why there are two baselines

A RAG answer fails in two separable ways. Either the retriever never found the
evidence, or it found it and the generator fumbled it. Only the second needs a
model, and conflating them means every future improvement is argued about
rather than attributed. So the baseline is split at that seam:

| config | what it does | cost |
|---|---|---|
| `baseline` | retrieve, then one chat call | 150 requests |
| `baseline-retrieval` | retrieve, and stop | nothing |

The split also decided which model generates. The free tier caps
`gemini-3.5-flash` at **20 requests a day**, so a 150-question run on it takes
eight days and a re-run takes eight more — an unclosable gate, since M4's own
wording asks for a run that fits the rate-limit budget and is reproducible. The
baseline therefore generates with `gemini-3.5-flash-lite`, which answers the
same 150 in under a minute. The results file records the model, so the weaker
generator is a stated fact rather than a hidden one, and a baseline being
beatable is the point of having one.

`baseline-retrieval` is not a lesser measurement. It is the **ceiling** on every
number the full baseline can produce, since no generator cites evidence it was
never shown, and it is the half that runs on any machine on any day.

The split is enforced rather than trusted. `generate=False` blanks every chat
field before the config is fingerprinted, so a retrieval run cannot silently
depend on which generator happens to be configured; the runner never constructs
a chat backend; and the scorecard reports exact-match, routing and abstention as
`--` rather than `0.0%`, because "routed everything wrong" and "does not route"
are different claims and a table that prints 0.0% for both is lying about one.

## What it found

```
| slice        |  n | exact | router | cite ok | cite gold | abstain |
| numeric      | 80 |  7.5% |     -- |  100.0% |     22.2% |      -- |
| narrative    | 60 |    -- |  51.7% |  100.0% |     12.9% |      -- |
| unanswerable | 10 |    -- | 100.0% |      -- |        -- |  100.0% |
| overall      |150 |  7.5% |  27.3% |  100.0% |     13.9% |  100.0% |
```

Three readings, in descending order of how much they should worry you.

**The system always cites, and rarely cites right.** `cite ok` is 100% and
`cite gold` is 13.9%: every answer resolves to a real chunk in the store, and
seven times in eight that chunk is not the one holding the evidence. A citation
that resolves is not a citation that supports, and only the second is worth
anything to a reader checking a filing.

**Numeric exact (7.5%) is higher than numeric hit@5 (2.5%).** The generator is
getting numbers right more often than the evidence for them is retrieved, which
has two possible causes that call for opposite responses: the gold spans are
under-annotated and the same figure appears in chunks not marked gold, or the
model is reciting figures it saw in pretraining. If it is the second, that 7.5%
is contamination rather than capability. Separating them is M5 work, and it is
the reason `cite gold` is reported next to `exact` instead of `exact` alone.

**Abstention is the one thing it does perfectly.** All ten unanswerable
questions are refused. That is worth stating plainly, because a naive system
that hedges everything would also score 100% here — and this one does not, since
it answers the other 140.

Underneath all of it sits the retrieval table, which is the ceiling:

```
| slice        |  n | hit@1 | hit@5 | hit@10 | ndcg@5 | LLM calls |
| numeric      | 80 |  2.5% |  2.5% |   5.0% |  0.008 |         0 |
| narrative    | 60 | 10.0% | 15.0% |  30.0% |  0.105 |         0 |
| overall      |150 |  5.7% |  7.9% |  15.7% |  0.050 |         0 |
```

Low enough to suspect the harness rather than the system, so the harness was
checked first. Handed a chunk's own text as the query, the retriever returns
that chunk at **rank 1, six times out of six, at cosine 0.95–0.98**, and every
id it returns is known to the chunk store. Every one of the 260 gold spans is
covered by an indexed chunk; none is missing from the index. The plumbing is
sound and the numbers are real.

## The number that matters more than hit@10

`hit@10 = 0` covers two completely different diagnoses — the evidence ranked
24th, or the evidence is nowhere — and they call for opposite work. So
`filing.eval depth` reports the whole curve, and beside it whether the search
at least reached the right *document*:

```
| slice     |  n | hit@10 | hit@50 | hit@100 | hit@500 | median rank | filing@10 | filing rank |
| narrative | 60 |  30.0% |  55.0% |   63.3% |   86.7% |          24 |     70.0% |           5 |
| numeric   | 80 |   5.0% |  12.5% |   18.8% |   43.8% |         147 |     26.2% |          35 |
```

**On narrative questions the retriever knows which filing to read and not where
to look in it.** The right document is at median rank 5 and inside the top ten
70% of the time; the right *passage* is at median rank 24. That is a ranking and
chunking problem, not a representation problem, and the curve prices the repair
in advance: a pipeline that fuses to depth 50 and reranks has a ceiling of 55%,
against the 30% dense-only achieves at 10. Everything M5 wants to do here has
room to work.

**On numeric questions it is not close.** The gold chunk sits at median rank
147, and for 45 of the 80 questions it is not in the top 500 at all — out of
48,934 chunks. No reranker repairs that, because reranking reorders a list the
passage is absent from. The right filing is only found at rank 35.

The reason is visible in the eval set's own geometry. A numeric gold span has a
median length of **six characters** — it is the digits of the figure and nothing
else — sitting in a 2,048-character slice of a financial statement, which from
the inside is a wall of numbers with almost no distinguishing prose. The
question says "revenue for fiscal 2021"; the passage that answers it says
`16,434` in a column. Dense similarity between those two is close to noise. The
slice is harder still because 79 of the 80 questions are answered by a
prior-year comparative column (see `docs/eval-set.md`), so the surrounding text
is about a *different* fiscal year than the question asks about.

This is the empirical case for the architecture the project set out to build.
The numeric route was going to SQL over the XBRL facts on the argument that
retrieving text to find a number is the wrong tool; the argument is now a
measurement, and its size is 2.5% hit@5.

## What this does not measure

Answer quality beyond exact match. A numeric answer is right or it is not, but
the narrative slice is scored by routing and citation, not by whether the prose
is any good — that needs a judge, and a judge is itself a model whose agreement
with a human has to be measured before its verdicts mean anything. Retrieval
quality is measured only against *this* gold, which is a span in a filing rather
than a chunk id, so the numbers move if the chunker changes and do not move if
the gold is re-cut — the property the whole eval set was built for.

## Three defects this shook out

**The runner cached failures.** A 429 came back as an outcome carrying an
`error`, and the cache stored it like any other result, so a run designed to
*resume* after a rate limit resumed by replaying its own rate limits from disk,
for as long as the fingerprint lived. Errors are now never cached; the
asymmetry justifies it, since re-answering a question that would have succeeded
costs one call, while remembering a 429 forever silently caps the score.

**A test was spending live API quota.** `test_a_second_run_costs_nothing_and_
needs_nothing` passed `None` for the backend and asserted the cache would serve
every question. On a cache hit that is fine. On a cache *miss* the runner does
exactly what it should and builds the real client — so when an unrelated bug
emptied the cache, a hermetic-looking test burned real requests and reported the
429s as ordinary failures. It now uses doubles that raise on contact, and
`tests/conftest.py` fails any test that opens a socket to a non-localhost
address. A suite's hermeticity should be enforced, not assumed. It also made the
suite faster: two files went from 121s to 1.9s, nearly all of it retry backoff.

**The runner read the backend from the environment, not from the config.** A
config named `baseline-local-alt` declared `chat_backend="ollama"`, was
fingerprinted, cached and written under `llama3.2:3b`, and sent all 180 of its
calls to Gemini — because the runner built its client from `LLM_BACKEND` in the
environment while the config's own field was used for nothing but the label.
The run completed cleanly, produced plausible numbers, and was wrong in the only
way an eval harness must never be: a results file that misnames the model that
produced it is worse than no file, because nobody re-checks a number that looks
fine. It surfaced by accident — a 503 quoting *"this model is currently
experiencing high demand"*, which is not a sentence a localhost server says.

The fix is one argument, and the test is worth more than the fix: it asserts
that the backend the runner constructs is the backend the fingerprint records,
so the label and the call can never drift apart again. The offending run and its
outcome cache were deleted rather than relabelled — the numbers were real, but
their provenance was not, and a cache keyed on a lie poisons every run that
inherits it.

There is a redeeming detail. The call cache is content-addressed on backend,
model and payload, so those Gemini calls were stored under Gemini's key all
along. When `baseline` later ran for real, 146 of its 150 answers replayed from
that cache and only the 4 that had failed with 503 were re-issued — four live
calls for a full 150-question baseline. The clause the gate asks for,
*reproducible from cache*, was demonstrated by the accident that broke it.
