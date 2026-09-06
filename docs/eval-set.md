# The eval set

What `data/eval/questions_v1.0.jsonl` contains, how each of the 150 questions and
every one of its 260 gold spans came to exist, and what the numbers measured on
it can and cannot support.

Code: `src/filing/eval/authoring.py` (the machinery),
`scripts/eval_v1_0/` (the record of the build itself, rerunnable),
`dataset.py` (reads and validates it), `metrics.py` (scores against it).
Tests: `tests/test_eval_dataset.py` — the `TestFrozenSet` class asserts the
claims on this page against the file itself, so a datasheet that has drifted
from its data fails CI rather than misleading a reader.

## Identity

| | |
|---|---|
| Version | `v1.0` |
| File | `data/eval/questions_v1.0.jsonl` |
| sha256 | `38ee1f9188cda1597dffbcfdaeea7daaccb6e762b248b1aec3cdb185de35692f` |
| Size | 171,915 bytes, 150 lines, LF |
| Frozen | yes — the file is committed and never edited in place |

```bash
python -m filing.eval dataset --show 3
```

The hash is not decoration. It goes inside the fingerprint of every run
(`EvalConfig.fingerprint`), which is written into `results/*.json`, so two
numbers are comparable only if they were measured on the same questions. That is
also why `dataset.write()` writes bytes with LF endings rather than going through
`Path.write_text`: on Windows the latter translates `\n` to `\r\n`, and a set
that hashed differently depending on which machine last touched it would silently
make every result incomparable across a clone.

## What it is for

Three questions, one slice each.

| slice | n | route it should take | what it measures |
|---|---|---|---|
| `numeric` | 80 | `sql` | does the system look the number up, or narrate one |
| `narrative` | 60 | `text` | does retrieval find the passage that answers the question |
| `unanswerable` | 10 | `refuse` | does the system say it does not know |

The third slice is the one most eval sets omit and the reason the other two are
worth reading. A system that never abstains scores full marks on a set that only
asks answerable things; ten questions is not enough to estimate an abstention
rate precisely, but it is enough to tell a system that abstains sometimes from
one that never does, and that is the distinction the number exists to draw.

## Gold is a span, not a chunk id

Every piece of evidence in this file is `(accn, char_start, char_end)` into the
filing's flattened text, plus a quote of what sits there. A retrieved chunk
counts as gold when it **overlaps** that range.

This is the single most consequential decision in the set, and it is a
deliberate repair of M3, whose eval labelled gold by *predicate* — a retrieval
was correct if the returned chunk came from the right Item. That cannot tell the
paragraph that answers the question from the forty other paragraphs of Item 1A,
and it rewards a system that returns the whole section.

Spans also survive a re-chunk. The naive baseline cuts every 2,048 characters
and the real system cuts on block boundaries with overlap; the two produce
entirely different chunk ids over the same documents. Chunk-id gold would have
meant two eval sets and no comparison. Character offsets mean both systems are
scored against identical ground truth, which is the whole basis of the M5 table.

The `quote` field (≤240 characters) exists so the file can be *read*. A gold
label nobody can check without running code is a gold label nobody checks.

### Multi-span gold, and why `hit_rate` is the headline

Sixty of the 150 questions carry more than one span — 50 numeric and 10
narrative — and the spans are **alternatives, not a set to be collected**.
Two things produce them:

- A number a filing prints more than once — in the statement, in the MD&A table,
  and again as the prior-year comparative. The median numeric question has two
  printings; the most-printed has eight.
- A disclosure a company repeats across annual reports almost verbatim.
  ExxonMobil's climate-transition risk factor is materially the same passage in
  three consecutive 10-Ks; so are Qualcomm's new-initiative risks, Chevron's
  legal/ESG risks and Target's non-GAAP reconciliation.

Scoring those as "must find all three" would mark a system wrong for finding the
2022 copy of a passage that also appears in 2021 and 2023. So `hit_rate@k` —
did *any* gold span appear in the top k — is the metric that matches the task,
and `recall@k` is reported beside it as the stricter number. On repeated
disclosures recall is bounded well below 1 by k itself, and that is a property of
the question, not a failure of the system.

## Composition

150 questions over **87 filings**, all `10-K`, drawn from the 407-filing corpus
described in [corpus.md](corpus.md). All 20 tickers in the universe appear.

| | |
|---|---|
| Gold spans | 260, mean length 270 chars (min 5, max 2,082) |
| Questions with >1 span | 60 |
| Cited filings' period ends | 2020-01-26 … 2025-02-01 |
| Fact periods asked about | 2018-01-28 … 2024-11-03 |
| Forms | 10-K only |

Per ticker: ABBV 8 · AMD 8 · AVGO 9 · COP 8 · COST 9 · CVX 7 · HD 9 · INTC 5 ·
JNJ 9 · LLY 8 · LOW 7 · MRK 8 · NVDA 9 · PFE 10 · PSX 10 · QCOM 7 · SLB 2 ·
TGT 2 · WMT 3 · XOM 2.

The spread is uneven and was not corrected. Coverage came from where usable gold
was found, and forcing it flat would have meant either inventing questions for
the thin companies or discarding good ones from the thick — both of which trade
a real property of the set for a cosmetic one. It does mean a per-company
breakdown of results is not meaningful for SLB, TGT, WMT or XOM.

Only 10-Ks appear even though the corpus is 74% 10-Q. Annual reports carry the
full risk-factor and MD&A discussion that the narrative slice needs, and their
XBRL facts are the ones a person actually asks about. The quarterly filings are
in the index and can be retrieved; nothing in the set has gold in one, so no
number here says anything about retrieval over 10-Qs.

## How each slice was built

The rule the whole authoring module obeys: **gold is found, not asserted.**
Nothing in the pipeline asks a language model what the answer is. A set whose
gold is a model's opinion measures agreement with that model and calls it
accuracy.

### numeric — 80, generated

Selected by SQL from `facts_current`: `unit = 'USD'`, `span IN ('FY','instant')`,
`|val| > 1e6`, `form = '10-K'`, and `tag` in a hand-written list of 26 headline
US-GAAP concepts. The tag list is the editorial part — "Deferred Tax Assets,
Other" is a real fact and a fake question — and the list's order is the
sampler's tie-break, so the headline figures are reached for first.

Caps of **5 per company and 5 per concept** stop the slice collapsing into eighty
variations of "what were revenues", which would make the slice mean a
measurement of one question asked repeatedly. The 80 questions cover 16 distinct
concepts, evenly at 5 each.

The question text is generated from the concept's own label, with two templates
so a duration and an instant read correctly in English:

```
What did NVDA report for Revenues for the fiscal year ended 2024-01-28?
What was PFE's Liabilities as of 2019-12-31?
```

Gold is then **located, not assumed**. The value is rendered the same several
ways M2's verifier renders it (units / thousands / millions, grouped or plain,
parenthesised when negative) and searched for in the filing's flattened text.
A rendering is accepted only if it has ≥4 digits and hits ≤8 places — `349,585`
is a fingerprint, `12` is a coincidence, and a span matching half the balance
sheet makes recall meaningless. Matches must be delimited: `349,585` inside
`1,349,585` is a different number. **A fact the store holds but the document
never prints is dropped**, not kept as unanswerable — it *is* answerable, from
the XBRL attachments, and labelling it otherwise would teach the router the
wrong lesson.

One consequence worth stating plainly, because it looks like a bug: a numeric
question's `period_end` is the **fact's** period, not the filing's. Facts run
back to 2018-01-28 while the corpus starts at 2020-01-26, and 79 of the 80
questions are answered by a filing published *after* the year they ask about —
32 by a lag of one year, 45 by two, 2 by three. The figure is there as a
prior-year comparative, and `accn` names the filing that prints it.

That was not designed in; it fell out of taking the *first* place each value
could be verified, and the sampler walks companies before years. It makes the
slice harder than it looks and more realistic than a same-year set would be —
comparatives are how financial statements are actually read — but it does mean
the slice is largely a test of finding a number in a comparative column, and a
system tuned to match the filing's own fiscal year to the question's would
score badly here for a reason that has nothing to do with retrieval quality.

Each numeric question's spans all live in that single filing, so multi-span
numeric gold always means *the same document prints the number more than once*,
never *two documents agree*.

Scored with `exact_match`: the answer text is scanned for numbers, every
plausible scale is tried (a filing says "16,434" and means millions; an answer
may say "$16.4 billion"), and the question is right if any reading lands within
0.5% of the XBRL value — the same tolerance M2 uses on the same values.

### narrative — 60, hand-written

A two-stage process, machine then human.

**Located by machine.** Inside Items 1A and 7 only — the two places a filing
says something rather than reports something — the parser's block structure is
scanned for a subsection heading: 24–170 characters, 4–24 words, no terminal
punctuation, not starting with `Item`/`See`/`As of`/`The following`, and not
mostly digits (which would be a table row that lost its columns). The heading
must be followed by ≥500 characters of prose. Gold is that heading plus up to
2,400 characters of what follows — a subsection often runs for pages, and calling
five pages gold would let a system score by returning any of them.

**Written by hand.** Every question was typed after reading the located span.
52 came from Item 7 (MD&A) and 8 from Item 1A (risk factors).

The rule while writing: **a question may not quote its own gold.** A question
that repeats the span's wording measures string matching, and BM25 would win the
slice on phrasing alone. This is enforced mechanically —
`test_narrative_questions_do_not_quote_their_own_gold` fails if any question
shares an 8-word run with any of its spans. Eight, not five, was measured rather
than chosen: at five words the check fired six times, and every one was an
unavoidable term of art ("selling, general and administrative expense",
"net periodic benefit cost for"). A question is allowed to name its subject in
the subject's own words; what it may not do is reproduce a clause.

Three questions were rewritten during the final read-through because the premise
did not match the span — a Phillips 66 question asked what PSX *received* under
an advance term loan when the passage reports what was outstanding *to* WRB; a
DCP LP question missed that the distribution was to unitholders *other than*
Phillips 66; a J&J question called fiscal 2022 "2023" because the period ends
on 2023-01-01. Each is the kind of error that produces a plausible wrong answer
and an unexplainable score.

### unanswerable — 10, hand-written

Five reasons, chosen so a system cannot pass by learning one refusal trigger:

| reason | n | example |
|---|---|---|
| `company-absent` | 2 | Apple's fiscal 2023 net sales |
| `period-absent` | 2 | Walmart's fiscal 1998 net sales |
| `document-absent` | 3 | NVIDIA's CEO compensation (that is DEF 14A, not 10-K) |
| `forecast` | 2 | Pfizer's expected 2027 dividend per share |
| `granularity` | 1 | Costco paid members in Japan specifically |

`document-absent` and `granularity` are the interesting ones. Both name a
company that *is* in the corpus and a fact that plausibly *sounds* like a 10-K
disclosure; the retrieval will return confident, on-topic, wrong context, which
is exactly the situation in which a RAG system confabulates. `company-absent`
is the easy case and is only two of the ten.

Scored with `abstention` — a case-insensitive match for the token
`INSUFFICIENT EVIDENCE`, which the prompt instructs the model to emit verbatim.
Reported beside `over_answered`, the same number from the other side, because
"answered 9 of 10 unanswerable questions" is the sentence a reader needs.

## Verification

Before the file was written, `authoring.verify_spans` re-parsed every one of the
87 cited filings from the raw HTML and confirmed that each span still holds the
text recorded in its quote. `scripts/eval_v1_0/freeze_dataset.py` refuses to
write on a single complaint. This is insurance against the one failure that
would quietly poison everything — a span recorded against one parse and scored
against a different one.

It is also a command, not a claim about something that happened once. Run it
against the frozen file whenever the parser changes:

```bash
python -m filing.eval verify
```

`dataset.validate` runs on every read, not just at freeze time. It rejects
duplicate ids, unknown slices, empty questions, undocumented provenance, numeric
questions without a value or unit or located span, narrative questions without
gold, unanswerable questions that carry gold, and empty spans. Every rule is
there because breaking it produces a *number* rather than an error.

## What this set does not measure

- **Answer quality beyond the number.** `exact_match` cannot tell a right figure
  stated for the wrong reason from a right one. There is no LLM judge, on
  purpose: it would make every result a function of a model that changes
  underneath it, cost a thousand calls a run against a 1,000-call-a-day budget,
  and grade the system with a sibling of the system.
- **Multi-hop and cross-company reasoning.** Every question is answerable from
  one company's filings. The graph store is not exercised here.
- **10-Q retrieval**, **8-K**, and anything outside Items 1A and 7 for narrative.
- **Calibration.** Ten unanswerable questions give a rate, not a confidence
  interval.
- **Generalisation past 20 large-cap US issuers.** The corpus is deliberately
  narrow; a small-cap filer's document structure is not represented.

## Changing it

Don't — not `v1.0`. Fix a typo in a question and every number ever measured on
this file becomes incomparable, which the fingerprint will correctly report as a
different experiment while a reader compares the rows anyway.

A new set is a new version: build it, write it to `questions_v1.1.jsonl`, add a
section here, and re-run the configs you want to compare. The old file stays.

`scripts/eval_v1_0/freeze_dataset.py` rebuilds this exact file — same bytes,
same sha256 — because `dataset.write` sorts by id and writes LF. But *rebuild*
is not *regenerate*: the numeric slice could be sampled again from
`authoring.build_numeric`, while the 70 hand-written questions are an *input* to
that script, not an output of anything.

That asymmetry is why `data/eval/` is one of the two things `.gitignore`
un-ignores under `data/`, and it is the exemption for the opposite reason to the
other. `manifest.duckdb` is committed because rebuilding it needs the corpus;
this file is committed because no amount of corpus would produce it again.
