# Parsing

How 1.12 GB of filing HTML becomes text a citation can point into, why the
parser is 200 lines of stdlib rather than a library, and what the corpus
actually looks like once it has been through it.

Code: `src/filing/stores/parse.py`. Tests: `tests/test_parse.py`.

## The requirement that picked the parser

A chunk carries `{accn, item, char_start, char_end}`, and the M3 gate re-reads
the source to prove `text[char_start:char_end]` is still that chunk. That is
only possible if the offsets index a string stored verbatim.

This rules out a converter, not on speed but on kind. Docling's markdown is a
*transformation* of the document — table pipes inserted, cells reordered into
rows, headings re-marked — and there is no function from an offset in the
markdown back to a span of the filing. It would parse the corpus fine and then
be unable to answer the one question M6 has to answer: *where in the filing does
this sentence come from?*

So the parser flattens to text and keeps block boundaries, and nothing
downstream is ever allowed to reflow that text.

## What the alternatives cost

Stratified nine-file sample, three from each size tercile, 19.89 MB total, one
CPU. `scripts` for this live in the scratchpad; the numbers were taken on
2026-09-06.

| Parser | Seconds | MB/min | Full corpus | Chars per MB |
|---|---:|---:|---:|---:|
| `filing.stores.parse.flatten` (stdlib `html.parser`) | 2.88 | 414 | **2.5 min** | 93k |
| selectolax `.text()` (C, Lexbor) | 0.25 | 4,748 | 13 s | 94k |
| docling `convert` → markdown | 50.89 | 23 | 44 min | 318k |

Three things worth reading off that table.

**The 11.6-hour estimate was wrong by two orders of magnitude.** M1 carried an
assumed 1.5 MB/min in `ASSUMED_PARSE_MB_PER_MIN` and reported the pass as 11.6
hours, which is what made the parquet block cache look like a schedule
necessity. It is 2.5 minutes. The constant is now `PARSE_MB_PER_MIN = 414`, and
the cache stays for a different and better reason: re-indexing has to be
idempotent, and re-deriving offsets on every run is a way to get them silently
wrong.

**Docling produces 3.4× more characters per MB**, because markdown table
syntax is mostly pipes and dashes. Those characters cost embedding tokens.

**selectolax is 11× faster than the stdlib path and is not used.** At 2.5
minutes for the whole corpus the speed buys nothing, and the stdlib parser is
the one whose block-boundary bookkeeping is already written and tested. If the
corpus grows an order of magnitude, this is the first thing to swap; the
`_Flattener` interface is two methods wide.

## Finding the sections

A naive `Item \d` regex over one flattened NVIDIA 10-K finds 55 matches for 23
sections. The extra 32 are the contents table and cross-references in prose
("see Item 15 of this Annual Report").

**A real heading occupies a block by itself.** A cross-reference never does,
because it sits inside a sentence. That single structural fact removes every
prose match, and it is why the flattener records block spans rather than just
emitting text.

What it does not remove is the contents table, which is also made of blocks that
are nothing but a heading. Every simple rule for that fails on some filer in
this corpus:

| Rule | Fails on |
|---|---|
| reject headings with no title | Walmart, Broadcom, ConocoPhillips — number and title in one cell |
| accept headings with no title | AMD's older 10-Qs — number alone, title in the next block |
| drop whatever comes first | Intel — its item table sits *after* the body, at offset 517k of 520k |
| drop any tightly packed run | every 10-Q — "Item 4. Mine Safety Disclosures: Not applicable" really is a 40-character section |

What is true of a contents table and of nothing else is that **it lists the
whole document in a corner of it**: an ascending run of headings covering ≥ 80%
of the distinct items found, packed into ≤ 15% of the text. The four measured
contents tables span 0.3%–0.8% of their documents, so that threshold is not a
close call.

A run also has to break when the numbering goes *backwards*, which is exactly
what happens where the contents table ends and the body's Item 1 begins — in
Costco's 10-Q those two are 83 characters apart.

### Three bugs the corpus found, in order

**Words split across styling spans.** Schlumberger's 2020 10-K renders its Item
8 heading as `<span>I</span><span>tem 8.</span>`. A flattener that inserts a
space at every text-run boundary produces `I tem 8. Financial Statement s and
Supplementary Data.` and loses the financial statements section. Whether two
runs are one word is decided by the source: the space belongs to the source or
nowhere.

**Part headings split across table cells.** ConocoPhillips lays its out as
three cells — `PART`, `I.`, `FINANCIAL INFORMATION` — so `PART` alone matches
nothing, the part stays on the `II` the contents table last set, and all six of
its 10-Qs file Part I under Part II.

**A part heading that is only a full stop.** Chevron announces its first part
as exactly `PART I.`. The rule that killed "Part I of this report on page 33."
was "a heading does not end in a full stop", and it killed this too. The real
discriminator is that a sentence has *words after the numeral*.

### The part is load-bearing for one form only

A 10-K numbers its items once through; a 10-Q restarts at 1 for Part II. So the
section key is `7` in a 10-K and `I.1` in a 10-Q. Keying a 10-K on parts breaks
on ConocoPhillips, which prints no part headings at all — Item 7 would file
under Part I in one filing and under nothing in the next.

The form comes from the manifest, never from the document. Guessing it from
heading count misclassified four 10-Qs as 10-Ks in a 24-filing sample, because a
contents table inflates the count past any threshold that separates them.

## Results over the whole corpus

407 filings, every one accounted for:

| Outcome | Filings | |
|---|---:|---|
| **split** into Item sections | 387 | 95% |
| **degraded** to one whole-document section | 20 | 5% — all Intel |
| **quarantined** | 0 | |

Section counts land where they should: 10-Ks at 17–23 sections (mode 23), 10-Qs
at 6–11 (modes 8 and 11 — a 10-Q's Part II tail varies by whether the filer
prints the "None" items).

### Why Intel is degraded rather than dropped

Intel's 10-Qs and its 2020–2024 10-Ks reorganise the narrative under their own
headings and carry no `Item N` heading in the body at all. Their only item list
is the end-placed table.

Dropping them would remove a whole company from the narrative index to satisfy a
splitter. Instead they are indexed as a single section spanning the document,
with the reason recorded in `ParsedFiling.degraded`. A question about Intel
still retrieves; it just cannot be filtered by item.

Recovering the sections by matching the contents table's *titles* against body
blocks was tried and rejected. On the 2024 10-K it finds 6 of 23 titles, and
Items 7 and 8 — MD&A and the financial statements, the two that matter — are not
among them. The spans between the six that do match would then carry the wrong
item label, and M6 cites that label. A wrong citation is worse than a missing
filter.

`quarantine` is reserved for text that cannot be indexed at all — under 20,000
characters, which in practice means a stored error page rather than a filing.
Nothing in the corpus currently hits it.
