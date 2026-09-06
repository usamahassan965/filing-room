"""Assemble the three slices, verify every span, and freeze questions_v1.0.jsonl.

This is the record of how ``data/eval/questions_v1.0.jsonl`` was made, kept in
the repository because two of its three slices cannot be regenerated: sixty
narrative questions and ten unanswerable ones were written by a person, and the
question text is the input, not the output.

Run it and it rebuilds the frozen file byte for byte -- ``dataset.write`` sorts
by id and writes LF, so the sha256 in ``docs/eval-set.md`` is reproducible from
a clone. It refuses to write if ``verify_spans`` returns a single complaint.

    python scripts/eval_v1_0/freeze_dataset.py

The three JSON files beside it are the located spans -- numeric facts found in
their filings by ``authoring.build_numeric``, narrative candidates found by
``authoring.find_candidates``. They are derived data, committed anyway because
deriving them again needs the 407-filing corpus, which is not in the repo.

A v1.1 is a new directory beside this one, not an edit to this file.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from narrative_questions import QUESTIONS  # noqa: E402
from unanswerable_questions import UNANSWERABLE  # noqa: E402

from filing.config import settings  # noqa: E402
from filing.eval import dataset  # noqa: E402
from filing.eval.authoring import Candidate, narrative_question, verify_spans  # noqa: E402
from filing.eval.dataset import EvalQuestion, Span  # noqa: E402

cfg = settings()

# ---------------------------------------------------------------- numeric
_numeric_json = json.loads((HERE / "numeric.json").read_text("utf-8"))
numeric = [EvalQuestion.from_json(d) for d in _numeric_json]

# -------------------------------------------------------------- narrative
base = [Candidate(**c) for c in json.loads((HERE / "narrative_cands.json").read_text("utf-8"))]
extra = [Candidate(**c) for c in json.loads((HERE / "narrative_extra.json").read_text("utf-8"))]


def cand(key: object) -> Candidate:
    return extra[int(str(key)[1:])] if isinstance(key, str) else base[key]


narrative: list[EvalQuestion] = []
for keys, text, note in QUESTIONS:
    cs = [cand(k) for k in keys]
    q = narrative_question(cs[0], text, note=note)
    if len(cs) > 1:
        q = replace(
            q,
            spans=tuple(
                Span(accn=c.accn, char_start=c.char_start, char_end=c.char_end, quote=c.body[:240])
                for c in cs
            ),
            gold_source=f"{q.gold_source}; the same passage recurs in {len(cs) - 1} more filing(s)",
        )
    narrative.append(q)

# ----------------------------------------------------------- unanswerable
unanswerable = [
    EvalQuestion(
        id="",
        slice="unanswerable",
        question=text,
        origin=f"hand-written; unanswerable because the {reason.split('-')[0]} is out of corpus"
        if reason.endswith("absent")
        else "hand-written to be answerable only by invention",
        gold_source=f"no gold: {reason}",
        note=note,
    )
    for text, reason, note in UNANSWERABLE
]

# ------------------------------------------------------------------ freeze
questions = dataset.renumber(numeric + narrative + unanswerable)
dataset.validate(questions)

problems = verify_spans(cfg, questions)
if problems:
    print(f"REFUSING TO FREEZE -- {len(problems)} problem(s):")
    for p in problems[:40]:
        print("  ", p)
    raise SystemExit(1)
print("span verification: clean")

path = dataset.dataset_path(cfg.data_dir)
digest = dataset.write(questions, path)

print(f"wrote {path}  ({path.stat().st_size:,} bytes)")
print(f"sha256 {digest}")
print("counts", dataset.counts(questions))
spans = [s for q in questions for s in q.spans]
print(f"spans {len(spans)} over {len({s.accn for s in spans})} filings")
print("tickers", dict(sorted(Counter(t for q in questions for t in q.tickers).items())))
print("forms", dict(Counter(f for q in questions for f in q.forms)))
