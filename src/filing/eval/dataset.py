"""The frozen question set: schema, gold, and the file it lives in.

The eval set is *data*, not code. It ships as one JSON Lines file under
``data/eval/``, tagged with a version and a content hash, and this module is
only the schema that reads it back. Nothing here generates questions -- that is
``authoring.py``, which runs once, is allowed to be slow, and is allowed to look
at the whole corpus. Keeping the two apart is what makes the set freezable: a
number measured on ``questions_v1.jsonl`` stays comparable because the file
cannot change underneath it without changing its hash.

**Gold is a span, not a chunk id.** A question's evidence is
``(accn, char_start, char_end)`` into the filing's flattened text -- the same
coordinate system chunks are cut from -- and a chunk counts as gold when it
overlaps that span. This is the single most important decision in M4 and it is a
direct repair of M3, whose eval set scored a retrieval as correct when the
returned chunk came from the right *item*. That is gold by predicate: it cannot
distinguish the paragraph that answers the question from the forty other
paragraphs of Item 1A, and it silently rewards a system that returns the whole
section. Spans also survive a re-chunk, which matters immediately, because the
naive baseline and the real system cut the same documents differently and have
to be scored against identical ground truth. Chunk-id gold could not have done
that; there would have been two eval sets and no comparison.

**Three slices, three jobs.**

* ``numeric`` -- the answer is a number the XBRL store already holds, and the
  question is whether the system routes to SQL instead of guessing from prose.
  Gold is the value, plus every place that value is printed in the filing.
* ``narrative`` -- the answer is a passage. Gold is the passage's span.
* ``unanswerable`` -- the corpus does not contain the answer, and the only
  correct behaviour is to say so. A set without these measures fluency, not
  truthfulness, because a system that never abstains scores full marks on one
  that only ever asks answerable things.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

DATASET_VERSION = "v1.0"

# numeric -> sql, narrative -> text, unanswerable -> refuse. Held here rather
# than inferred at scoring time so router accuracy is scored against a label in
# the file, which a reader can disagree with, rather than against a rule in the
# scorer, which they cannot see.
ROUTES = {"numeric": "sql", "narrative": "text", "unanswerable": "refuse"}

SLICES = ("numeric", "narrative", "unanswerable")

# Filings round to the cent; the store carries full precision. Same tolerance
# M2 uses on the same values, for the same reason -- see stores/questions.py.
NUMERIC_TOLERANCE = 0.005


@dataclass(frozen=True, slots=True)
class Span:
    """A stretch of one filing's flattened text, and what it says.

    ``quote`` is stored so the file can be *read*. A gold label nobody can check
    without running code is a gold label nobody checks; 240 characters is enough
    for a reviewer to see whether the span is really the evidence, and small
    enough that 150 questions stay a file rather than a corpus.
    """

    accn: str
    char_start: int
    char_end: int
    quote: str = ""

    def overlaps(self, accn: str, start: int, end: int) -> bool:
        return accn == self.accn and start < self.char_end and end > self.char_start


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    slice: str
    question: str
    # Gold. Which fields are populated depends on the slice, and ``validate``
    # is what enforces that rather than a comment hoping for it.
    value: float | None = None
    unit: str = ""
    tag: str = ""
    spans: tuple[Span, ...] = ()
    # What the question is *about*, which is provenance and not a hint: nothing
    # in the runner passes these to a retriever as a filter.
    tickers: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    period_end: str = ""
    accn: str = ""
    item_key: str = ""
    # How this question and its gold came to exist. Every question carries it,
    # and the datasheet is a summary of these two fields.
    origin: str = ""
    gold_source: str = ""
    note: str = ""

    @property
    def route(self) -> str:
        return ROUTES[self.slice]

    @property
    def answerable(self) -> bool:
        return self.slice != "unanswerable"

    def is_gold(self, accn: str, char_start: int, char_end: int) -> bool:
        """Does this chunk overlap any gold span? The scorer's whole definition."""
        return any(s.overlaps(accn, char_start, char_end) for s in self.spans)

    # ---------------------------------------------------------------- codec

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["spans"] = [asdict(s) for s in self.spans]
        for key in ("tickers", "forms"):
            d[key] = list(d[key])
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EvalQuestion:
        d = dict(d)
        d["spans"] = tuple(Span(**s) for s in d.get("spans", ()))
        for key in ("tickers", "forms"):
            d[key] = tuple(d.get(key, ()))
        return cls(**d)


class DatasetError(ValueError):
    """The file on disk is not a usable eval set. Never caught in the runner."""


def validate(questions: list[EvalQuestion]) -> None:
    """Reject a set that cannot be scored, loudly, before anything is measured.

    Every rule here exists because breaking it produces a *number* rather than
    an error -- a duplicate id silently halves a slice, a numeric question with
    no value scores as a miss for every system equally, an unanswerable question
    with a gold span is a contradiction that rewards whichever behaviour the
    scorer happened to implement first.
    """
    seen: set[str] = set()
    for q in questions:
        where = f"{q.id!r}"
        if q.id in seen:
            raise DatasetError(f"duplicate question id {where}")
        seen.add(q.id)
        if q.slice not in SLICES:
            raise DatasetError(f"{where}: unknown slice {q.slice!r}")
        if not q.question.strip():
            raise DatasetError(f"{where}: empty question")
        if not q.origin or not q.gold_source:
            raise DatasetError(f"{where}: undocumented provenance")
        if q.slice == "numeric":
            if q.value is None:
                raise DatasetError(f"{where}: numeric question with no gold value")
            if not q.unit:
                raise DatasetError(f"{where}: numeric question with no unit")
            if not q.spans:
                raise DatasetError(f"{where}: numeric gold was never found in a filing")
        if q.slice == "narrative" and not q.spans:
            raise DatasetError(f"{where}: narrative question with no gold span")
        if q.slice == "unanswerable":
            if q.spans:
                raise DatasetError(f"{where}: unanswerable question carries gold evidence")
            if q.value is not None:
                raise DatasetError(f"{where}: unanswerable question carries a gold value")
        for s in q.spans:
            if s.char_end <= s.char_start:
                raise DatasetError(f"{where}: empty span {s.char_start}:{s.char_end}")


def dataset_path(data_dir: Path, version: str = DATASET_VERSION) -> Path:
    return data_dir / "eval" / f"questions_{version}.jsonl"


def write(questions: list[EvalQuestion], path: Path) -> str:
    """Write the set and return its sha256. Sorted by id, so the hash is stable.

    Written as bytes with LF endings rather than through ``write_text``, which
    would translate them to CRLF on Windows. The hash of this file *is* the
    identity of the experiment -- it goes inside every run fingerprint -- and a
    set that hashed differently depending on which machine last wrote it would
    make every result incomparable across a clone.
    """
    validate(questions)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(q.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
        for q in sorted(questions, key=lambda q: q.id)
    )
    raw = body.encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> list[EvalQuestion]:
    if not path.exists():
        raise DatasetError(
            f"no eval set at {path}. Rebuild it with `python scripts/eval_v1_0/freeze_dataset.py`."
        )
    questions = [
        EvalQuestion.from_json(json.loads(line)) for line in path.read_text("utf-8").splitlines()
    ]
    validate(questions)
    return questions


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(questions: list[EvalQuestion]) -> dict[str, int]:
    out = dict.fromkeys(SLICES, 0)
    for q in questions:
        out[q.slice] += 1
    return out


def renumber(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    """Give every question a stable id of the form ``num-007``.

    Ids are positional within a slice and assigned once, at freeze time. They
    are not meaningful; they are short enough to name a question in a results
    table and a bug report, which is all an id is for.
    """
    prefix = {"numeric": "num", "narrative": "nar", "unanswerable": "una"}
    out: list[EvalQuestion] = []
    for name in SLICES:
        group = [q for q in questions if q.slice == name]
        for i, q in enumerate(group, start=1):
            out.append(replace(q, id=f"{prefix[name]}-{i:03d}"))
    return out
