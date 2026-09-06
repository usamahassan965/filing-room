"""The harness: run a named configuration over the frozen set, cache, score.

Three properties matter more here than anything about the systems being
measured, because without them a results table is an anecdote.

**A configuration is a value, and it is hashed.** :class:`EvalConfig` holds every
input that could change a number -- which system, which chat model, which
embedding backend, which collection, which prompt, what k -- and
:meth:`EvalConfig.fingerprint` folds it together with the sha256 of the question
file. Two runs with the same fingerprint are the same experiment; two runs with
different fingerprints are not comparable, and the fingerprint is written into
the results file so that claim can be checked rather than trusted.

**Every answer is cached under that fingerprint.** One outcome per file, keyed by
question id. A run that dies at question 90 resumes at 90. A re-run of a finished
config costs zero HTTP calls, which is what makes the definition of done
("reproducible from cache") a real property rather than a hope -- and on a tier
that allows ~250 chat calls a day, a harness that cannot resume is a harness that
gets one attempt per day.

**Scoring is separate from running.** ``run`` produces outcomes; ``score``
consumes them. Changing a metric never costs a model call, and the same outcomes
can be rescored by a later gate's metrics without re-answering 150 questions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from filing.config import Settings, model_for
from filing.eval import dataset, metrics
from filing.eval.dataset import EvalQuestion
from filing.eval.metrics import Outcome, RetrievedChunk, Scorecard
from filing.eval.naive import NAIVE_CHUNKER, NAIVE_K, NaiveRetriever

# Bump when the prompt below changes. It is part of the fingerprint, so a
# reworded instruction produces a different experiment rather than a silently
# different number under the same name.
PROMPT_VERSION = "p1"

# The token the model is told to emit when the context does not answer the
# question. Scoring for abstention has to be deterministic -- an LLM judge
# deciding "did it refuse?" would put a model call in the metric path, which is
# exactly what M4 forbids -- so refusal is a string match on a token nothing
# else in an answer would contain.
REFUSAL = "INSUFFICIENT EVIDENCE"

SYSTEM_PROMPT = (
    "You answer questions about SEC filings using only the numbered excerpts provided.\n"
    "Rules:\n"
    f"1. If the excerpts do not contain the answer, reply with exactly: {REFUSAL}\n"
    "2. Never use knowledge from outside the excerpts, and never estimate a number.\n"
    "3. Cite the excerpts you used as bracketed numbers, e.g. [2] or [1][3].\n"
    "4. Be brief: two sentences at most, and give figures exactly as the filing states them."
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Everything that could move a number, in one hashable value."""

    name: str
    system: str = "naive"
    k: int = NAIVE_K
    chat_role: str = "chat"
    chat_backend: str = ""
    chat_model: str = ""
    embed_backend: str = ""
    embed_model: str = ""
    chunker: str = NAIVE_CHUNKER
    collection: str = ""
    dataset_version: str = dataset.DATASET_VERSION
    prompt_version: str = PROMPT_VERSION
    temperature: float = 0.0
    max_tokens: int = 512
    note: str = ""

    def resolved(self, cfg: Settings) -> EvalConfig:
        """Fill in the model ids and collection name from the environment.

        Done once, explicitly, and then written to the results file -- because
        "gemini" is not a reproducible description of an experiment and
        ``gemini-3.5-flash`` is.
        """
        from dataclasses import replace

        from filing.stores.index import collection_name

        chat_backend = self.chat_backend or cfg.llm_backend
        embed_backend = self.embed_backend or cfg.embed_backend
        chat = model_for(self.chat_role, chat_backend)  # type: ignore[arg-type]
        embed = model_for("embed", embed_backend)  # type: ignore[arg-type]
        return replace(
            self,
            chat_backend=chat_backend,
            chat_model=chat.id,
            embed_backend=embed_backend,
            embed_model=embed.id,
            collection=self.collection
            or collection_name(embed_backend, embed.id, embed.dim, self.chunker),
        )

    def fingerprint(self, dataset_sha: str) -> str:
        """sha256 over the config and the question file together.

        The dataset hash belongs inside the fingerprint rather than beside it:
        the same config over a different question set is a different experiment,
        and the commonest way an eval quietly stops being comparable is that
        someone fixed a typo in a question.
        """
        body = json.dumps(asdict(self) | {"dataset_sha256": dataset_sha}, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


CONFIGS: dict[str, EvalConfig] = {
    "baseline": EvalConfig(
        name="baseline",
        system="naive",
        note="fixed 2,048-char chunks over the whole filing, dense top-5, one LLM call",
    ),
}


def get_config(name: str) -> EvalConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        known = ", ".join(sorted(CONFIGS))
        raise KeyError(f"unknown eval config {name!r}; have: {known}") from None


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


def results_dir(cfg: Settings) -> Path:
    return Path(cfg.data_dir).parent / "results"


class OutcomeCache:
    """One JSON file per answered question, under the config's fingerprint.

    A directory rather than a database because the unit of reuse is a single
    question: a partial run is just a partly-full directory, and the failure
    mode of a corrupt entry is one wasted call rather than a lost run.
    """

    def __init__(self, root: Path, fingerprint: str) -> None:
        self.dir = root / "cache" / fingerprint[:16]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0

    def get(self, qid: str) -> Outcome | None:
        path = self.dir / f"{qid}.json"
        if not path.exists():
            return None
        try:
            out = Outcome.from_json(json.loads(path.read_text("utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None
        self.hits += 1
        return out

    def put(self, outcome: Outcome) -> None:
        (self.dir / f"{outcome.qid}.json").write_text(
            json.dumps(outcome.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# the baseline system
# --------------------------------------------------------------------------


def build_prompt(question: str, chunks: list[Any]) -> list[dict[str, str]]:
    """Numbered excerpts, then the question. The citation contract lives here.

    Excerpts are numbered ``[1]``..``[k]`` rather than cited by chunk id: a UUID
    is not something a model can reproduce reliably, so asking for one would
    measure transcription rather than grounding. The number is mapped back to
    the id after the call, which is deterministic and keeps citation resolution
    meaningful.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {c.ticker} {c.form} (period ending {c.period_end})\n{c.text}")
    body = "\n\n".join(blocks) if blocks else "(no excerpts were retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{body}\n\nQuestion: {question}"},
    ]


def parse_citations(answer: str, chunks: list[Any]) -> tuple[str, ...]:
    """Map ``[n]`` markers back to chunk ids, in order, without duplicates.

    Out-of-range markers are dropped rather than kept as unresolvable ids --
    the interesting citation failure is a marker pointing at an excerpt that
    does not support the claim, not one pointing at nothing.
    """
    import re

    out: list[str] = []
    for m in re.finditer(r"\[(\d{1,2})\]", answer):
        i = int(m.group(1))
        if 1 <= i <= len(chunks):
            cid = chunks[i - 1].chunk_id
            if cid not in out:
                out.append(cid)
    return tuple(out)


def answer_naive(
    question: EvalQuestion,
    *,
    retriever: NaiveRetriever,
    backend: Any,
    config: EvalConfig,
) -> Outcome:
    """Embed, search, one chat call. No router, no tools, no second pass."""
    started = time.monotonic()
    before = backend.usage()
    try:
        chunks = retriever.search(question.question, k=config.k)
        text = backend.chat(
            build_prompt(question.question, chunks),
            role=config.chat_role,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ).strip()
        error = ""
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        chunks, text, error = [], "", f"{type(exc).__name__}: {exc}"

    refused = REFUSAL.lower() in text.lower()
    after = backend.usage()
    return Outcome(
        qid=question.id,
        # The baseline has no router. It reads text, always -- unless it
        # declines to answer, which is the one routing decision it can make.
        route="refuse" if refused else "text",
        answer=text,
        refused=refused,
        retrieved=tuple(
            RetrievedChunk(
                chunk_id=c.chunk_id, accn=c.accn, char_start=c.char_start, char_end=c.char_end
            )
            for c in chunks
        ),
        citations=() if refused else parse_citations(text, chunks),
        llm_calls=after.http_calls - before.http_calls,
        seconds=time.monotonic() - started,
        error=error,
    )


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


@dataclass
class RunReport:
    config: EvalConfig
    fingerprint: str
    dataset_path: str
    dataset_sha256: str
    counts: dict[str, int]
    answered: int = 0
    from_cache: int = 0
    llm_calls: int = 0
    errors: int = 0
    seconds: float = 0.0
    card: Scorecard | None = None
    outcomes: list[Outcome] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "fingerprint": self.fingerprint,
            "dataset": {
                "path": self.dataset_path,
                "sha256": self.dataset_sha256,
                "counts": self.counts,
            },
            "run": {
                "answered": self.answered,
                "from_cache": self.from_cache,
                "llm_calls": self.llm_calls,
                "errors": self.errors,
                "seconds": round(self.seconds, 2),
            },
            "scores": self.card.to_json() if self.card else {},
            "outcomes": [o.to_json() for o in self.outcomes],
        }


def run(
    cfg: Settings,
    *,
    config: str | EvalConfig = "baseline",
    limit: int | None = None,
    slices: tuple[str, ...] = dataset.SLICES,
    use_cache: bool = True,
    write: bool = True,
    backend: Any = None,
    retriever: NaiveRetriever | None = None,
    on_question: Callable[[int, int, Outcome], None] | None = None,
) -> RunReport:
    """Answer the frozen set under one configuration, score it, write it down."""
    ec = (get_config(config) if isinstance(config, str) else config).resolved(cfg)
    if ec.system != "naive":  # pragma: no cover - M5 adds the other one
        raise ValueError(f"no runner for system {ec.system!r} yet")

    path = dataset.dataset_path(Path(cfg.data_dir), ec.dataset_version)
    questions = [q for q in dataset.read(path) if q.slice in slices]
    if limit:
        questions = questions[:limit]
    sha = dataset.fingerprint(path)
    fp = ec.fingerprint(sha)

    report = RunReport(
        config=ec,
        fingerprint=fp,
        dataset_path=str(path),
        dataset_sha256=sha,
        counts=dataset.counts(questions),
    )

    root = results_dir(cfg)
    cache = OutcomeCache(root, fp) if use_cache else None
    started = time.monotonic()

    ret = retriever
    be = backend
    outcomes: list[Outcome] = []
    for i, q in enumerate(questions, start=1):
        hit = cache.get(q.id) if cache else None
        if hit is not None:
            outcomes.append(hit)
            report.from_cache += 1
        else:
            # Built lazily so a fully cached re-run needs neither Qdrant nor a
            # network: "reproducible from cache" has to mean reproducible with
            # nothing running.
            if ret is None:
                ret = NaiveRetriever(cfg)
                ret.require()
            if be is None:
                from filing.llm.factory import build_backend

                be = build_backend(cfg)
            hit = answer_naive(q, retriever=ret, backend=be, config=ec)
            outcomes.append(hit)
            report.answered += 1
            # Counted only on a miss. A cached outcome remembers what it cost
            # when it was made, and adding that back would make a free re-run
            # report a full day's quota -- the one number a budgeted harness
            # must not lie about. The per-outcome cost is still in the file.
            report.llm_calls += hit.llm_calls
            if cache:
                cache.put(hit)
        report.errors += bool(hit.error)
        if on_question:
            on_question(i, len(questions), hit)

    report.seconds = time.monotonic() - started
    report.outcomes = outcomes
    known = {c.chunk_id for o in outcomes for c in o.retrieved}
    report.card = metrics.score(questions, outcomes, known_chunks=known)

    if write:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{ec.name}.json").write_text(
            json.dumps(report.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (root / f"{ec.name}.md").write_text(
            metrics.to_markdown(report.card, title=f"{ec.name} ({ec.chat_model}, k={ec.k})"),
            encoding="utf-8",
        )
    return report
