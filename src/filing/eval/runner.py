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

# Which answering systems this harness knows how to run. A set rather than a
# chain of ifs so that an unknown name fails at the top of the run with the list
# of valid ones, instead of forty questions in.
SYSTEMS = ("naive", "agent")

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
    generate: bool = True
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

        When ``generate`` is off, every chat field is blanked instead. That is
        not tidiness: those fields are in the fingerprint, and a retrieval-only
        run that inherited whichever chat model happened to be configured would
        appear to be a different experiment each time the generator changed --
        while producing, correctly, the identical numbers. The retriever does
        not know what will read its output, so the retrieval fingerprint must
        not either.
        """
        from dataclasses import replace

        from filing.stores.index import collection_name

        embed_backend = self.embed_backend or cfg.embed_backend
        embed = model_for("embed", embed_backend)  # type: ignore[arg-type]
        chat: dict[str, Any] = {"chat_backend": "", "chat_model": "", "chat_role": ""}
        if self.generate:
            backend = self.chat_backend or cfg.llm_backend
            model = model_for(self.chat_role, backend)  # type: ignore[arg-type]
            chat = {
                "chat_backend": backend,
                "chat_model": model.id,
                "chat_role": self.chat_role,
            }
        else:
            chat |= {"prompt_version": "", "temperature": 0.0, "max_tokens": 0}
        return replace(
            self,
            **chat,
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
        # chat_fast, not chat, and the gate is the reason. M4 asks for a full
        # baseline run that "fits inside the rate-limit budget and is
        # reproducible from cache" -- and the free tier caps gemini-3.5-flash
        # at 20 requests per DAY, so 150 questions is an eight-day baseline and
        # a re-run is eight more. A number nobody can reproduce is not a
        # baseline, it is an anecdote. flash-lite answers the same 150 in about
        # fifteen minutes, and the results file records which model produced
        # them, so the weaker generator is stated rather than hidden. That the
        # baseline is beatable is the point of a baseline.
        chat_role="chat_fast",
        note="fixed 2,048-char chunks over the whole filing, dense top-5, one LLM call",
    ),
    # The same retriever with the generator removed. It exists because the
    # ceiling on every downstream number is set here: a generator cannot cite
    # evidence the search never returned, so hit@k is not a diagnostic for the
    # baseline's answers, it is the bound on them. It also costs nothing and
    # needs no key, which means it is the half of the baseline that can be run
    # on any machine, any day, without a quota.
    "baseline-retrieval": EvalConfig(
        name="baseline-retrieval",
        system="naive",
        generate=False,
        k=max(metrics.KS),
        note="the baseline retriever alone: dense top-10, no LLM call at all",
    ),
    # The generator ablation, three ways. Identical retrieval to `baseline` --
    # same naive index, same top-5, same prompt -- so the only thing that moves
    # between these tables and the baseline's is which model reads the excerpts.
    # That is the whole point: it separates the retriever's ceiling from the
    # generator's, which a single run cannot do.
    #
    # All three free tiers were probed live, and two of them still surprised
    # the run. Cohere meters CALLS (1,000 a month): 150 questions is 15% of the
    # month and finishes in nine minutes. Groq meters TOKENS PER DAY (200,000,
    # per model) -- roughly 47 questions -- so its full run is a three-day run
    # and `--limit` is the only honest way to score it in one sitting. OVHcloud
    # meters a shared anonymous pool that belongs to nobody, so it is available
    # exactly when it is available. None of that is visible from a docs page,
    # and only the Cohere number was visible before the run started.
    "baseline-cohere": EvalConfig(
        name="baseline-cohere",
        system="naive",
        chat_backend="cohere",
        chat_role="chat",
        note="the baseline retriever with command-a -- the generator ablation",
    ),
    "baseline-groq": EvalConfig(
        name="baseline-groq",
        system="naive",
        chat_backend="groq",
        chat_role="chat",
        note="the baseline retriever with gpt-oss-120b -- the generator ablation",
    ),
    # No key and no account. If every other provider on this list refuses the
    # country, this row still runs, which is the only reason a 2-rpm endpoint
    # is worth an hour.
    "baseline-ovh": EvalConfig(
        name="baseline-ovh",
        system="naive",
        chat_backend="ovh",
        chat_role="chat",
        note="the baseline retriever with Qwen3.5-397B, anonymously",
    ),
    # The generating half, run on this machine instead of on a quota. Two of
    # them, because the registry holds two local chat models that both fit in
    # memory and there is no way to know from the outside which one answers
    # filing questions better. They differ in exactly one field -- the role the
    # chat model is resolved from -- so the fingerprints differ, the results
    # land in separate files, and the comparison is the same 150 questions
    # rather than an opinion about model families.
    "baseline-local": EvalConfig(
        name="baseline-local",
        system="naive",
        chat_backend="ollama",
        chat_role="chat",
        note="the baseline, generated locally by the registry's chat model",
    ),
    "baseline-local-alt": EvalConfig(
        name="baseline-local-alt",
        system="naive",
        chat_backend="ollama",
        chat_role="chat_fast",
        note="the same baseline generated by the registry's chat_fast model",
    ),
    # M5. The same 150 questions, the same generator as `baseline`, and the
    # same refusal token and citation contract -- imported rather than
    # restated, so both systems are scored by one rule. What changes is
    # everything between the question and the prompt: a router, three stores, a
    # deterministic grader and a repair loop bounded at two.
    #
    # chat_role is `chat_fast` for the reason the baseline gives: the free tier
    # caps gemini-3.5-flash at 20 requests a day, and the agent spends up to two
    # per question. flash-lite makes the agent's 150 a run rather than a
    # fortnight, and holding the generator fixed across the two configs is what
    # makes the comparison about the agent instead of about the model.
    "agent": EvalConfig(
        name="agent",
        system="agent",
        k=5,
        chat_role="chat_fast",
        chunker="semantic",
        note="plan/route -> sql|text|graph -> rerank -> grade -> repair(<=2) -> synthesise",
    ),
    # The agent's retriever without a generator: no plan call, so the router is
    # the heuristic floor and every question goes to text. It measures the one
    # thing the agent shares with M3 -- hybrid retrieval plus the cross-encoder
    # -- against the naive dense top-10 the baseline used, with no quota and no
    # key, on any machine.
    "agent-retrieval": EvalConfig(
        name="agent-retrieval",
        system="agent",
        generate=False,
        k=max(metrics.KS),
        chunker="semantic",
        note="the agent's hybrid retriever and reranker alone, no LLM call at all",
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

    # gpt-oss-120b cites with FULLWIDTH brackets -- U+3010/U+3011, the CJK lenticular
    # pair -- so the ASCII-only pattern scored it at zero citations while it was in
    # fact citing correctly on every answer. That is a broken metric, not a finding
    # about the model, and the contract in SYSTEM_PROMPT says "bracketed numbers"
    # without promising a codepoint. Normalise before matching.
    answer = answer.translate(
        str.maketrans({"\u3010": "[", "\u3011": "]", "\uff3b": "[", "\uff3d": "]"})
    )
    out: list[str] = []
    for m in re.finditer(r"\[(\d{1,2})\]", answer):
        i = int(m.group(1))
        if 1 <= i <= len(chunks):
            cid = chunks[i - 1].chunk_id
            # Empty means the cited evidence is not a chunk -- an XBRL fact,
            # which the agent cites and which has no span for the scorer to
            # check. Dropping it reports the numeric slice as unmeasured rather
            # than counting a correct citation as an unresolvable one.
            if cid and cid not in out:
                out.append(cid)
    return tuple(out)


def answer_naive(
    question: EvalQuestion,
    *,
    retriever: NaiveRetriever,
    backend: Any,
    config: EvalConfig,
) -> Outcome:
    """Embed, search, one chat call. No router, no tools, no second pass.

    With ``config.generate`` off the chat call is not made, and the outcome
    carries the retrieved chunks and nothing else -- no answer, no route, no
    citations. :func:`filing.eval.metrics.score` reads that emptiness and
    reports the generation columns as unmeasured rather than as zero.
    """
    started = time.monotonic()
    # Not merely "if a backend was passed": with generation off the backend is
    # not consulted at all, not even for its call counter, so a retrieval-only
    # run holds no opinion about which generator is configured and cannot be
    # made to build one.
    before = backend.usage() if config.generate else None
    scored: list[tuple[Any, float]] = []
    text = ""
    error = ""
    try:
        scored = retriever.search_scored(question.question, k=config.k)
        if config.generate:
            text = backend.chat(
                build_prompt(question.question, [c for c, _ in scored]),
                role=config.chat_role,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            ).strip()
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        scored, text, error = [], "", f"{type(exc).__name__}: {exc}"

    chunks = [c for c, _ in scored]
    refused = bool(text) and REFUSAL.lower() in text.lower()
    calls = backend.usage().http_calls - before.http_calls if before is not None else 0
    return Outcome(
        qid=question.id,
        # The baseline has no router. It reads text, always -- unless it
        # declines to answer, which is the one routing decision it can make.
        # A retrieval-only run makes no such decision and says so with "".
        route=("refuse" if refused else "text") if config.generate else "",
        answer=text,
        refused=refused,
        retrieved=tuple(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                accn=c.accn,
                char_start=c.char_start,
                char_end=c.char_end,
                score=round(score, 6),
            )
            for c, score in scored
        ),
        citations=() if refused else parse_citations(text, chunks),
        llm_calls=calls,
        seconds=time.monotonic() - started,
        error=error,
    )


def build_agent_tools(cfg: Settings, *, backend: Any, config: EvalConfig) -> Any:
    """Open every store the agent can route to, and skip the ones that are absent.

    A missing store is a downgrade, not a crash: the route node sends `sql` to
    `text` when there is no facts file and `text` to `refuse` when there is no
    index. That is what lets the agent be run on a machine that has built part
    of the corpus, and it is also why the router's downgrades are logged --
    "the router chose text" and "the router wanted SQL and could not have it"
    are different facts and only one of them is about the router.
    """
    from filing.agent.entities import GraphTool
    from filing.agent.nodes import Tools
    from filing.agent.sql import SqlTool
    from filing.stores.facts import FactsStore
    from filing.stores.graph import GraphStore
    from filing.stores.retrieve import Retriever

    retriever = Retriever(cfg)
    retriever.require()

    sql = None
    if cfg.facts_path.exists():
        sql = SqlTool(FactsStore(cfg.facts_path))

    graph_store = GraphStore(cfg.graph_dir)
    graph = GraphTool(graph_store) if graph_store.exists else None

    return Tools(
        backend=backend,
        sql=sql,
        retriever=retriever,
        graph=graph,
        top_n=config.k,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        chat_role=config.chat_role,
        generate=config.generate,
    )


def answer_agent(
    question: EvalQuestion,
    *,
    tools: Any,
    config: EvalConfig,
    graph: Any = None,
) -> Outcome:
    """One question through the agent graph, flattened into M4's outcome shape.

    Scoring the agent with the baseline's metrics is the entire point of having
    a baseline, so nothing here reshapes the metric to suit the system. That
    costs the agent something, and it is worth stating rather than burying:

    ``retrieved`` and ``citations`` are **text-chunk** concepts. M4 defined a
    citation as supported when the cited chunk's character span overlaps a gold
    span, and a fact from DuckDB has no character span -- the XBRL value and the
    printed number in the filing are the same fact reached two different ways,
    and only one of them carries offsets. So a fact citation is recorded in the
    answer and deliberately **not** counted: the numeric slice's retrieval and
    citation columns come back unmeasured for the agent rather than zero, which
    is the same convention a retrieval-only run already uses. The numeric claim
    therefore rests on ``exact_match``, and the accession-level check on fact
    citations is computed in the M5 report where it can be named for what it is.

    Inventing a span for a fact so the existing metric would score it is the
    move this docstring exists to refuse.
    """
    from filing.agent.graph import run_question

    started = time.monotonic()
    state = run_question(question.question, tools=tools, qid=question.id, graph=graph)

    evidence = list(state.get("evidence") or [])
    text_evidence = [e for e in evidence if e.chunk_id]
    answer = str(state.get("answer") or "")
    refused = bool(state.get("refused")) or (bool(answer) and REFUSAL.lower() in answer.lower())

    route = str(state.get("route") or "")
    if refused:
        # A route of `sql` that ended in an abstention was, in the end, a
        # refusal -- and the baseline is scored the same way.
        route = "refuse"
    if not config.generate:
        route = ""

    return Outcome(
        qid=question.id,
        route=route,
        answer=answer,
        refused=refused,
        retrieved=tuple(
            RetrievedChunk(
                chunk_id=e.chunk_id,
                accn=e.accn,
                char_start=e.char_start,
                char_end=e.char_end,
                score=round(float(e.score), 6),
            )
            for e in text_evidence
        ),
        # Numbered against the full evidence list, because that is the list the
        # model was shown; fact evidence simply resolves to no chunk id.
        citations=() if refused else parse_citations(answer, evidence),
        llm_calls=int(state.get("llm_calls") or 0),
        seconds=time.monotonic() - started,
        error=str(state.get("error") or ""),
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
    # Whether this run covered the whole frozen set. In the JSON as well as in
    # the filename, so a file that gets copied, renamed or pasted into a table
    # still carries the scope its numbers are only meaningful under.
    full: bool = True
    written: Path | None = None

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
                # First field on purpose: a reader who checks one thing should
                # hit the scope before the scores.
                "full_set": self.full,
                "answered": self.answered,
                "from_cache": self.from_cache,
                "llm_calls": self.llm_calls,
                "errors": self.errors,
                "seconds": round(self.seconds, 2),
            },
            "scores": self.card.to_json() if self.card else {},
            "outcomes": [o.to_json() for o in self.outcomes],
        }


def title_for(ec: EvalConfig, *, partial: bool = False) -> str:
    """How a scorecard names itself. The generator, or the fact there isn't one."""
    scope = ", PARTIAL RUN" if partial else ""
    return f"{ec.name} ({ec.chat_model or 'retrieval only, no generation'}, k={ec.k}{scope})"


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
    agent_tools: Any = None,
    on_question: Callable[[int, int, Outcome], None] | None = None,
) -> RunReport:
    """Answer the frozen set under one configuration, score it, write it down."""
    ec = (get_config(config) if isinstance(config, str) else config).resolved(cfg)
    if ec.system not in SYSTEMS:
        known = ", ".join(sorted(SYSTEMS))
        raise ValueError(f"no runner for system {ec.system!r}; have: {known}")

    path = dataset.dataset_path(Path(cfg.data_dir), ec.dataset_version)
    questions = [q for q in dataset.read(path) if q.slice in slices]
    if limit:
        questions = questions[:limit]
    # Full means "every question the frozen set holds", not "every question
    # this call asked for" -- so it is decided against the dataset, not against
    # whether the caller happened to pass a flag.
    is_full = len(questions) == len(dataset.read(path))
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
    tools = agent_tools
    app: Any = None
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
            if be is None and ec.generate:
                from filing.llm.factory import build_backend

                # ec.chat_backend, not cfg.llm_backend. The config already
                # names the backend it is fingerprinted under, and reading it
                # from the environment instead let a whole 150-question run
                # bill itself to Gemini while the results file recorded
                # llama3.2:3b -- a file that lies about which model produced
                # its numbers is worse than no file. Empty falls back to the
                # environment, which is what the plain `baseline` config wants.
                be = build_backend(cfg, ec.chat_backend or None)
            if ec.system == "agent":
                from filing.agent.graph import build_graph

                if tools is None:
                    tools = build_agent_tools(cfg, backend=be, config=ec)
                if app is None:
                    # Compiled once for the run, not once per question, and on
                    # its own condition rather than folded into the one above --
                    # a caller that injects its own tools still gets one graph,
                    # which is the case the shipped path does not exercise and
                    # therefore the case that silently rebuilt 150 times.
                    app = build_graph(tools)
                hit = answer_agent(q, tools=tools, config=ec, graph=app)
            else:
                if ret is None:
                    ret = NaiveRetriever(cfg)
                    ret.require()
                hit = answer_naive(q, retriever=ret, backend=be, config=ec)
            outcomes.append(hit)
            report.answered += 1
            # Counted only on a miss. A cached outcome remembers what it cost
            # when it was made, and adding that back would make a free re-run
            # report a full day's quota -- the one number a budgeted harness
            # must not lie about. The per-outcome cost is still in the file.
            report.llm_calls += hit.llm_calls
            # Errors are never cached. A rate limit, a dropped connection and a
            # timeout are all facts about the afternoon, not about the system,
            # and caching one turns a resumable run into a run that replays its
            # own failures for as long as the fingerprint lives. The cost of
            # being wrong here is asymmetric: re-answering a question that
            # would have succeeded wastes one call, while remembering a 429
            # forever silently caps the score.
            if cache and not hit.error:
                cache.put(hit)
        report.errors += bool(hit.error)
        if on_question:
            on_question(i, len(questions), hit)

    report.full = is_full
    report.seconds = time.monotonic() - started
    report.outcomes = outcomes
    known = {c.chunk_id for o in outcomes for c in o.retrieved}
    report.card = metrics.score(questions, outcomes, known_chunks=known, generated=ec.generate)

    if write:
        root.mkdir(parents=True, exist_ok=True)
        # A subset run never takes the canonical filename. `--limit 3` and the
        # full 150 are different experiments, and the one thing that must not
        # happen is the small one landing at results/<config>.json, where every
        # later reader -- a commit, a README table, me next week -- takes it for
        # the run the gate asks for. It has happened twice: a 3-question
        # baseline.json survived a rate-limited afternoon, and a --limit 3
        # smoke test overwrote baseline-local-alt.json. Neither announced
        # itself, because a results file carries its numbers, not its scope.
        stem = ec.name if is_full else f"{ec.name}.partial"
        (root / f"{stem}.json").write_text(
            json.dumps(report.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (root / f"{stem}.md").write_text(
            metrics.to_markdown(report.card, title=title_for(ec, partial=not is_full)),
            encoding="utf-8",
        )
        report.written = root / f"{stem}.json"
    return report
