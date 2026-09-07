"""The ablation ladder: one table that says what each stage was worth.

M8 asks for a sweep "naive -> +hybrid -> +rerank -> +router -> +grader/repair
-> +verifier" and a quality/cost/latency table regenerable by one command. This
module builds that table out of the results files the earlier gates already
committed, so the sweep is an aggregation rather than a re-run: the numbers in
the table are the same numbers in ``results/*.json``, and the fingerprint of
every source file is carried into the output so a reader can check that.

**The ladder is two lanes, not one chain, and that is a finding rather than a
compromise.** The plan drew a single monotone chain. The system does not have
one, because the agent graph turns the router, the grader and the repair loop on
together -- there is no configuration in which retrieval is hybrid but the route
is still fixed *and* a model is writing the answer. What the committed evidence
does support is two clean chains that share a corpus and a question set:

* a **retrieval lane** of three rungs that runs with no model at all, isolating
  the two retrieval changes one node at a time, and
* an **answer lane** of three rungs at a fixed generator, isolating the agent
  and then the verifier.

Splitting them is what keeps each delta attributable to one node. Forcing them
into one column would have produced a chain in which two rungs moved four things
at once, which is a table that looks like an ablation and is not one.

**Cost and latency cannot come from the results files, and the reason is the
cache.** Every committed run is served from ``OutcomeCache`` -- that is what
makes it reproducible with nothing running -- and a replayed outcome records the
replay's timings, not the original's. So ``results/agent.json`` reports 290 LLM
calls and ``results/agent-guarded.json`` reports zero for the same graph, purely
because of which run happened to populate the cache. Timing therefore comes from
a separate, deliberately uncached pass over a small sample, written to its own
file with its own sample size attached, and the table says ``--`` rather than
guessing when that file is absent.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from filing.config import Settings
from filing.eval import dataset
from filing.eval.runner import CONFIGS, get_config, results_dir, run

TIMING_FILE = "ablation-timing.json"
TABLE_JSON = "ablation.json"
TABLE_MD = "ablation.md"

#: Questions per rung in the timing pass. Small on purpose: this measures the
#: shape of a call, not the score, and the scores come from the 150-question
#: runs that are already committed.
TIMING_SAMPLE = 8


@dataclass(frozen=True, slots=True)
class Rung:
    """One step of the ladder: a committed config plus what it turned on."""

    config: str
    lane: str
    adds: str
    detail: str


#: The order is the ladder. Within a lane each rung differs from the one above
#: it by the node named in ``adds`` and by nothing else, which is the only
#: property that makes a difference column mean anything.
LADDER: tuple[Rung, ...] = (
    Rung(
        config="baseline-retrieval",
        lane="retrieval",
        adds="naive",
        detail="fixed 2,048-char chunks, dense top-10, no fusion and no reranker",
    ),
    Rung(
        config="agent-retrieval-norerank",
        lane="retrieval",
        adds="+ hybrid",
        detail="semantic chunks, dense and BM25 fused by RRF, fused order kept",
    ),
    Rung(
        config="agent-retrieval",
        lane="retrieval",
        adds="+ rerank",
        detail="the same fusion, reordered by the local MiniLM cross-encoder",
    ),
    Rung(
        config="baseline",
        lane="answer",
        adds="naive + generate",
        detail="dense top-5 into one prompt, one LLM call, no router and no grader",
    ),
    Rung(
        config="agent",
        lane="answer",
        adds="+ router, grader, repair",
        detail="plan, route to sql|text|graph, rerank, grade, repair at most twice",
    ),
    Rung(
        config="agent-guarded",
        lane="answer",
        adds="+ verifier",
        detail="every figure checked against evidence and every citation resolved",
    ),
)

LANES: tuple[str, ...] = ("retrieval", "answer")


@dataclass(frozen=True, slots=True)
class Timing:
    """What one uncached sample cost, in calls and in seconds.

    ``warmup`` is the first question of the rung, held out of the percentiles.
    The first version did not hold it out, and the first version reported that
    the naive dense retriever has a p95 of 53 seconds against a median of 0.19
    -- which is true of the measurement and false of the retriever. The 53
    seconds is a sentence-transformers model loading off disk into a cold
    process. Every rung had one and the ones with a bigger model had a bigger
    one, so the column was ranking start-up cost while claiming to rank tail
    latency.

    It is kept rather than dropped, because how long a rung takes to become
    ready is a real number that a reader deploying this would want; it just is
    not a percentile of anything.
    """

    config: str
    n: int
    llm_calls: int
    seconds: tuple[float, ...]
    warmup: float = 0.0
    #: Every question the pass ran, warm-up included. Zero means "not
    #: recorded", and the property below falls back to the timed count.
    ran: int = 0

    @property
    def questions(self) -> int:
        if self.ran:
            return self.ran
        # A timing file written before ``questions_run`` existed still records
        # a warm-up, and a recorded warm-up is evidence that one more question
        # ran than was timed.
        return self.n + 1 if self.warmup else self.n

    @property
    def calls_per_question(self) -> float:
        """Calls over *every* question the pass ran, warm-up included.

        The warm-up leaves the latency percentiles and stays in the cost
        average, and the asymmetry is the point: a cold process is slow because
        a model is loading off disk, but it does not make an extra API call.
        The warm-up question costs exactly what the eight after it cost. So
        dividing a nine-question call total by the eight timed questions -- the
        first version of this fix did -- inflates every cost figure in the
        table by an eighth, which is how holding out a contaminated number
        contaminates a clean one.
        """
        return self.llm_calls / self.questions if self.questions else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.seconds) if self.seconds else 0.0

    @property
    def p95(self) -> float:
        if not self.seconds:
            return 0.0
        ordered = sorted(self.seconds)
        # Nearest-rank, and clamped: at n=8 the 95th percentile is the slowest
        # question, and pretending otherwise by interpolating would invent a
        # number the sample cannot support.
        idx = min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1)
        return ordered[max(idx, 0)]

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "n": self.n,
            "llm_calls": self.llm_calls,
            "questions_run": self.questions,
            "calls_per_question": round(self.calls_per_question, 2),
            "warmup_seconds": round(self.warmup, 2),
            "p50_seconds": round(self.p50, 2),
            "p95_seconds": round(self.p95, 2),
            "seconds": [round(s, 3) for s in self.seconds],
        }


@dataclass
class Row:
    """One line of the table: a rung, its scores, and what it moved."""

    rung: Rung
    fingerprint: str
    dataset_sha256: str
    n: int
    retrieval_n: int
    full_set: bool
    quality: dict[str, float | None]
    timing: Timing | None = None
    deltas: dict[str, float | None] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rung.config

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.rung.config,
            "lane": self.rung.lane,
            "adds": self.rung.adds,
            "detail": self.rung.detail,
            "fingerprint": self.fingerprint,
            "dataset_sha256": self.dataset_sha256,
            "n": self.n,
            "retrieval_n": self.retrieval_n,
            "full_set": self.full_set,
            "quality": self.quality,
            "deltas": self.deltas,
            "timing": self.timing.to_json() if self.timing else None,
        }


class MissingResults(FileNotFoundError):
    """A rung has no committed results file, so the ladder has a hole in it."""


def _quality(scores: dict[str, Any], *, generates: bool) -> dict[str, float | None]:
    """The columns, pulled from a scorecard's overall row.

    A retrieval rung leaves the answer columns empty rather than reporting a
    zero: nothing generated an answer, and a zero would read as a bad one.
    """
    hit = {str(k): v for k, v in (scores.get("hit_rate") or {}).items()}
    ndcg = {str(k): v for k, v in (scores.get("ndcg") or {}).items()}
    out: dict[str, float | None] = {
        "hit@5": hit.get("5"),
        "hit@10": hit.get("10"),
        "ndcg@10": ndcg.get("10"),
    }
    if not generates:
        out |= {"exact_match": None, "citations_supported": None, "hallucinated": None}
        return out
    out |= {
        "exact_match": scores.get("exact_match"),
        "citations_supported": scores.get("citations_supported"),
        "hallucinated": scores.get("hallucinated"),
    }
    return out


def load_row(root: Path, rung: Rung) -> Row:
    path = root / f"{rung.config}.json"
    if not path.exists():
        raise MissingResults(
            f"{path} is missing -- run: python -m filing.eval run --config {rung.config}"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    scores = (blob.get("scores") or {}).get("overall") or {}
    generates = bool(blob.get("config", {}).get("generate", True))
    return Row(
        rung=rung,
        fingerprint=blob.get("fingerprint", ""),
        dataset_sha256=(blob.get("dataset") or {}).get("sha256", ""),
        n=int(scores.get("n") or 0),
        retrieval_n=int(scores.get("retrieval_n") or 0),
        full_set=bool((blob.get("run") or {}).get("full_set", False)),
        quality=_quality(scores, generates=generates),
    )


def load_timings(root: Path) -> dict[str, Timing]:
    path = root / TIMING_FILE
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Timing] = {}
    for entry in blob.get("timings", []):
        out[entry["config"]] = Timing(
            config=entry["config"],
            n=int(entry["n"]),
            llm_calls=int(entry["llm_calls"]),
            seconds=tuple(float(s) for s in entry.get("seconds", [])),
            warmup=float(entry.get("warmup_seconds", 0.0)),
            ran=int(entry.get("questions_run") or 0),
        )
    return out


def _fill_deltas(rows: list[Row]) -> None:
    """Each rung against the one above it *in its own lane*.

    Across lanes the difference is meaningless -- the first answer rung is a
    different experiment from the last retrieval rung, not one node further on
    -- so the first row of each lane has no deltas at all.
    """
    previous: dict[str, Row] = {}
    for row in rows:
        prior = previous.get(row.rung.lane)
        if prior is not None:
            for key, value in row.quality.items():
                before = prior.quality.get(key)
                row.deltas[key] = None if value is None or before is None else value - before
        previous[row.rung.lane] = row


def build(root: Path, *, ladder: tuple[Rung, ...] = LADDER) -> list[Row]:
    """Every rung, scored, with per-lane deltas and timings attached."""
    timings = load_timings(root)
    rows = [load_row(root, rung) for rung in ladder]
    for row in rows:
        row.timing = timings.get(row.name)
    _fill_deltas(rows)
    return rows


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

DASH = "--"


def _pct(value: float | None) -> str:
    return DASH if value is None else f"{value * 100:.1f}%"


def _signed_pct(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) < 5e-5:
        return " (=)"
    return f" ({value * 100:+.1f})"


def _cell(row: Row, key: str) -> str:
    return _pct(row.quality.get(key)) + _signed_pct(row.deltas.get(key))


def _timing_cells(row: Row) -> tuple[str, str, str, str]:
    if row.timing is None:
        return DASH, DASH, DASH, DASH
    return (
        f"{row.timing.calls_per_question:.2f}",
        f"{row.timing.p50:.1f}",
        f"{row.timing.p95:.1f}",
        f"{row.timing.warmup:.1f}",
    )


LANE_TITLES = {
    "retrieval": (
        "### Retrieval lane -- no model, no key, no quota",
        "Every rung here is the same 150 questions with the generator switched off, so "
        "the columns are the ceiling on anything downstream: a writer cannot cite "
        "evidence the search never returned. A dash in the answer columns means nothing "
        "generated an answer, not that it generated a bad one.",
    ),
    "answer": (
        "### Answer lane -- one generator, held fixed",
        "The same 150 questions end to end at `gemini-3.5-flash-lite`, held constant "
        "across all three rungs so the deltas are about the graph rather than about the "
        "model. Retrieval columns are scored over the questions that actually reached a "
        "retriever, which is why `n` differs between the naive rung and the agent's.",
    ),
}

_HEADER = (
    "| rung | config | n | hit@5 | hit@10 | nDCG@10 | exact | cite->gold "
    "| halluc | calls/q | p50 s | p95 s | ready s |"
)
_RULE = "|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|"


def to_markdown(rows: list[Row], *, timings_present: bool) -> str:
    lines: list[str] = ["## Ablation ladder", ""]
    lines += [
        "What each stage was worth, measured on the frozen 150-question set. "
        "Deltas in brackets are against the rung above, within the same lane.",
        "",
    ]
    for lane in LANES:
        lane_rows = [r for r in rows if r.rung.lane == lane]
        if not lane_rows:
            continue
        title, blurb = LANE_TITLES[lane]
        lines += [title, "", blurb, "", _HEADER, _RULE]
        for row in lane_rows:
            calls, p50, p95, ready = _timing_cells(row)
            n = row.retrieval_n if lane == "retrieval" else row.n
            lines.append(
                f"| {row.rung.adds} | `{row.name}` | {n} | "
                f"{_cell(row, 'hit@5')} | {_cell(row, 'hit@10')} | {_cell(row, 'ndcg@10')} | "
                f"{_cell(row, 'exact_match')} | {_cell(row, 'citations_supported')} | "
                f"{_pct(row.quality.get('hallucinated'))} | "
                f"{calls} | {p50} | {p95} | {ready} |"
            )
        lines.append("")
        for row in lane_rows:
            lines.append(f"- **{row.rung.adds}** (`{row.name}`) -- {row.rung.detail}")
        lines.append("")
    if timings_present:
        lines += [
            "`calls/q`, `p50` and `p95` come from a separate pass with both caches "
            "off, over a small sample per rung -- the committed runs are served from "
            "the outcome cache and can only report what a replay cost. `ready s` is "
            "that rung's first question, held out of the percentiles: it is a model "
            "loading off disk, not a slow query, and leaving it in had the naive "
            "retriever reporting a 53-second p95 against a 0.2-second median.",
            "",
        ]
    else:
        lines += [
            "> `calls/q`, `p50` and `p95` are unmeasured. The committed results files "
            "cannot supply them: every one of those runs is served from the outcome "
            "cache, so their recorded call counts and durations belong to the replay "
            "rather than to the work. `python -m filing.eval ablation --time` measures "
            f"them on an uncached sample and writes `results/{TIMING_FILE}`.",
            "",
        ]
    return "\n".join(lines)


def to_json(rows: list[Row]) -> dict[str, Any]:
    shas = {r.dataset_sha256 for r in rows if r.dataset_sha256}
    return {
        "ladder": [asdict(rung) for rung in LADDER],
        # One hash for every rung, or the table is comparing question sets.
        "dataset_sha256": sorted(shas)[0] if len(shas) == 1 else sorted(shas),
        "comparable": len(shas) == 1,
        "rows": [r.to_json() for r in rows],
    }


def write(cfg: Settings, *, root: Path | None = None) -> tuple[Path, Path, list[Row]]:
    """Regenerate both artefacts. This is what the one command does."""
    root = root or results_dir(cfg)
    rows = build(root)
    timings_present = any(r.timing for r in rows)
    json_path = root / TABLE_JSON
    md_path = root / TABLE_MD
    json_path.write_text(json.dumps(to_json(rows), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(rows, timings_present=timings_present) + "\n", encoding="utf-8")
    return json_path, md_path, rows


# --------------------------------------------------------------------------
# the timing pass
# --------------------------------------------------------------------------


def measure(
    cfg: Settings,
    *,
    ladder: tuple[Rung, ...] = LADDER,
    sample: int = TIMING_SAMPLE,
    root: Path | None = None,
    on_rung: Any = None,
) -> Path:
    """Run each rung uncached over ``sample`` questions and record what it cost.

    Uncached is the whole point, and it is also why this is a sample rather than
    the frozen set: the numbers the table needs are what a question costs when
    the work actually happens, and paying for 150 of those per rung to learn the
    shape of one call would be spending a day's quota on a latency column.

    The sample is written into the output beside the numbers, because a p95 over
    eight questions is a different claim from a p95 over a hundred and fifty and
    a table that hid the difference would be the dishonest kind.

    *Both* caches come off, not just the outcome cache. Turning off only the
    outcome cache leaves the response cache underneath it answering every prompt
    the earlier runs already sent -- which is most of them -- so the pass would
    report zero LLM calls and millisecond latencies for a graph that in fact
    makes two hosted round trips per question. That is the same replay artefact
    this file exists to route around, one layer down.
    """
    root = root or results_dir(cfg)
    cfg = cfg.model_copy(update={"cache_enabled": False})
    out: list[Timing] = []
    for rung in ladder:
        ec = get_config(rung.config)
        seconds: list[float] = []

        def tick(_i: int, _total: int, outcome: Any, sink: list[float] = seconds) -> None:
            sink.append(float(getattr(outcome, "seconds", 0.0)))

        started = time.monotonic()
        # One question more than asked for: the first is the warm-up and does
        # not enter the percentiles, so ``--sample 8`` still means eight timed
        # questions rather than seven and a model load.
        report = run(
            cfg,
            config=ec,
            limit=sample + 1,
            use_cache=False,
            write=False,
            on_question=tick,
        )
        wall = time.monotonic() - started
        warmup, timed = (seconds[0], seconds[1:]) if seconds else (0.0, [])
        timing = Timing(
            config=rung.config,
            n=len(timed) or max(report.answered - 1, 0),
            llm_calls=report.llm_calls,
            seconds=tuple(timed),
            warmup=warmup,
            ran=len(seconds) or report.answered,
        )
        out.append(timing)
        if on_rung is not None:
            on_rung(rung, timing, wall)
    path = root / TIMING_FILE
    path.write_text(
        json.dumps(
            {
                "sample": sample,
                "dataset_version": dataset.DATASET_VERSION,
                "cached": False,
                "note": (
                    "Measured with the outcome cache disabled. Quality never comes from "
                    "this file and cost never comes from the results files."
                ),
                "timings": [t.to_json() for t in out],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def known_configs() -> tuple[str, ...]:
    """Every rung's config name, checked against the registry at import time."""
    return tuple(r.config for r in LADDER)


_unknown = [c for c in known_configs() if c not in CONFIGS]
if _unknown:  # pragma: no cover - a typo in LADDER should not be a runtime surprise
    raise RuntimeError(f"ablation ladder names configs that do not exist: {_unknown}")
