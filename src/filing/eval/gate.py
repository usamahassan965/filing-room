"""The regression gate: what CI runs on every pull request.

M8 asks for "deterministic metrics on a 40-question subset, per PR". The subset
was there to bound cost, and this gate has no cost to bound, so it checks all
150 instead -- because it never answers a question. It reads the committed
results files, rebuilds each run's outcomes from the JSON, and re-scores them
with the scoring code as it stands in the working tree. No key, no Qdrant, no
DuckDB, no network, a few seconds.

That inversion is the point. A gate that re-answers questions on every PR is
measuring the weather -- a provider's mood, a rate limit, a model that changed
under a stable name -- and it can only run where the secrets are. A gate that
re-scores frozen answers measures exactly one thing, the code in the diff, and
it runs on a fork. What it cannot catch is a change to the *answering* path,
which is what the eval command itself is for.

Three checks, and each catches a different way the record goes wrong:

``rescore``
    The scorecard in the file must be the scorecard today's metrics code
    computes from the same outcomes. This is the one that fails when somebody
    changes a metric -- fixing an off-by-one in nDCG, widening the numeric
    tolerance -- and every committed table silently starts describing a
    different measurement.

``fingerprint``
    Every results file must still hash to the fingerprint it carries. This
    fails when a default moves in :class:`~filing.eval.runner.EvalConfig`, which
    is the quiet way two runs stop being comparable.

``floors``
    Each config's headline numbers must not fall below a written-down floor.
    This is the one that fails when a re-run is worse than the run it replaced
    -- the ordinary regression, the one where nothing is broken and the system
    is simply doing less well than it did.

The floors are deliberately a little under the numbers they guard, so ordinary
noise does not page anybody and a real drop does. They are values in this file
rather than in a data file because a change to a floor is a change to the claim
the project makes, and it should show up in a diff next to the reason for it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filing.eval import dataset, metrics
from filing.eval.runner import CONFIGS, get_config

#: How close a recomputed metric has to be to the committed one. Floating point
#: means "identical" is not a thing you can ask for across a JSON round trip.
TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Floor:
    """One published claim, and which direction breaks it."""

    metric: str
    op: str  # ">=" or "<="
    value: float
    why: str

    def holds(self, actual: float | None) -> bool:
        if actual is None:
            return False
        return actual >= self.value if self.op == ">=" else actual <= self.value

    def describe(self, actual: float | None) -> str:
        got = "missing" if actual is None else f"{actual:.4f}"
        return f"{self.metric} {got} {self.op} {self.value:.4f}"


#: Read these as the promises the README makes. Anything not listed is measured
#: and reported but not defended -- retrieval quality on the agent, for one,
#: because M7 found two retrieval defects that a floor would only freeze in.
FLOORS: dict[str, tuple[Floor, ...]] = {
    "baseline-retrieval": (
        Floor("hit_rate.10", ">=", 0.14, "the naive dense floor the whole ladder is read against"),
    ),
    "agent-retrieval-norerank": (
        Floor("hit_rate.10", ">=", 0.29, "hybrid fusion has to keep beating naive dense"),
        Floor("ndcg.10", ">=", 0.16, "and beat it on ordering, not only on presence"),
    ),
    "agent-retrieval": (
        Floor("hit_rate.10", ">=", 0.23, "the reranked order, which is currently the worse one"),
    ),
    "baseline": (
        Floor("abstention", ">=", 1.0, "the baseline refuses every unanswerable question"),
        Floor("over_answered", "<=", 0.0, "and answers none of them"),
        Floor("citations_resolvable", ">=", 1.0, "every bracket it prints points at a real chunk"),
    ),
    "agent": (
        Floor("exact_match", ">=", 0.95, "the numeric slice is the headline claim"),
        Floor("hit_rate.10", ">=", 0.38, "narrative retrieval on the answering path"),
        Floor("abstention", ">=", 1.0, "M5's whole point: it declines when it should"),
        Floor("over_answered", "<=", 0.0, "no answer to a question with no answer in the corpus"),
        Floor("citations_resolvable", ">=", 1.0, "no bracket pointing at nothing"),
        Floor("router_accuracy", ">=", 0.80, "the router picks the intended store"),
    ),
    "agent-guarded": (
        Floor("exact_match", ">=", 0.95, "the verifier must not cost accuracy"),
        Floor("hallucinated", "<=", 0.0, "no figure printed that the evidence does not contain"),
        Floor("locator_rate", ">=", 1.0, "every citation resolves to a filing locator"),
        Floor("abstention", ">=", 1.0, "the guard does not soften the refusal contract"),
    ),
}


@dataclass
class Check:
    """One assertion, and what it found."""

    config: str
    kind: str
    label: str
    ok: bool
    detail: str = ""


@dataclass
class GateReport:
    checks: list[Check] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def ok(self) -> bool:
        return not self.failures and not self.missing

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": len(self.checks),
            "failed": len(self.failures),
            "missing": self.missing,
            "checks": [
                {
                    "config": c.config,
                    "kind": c.kind,
                    "label": c.label,
                    "ok": c.ok,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def _dig(scores: dict[str, Any], path: str) -> float | None:
    """``hit_rate.10`` out of a scorecard's nested dicts."""
    node: Any = scores
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part, node.get(str(part)))
        if node is None:
            return None
    return float(node) if isinstance(node, int | float) else None


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a scorecard, keyed by dotted path, for a value-by-value diff."""
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out |= _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    else:
        out[prefix] = node
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE)
    return a == b


def check_rescore(blob: dict[str, Any], questions: list[Any]) -> Check:
    """Re-score the committed outcomes with today's metrics code."""
    name = blob["config"]["name"]
    outcomes = [metrics.Outcome.from_json(o) for o in blob.get("outcomes", [])]
    if not outcomes:
        return Check(name, "rescore", "outcomes re-scored", False, "the file carries no outcomes")
    asked = {o.qid for o in outcomes}
    subset = [q for q in questions if q.id in asked]
    known = {c.chunk_id for o in outcomes for c in o.retrieved}
    card = metrics.score(
        subset, outcomes, known_chunks=known, generated=bool(blob["config"].get("generate", True))
    )
    now = _flatten(card.to_json())
    then = _flatten(blob.get("scores") or {})
    drifted = [
        f"{key}: file {then[key]!r} -> now {now.get(key)!r}"
        for key in sorted(then)
        if not _same(then[key], now.get(key))
    ]
    return Check(
        name,
        "rescore",
        f"{len(outcomes)} outcomes re-scored",
        not drifted,
        "; ".join(drifted[:6]),
    )


def check_fingerprint(blob: dict[str, Any]) -> Check:
    """The config in the file must still hash to the fingerprint in the file."""
    name = blob["config"]["name"]
    stored = blob.get("fingerprint", "")
    sha = (blob.get("dataset") or {}).get("sha256", "")
    from dataclasses import replace

    fields = {k: v for k, v in blob["config"].items() if k != "name"}
    try:
        ec = replace(get_config(name), **fields)
    except (KeyError, TypeError) as exc:
        return Check(name, "fingerprint", "config rebuilt from file", False, str(exc))
    now = ec.fingerprint(sha)
    return Check(
        name,
        "fingerprint",
        f"{stored[:12]}",
        now == stored,
        "" if now == stored else f"recomputes to {now[:12]}",
    )


def check_floors(blob: dict[str, Any]) -> list[Check]:
    name = blob["config"]["name"]
    scores = (blob.get("scores") or {}).get("overall") or {}
    out: list[Check] = []
    for floor in FLOORS.get(name, ()):
        actual = _dig(scores, floor.metric)
        ok = floor.holds(actual)
        out.append(Check(name, "floor", floor.describe(actual), ok, "" if ok else floor.why))
    return out


def run_gate(root: Path, data_dir: Path, *, configs: tuple[str, ...] | None = None) -> GateReport:
    """Every check over every config that has a committed results file."""
    names = configs or tuple(FLOORS)
    report = GateReport()
    questions = dataset.read(dataset.dataset_path(data_dir))
    for name in names:
        if name not in CONFIGS:
            report.missing.append(f"{name}: not a known config")
            continue
        path = root / f"{name}.json"
        if not path.exists():
            report.missing.append(f"{name}: {path} is not committed")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not (blob.get("run") or {}).get("full_set", False):
            report.missing.append(f"{name}: {path} is a partial run, not the frozen set")
            continue
        report.checks.append(check_fingerprint(blob))
        report.checks.append(check_rescore(blob, questions))
        report.checks.extend(check_floors(blob))
    return report


def to_markdown(report: GateReport) -> str:
    lines = ["| config | check | assertion | result |", "|---|---|---|---|"]
    for c in report.checks:
        mark = "pass" if c.ok else "**FAIL**"
        detail = f" -- {c.detail}" if c.detail and not c.ok else ""
        lines.append(f"| `{c.config}` | {c.kind} | {c.label}{detail} | {mark} |")
    for m in report.missing:
        lines.append(f"| -- | missing | {m} | **FAIL** |")
    return "\n".join(lines)
