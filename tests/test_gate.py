"""The regression gate, and the regressions it is supposed to catch.

A gate is only worth having if it fails. Most of this file injects a specific
regression into a results file and asserts the gate goes red -- one test per
way the committed record can rot -- and one test runs the real gate over the
real ``results/`` directory, which is the assertion the CI workflow makes on
every pull request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from filing.config import settings
from filing.eval import gate
from filing.eval.runner import CONFIGS

DATA = Path(settings().data_dir)


def _load(results_root: Path, name: str) -> dict:
    return json.loads((results_root / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def sandbox(tmp_path, results_root):
    """A writable copy of one committed run, for injecting damage into."""

    def copy(name: str) -> Path:
        (tmp_path / f"{name}.json").write_text(
            json.dumps(_load(results_root, name)), encoding="utf-8"
        )
        return tmp_path

    return copy


# --------------------------------------------------------------------------
# the gate on the real committed record -- this is what CI asserts
# --------------------------------------------------------------------------


def test_the_gate_passes_on_the_committed_results(results_root):
    report = gate.run_gate(results_root, DATA)
    assert report.ok, "\n".join(
        f"{c.config} {c.kind}: {c.label} {c.detail}" for c in report.failures
    ) + "\n".join(report.missing)


def test_the_gate_checks_every_config_it_holds_floors_for(results_root):
    report = gate.run_gate(results_root, DATA)
    checked = {c.config for c in report.checks}
    assert checked == set(gate.FLOORS)
    # Three kinds, and all three actually ran. A gate that quietly stopped
    # re-scoring would still be green.
    assert {c.kind for c in report.checks} == {"fingerprint", "rescore", "floor"}


def test_every_floor_names_a_config_that_exists():
    assert set(gate.FLOORS) <= set(CONFIGS)


def test_every_floor_carries_a_reason(results_root):
    for name, floors in gate.FLOORS.items():
        for floor in floors:
            assert floor.why, f"{name}.{floor.metric} has no stated reason"
            assert floor.op in (">=", "<=")


def test_the_gate_never_opens_a_socket(results_root):
    # The autouse fixture in conftest turns an outbound connection into a failed
    # test, so this passing at all is the assertion: the gate is offline, which
    # is what lets it run on a fork with no secrets.
    assert gate.run_gate(results_root, DATA).ok


# --------------------------------------------------------------------------
# injected regressions
# --------------------------------------------------------------------------


def test_a_dropped_headline_number_fails_the_floor(sandbox):
    root = sandbox("agent")
    blob = _load(root, "agent")
    blob["scores"]["overall"]["exact_match"] = 0.80
    (root / "agent.json").write_text(json.dumps(blob), encoding="utf-8")
    report = gate.run_gate(root, DATA, configs=("agent",))
    assert not report.ok
    failed = [c for c in report.failures if c.kind == "floor"]
    assert any("exact_match" in c.label for c in failed)
    assert any("numeric slice" in c.detail for c in failed)


def test_a_scorecard_that_no_longer_matches_its_own_outcomes_fails_the_rescore(sandbox):
    """The regression that matters most: a metric changed under a frozen run.

    Nothing about the answers moved. Somebody edited a number in the file -- or,
    in the case this exists for, edited the code that computes it -- and every
    committed table silently began describing a different measurement.
    """
    root = sandbox("agent-guarded")
    blob = _load(root, "agent-guarded")
    blob["scores"]["overall"]["hit_rate"]["10"] = 0.99
    (root / "agent-guarded.json").write_text(json.dumps(blob), encoding="utf-8")
    report = gate.run_gate(root, DATA, configs=("agent-guarded",))
    failed = [c for c in report.failures if c.kind == "rescore"]
    assert failed, [c.label for c in report.checks]
    assert "0.99" in failed[0].detail


def test_a_config_edited_after_the_run_fails_the_fingerprint(sandbox):
    root = sandbox("agent")
    blob = _load(root, "agent")
    blob["config"]["k"] = 7
    (root / "agent.json").write_text(json.dumps(blob), encoding="utf-8")
    report = gate.run_gate(root, DATA, configs=("agent",))
    failed = [c for c in report.failures if c.kind == "fingerprint"]
    assert failed and "recomputes to" in failed[0].detail


def test_an_answer_swapped_for_a_wrong_one_fails_the_rescore(sandbox):
    """Editing an outcome moves the recomputed score away from the stored one."""
    root = sandbox("agent")
    blob = _load(root, "agent")
    for outcome in blob["outcomes"]:
        if outcome["qid"].startswith("num-"):
            outcome["answer"] = "The figure is 42."
            break
    (root / "agent.json").write_text(json.dumps(blob), encoding="utf-8")
    report = gate.run_gate(root, DATA, configs=("agent",))
    assert any(c.kind == "rescore" for c in report.failures)


def test_a_partial_run_is_refused_rather_than_scored(sandbox):
    root = sandbox("agent")
    blob = _load(root, "agent")
    blob["run"]["full_set"] = False
    (root / "agent.json").write_text(json.dumps(blob), encoding="utf-8")
    report = gate.run_gate(root, DATA, configs=("agent",))
    assert not report.ok
    assert any("partial run" in m for m in report.missing)
    assert not report.checks, "a partial file must not be scored at all"


def test_a_missing_results_file_is_a_failure_not_a_skip(tmp_path):
    report = gate.run_gate(tmp_path, DATA, configs=("agent",))
    assert not report.ok
    assert any("not committed" in m for m in report.missing)


def test_an_unknown_config_name_is_reported(tmp_path):
    report = gate.run_gate(tmp_path, DATA, configs=("no-such-config",))
    assert any("not a known config" in m for m in report.missing)


# --------------------------------------------------------------------------
# the plumbing
# --------------------------------------------------------------------------


def test_floors_read_nested_metric_paths():
    scores = {"hit_rate": {"10": 0.4}, "exact_match": 0.9}
    assert gate._dig(scores, "hit_rate.10") == pytest.approx(0.4)
    assert gate._dig(scores, "exact_match") == pytest.approx(0.9)
    assert gate._dig(scores, "hit_rate.99") is None
    assert gate._dig(scores, "nope.10") is None


def test_a_missing_metric_fails_its_floor_rather_than_passing_it():
    floor = gate.Floor("exact_match", ">=", 0.9, "because")
    assert not floor.holds(None)
    assert floor.holds(0.9) and not floor.holds(0.89)


def test_an_upper_bound_floor_runs_the_other_way():
    floor = gate.Floor("hallucinated", "<=", 0.0, "because")
    assert floor.holds(0.0) and not floor.holds(0.01)


def test_the_markdown_marks_failures_loudly(results_root):
    report = gate.run_gate(results_root, DATA, configs=("agent",))
    assert "**FAIL**" not in gate.to_markdown(report)
    report.checks.append(gate.Check("x", "floor", "y", False, "z"))
    assert "**FAIL**" in gate.to_markdown(report)
