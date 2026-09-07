"""The ladder table, and the two things it must never quietly do.

A results-file aggregator has exactly two ways to lie. It can compare rungs that
were scored against different question sets, and it can present a number it did
not measure -- a latency invented out of a cached run's replay timings, say. Both
have a test here, because both would look completely fine in the output.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from filing.eval import ablation
from filing.eval.runner import CONFIGS

SHA = "a" * 64


def _results(
    tmp_path,
    name: str,
    *,
    generate: bool = True,
    sha: str = SHA,
    full: bool = True,
    hit10: float = 0.4,
    exact: float | None = 0.9,
):
    scores = {
        "name": "overall",
        "n": 150,
        "retrieval_n": 140,
        "hit_rate": {"1": 0.1, "5": 0.2, "10": hit10},
        "ndcg": {"1": 0.1, "5": 0.2, "10": 0.25},
        "exact_match": exact if generate else None,
        "citations_supported": 0.3 if generate else None,
        "hallucinated": 0.0 if generate else None,
    }
    blob = {
        "config": dict(name=name, generate=generate, k=5),
        "fingerprint": "f" * 64,
        "dataset": {"sha256": sha, "counts": {}},
        "run": {"full_set": full, "llm_calls": 0, "seconds": 1.0},
        "scores": {"overall": scores, "slices": {}},
        "outcomes": [],
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(blob), encoding="utf-8")


@pytest.fixture
def ladder_dir(tmp_path):
    for rung in ablation.LADDER:
        _results(tmp_path, rung.config, generate=rung.lane == "answer")
    return tmp_path


# --------------------------------------------------------------------------
# the ladder itself
# --------------------------------------------------------------------------


def test_every_rung_names_a_config_that_exists():
    # The module raises at import time if this is false; the test says so out
    # loud, because a ladder rung pointing at a deleted config is a table with a
    # hole in it and the hole is only visible at the moment somebody runs it.
    assert all(rung.config in CONFIGS for rung in ablation.LADDER)


def test_the_ladder_has_six_rungs_in_two_lanes():
    assert len(ablation.LADDER) == 6
    lanes = [rung.lane for rung in ablation.LADDER]
    assert set(lanes) == set(ablation.LANES)
    # Contiguous: a lane that interleaves with the other would make the delta
    # column compare a rung against something two steps away.
    assert lanes == sorted(lanes, key=ablation.LANES.index)


def test_the_retrieval_lane_generates_nothing(ladder_dir):
    rows = ablation.build(ladder_dir)
    for row in rows:
        if row.rung.lane != "retrieval":
            continue
        assert row.quality["exact_match"] is None
        assert row.quality["citations_supported"] is None


# --------------------------------------------------------------------------
# deltas
# --------------------------------------------------------------------------


def test_deltas_are_within_a_lane_and_never_across_one(tmp_path):
    for rung in ablation.LADDER:
        _results(
            tmp_path,
            rung.config,
            generate=rung.lane == "answer",
            hit10=0.1 if rung.lane == "retrieval" else 0.9,
        )
    rows = ablation.build(tmp_path)
    first_of_lane = {}
    for row in rows:
        if row.rung.lane not in first_of_lane:
            first_of_lane[row.rung.lane] = row
            assert row.deltas == {}, f"{row.name} should open its lane with no delta"
    # The answer lane opens at 0.9 against a retrieval lane sitting at 0.1. If
    # deltas crossed the boundary the first answer rung would show +80 points.
    assert first_of_lane["answer"].deltas == {}


def test_a_delta_can_be_negative_and_is_rendered_as_such(tmp_path):
    for i, rung in enumerate(ablation.LADDER):
        _results(tmp_path, rung.config, generate=rung.lane == "answer", hit10=0.5 - 0.1 * i)
    rows = ablation.build(tmp_path)
    second = rows[1]
    assert second.deltas["hit@10"] == pytest.approx(-0.1)
    assert "(-10.0)" in ablation.to_markdown(rows, timings_present=False)


# --------------------------------------------------------------------------
# the two ways a table like this lies
# --------------------------------------------------------------------------


def test_rungs_scored_against_different_question_sets_are_marked_not_comparable(tmp_path):
    for i, rung in enumerate(ablation.LADDER):
        _results(tmp_path, rung.config, generate=rung.lane == "answer", sha=SHA if i else "b" * 64)
    blob = ablation.to_json(ablation.build(tmp_path))
    assert blob["comparable"] is False
    assert isinstance(blob["dataset_sha256"], list)


def test_one_question_set_across_every_rung_is_comparable(ladder_dir):
    blob = ablation.to_json(ablation.build(ladder_dir))
    assert blob["comparable"] is True
    assert blob["dataset_sha256"] == SHA


def test_with_no_timing_file_the_cost_columns_are_dashes_and_say_why(ladder_dir):
    rows = ablation.build(ladder_dir)
    assert all(row.timing is None for row in rows)
    md = ablation.to_markdown(rows, timings_present=False)
    assert "| -- | -- | -- | -- |" in md
    # The note is the point: a reader who sees a dash should learn that the
    # committed runs *cannot* supply the number, not that nobody bothered.
    assert "outcome cache" in md
    assert "ablation --time" in md


def test_a_timing_file_fills_the_cost_columns(ladder_dir):
    (ladder_dir / ablation.TIMING_FILE).write_text(
        json.dumps(
            {
                "sample": 4,
                "timings": [
                    {
                        "config": "agent",
                        "n": 4,
                        "llm_calls": 9,
                        "warmup_seconds": 53.0,
                        "seconds": [1.0, 2.0, 3.0, 40.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = ablation.build(ladder_dir)
    agent = next(r for r in rows if r.name == "agent")
    assert agent.timing is not None
    # Nine calls over *five* questions, not over the four that were timed. The
    # warm-up is held out of the percentiles and kept in the cost average --
    # see Timing.calls_per_question for why that asymmetry is the correct one.
    assert agent.timing.questions == 5
    assert agent.timing.calls_per_question == pytest.approx(1.8)
    assert agent.timing.p50 == pytest.approx(2.5)
    # Nearest-rank at n=4 is the slowest question. Interpolating would report a
    # p95 of about 28 seconds, a number no question in the sample took.
    assert agent.timing.p95 == pytest.approx(40.0)
    # The warm-up is carried, and it is carried *outside* the percentiles: a
    # 53-second model load next to a 2.5-second median is exactly the shape
    # that made the first version of this column meaningless.
    assert agent.timing.warmup == pytest.approx(53.0)
    md = ablation.to_markdown(rows, timings_present=True)
    assert "| 1.80 | 2.5 | 40.0 | 53.0 |" in md
    # Rungs without a timing entry keep their dashes rather than borrowing one.
    assert "| -- | -- | -- | -- |" in md


def test_the_readme_tables_do_not_drift_from_the_generated_one():
    """The README's ladder is hand-copied, and a hand copy goes stale.

    It did, immediately: a re-measurement moved the agent's p50 from 14.8s to
    10.9s in ``results/ablation.md`` and the README went on quoting 14.8 --
    a number wrong about its own run, which is precisely the class of failure
    the gate exists to catch and the one place the gate could not see. Cheaper
    to assert than to remember.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    generated = (root / "results" / ablation.TABLE_MD).read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    def rows(text: str) -> dict[str, list[list[str]]]:
        """Every markdown row that names a config, keyed by that name.

        A list per name rather than one row, because the README mentions some
        of these configs in more than one table and the ladder is not always
        the last of them.
        """
        out: dict[str, list[list[str]]] = {}
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            parts = [c.strip() for c in line.strip().strip("|").split("|")]
            named = [c for c in parts if c.startswith("`") and c.endswith("`")]
            if named:
                out.setdefault(named[0].strip("`"), []).append(parts)
        return out

    gen, doc = rows(generated), rows(readme)
    assert gen, "no rows parsed out of the generated table"
    checked = set()
    for name, candidates in doc.items():
        if name not in gen:
            continue
        # A README ladder row is one that ends in a p50: "10.9s". The other
        # tables that name a config are not this one.
        for row in candidates:
            if not re.fullmatch(r"\d+(?:\.\d+)?s", row[-1]):
                continue
            # p50 is third from the end in the generated table (p50, p95, ready).
            assert row[-1].rstrip("s") == gen[name][0][-3], (
                f"{name}: README says p50 {row[-1]}, results/{ablation.TABLE_MD} "
                f"says {gen[name][0][-3]}s -- regenerate the table and copy it across"
            )
            checked.add(name)
    missing = {r.config for r in ablation.LADDER} - checked
    assert not missing, f"no README ladder row found for {sorted(missing)}"


def test_a_missing_rung_names_the_command_that_would_produce_it(tmp_path):
    _results(tmp_path, ablation.LADDER[0].config, generate=False)
    with pytest.raises(ablation.MissingResults) as excinfo:
        ablation.build(tmp_path)
    assert "python -m filing.eval run --config" in str(excinfo.value)


def test_the_warmup_question_leaves_the_percentiles_but_not_the_cost():
    """The one number the warm-up fix could quietly have got wrong.

    Holding the first question out of p50/p95 is right -- it is a model
    loading off disk. Holding it out of ``calls/q`` is wrong, because a cold
    process makes no extra API call: the warm-up costs what every other
    question costs. Divide the run's calls by the timed count and every cost
    figure in the table comes out an eighth too high.
    """
    timed = ablation.Timing(config="x", n=8, llm_calls=18, seconds=(1.0,) * 8, warmup=40.0, ran=9)
    assert timed.calls_per_question == pytest.approx(2.0)
    assert timed.p50 == pytest.approx(1.0)  # the 40s warm-up is nowhere in here
    # An older file has no questions_run; the recorded warm-up still counts.
    legacy = ablation.Timing(config="x", n=8, llm_calls=18, seconds=(1.0,) * 8, warmup=40.0)
    assert legacy.questions == 9
    # And with no warm-up recorded at all, the timed count is all there is.
    assert ablation.Timing(config="x", n=8, llm_calls=8, seconds=(1.0,) * 8).questions == 8


def test_an_empty_timing_list_is_not_a_zero_second_answer():
    timing = ablation.Timing(config="x", n=0, llm_calls=0, seconds=())
    assert timing.p50 == 0.0 and timing.p95 == 0.0
    assert timing.calls_per_question == 0.0


# --------------------------------------------------------------------------
# the committed table
# --------------------------------------------------------------------------


def test_the_committed_table_covers_the_whole_ladder(results_root):
    rows = ablation.build(results_root)
    assert [r.name for r in rows] == [rung.config for rung in ablation.LADDER]
    assert all(row.full_set for row in rows), "a partial run must never reach the table"


def test_the_committed_rungs_all_scored_the_same_question_set(results_root):
    blob = ablation.to_json(ablation.build(results_root))
    assert blob["comparable"] is True, blob["dataset_sha256"]


def test_the_reranker_rung_is_the_one_that_goes_backwards(results_root):
    """M5's finding, asserted where a reader will meet it.

    The whole reason for a ladder is that it can report a stage as worthless.
    If this ever passes by turning positive, the table has stopped describing
    the system that was measured and somebody should look at why.
    """
    rows = {r.name: r for r in ablation.build(results_root)}
    rerank = rows["agent-retrieval"]
    assert rerank.deltas["hit@10"] < 0
    assert rerank.deltas["ndcg@10"] < 0
