"""The harness: fingerprinting, caching, the prompt contract, the run loop.

Every test here runs against a fake retriever and a fake backend. That is not a
convenience -- it is the property being tested. If any of this needed Qdrant or a
network to exercise, "a finished run re-scores from cache with nothing running"
would be a claim rather than a fact, and the whole reproducibility argument in
the runner's docstring would rest on nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from conftest import make_chunk
from filing.config import Settings
from filing.eval import dataset, runner
from filing.eval.dataset import EvalQuestion, Span
from filing.eval.metrics import Outcome, RetrievedChunk
from filing.eval.runner import OutcomeCache

# ------------------------------------------------------------------- doubles


@dataclass
class Usage:
    http_calls: int = 0


class FakeBackend:
    """Answers from a script, and counts its own calls the way a real one does."""

    def __init__(self, answers: list[str] | str = "Revenue was $1,000. [1]") -> None:
        self.script = answers if isinstance(answers, list) else None
        self.fixed = None if isinstance(answers, list) else answers
        self.calls: list[list[dict[str, str]]] = []

    def usage(self) -> Usage:
        return Usage(http_calls=len(self.calls))

    def chat(self, messages, *, role="chat", temperature=0.0, max_tokens=512) -> str:  # noqa: ANN001
        self.calls.append(messages)
        if self.fixed is not None:
            return self.fixed
        return self.script[(len(self.calls) - 1) % len(self.script)]


class FakeRetriever:
    def __init__(self, chunks=None) -> None:
        self.chunks = chunks if chunks is not None else [make_chunk("a body", char_start=0)]
        self.queries: list[str] = []

    def search(self, query: str, *, k: int = 5):
        self.queries.append(query)
        return self.chunks[:k]


class ExplodingBackend:
    def usage(self) -> Usage:
        return Usage(http_calls=0)

    def chat(self, *a, **kw) -> str:  # noqa: ANN002, ANN003
        raise RuntimeError("429 quota exhausted")


def question(qid="num-001", slice_="numeric", **over) -> EvalQuestion:
    base = dict(
        id=qid,
        slice=slice_,
        question="What did NVDA report for revenue?",
        value=1000.0 if slice_ == "numeric" else None,
        unit="USD" if slice_ == "numeric" else "",
        tag="Revenues" if slice_ == "numeric" else "",
        spans=() if slice_ == "unanswerable" else (Span("0000000000-00-000000", 0, 6, "a body"),),
        origin="test",
        gold_source="test",
    )
    return EvalQuestion(**(base | over))


@pytest.fixture
def env(tmp_path):
    """A Settings pointed at a throwaway data dir, with a two-question set in it."""
    cfg = Settings(data_dir=tmp_path / "data", llm_backend="gemini", embed_backend="local")
    questions = [question(), question("una-001", "unanswerable")]
    dataset.write(questions, dataset.dataset_path(cfg.data_dir))
    return cfg


# ------------------------------------------------------------- configuration


def test_the_shipped_config_is_the_one_the_gate_names():
    assert runner.get_config("baseline").name == "baseline"
    assert runner.get_config("baseline").system == "naive"


def test_an_unknown_config_lists_the_known_ones():
    with pytest.raises(KeyError, match="baseline"):
        runner.get_config("nope")


def test_resolved_replaces_backend_names_with_model_ids(env):
    ec = runner.get_config("baseline").resolved(env)
    assert ec.chat_model and ec.chat_model != "gemini"
    assert ec.embed_model and ec.collection.startswith("filing__local__")


@pytest.mark.parametrize(
    "change",
    [
        {"k": 10},
        {"chat_model": "some-other-model"},
        {"prompt_version": "p2"},
        {"temperature": 0.7},
        {"collection": "elsewhere"},
        {"chunker": "naive-4096"},
    ],
)
def test_every_field_that_could_move_a_number_moves_the_fingerprint(change):
    base = runner.get_config("baseline")
    assert base.fingerprint("sha") != replace(base, **change).fingerprint("sha")


def test_a_different_question_file_is_a_different_experiment():
    base = runner.get_config("baseline")
    assert base.fingerprint("aaa") != base.fingerprint("bbb")


def test_the_same_config_hashes_the_same_twice():
    base = runner.get_config("baseline")
    assert base.fingerprint("sha") == replace(base, name="baseline").fingerprint("sha")


def test_the_note_is_prose_and_still_part_of_the_hash():
    """A config whose description changed is a config a reader would compare wrongly."""
    base = runner.get_config("baseline")
    assert base.fingerprint("s") != replace(base, note="different story").fingerprint("s")


# --------------------------------------------------------------------- cache


def test_cache_round_trips_an_outcome(tmp_path):
    cache = OutcomeCache(tmp_path, "f" * 64)
    out = Outcome(qid="num-001", answer="a", retrieved=(RetrievedChunk("c", "A", 0, 5),))
    cache.put(out)
    assert OutcomeCache(tmp_path, "f" * 64).get("num-001") == out


def test_cache_is_partitioned_by_fingerprint(tmp_path):
    OutcomeCache(tmp_path, "a" * 64).put(Outcome(qid="num-001", answer="one"))
    assert OutcomeCache(tmp_path, "b" * 64).get("num-001") is None


def test_a_corrupt_entry_costs_one_call_not_the_run(tmp_path):
    cache = OutcomeCache(tmp_path, "c" * 64)
    (cache.dir / "num-001.json").write_text("{not json", encoding="utf-8")
    assert cache.get("num-001") is None


def test_a_missing_entry_is_not_an_error(tmp_path):
    assert OutcomeCache(tmp_path, "d" * 64).get("nobody") is None


# -------------------------------------------------------------------- prompt


def test_the_prompt_numbers_the_excerpts_from_one():
    chunks = [make_chunk("first"), make_chunk("second")]
    user = runner.build_prompt("why?", chunks)[1]["content"]
    assert "[1] NVDA 10-K" in user and "[2] NVDA 10-K" in user
    assert user.rstrip().endswith("Question: why?")


def test_the_prompt_says_what_to_do_with_nothing():
    user = runner.build_prompt("why?", [])[1]["content"]
    assert "no excerpts" in user


def test_the_system_prompt_names_the_exact_refusal_token():
    system = runner.build_prompt("q", [])[0]["content"]
    assert runner.REFUSAL in system


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Because [1].", ("c1",)),
        ("Both [1] and [2].", ("c1", "c2")),
        ("Twice [2][2].", ("c2",)),
        ("Backwards [2][1].", ("c2", "c1")),
        ("Out of range [9].", ()),
        ("Zero is not an excerpt [0].", ()),
        ("No citation at all.", ()),
    ],
)
def test_citations_map_markers_back_to_chunk_ids(answer, expected):
    chunks = [make_chunk("a", chunk_id="c1"), make_chunk("b", chunk_id="c2")]
    assert runner.parse_citations(answer, chunks) == expected


# ------------------------------------------------------------- answer_naive


def test_an_answer_carries_its_evidence_and_its_cost():
    ret, be = FakeRetriever(), FakeBackend("Revenue was $1,000. [1]")
    out = runner.answer_naive(
        question(), retriever=ret, backend=be, config=runner.get_config("baseline")
    )
    assert out.route == "text"
    assert not out.refused
    assert out.citations == (ret.chunks[0].chunk_id,)
    assert out.retrieved[0].accn == ret.chunks[0].accn
    assert out.llm_calls == 1
    assert out.error == ""


def test_the_refusal_token_is_recognised_whatever_its_case():
    be = FakeBackend("insufficient evidence for this question")
    out = runner.answer_naive(
        question(), retriever=FakeRetriever(), backend=be, config=runner.get_config("baseline")
    )
    assert out.refused and out.route == "refuse"


def test_a_refusal_cites_nothing_even_if_it_prints_a_marker():
    """Otherwise an abstention would score citations, which are about answers."""
    be = FakeBackend(f"{runner.REFUSAL} [1]")
    out = runner.answer_naive(
        question(), retriever=FakeRetriever(), backend=be, config=runner.get_config("baseline")
    )
    assert out.citations == ()


def test_a_failed_question_becomes_an_error_not_an_exception():
    out = runner.answer_naive(
        question(),
        retriever=FakeRetriever(),
        backend=ExplodingBackend(),
        config=runner.get_config("baseline"),
    )
    assert "RuntimeError" in out.error and "429" in out.error
    assert out.answer == "" and out.retrieved == ()


def test_k_is_honoured():
    ret = FakeRetriever([make_chunk(f"body {i}", chunk_id=f"c{i}") for i in range(9)])
    ec = replace(runner.get_config("baseline"), k=3)
    out = runner.answer_naive(question(), retriever=ret, backend=FakeBackend(), config=ec)
    assert len(out.retrieved) == 3


# ----------------------------------------------------------------- the run


def test_a_run_answers_scores_and_writes(env):
    be, ret = FakeBackend(["Revenue was $1,000. [1]", runner.REFUSAL]), FakeRetriever()
    report = runner.run(env, backend=be, retriever=ret)

    assert report.answered == 2 and report.from_cache == 0
    assert report.card.slices["numeric"].exact_match == 1.0
    assert report.card.slices["unanswerable"].abstention == 1.0
    out = runner.results_dir(env)
    written = json.loads((out / "baseline.json").read_text("utf-8"))
    assert written["fingerprint"] == report.fingerprint
    assert written["dataset"]["sha256"] == report.dataset_sha256
    assert len(written["outcomes"]) == 2
    assert "| numeric |" in (out / "baseline.md").read_text("utf-8")


def test_a_second_run_costs_nothing_and_needs_nothing(env):
    runner.run(env, backend=FakeBackend(), retriever=FakeRetriever())
    # No backend, no retriever: if the loop reached for either, this raises.
    again = runner.run(env)
    assert again.from_cache == 2 and again.answered == 0
    assert again.llm_calls == 0


def test_no_cache_re_answers(env):
    runner.run(env, backend=FakeBackend(), retriever=FakeRetriever())
    be = FakeBackend()
    again = runner.run(env, backend=be, retriever=FakeRetriever(), use_cache=False)
    assert again.answered == 2 and len(be.calls) == 2


def test_a_changed_config_does_not_read_the_old_cache(env):
    runner.run(env, backend=FakeBackend(), retriever=FakeRetriever())
    other = replace(runner.get_config("baseline"), name="baseline", k=10)
    be = FakeBackend()
    again = runner.run(env, config=other, backend=be, retriever=FakeRetriever())
    assert again.answered == 2 and again.from_cache == 0


def test_a_run_can_be_limited_to_one_slice(env):
    report = runner.run(
        env, backend=FakeBackend(), retriever=FakeRetriever(), slices=("unanswerable",)
    )
    assert report.counts == {"numeric": 0, "narrative": 0, "unanswerable": 1}


def test_limit_takes_the_first_n(env):
    report = runner.run(env, backend=FakeBackend(), retriever=FakeRetriever(), limit=1)
    assert report.answered == 1


def test_write_false_leaves_no_results_file(env):
    runner.run(env, backend=FakeBackend(), retriever=FakeRetriever(), write=False)
    assert not (runner.results_dir(env) / "baseline.json").exists()


def test_errors_are_counted_and_the_run_finishes(env):
    report = runner.run(env, backend=ExplodingBackend(), retriever=FakeRetriever())
    assert report.errors == 2
    assert len(report.outcomes) == 2


def test_progress_is_reported_per_question(env):
    seen: list[tuple[int, int, str]] = []
    runner.run(
        env,
        backend=FakeBackend(),
        retriever=FakeRetriever(),
        on_question=lambda i, n, o: seen.append((i, n, o.qid)),
    )
    assert [(i, n) for i, n, _ in seen] == [(1, 2), (2, 2)]


def test_a_missing_dataset_says_how_to_build_one(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data")
    with pytest.raises(dataset.DatasetError, match="freeze_dataset"):
        runner.run(cfg, backend=FakeBackend(), retriever=FakeRetriever())


def test_the_results_file_records_the_config_in_full(env):
    report = runner.run(env, backend=FakeBackend(), retriever=FakeRetriever())
    written = json.loads((runner.results_dir(env) / "baseline.json").read_text("utf-8"))
    assert written["config"]["chat_model"] == report.config.chat_model
    assert written["config"]["dataset_version"] == dataset.DATASET_VERSION
    assert written["config"]["prompt_version"] == runner.PROMPT_VERSION
