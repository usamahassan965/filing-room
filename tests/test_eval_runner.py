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
from filing.eval import dataset, metrics, runner
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
        return [c for c, _ in self.search_scored(query, k=k)]

    def search_scored(self, query: str, *, k: int = 5):
        self.queries.append(query)
        # Descending, so a test can tell the order came from the retriever.
        return [(c, 0.9 - 0.1 * i) for i, c in enumerate(self.chunks[:k])]


class ExplodingBackend:
    def usage(self) -> Usage:
        return Usage(http_calls=0)

    def chat(self, *a, **kw) -> str:  # noqa: ANN002, ANN003
        raise RuntimeError("429 quota exhausted")


class ForbiddenBackend:
    """For tests whose claim is that nothing was reached for.

    Passing ``None`` and trusting the cache expresses the same intent and is a
    trap: on a cache miss the runner builds the *real* backend and the test
    quietly spends live quota instead of failing. A double that raises on
    contact turns that into a red test, which is what the assertion meant.
    """

    def usage(self) -> Usage:
        raise AssertionError("the run reached for a chat backend; the cache should have served")

    def chat(self, *a, **kw) -> str:  # noqa: ANN002, ANN003
        raise AssertionError("the run made a chat call; the cache should have served")


class ForbiddenRetriever:
    def search(self, *a, **kw):  # noqa: ANN002, ANN003
        raise AssertionError("the run reached for a retriever; the cache should have served")

    search_scored = search


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
    first = runner.run(env, backend=FakeBackend(), retriever=FakeRetriever())
    assert first.answered == 2 and first.errors == 0  # or the cache is empty below
    # Doubles that raise on contact, not None: see ForbiddenBackend.
    again = runner.run(env, backend=ForbiddenBackend(), retriever=ForbiddenRetriever())
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


# ------------------------------------------------------- errors are not results


def test_a_transient_error_is_never_cached(env):
    """The bug this exists to prevent cost a day of quota to find.

    A rate limit came back as an ``Outcome`` with an ``error``, the cache stored
    it like any other, and every later run replayed the failure from disk
    without retrying -- so the run that was meant to resume where it stopped
    resumed by re-reporting its own 429s, permanently, for as long as the
    fingerprint lived.
    """
    first = runner.run(env, backend=ExplodingBackend(), retriever=FakeRetriever())
    assert first.errors == 2

    be = FakeBackend()
    again = runner.run(env, backend=be, retriever=FakeRetriever())
    assert again.from_cache == 0, "a failed question came back from the cache"
    assert again.answered == 2 and again.errors == 0
    assert len(be.calls) == 2


def test_a_partial_run_caches_only_what_worked(env):
    """One good answer and one failure: the good one is banked, the bad one is not."""
    backend = FakeBackend()
    backend.chat = _fails_on_second(backend)
    runner.run(env, backend=backend, retriever=FakeRetriever())
    cached = list((runner.results_dir(env) / "cache").rglob("*.json"))
    assert [p.stem for p in cached] == ["num-001"]


def _fails_on_second(backend):
    real = backend.chat

    def chat(messages, **kw):
        if len(backend.calls) >= 1:
            raise RuntimeError("429 quota exhausted")
        return real(messages, **kw)

    return chat


# ------------------------------------------------------------ retrieval only


def test_retrieval_only_makes_no_chat_call_at_all(env):
    ec = runner.get_config("baseline-retrieval")
    report = runner.run(env, config=ec, backend=ForbiddenBackend(), retriever=FakeRetriever())
    assert report.llm_calls == 0 and report.errors == 0
    assert all(o.answer == "" and o.route == "" and not o.citations for o in report.outcomes)
    assert all(o.retrieved for o in report.outcomes)


def test_retrieval_only_reports_generation_as_unmeasured_not_zero(env):
    """0.0% and "not attempted" are different claims and must not share a cell."""
    ec = runner.get_config("baseline-retrieval")
    report = runner.run(env, config=ec, backend=ForbiddenBackend(), retriever=FakeRetriever())
    numeric = report.card.slices["numeric"]
    assert numeric.exact_match is None
    assert numeric.router_accuracy is None
    assert report.card.slices["unanswerable"].abstention is None
    assert numeric.hit_rate[1] == 1.0  # retrieval, however, was measured
    assert "| -- |" in metrics.to_markdown(report.card)


def test_the_retrieval_fingerprint_ignores_the_chat_model(env):
    """Otherwise the same retrieval numbers would arrive under two identities."""
    ec = runner.get_config("baseline-retrieval")
    a = ec.resolved(env)
    b = replace(ec, chat_model="something-else", chat_backend="ollama").resolved(env)
    assert a.fingerprint("sha") == b.fingerprint("sha")
    assert a.chat_model == "" and a.chat_backend == ""


def test_retrieval_only_and_the_full_baseline_do_not_share_a_cache(env):
    """Their outcomes differ, so a cache hit across them would be a wrong answer."""
    sha = "sha"
    full = runner.get_config("baseline").resolved(env).fingerprint(sha)
    retrieval = runner.get_config("baseline-retrieval").resolved(env).fingerprint(sha)
    assert full != retrieval


def test_a_retrieved_chunk_carries_its_score(env):
    report = runner.run(
        env,
        config=runner.get_config("baseline-retrieval"),
        backend=ForbiddenBackend(),
        retriever=FakeRetriever(),
    )
    assert report.outcomes[0].retrieved[0].score == 0.9


def test_a_subset_run_never_takes_the_canonical_filename(env):
    """`--limit 3` must not land at results/<config>.json.

    A results file carries its numbers, not its scope, so nothing downstream
    can tell a three-question smoke test from the run the gate asks for once
    it is sitting at the canonical path. This has happened twice in this repo.
    """
    root = runner.results_dir(env)
    full = runner.run(env, backend=FakeBackend(), retriever=FakeRetriever())
    assert full.full and full.written == root / "baseline.json"

    part = runner.run(env, backend=FakeBackend(), retriever=FakeRetriever(), limit=1)
    assert not part.full
    assert part.written == root / "baseline.partial.json"
    assert "PARTIAL RUN" in (root / "baseline.partial.md").read_text(encoding="utf-8")

    # and the full run's file is still the full run's
    body = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
    assert body["run"]["full_set"] is True and body["run"]["answered"] == full.answered


def test_the_scope_is_recorded_inside_the_file_too(env):
    """A file that is copied or renamed keeps the claim its numbers need."""
    root = runner.results_dir(env)
    runner.run(env, backend=FakeBackend(), retriever=FakeRetriever(), limit=1)
    body = json.loads((root / "baseline.partial.json").read_text(encoding="utf-8"))
    assert body["run"]["full_set"] is False


def test_the_config_picks_the_backend_not_the_environment(env, monkeypatch):
    """A config that names ollama must not be generated by whatever LLM_BACKEND says.

    This shipped broken once. `baseline-local-alt` declared chat_backend="ollama"
    and was fingerprinted, cached and written under llama3.2:3b, while every one
    of its 180 calls went to Gemini because the runner built the backend from
    settings. The results file was not merely wrong, it was confidently
    mislabelled -- the one failure an eval harness cannot be allowed.
    """
    asked: list[object] = []

    def spy(cfg, backend=None):
        asked.append(backend)
        return FakeBackend()

    monkeypatch.setattr("filing.llm.factory.build_backend", spy)

    ec = replace(runner.CONFIGS["baseline"], chat_backend="ollama")
    runner.run(env, config=ec, retriever=FakeRetriever(), use_cache=False, write=False)
    assert asked == ["ollama"], asked

    # An unset chat_backend still ends up naming one, because resolved() fills
    # it from settings before the run starts. That is the property worth
    # holding: whatever the fingerprint records as the backend is the backend
    # that gets built, with no second reading of the environment in between.
    asked.clear()
    ec = replace(runner.CONFIGS["baseline"], chat_backend="")
    report = runner.run(env, config=ec, retriever=FakeRetriever(), use_cache=False, write=False)
    assert asked == [report.config.chat_backend] == ["gemini"], asked
