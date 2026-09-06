"""The eval set's schema, its validation rules, and the frozen file itself.

Two kinds of test here. Most exercise the schema against hand-built questions.
The last group opens ``data/eval/questions_v1.0.jsonl`` if it is present and
asserts the properties the gate promises about it -- a file that has drifted
from its own datasheet is worse than no file, because every number measured on
it still looks valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from filing.eval import dataset
from filing.eval.dataset import DatasetError, EvalQuestion, Span

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "data" / "eval" / "questions_v1.0.jsonl"


def numeric(**over) -> EvalQuestion:
    base = dict(
        id="num-001",
        slice="numeric",
        question="What did NVDA report for Revenues for the fiscal year ended 2024-01-28?",
        value=60922000000.0,
        unit="USD",
        tag="Revenues",
        spans=(Span("0001045810-24-000029", 100, 200, "Revenue $ 60,922"),),
        origin="generated from facts_current",
        gold_source="value located in the filing text",
    )
    return EvalQuestion(**(base | over))


def narrative(**over) -> EvalQuestion:
    base = dict(
        id="nar-001",
        slice="narrative",
        question="How does the company describe its supply concentration?",
        spans=(Span("0001045810-24-000029", 500, 900, "We depend on a limited number"),),
        origin="hand-written",
        gold_source="hand-checked",
    )
    return EvalQuestion(**(base | over))


def unanswerable(**over) -> EvalQuestion:
    base = dict(
        id="una-001",
        slice="unanswerable",
        question="What was Apple's revenue in 2023?",
        origin="hand-written",
        gold_source="no gold: company-absent",
    )
    return EvalQuestion(**(base | over))


# --------------------------------------------------------------------- spans


def test_span_overlap_is_half_open():
    s = Span("A", 100, 200)
    assert s.overlaps("A", 150, 250)
    assert s.overlaps("A", 0, 101)
    # touching is not overlapping, in both directions
    assert not s.overlaps("A", 200, 300)
    assert not s.overlaps("A", 0, 100)


def test_span_overlap_requires_the_same_filing():
    assert not Span("A", 100, 200).overlaps("B", 100, 200)


def test_is_gold_is_true_for_any_span():
    q = narrative(spans=(Span("A", 0, 10), Span("B", 50, 60)))
    assert q.is_gold("B", 55, 65)
    assert not q.is_gold("B", 0, 10)


# ------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    ("q", "route"),
    [(numeric(), "sql"), (narrative(), "text"), (unanswerable(), "refuse")],
)
def test_route_follows_the_slice(q, route):
    assert q.route == route


def test_only_unanswerable_is_unanswerable():
    assert numeric().answerable
    assert narrative().answerable
    assert not unanswerable().answerable


# ---------------------------------------------------------------- validation


def test_validate_accepts_one_of_each():
    dataset.validate([numeric(), narrative(), unanswerable()])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(numeric(value=None), id="numeric-without-value"),
        pytest.param(numeric(unit=""), id="numeric-without-unit"),
        pytest.param(numeric(spans=()), id="numeric-never-found-in-a-filing"),
        pytest.param(narrative(spans=()), id="narrative-without-gold"),
        pytest.param(unanswerable(spans=(Span("A", 0, 5),)), id="unanswerable-with-gold"),
        pytest.param(unanswerable(value=1.0), id="unanswerable-with-a-value"),
        pytest.param(narrative(slice="opinion"), id="unknown-slice"),
        pytest.param(narrative(question="   "), id="empty-question"),
        pytest.param(narrative(origin=""), id="no-provenance"),
        pytest.param(narrative(gold_source=""), id="no-gold-source"),
        pytest.param(narrative(spans=(Span("A", 200, 200),)), id="empty-span"),
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(DatasetError):
        dataset.validate([bad])


def test_validate_rejects_duplicate_ids():
    with pytest.raises(DatasetError, match="duplicate"):
        dataset.validate([narrative(), narrative()])


# ---------------------------------------------------------------------- i/o


def test_round_trip_preserves_every_field(tmp_path):
    original = [numeric(), narrative(), unanswerable()]
    path = tmp_path / "q.jsonl"
    dataset.write(original, path)
    # write() sorts by id, so compare as a set of questions rather than a list.
    assert dataset.read(path) == sorted(original, key=lambda q: q.id)


def test_write_is_byte_stable_and_order_independent(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first = dataset.write([numeric(), narrative()], a)
    second = dataset.write([narrative(), numeric()], b)
    assert first == second == dataset.fingerprint(a) == dataset.fingerprint(b)


def test_write_uses_lf_endings_on_every_platform(tmp_path):
    path = tmp_path / "q.jsonl"
    dataset.write([numeric()], path)
    assert b"\r\n" not in path.read_bytes()


def test_read_validates(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_bytes(json.dumps(narrative(spans=()).to_json()).encode() + b"\n")
    with pytest.raises(DatasetError):
        dataset.read(path)


def test_read_says_how_to_build_a_missing_set(tmp_path):
    """And names a script that is in the repository, not one that used to be."""
    with pytest.raises(DatasetError, match="freeze_dataset"):
        dataset.read(tmp_path / "nope.jsonl")


def test_renumber_is_positional_within_a_slice():
    out = dataset.renumber([narrative(id="x"), numeric(id="y"), narrative(id="z")])
    assert [q.id for q in out] == ["num-001", "nar-001", "nar-002"]


def test_counts_reports_every_slice_even_when_empty():
    assert dataset.counts([numeric()]) == {"numeric": 1, "narrative": 0, "unanswerable": 0}


def test_dataset_path_carries_the_version(tmp_path):
    assert dataset.dataset_path(tmp_path, "v9.9").name == "questions_v9.9.jsonl"


# ------------------------------------------------------------- the real file


@pytest.mark.skipif(not FROZEN.exists(), reason="eval set v1.0 not built in this checkout")
class TestFrozenSet:
    """Properties the datasheet claims. If one fails, the datasheet is a lie."""

    @pytest.fixture(scope="class")
    @staticmethod
    def questions():
        return dataset.read(FROZEN)

    def test_slice_sizes_match_the_gate(self, questions):
        assert dataset.counts(questions) == {
            "numeric": 80,
            "narrative": 60,
            "unanswerable": 10,
        }

    def test_every_answerable_question_has_gold(self, questions):
        assert all(q.spans for q in questions if q.answerable)

    def test_no_unanswerable_question_has_gold(self, questions):
        assert not any(q.spans or q.value is not None for q in questions if not q.answerable)

    def test_every_numeric_question_names_a_tag_and_a_filing(self, questions):
        nums = [q for q in questions if q.slice == "numeric"]
        assert all(q.tag and q.accn and q.tickers for q in nums)

    def test_narrative_questions_do_not_quote_their_own_gold(self, questions):
        """A question that repeats its span's wording measures string matching.

        Eight consecutive words, not five. A question is *allowed* to name its
        subject in the subject's own words -- there is no other way to ask about
        "selling, general and administrative expense", and demanding a paraphrase
        would produce questions no analyst would type. What it may not do is
        reproduce a clause: at eight words the shared run has stopped being a
        term of art and started being the sentence, which is what would hand
        BM25 the slice on wording alone.
        """

        def shingles(text: str) -> set[str]:
            words = [w.strip(".,$()").lower() for w in text.split()]
            return {" ".join(words[i : i + 8]) for i in range(len(words) - 7)}

        leaks = [
            q.id
            for q in questions
            if q.slice == "narrative"
            for s in q.spans
            if shingles(q.question) & shingles(s.quote)
        ]
        assert leaks == []

    def test_gold_spans_are_well_formed(self, questions):
        for q in questions:
            for s in q.spans:
                assert 0 <= s.char_start < s.char_end
                assert s.quote.strip()

    def test_the_set_covers_the_whole_universe(self, questions):
        tickers = {t for q in questions for t in q.tickers}
        assert len(tickers) >= 19

    def test_the_numeric_slice_is_spread_across_concepts(self, questions):
        """The datasheet's "16 concepts, five questions each" -- the caps worked."""
        tags = [q.tag for q in questions if q.slice == "numeric"]
        assert len(set(tags)) == 16
        assert max(tags.count(t) for t in set(tags)) <= 5

    def test_numeric_gold_never_spans_two_filings(self, questions):
        """Multi-span numeric gold means one document printing a number twice.

        If a value were ever located in two different filings, ``recall`` would
        be scoring agreement between documents rather than coverage of one, and
        the datasheet's account of what the metric means would be wrong.
        """
        for q in questions:
            if q.slice == "numeric":
                assert len({s.accn for s in q.spans}) == 1

    def test_the_frozen_file_has_no_crlf(self):
        """A clone that checked this out CRLF has a different sha256.

        ``.gitattributes`` pins ``eol=lf`` for exactly this. Without it the
        repo's ``text=auto`` rewrites the file on a Windows checkout, and every
        fingerprint measured on it stops matching the one in the datasheet.
        """
        assert b"\r\n" not in FROZEN.read_bytes()

    def test_the_file_on_disk_is_what_write_would_produce(self, questions, tmp_path):
        again = tmp_path / FROZEN.name
        assert dataset.write(questions, again) == dataset.fingerprint(FROZEN)
