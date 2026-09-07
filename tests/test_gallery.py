"""The failure gallery's query construction and its broken-vs-empty check.

Nothing here talks to Phoenix. What is worth pinning is the part that was
actually wrong for an hour: the attribute path. Phoenix answers a misspelt
path with an empty result rather than an error, so the only defence is that
the one working form is written down once and asserted.
"""

from __future__ import annotations

import pytest

from filing.agent.verify import TAXONOMY
from filing.gallery import QUERIES, Gallery, filter_for


def _gallery(**counts: int) -> Gallery:
    return Gallery(project="filing-room", endpoint="http://localhost:6006", counts=dict(counts))


def test_the_attribute_path_is_bracket_chained_not_dotted():
    # The dotted and single-string forms both parse and both return nothing,
    # which is why this is an assertion and not a comment.
    expr = filter_for("retrieval_miss")
    assert '["filing"]["failure"]["kinds"]' in expr
    assert "filing.failure.kinds" not in expr
    assert "attributes.filing" not in expr


def test_every_tag_in_the_taxonomy_has_a_filter():
    for kind in TAXONOMY:
        assert kind in QUERIES
        assert repr(kind) in QUERIES[kind]


@pytest.mark.parametrize("label", ["verified", "flagged", "blocked", "tagged"])
def test_the_run_level_filters_index_the_same_way(label):
    assert '["filing"]["' in QUERIES[label]


def test_a_run_with_more_tags_than_tagged_spans_is_consistent():
    # Tags are a set per span, so the per-tag counts over-count on purpose.
    g = _gallery(
        tagged=83, retrieval_miss=70, router_wrong=35, synthesis_drift=6, grader_false_positive=6
    )
    assert g.consistent


def test_a_tag_filter_that_matched_nothing_is_caught():
    # The bug this exists for: 83 spans carry a tag, but every per-tag filter
    # comes back empty because the path was wrong. Without the cross-check
    # that reads as a clean run.
    g = _gallery(
        tagged=83, retrieval_miss=0, router_wrong=0, synthesis_drift=0, grader_false_positive=0
    )
    assert not g.consistent


def test_a_corpus_with_no_failures_at_all_is_not_called_consistent():
    # Zero tagged spans cannot corroborate anything, so it reports as
    # unconfirmed rather than as proof the filters work.
    assert not _gallery(tagged=0, retrieval_miss=0).consistent


def test_a_dead_phoenix_is_reported_not_raised():
    g = Gallery(project="p", endpoint="http://localhost:6006", errors={"tagged": "ConnectError"})
    assert not g.ok
    assert not g.consistent
