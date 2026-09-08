"""The two surfaces render one document. These are the tests of the document.

:mod:`tests.test_ui` already asserts that every outcome the API can emit has a
banner and every figure status has a colour. Those assertions live there and
still pass, because :mod:`filing.ui` imports the constants from
:mod:`filing.render` rather than keeping its own -- which is the property the
first test here pins down, since an inlined copy would keep them passing while
the two pages drifted apart.

The rest is about the renderers themselves, and the questions worth asking of a
renderer are all answerable without a corpus or a model: does an abstention
print the refusal, does a block say who blocked it, does a figure get the
colour its status calls for, and -- the one that is a security bug rather than
a cosmetic one -- does a span of filing text containing a ``<`` reach the
browser as text.
"""

from __future__ import annotations

import pytest

from filing import render, ui
from filing.api import payload_for

PANELS = (
    render.outcome_banner,
    render.answer_panel,
    render.claims_panel,
    render.verification_panel,
    render.evidence_panel,
    render.path_panel,
)


def test_the_page_and_the_space_share_one_set_of_judgements() -> None:
    """Not equal values -- the same objects.

    Equality would pass against two copies that happen to agree today, which is
    exactly the state this module exists to prevent.
    """
    assert ui.OUTCOMES is render.OUTCOMES
    assert ui.STATUS_COLOUR is render.STATUS_COLOUR
    assert ui.HOW is render.HOW
    assert ui.EXAMPLES is render.EXAMPLES


def test_the_renderers_import_no_surface() -> None:
    """`render` must not pull in streamlit or gradio.

    It is imported by both, and by the API's tests. A stray import would make
    the lighter surface pay for the heavier one and would turn a missing extra
    into an ImportError three modules away from the cause.
    """
    src = (render.__file__ or "").replace("\\", "/")
    text = open(src, encoding="utf-8").read()  # noqa: SIM115, PTH123
    assert "import streamlit" not in text
    assert "import gradio" not in text


# --------------------------------------------------------------------------
# escaping
# --------------------------------------------------------------------------


def test_a_filing_span_containing_markup_reaches_the_browser_as_text() -> None:
    """The one that is a bug rather than a blemish.

    ``st.markdown`` escapes by default and the Streamlit page got this free.
    A string built here goes to the browser as written, and evidence bodies are
    spans of SEC filings -- text that genuinely contains ``<``, ``&`` and the
    occasional stray tag.
    """
    payload = {
        "outcome": "answered",
        "evidence": [
            {
                "marker": 1,
                "citation": "NVDA 10-K",
                "kind": "text",
                "score": 0.5,
                "cited": True,
                "locatable": True,
                "body": "<script>alert(1)</script> & cost of revenue < 10%",
            }
        ],
    }
    html = render.evidence_panel(payload)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_the_answer_itself_is_escaped() -> None:
    html = render.answer_panel({"outcome": "answered", "answer": "5 < 6 & 7 > 6"})
    assert "&lt;" in html and "&amp;" in html and "&gt;" in html


@pytest.mark.parametrize("panel", PANELS)
def test_no_panel_raises_on_an_empty_payload(panel) -> None:  # noqa: ANN001
    """Every panel has to survive the payload of a run that produced nothing.

    An error state that crashes the renderer costs the reader the one screen
    that would have told them what went wrong.
    """
    assert isinstance(panel({}), str)


@pytest.mark.parametrize("panel", PANELS)
@pytest.mark.parametrize("outcome", ["answered", "abstained", "blocked", "error"])
def test_no_panel_raises_on_any_outcome(panel, outcome) -> None:  # noqa: ANN001
    assert isinstance(panel({"outcome": outcome}), str)


# --------------------------------------------------------------------------
# the judgements, rendered
# --------------------------------------------------------------------------


def test_an_abstention_prints_the_refusal_and_does_not_call_it_a_failure() -> None:
    html = render.answer_panel({"outcome": "abstained", "answer": "INSUFFICIENT EVIDENCE"})
    assert "INSUFFICIENT EVIDENCE" in html
    banner = render.outcome_banner({"outcome": "abstained"})
    assert "fail" not in banner.lower()
    assert "declined" in banner.lower()


def test_a_block_names_the_verifier_as_the_one_that_stopped_it() -> None:
    """The distinction M6 exists to make, on the surface that shows it.

    An abstention and a block both print the refusal token. If the page
    collapsed them, the guard would be invisible in exactly the runs where it
    acted.
    """
    payload = {
        "outcome": "blocked",
        "answer": "INSUFFICIENT EVIDENCE",
        "verification": {"enabled": True, "reasons": ["a figure appears in no evidence record"]},
    }
    html = render.answer_panel(payload)
    assert "verifier" in html.lower()
    assert "a figure appears in no evidence record" in html
    assert "verifier" in render.outcome_banner(payload).lower()


def test_verification_off_says_so_rather_than_showing_zero_problems() -> None:
    html = render.verification_panel({"verification": {"enabled": False}})
    assert "off in this config" in html
    assert "not the same as nothing being wrong" in html


@pytest.mark.parametrize("status", sorted(render.STATUS_COLOUR))
def test_a_figure_wears_the_colour_its_status_calls_for(status: str) -> None:
    html = render.verification_panel(
        {
            "verification": {
                "enabled": True,
                "numbers": [{"text": "1,234", "status": status, "how": "digits"}],
            }
        }
    )
    if status == "context":
        # Deliberately not listed: a period or a figure echoed from the question
        # is not a claim, and printing every one of them would bury the ones
        # that are.
        assert "1,234" not in html
    else:
        assert render.STATUS_COLOUR[status] in html


def test_every_way_a_number_can_be_matched_renders_its_sentence() -> None:
    for how, sentence in render.HOW.items():
        html = render.verification_panel(
            {
                "verification": {
                    "enabled": True,
                    "numbers": [{"text": "9", "status": "supported", "how": how}],
                }
            }
        )
        assert sentence in html


def test_evidence_with_nothing_to_look_up_is_called_out() -> None:
    html = render.evidence_panel(
        {"evidence": [{"marker": 1, "citation": "", "score": 0.1, "locatable": False, "body": ""}]}
    )
    assert "nothing a reader could look up" in html


def test_the_repair_log_is_printed_in_order() -> None:
    html = render.path_panel(
        {"repairs": 2, "repair_log": ["drop the period", "fall back to the text retriever"]}
    )
    assert html.index("drop the period") < html.index("fall back to the text retriever")
    assert "2 of 2" in html


# --------------------------------------------------------------------------
# against a real payload
# --------------------------------------------------------------------------


def test_the_renderers_accept_what_the_api_actually_builds() -> None:
    """The hand-built payloads above test the branches; this tests the shape.

    ``payload_for`` is pure over a finished state, so the document the page
    receives in production can be produced here without a corpus -- and a field
    renamed on the API would otherwise break the page silently.
    """
    payload = payload_for(
        {"answer": "", "evidence": [], "blocked": False},
        question="What was the population of France in 1780?",
        qid="t",
    ).model_dump(mode="json")
    for panel in PANELS:
        assert isinstance(panel(payload), str)
    assert payload["outcome"] in render.OUTCOMES
