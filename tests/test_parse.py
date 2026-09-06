"""The filing splitter, tested on the layouts that actually broke it.

Every case here is a reduction of a real document. The corpus itself is checked
by the M3 gate, which needs 1.1 GB on disk; these run in a second and fail for
one reason at a time.

Two properties matter more than the section count. Offsets must round-trip --
``text[section.start:section.end]`` is what a citation will quote, so if the
canonical text is ever reflowed after the offsets are taken, every citation
moves. And a contents table must never become sections, because a contents table
lists every item in the document and would otherwise win every retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from filing.stores.parse import (
    MIN_USABLE_CHARS,
    WHOLE_DOC_ITEM,
    Section,
    find_sections,
    flatten,
    parse_filing,
)


def html(body: str) -> str:
    return f"<html><body>{body}</body></html>"


def block(*rows: str) -> str:
    return "".join(f"<div>{r}</div>" for r in rows)


def section_body(item: str, chars: int = 400) -> str:
    return f"<div>Item {item}</div><p>{'body text ' * (chars // 10)}</p>"


# --------------------------------------------------------------------------
# flatten
# --------------------------------------------------------------------------


def test_offsets_index_the_returned_text():
    text, blocks = flatten(html(block("Alpha", "Beta", "Gamma")))
    assert [text[s:e] for s, e in blocks] == ["Alpha", "Beta", "Gamma"]


def test_words_split_across_styling_spans_are_not_torn_apart():
    """``<span>I</span><span>tem 8.</span>`` is one word in the browser.

    Schlumberger's 2020 10-K lays its Item 8 heading out exactly like this. A
    flattener that inserts a space at every text-run boundary produces "I tem 8."
    and loses the financial statements section entirely.
    """
    text, _ = flatten(html("<p><span>I</span><span>tem 8.</span> Financial</p>"))
    assert "Item 8. Financial" in text


def test_a_space_in_the_source_survives_the_run_boundary():
    text, _ = flatten(html("<p><span>Risk </span><span>Factors</span></p>"))
    assert "Risk Factors" in text


def test_non_breaking_spaces_become_spaces():
    text, _ = flatten(html("<p>Item\xa07.\xa0MD&amp;A</p>"))
    assert "Item 7. MD&A" in text


def test_script_and_style_are_dropped():
    text, _ = flatten(html("<script>var x = 1;</script><p>Item 1</p><style>p{}</style>"))
    assert "var x" not in text
    assert "Item 1" in text


# --------------------------------------------------------------------------
# find_sections
# --------------------------------------------------------------------------


def split(body: str, *, parts: bool = False) -> list[Section]:
    text, blocks = flatten(html(body))
    return find_sections(text, blocks, parts=parts)


def test_a_cross_reference_inside_a_sentence_is_not_a_heading():
    got = split(section_body("1") + "<p>See Item 15 of this Annual Report for more.</p>")
    assert [s.item for s in got] == ["1"]


def test_a_contents_table_is_dropped_wherever_it_sits():
    """Intel puts its item table *after* the body, so position cannot be the rule."""
    toc = block("Item 1", "Item 1A", "Item 7", "Item 8")
    body = "".join(section_body(i) for i in ("1", "1A", "7", "8"))
    assert [s.item for s in split(toc + body)] == ["1", "1A", "7", "8"]
    assert [s.item for s in split(body + toc)] == ["1", "1A", "7", "8"]


def test_a_short_tail_of_real_sections_is_not_mistaken_for_a_contents_table():
    """ "Item 4. Mine Safety Disclosures: Not applicable" really is 40 characters.

    Density alone would take the whole Part II tail of every 10-Q with it. What
    separates a contents table is that it lists the *whole* document.
    """
    body = (
        section_body("1", 4000)
        + section_body("2", 4000)
        + block("Item 3. Defaults Upon Senior Securities: None")
        + block("Item 4. Mine Safety Disclosures: Not applicable")
        + block("Item 5. Other Information: None")
    )
    assert [s.item for s in split(body)] == ["1", "2", "3", "4", "5"]


def test_a_title_in_the_next_cell_is_borrowed():
    """AMD's older 10-Qs put the number and the title in adjacent blocks."""
    got = split(block("Item 2") + block("Management's Discussion and Analysis") + "<p>x</p>")
    assert got[0].title == "Management's Discussion and Analysis"


def test_a_repeated_heading_belongs_to_the_first_one():
    body = section_body("1A", 2000) + block("Item 1A. Risk Factors (continued)") + "<p>more</p>"
    got = split(body)
    assert [s.item for s in got] == ["1A"]
    assert got[0].end == len(flatten(html(body))[0])


def test_parts_are_tracked_when_the_form_restarts_its_numbering():
    body = block("PART I") + section_body("1") + block("PART II") + section_body("1")
    assert [s.key for s in split(body, parts=True)] == ["I.1", "II.1"]


def test_a_part_split_across_table_cells_is_still_a_part():
    """ConocoPhillips lays it out as three cells: "PART", "I.", "FINANCIAL...".

    Without this the part stays on whatever the contents table last set, and the
    whole quarter files under Part II.
    """
    body = block("PART", "II") + block("PART", "I.", "FINANCIAL INFORMATION") + section_body("1")
    assert [s.key for s in split(body, parts=True)] == ["I.1"]


def test_a_part_named_in_a_sentence_does_not_move_the_part():
    body = (
        block("PART I")
        + section_body("1")
        + "<p>Part II of this report on page 33.</p>"
        + section_body("2")
    )
    assert [s.key for s in split(body, parts=True)] == ["I.1", "I.2"]


def test_a_bare_part_number_with_a_full_stop_is_still_a_heading():
    """Chevron announces its first part as exactly ``PART I.`` and nothing else."""
    body = block("PART II") + block("PART I.") + section_body("1")
    assert [s.key for s in split(body, parts=True)] == ["I.1"]


def test_a_ten_k_key_carries_no_part():
    """ConocoPhillips prints no part headings at all in its 10-K.

    Keying on a part that was never announced files Item 7 under Part I in one
    filing and under nothing in the next.
    """
    assert [s.key for s in split(section_body("7"))] == ["7"]


def test_prose_starting_with_the_word_item_is_not_a_heading():
    long_line = "Item costs of {} rose".format("x" * 300)
    assert split(f"<p>{long_line}</p>" + section_body("1"))[0].item == "1"


def test_a_section_spans_to_the_next_heading():
    text, blocks = flatten(html(section_body("1", 400) + section_body("2", 400)))
    got = find_sections(text, blocks, parts=False)
    assert got[0].end == got[1].start
    assert got[-1].end == len(text)
    assert text[got[0].start :].startswith("Item 1")


def test_offsets_round_trip():
    """The M3 gate re-reads the source to prove a chunk is still its offsets."""
    text, blocks = flatten(html(section_body("1", 900) + section_body("7", 900)))
    for s in find_sections(text, blocks, parts=False):
        assert text[s.start : s.end].startswith(f"Item {s.item}")


def test_a_stub_section_is_flagged_not_dropped():
    body = section_body("1", 4000) + block("Item 6. [Reserved]") + section_body("7", 4000)
    got = {s.item: s for s in split(body)}
    assert got["6"].is_stub
    assert not got["7"].is_stub


# --------------------------------------------------------------------------
# parse_filing
# --------------------------------------------------------------------------


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "filing.htm"
    p.write_bytes(html(body).encode("utf-8"))
    return p


def filler(chars: int) -> str:
    return f"<p>{'filler text ' * (chars // 12)}</p>"


def test_a_ten_k_with_its_required_sections_is_split(tmp_path):
    body = "".join(section_body(i, 9000) for i in ("1", "1A", "7", "8"))
    pf = parse_filing(write(tmp_path, body), accn="x", form="10-K")
    assert pf.split
    assert pf.section("7") is not None


def test_a_ten_q_keys_on_the_part(tmp_path):
    body = block("PART I") + section_body("1", 9000) + section_body("2", 9000)
    pf = parse_filing(write(tmp_path, body), accn="x", form="10-Q")
    assert pf.split
    assert pf.section("I.1") is not None


def test_a_filing_with_no_findable_headings_is_degraded_not_dropped(tmp_path):
    """Intel's 10-Qs and recent 10-Ks carry no ``Item N`` heading in the body.

    Twenty of Intel's filings land here. Quarantining them would drop a whole
    company from the narrative index to satisfy a splitter, so they are indexed
    whole, with the reason recorded.
    """
    pf = parse_filing(write(tmp_path, filler(MIN_USABLE_CHARS + 1000)), accn="x", form="10-K")
    assert pf.ok and not pf.split
    assert pf.degraded == "no Item heading in the body"
    assert [s.item for s in pf.sections] == [WHOLE_DOC_ITEM]
    assert pf.sections[0].end == len(pf.text)


def test_a_filing_missing_only_some_required_sections_is_degraded(tmp_path):
    body = section_body("1", 9000) + section_body("1A", 9000) + section_body("7", 9000)
    pf = parse_filing(write(tmp_path, body), accn="x", form="10-K")
    assert pf.degraded == "no heading for 8"


def test_text_too_short_to_index_is_quarantined(tmp_path):
    pf = parse_filing(
        write(tmp_path, "<p>Not the document you asked for.</p>"), accn="x", form="10-K"
    )
    assert not pf.ok
    assert "only 32 chars" in pf.quarantine
    assert pf.sections == ()


@pytest.mark.parametrize("form", ["10-K", "10-Q"])
def test_every_filing_is_split_degraded_or_quarantined(tmp_path, form):
    """The M3 definition of done: no filing is silently half-parsed."""
    pf = parse_filing(write(tmp_path, filler(50_000)), accn="x", form=form)
    assert sum([pf.split, bool(pf.degraded), bool(pf.quarantine)]) == 1
