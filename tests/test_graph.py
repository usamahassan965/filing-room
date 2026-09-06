"""The rules the extractor is made of, each pinned to the sentence that found it.

Every assertion here started as real output from the corpus. The extractor is
rules over prose, which means its failures are specific and quotable -- a node
called "u.s" with 867 edges, a company named "Strategy", an acquisition verb
claimed by a pair it was never about -- and each rule that fixed one of those
gets a test, so a later "simplification" has to argue with the evidence.

Nothing here touches DuckDB or NetworkX except the two round-trip tests at the
bottom. Extraction is pure text in, edges out, and that is the whole point of
it being auditable.
"""

from __future__ import annotations

import pytest

from conftest import make_chunk
from filing.stores.graph import (
    AGENCIES,
    MIN_ORG_FILINGS,
    Edge,
    GraphStore,
    Mention,
    canonical,
    cue_span,
    discover_organizations,
    extract,
    mentions,
    neighbours,
    sentences,
)

# --------------------------------------------------------------------------
# sentence splitting
# --------------------------------------------------------------------------


def spans(text: str) -> list[str]:
    return [text[s:e] for s, e in sentences(text)]


def test_a_corporate_suffix_is_not_the_end_of_a_sentence():
    """ "Xilinx, Inc. was acquired" is one sentence, and splitting it loses the verb."""
    text = "Xilinx, Inc. was acquired in 2022. Revenue grew."
    assert spans(text) == ["Xilinx, Inc. was acquired in 2022.", "Revenue grew."]


def test_a_single_initial_is_not_the_end_of_a_sentence():
    text = "J. Smith signed the agreement. He resigned later."
    assert spans(text) == ["J. Smith signed the agreement.", "He resigned later."]


def test_a_newline_always_ends_a_sentence_even_after_an_abbreviation():
    """Otherwise "Item 1A." swallows the heading below it and the whole section reads as one."""
    assert spans("Item 1A.\nRisk Factors follow.") == ["Item 1A.", "Risk Factors follow."]


def test_a_blank_line_is_one_boundary_and_not_two_empty_sentences():
    assert spans("First block.\n\nSecond block.") == ["First block.", "Second block."]


def test_the_offsets_resolve_back_to_the_text_they_name():
    """The same promise the chunker makes: an offset a reader can check."""
    text = "We rely on TSMC. Approx. 60% of wafers come from Taiwan.\nSupply is concentrated."
    for start, end in sentences(text):
        assert text[start:end] == text[start:end].strip()
    assert spans(text)[1] == "Approx. 60% of wafers come from Taiwan."


def test_trailing_text_with_no_final_period_is_still_a_sentence():
    assert spans("A complete one. An unterminated tail") == [
        "A complete one.",
        "An unterminated tail",
    ]


# --------------------------------------------------------------------------
# canonical names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    ["Xilinx, Inc.", "Xilinx Inc", "Xilinx Incorporated", "Xilinx"],
)
def test_every_way_a_filing_punctuates_a_name_is_one_node(written):
    assert canonical(written)[0] == "xilinx"


def test_the_display_name_keeps_its_capitals():
    assert canonical("Taiwan Semiconductor Manufacturing Company") == (
        "taiwan semiconductor manufacturing",
        "Taiwan Semiconductor Manufacturing",
    )


def test_a_trailing_connector_is_dropped_with_the_suffix():
    """ "The Bank of Company" is not a thing; "Bank of" as a node would be."""
    assert canonical("Bank of Company")[0] == "bank"


# --------------------------------------------------------------------------
# discovering organisations
# --------------------------------------------------------------------------


def org_chunks(*texts: str) -> list:
    return [make_chunk(t, accn=f"{i:010d}-00-000000") for i, t in enumerate(texts)]


def test_a_name_two_filings_spell_out_becomes_a_node():
    got = discover_organizations(
        org_chunks(
            "We buy wafers from Taiwan Semiconductor Manufacturing Company.",
            "Our foundry is Taiwan Semiconductor Manufacturing Company.",
        )
    )
    assert got == {"taiwan semiconductor manufacturing": "Taiwan Semiconductor Manufacturing"}


def test_a_name_only_one_filing_spells_out_does_not():
    assert (
        discover_organizations(
            org_chunks("We buy wafers from Taiwan Semiconductor Manufacturing Company.")
        )
        == {}
    )


def test_the_threshold_counts_filings_and_not_mentions():
    """Forty repetitions inside one document is one company's word for something."""
    text = "Taiwan Semiconductor Manufacturing Company. " * 40
    assert discover_organizations([make_chunk(text)]) == {}


def test_min_org_filings_is_the_threshold_the_tests_assume():
    assert MIN_ORG_FILINGS == 2


def test_a_heading_whose_words_the_text_also_lower_cases_is_not_a_company():
    """ "Liquidity and Capital Resources" ends in a suffix and is not a company.

    The corpus settles it: the same filings write "liquidity", "capital" and
    "and" in ordinary prose, so the capitalised run is a heading.
    """
    body = (
        "Liquidity and Capital Resources\n"
        "Our liquidity depends on capital markets and on operating cash flow."
    )
    assert "liquidity and capital" not in discover_organizations(org_chunks(body, body))


def test_an_initialism_is_not_a_name():
    """ "U.S. Bank" canonicalises to "u.s", which then matches every "U.S." in the corpus."""
    text = "U.S. Bank provides our revolving credit facility."
    assert discover_organizations(org_chunks(text, text)) == {}


def test_a_capitalised_run_does_not_cross_a_newline():
    """A heading sits on its own line. "Strategy\\nOur Company" is not a company."""
    text = "Strategy\nWestern Digital Corporation is a supplier."
    got = discover_organizations(org_chunks(text, text))
    assert got == {"western digital": "Western Digital"}


# --------------------------------------------------------------------------
# mentions
# --------------------------------------------------------------------------

GAZ = {
    "advanced micro devices": ("amd", "company"),
    "advanced": ("advanced-energy", "company"),
    "intel": ("intel", "company"),
    "nvidia": ("nvidia", "company"),
    "xilinx": ("xilinx", "company"),
    "microsoft": ("microsoft", "company"),
    "food and drug administration": ("food and drug administration", "agency"),
    "fda": ("food and drug administration", "agency"),
}


def test_the_longest_matching_name_wins():
    """ "Advanced Micro Devices" is not a mention of a company called "Advanced"."""
    [m] = mentions("Advanced Micro Devices ships accelerators.", GAZ)
    assert m.key == "amd"


def test_a_mention_reports_where_it_sits_in_the_sentence():
    sentence = "We compete with Intel in data centres."
    [m] = mentions(sentence, GAZ)
    assert sentence[m.start : m.end] == "Intel"


def test_the_same_company_twice_in_a_row_is_one_mention():
    """Otherwise a self-edge, from a sentence that names one company emphatically."""
    assert [m.key for m in mentions("NVIDIA Corporation and NVIDIA are the same.", GAZ)] == [
        "nvidia"
    ]


def test_a_run_broken_by_a_newline_still_finds_the_name_after_it():
    got = mentions("Our Strategy\nMicrosoft Corporation supplies us.", GAZ)
    assert [m.key for m in got] == ["microsoft"]


def test_an_agency_alias_and_its_full_name_are_the_same_node():
    assert AGENCIES["fda"] == AGENCIES["food and drug administration"]
    a = mentions("The FDA cleared it.", GAZ)
    b = mentions("The Food and Drug Administration cleared it.", GAZ)
    assert [m.key for m in a] == [m.key for m in b] == ["food and drug administration"]


# --------------------------------------------------------------------------
# the cue window
# --------------------------------------------------------------------------

CUE_SENTENCE = (
    "Broadcom acquired VMware, and Intel competes with Advanced Micro Devices in servers."
)
CUE_MENTIONS = (
    Mention("broadcom", "Broadcom", "company", 0, 8),
    Mention("vmware", "VMware", "company", 18, 24),
    Mention("intel", "Intel", "company", 30, 35),
    Mention("amd", "Advanced Micro Devices", "company", 49, 71),
)


def test_a_cue_window_stops_before_the_next_mention():
    """The verb that belongs to (Intel, AMD) is not evidence about (Broadcom, VMware)."""
    span = cue_span(CUE_SENTENCE, CUE_MENTIONS, 0)
    assert "acquired" in span
    assert "competes" not in span


def test_a_cue_window_starts_after_the_previous_one():
    span = cue_span(CUE_SENTENCE, CUE_MENTIONS, 2)
    assert "competes" in span
    assert "acquired" not in span


# --------------------------------------------------------------------------
# relations
# --------------------------------------------------------------------------

FILER = Mention(key="amd", name="Advanced Micro Devices", kind="company", start=-1, end=-1)


def extract_one(text: str, *, filer: Mention | None = FILER) -> list[Edge]:
    return extract(make_chunk(text), GAZ, filer)


def test_the_filer_is_inserted_where_a_sentence_says_we():
    """Filings write "we acquired Xilinx", never "AMD acquired Xilinx"."""
    [edge] = extract_one("In 2022 we completed our acquisition of Xilinx, Inc. for stock.")
    assert (edge.source, edge.kind, edge.target) == ("amd", "acquired", "xilinx")


def test_a_sentence_with_no_first_person_and_one_name_yields_nothing():
    assert extract_one("The acquisition of Xilinx closed during the fiscal year 2022.") == []


def test_the_specific_relational_noun_beats_the_general_one():
    """ "Our primary competitor in the supply of microprocessors is Intel" is competition.

    Both cues are in the sentence. Ordering ``competes_with`` above ``supplies``
    is what stops every competitor from also being recorded as a vendor.
    """
    [edge] = extract_one("Our primary competitor in the supply of microprocessors is Intel.")
    assert edge.kind == "competes_with"


def test_a_symmetric_edge_is_stored_in_one_direction_only():
    """One fact is one edge, whichever order the sentence happened to use."""
    [edge] = extract_one("We compete with Intel across the whole of our product portfolio.")
    assert (edge.source, edge.target) == ("amd", "intel")
    [other] = extract(
        make_chunk("Intel competes with us across the whole of our product portfolio."),
        GAZ,
        FILER,
    )
    assert (other.source, other.target) == ("amd", "intel")


def test_the_passive_voice_reverses_the_direction():
    text = "The acquisition of Xilinx, Inc. was completed by Advanced Micro Devices, Inc."
    [edge] = extract(make_chunk(text), GAZ, None)
    assert (edge.source, edge.kind, edge.target) == ("amd", "acquired", "xilinx")


def test_a_regulator_needs_a_regulatory_cue():
    [edge] = extract_one("We received clearance from the Food and Drug Administration in May.")
    assert edge.kind == "regulated_by"
    assert edge.target_kind == "agency"


def test_a_regulator_in_a_sentence_about_something_else_is_not_a_supplier():
    """ "Sales to the FDA" is a sentence that mentioned a regulator, not a supply chain."""
    assert extract_one("We record sales to the Food and Drug Administration as revenue.") == []


def test_a_sentence_too_short_to_be_a_claim_is_skipped():
    assert extract_one("We compete with Intel.") == []


def test_a_sentence_cut_off_by_the_chunk_boundary_is_skipped():
    """The overlap means the whole sentence is in the next chunk; only the half is lost."""
    assert extract_one("We rely on Intel for the supply of chipsets used in our platform") == []


def test_an_edge_carries_the_sentence_and_absolute_offsets():
    """An edge you cannot read back to the filing is a confident-looking way to be wrong."""
    text = "We compete with Intel across the whole of our product portfolio."
    [edge] = extract(make_chunk(text, char_start=5000), GAZ, FILER)
    assert edge.sentence == text
    assert (edge.char_start, edge.char_end) == (5000, 5000 + len(text))
    assert edge.item_key == "1A"


def test_the_sentence_on_an_edge_has_its_whitespace_flattened():
    """A citation is one line. The parser leaves runs of spaces where a table was."""
    text = "We  compete   with Intel\tacross the whole  of our product portfolio."
    [edge] = extract(make_chunk(text), GAZ, FILER)
    assert edge.sentence == "We compete with Intel across the whole of our product portfolio."


# --------------------------------------------------------------------------
# storage and the graph view
# --------------------------------------------------------------------------


def an_edge(source="amd", target="intel", kind="competes_with", **kw) -> Edge:
    fields = dict(
        source=source,
        target=target,
        kind=kind,
        source_kind="company",
        target_kind="company",
        ticker="AMD",
        accn="0000000000-00-000000",
        form="10-K",
        period_end="2024-12-28",
        item_key="1A",
        char_start=10,
        char_end=60,
        sentence="We compete with Intel.",
    )
    fields.update(kw)
    return Edge(**fields)


def test_edges_round_trip_through_parquet_unchanged(tmp_path):
    store = GraphStore(tmp_path)
    assert store.exists is False
    store.write([an_edge(), an_edge(target="nvidia")])
    assert GraphStore(tmp_path).edges() == [an_edge(), an_edge(target="nvidia")]


def test_an_absent_store_reads_as_no_edges_rather_than_an_error(tmp_path):
    assert GraphStore(tmp_path).edges() == []


def test_neighbours_looks_both_ways():
    """ "Who did AMD acquire" and "who acquired AMD" are the same question to ask."""
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_edge("amd", "xilinx", kind="acquired", period_end="2022-12-31")
    g.add_edge("intel", "amd", kind="competes_with", period_end="2024-12-28")
    got = {(n["other"], n["direction"]) for n in neighbours(g, "amd")}
    assert got == {("xilinx", "out"), ("intel", "in")}


def test_neighbours_can_be_asked_for_one_kind_of_edge():
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_edge("amd", "xilinx", kind="acquired", period_end="2022-12-31")
    g.add_edge("amd", "intel", kind="competes_with", period_end="2024-12-28")
    got = neighbours(g, "amd", kinds=["acquired"])
    assert [n["other"] for n in got] == ["xilinx"]
