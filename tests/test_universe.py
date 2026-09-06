"""universe.yaml is the declaration of scope, so it is validated strictly.

Every test here is about failing loudly. A universe that loads with a silent
hole in it produces a corpus that is wrong in exactly one place and complete
everywhere anyone counts.
"""

from __future__ import annotations

import pytest

from filing.config import settings
from filing.ingest.universe import Universe, UniverseError, load_universe

YAML = """
window:
  period_end_from: "2020-01-01"
  period_end_to: "2025-02-28"
forms: [10-K, 10-Q]
sectors:
  semis:
    - { ticker: NVDA, name: NVIDIA Corporation }
    - { ticker: amd,  name: Advanced Micro Devices }
  retail:
    - { ticker: WMT,  name: Walmart Inc. }
"""


def write(tmp_path, text):
    path = tmp_path / "universe.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_and_normalises(tmp_path):
    uni = load_universe(write(tmp_path, YAML))
    assert [c.ticker for c in uni.companies] == ["NVDA", "AMD", "WMT"]
    assert uni.forms == frozenset({"10-K", "10-Q"})
    assert uni.period_end_from == "2020-01-01"
    assert set(uni.sectors) == {"semis", "retail"}
    assert len(uni.sectors["semis"]) == 2


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(UniverseError, match="no universe file"):
        load_universe(tmp_path / "nope.yaml")


WINDOW = 'window: {period_end_from: "2020-01-01", period_end_to: "2025-01-01"}'
FORMS = "forms: [10-K]"
SECTORS = "sectors:\n  s:\n    - { ticker: NVDA }"


@pytest.mark.parametrize(
    ("section", "text"),
    [
        ("window", f"{FORMS}\n{SECTORS}\n"),
        ("forms", f"{WINDOW}\n{SECTORS}\n"),
        ("sectors", f"{WINDOW}\n{FORMS}\n"),
    ],
)
def test_each_section_is_required(tmp_path, section, text):
    with pytest.raises(UniverseError, match=section):
        load_universe(write(tmp_path, text))


def test_duplicate_ticker_is_an_error(tmp_path):
    """A duplicate would double every count it appears in, quietly."""
    doubled = YAML.replace("- { ticker: WMT,", "- { ticker: nvda,")
    with pytest.raises(UniverseError, match="NVDA appears twice"):
        load_universe(write(tmp_path, doubled))


def test_resolve_attaches_ciks(tmp_path):
    uni = load_universe(write(tmp_path, YAML))
    resolved = uni.resolved(
        {
            "NVDA": ("0001045810", "NVIDIA CORP"),
            "AMD": ("0000002488", "ADVANCED MICRO DEVICES INC"),
            "WMT": ("0000104169", "Walmart Inc."),
        }
    )
    assert [c.cik for c in resolved.companies] == ["0001045810", "0000002488", "0000104169"]
    # The declared name survives resolution; EDGAR's own name is stored separately.
    assert resolved.companies[0].name == "NVIDIA Corporation"
    assert resolved.forms == uni.forms


def test_resolve_reports_every_failure_at_once(tmp_path):
    """Not the first one. Nineteen of twenty is the dangerous outcome."""
    uni = load_universe(write(tmp_path, YAML))
    with pytest.raises(UniverseError) as exc:
        uni.resolved({"NVDA": ("0001045810", "NVIDIA CORP")})
    message = str(exc.value)
    assert "AMD" in message and "WMT" in message
    assert "2 ticker(s)" in message


def test_committed_universe_is_the_shape_m1_asks_for():
    """Twenty companies, four sectors of five -- so peers exist to compare."""
    uni = load_universe(cfg=settings())
    assert len(uni.companies) == 20
    assert len(uni.sectors) == 4
    assert {len(v) for v in uni.sectors.values()} == {5}
    assert uni.forms == frozenset({"10-K", "10-Q"})


def test_universe_is_frozen():
    assert isinstance(load_universe(cfg=settings()), Universe)
    with pytest.raises((AttributeError, TypeError)):
        load_universe(cfg=settings()).companies = ()  # type: ignore[misc]


def test_a_declared_cik_is_never_looked_up(tmp_path):
    """SEC's ticker map points at whoever trades under the ticker today.

    After a holding-company reorganisation that is a brand-new registrant with
    no filing history, while the old one keeps the 10-Ks and loses the ticker.
    Nothing errors -- the corpus is just silently short one company. See XOM.
    """
    pinned = YAML.replace(
        "- { ticker: WMT,  name: Walmart Inc. }",
        '- { ticker: WMT, name: Walmart Inc., cik: "104169" }',
    )
    uni = load_universe(write(tmp_path, pinned))
    assert uni.companies[-1].cik == "0000104169"  # zero-padded on the way in

    resolved = uni.resolved(
        {
            "NVDA": ("0001045810", "NVIDIA CORP"),
            "AMD": ("0000002488", "AMD"),
            "WMT": ("0009999999", "SOME NEW HOLDCO"),  # what the map would say
        }
    )
    assert resolved.companies[-1].cik == "0000104169"


def test_a_declared_cik_survives_a_ticker_the_map_has_never_heard_of(tmp_path):
    pinned = YAML.replace(
        "- { ticker: WMT,  name: Walmart Inc. }",
        '- { ticker: WMT, name: Walmart Inc., cik: "0000104169" }',
    )
    uni = load_universe(write(tmp_path, pinned))
    resolved = uni.resolved({"NVDA": ("0001045810", "NVIDIA CORP"), "AMD": ("0000002488", "AMD")})
    assert [c.ticker for c in resolved.companies] == ["NVDA", "AMD", "WMT"]


def test_the_committed_universe_pins_xom():
    """A regression guard on a real, silent data loss -- not a style preference."""
    uni = load_universe(cfg=settings())
    xom = next(c for c in uni.companies if c.ticker == "XOM")
    assert xom.cik == "0000034088"
