"""The manifest is the source of truth about the corpus, so it is tested as one.

The behaviours that matter are the ones idempotency rests on: upserts that do
not duplicate, paths that survive the project being moved, and a ``has_``
predicate that answers "do I need to fetch this?" rather than "is there a row?".
"""

from __future__ import annotations

import pytest

from filing.ingest.manifest import Manifest, sha256_bytes

FILING = dict(
    accn="0001045810-25-000023",
    cik="0001045810",
    ticker="NVDA",
    form="10-K",
    fy=2025,
    filed_date="2025-02-26",
    period_end="2025-01-26",
    primary_doc="nvda-20250126.htm",
    bytes=1234,
    sha256="deadbeef",
)


@pytest.fixture
def manifest(tmp_path):
    with Manifest(tmp_path / "m.duckdb", tmp_path) as m:
        yield m


def landed(manifest, rel: str, blob: bytes = b"x") -> str:
    """Put bytes on disk where a manifest row would point, and return the path."""
    dest = manifest.data_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return rel


def test_sha256_is_of_the_bytes_as_served():
    assert sha256_bytes(b"") == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_paths_are_stored_relative_to_data_dir(manifest):
    absolute = manifest.data_dir / "raw" / "filings" / "x.htm"
    rel = manifest.relative(absolute)
    assert rel == "raw/filings/x.htm"  # posix, so it survives crossing platforms
    assert manifest.resolve(rel) == absolute


def test_upsert_filing_is_idempotent(manifest):
    """Re-recording the same accession updates one row; it never adds a second."""
    manifest.upsert_filing(path=landed(manifest, "a.htm"), **FILING)
    manifest.upsert_filing(path=landed(manifest, "a.htm"), **{**FILING, "bytes": 999})
    stats = manifest.stats()
    assert stats.filings == 1
    assert stats.filing_bytes == 999


def test_has_filing_needs_both_the_row_and_the_bytes(manifest):
    """Two failure modes -- an unfinished download and a cleaned-out data/ --
    and the caller should not have to care which one happened."""
    assert not manifest.has_filing(FILING["accn"])  # no row at all

    rel = landed(manifest, "a.htm")
    manifest.upsert_filing(path=rel, **FILING)
    assert manifest.has_filing(FILING["accn"])

    (manifest.data_dir / rel).unlink()
    assert not manifest.has_filing(FILING["accn"])


def test_missing_files_reports_rows_whose_bytes_are_gone(manifest):
    manifest.upsert_filing(path=landed(manifest, "a.htm"), **FILING)
    manifest.upsert_facts(
        cik="0001045810",
        ticker="NVDA",
        path=landed(manifest, "f.json"),
        bytes=10,
        sha256="x",
        n_concepts=500,
    )
    assert manifest.missing_files() == []

    (manifest.data_dir / "a.htm").unlink()
    (manifest.data_dir / "f.json").unlink()
    gone = dict(manifest.missing_files())
    assert gone == {FILING["accn"]: "a.htm", "0001045810": "f.json"}


def test_stats_counts_by_form_and_distinct_companies(manifest):
    manifest.upsert_company(cik="0001045810", ticker="NVDA", name="NVIDIA", sector="semis")
    for i, form in enumerate(["10-K", "10-Q", "10-Q"]):
        manifest.upsert_filing(
            path=landed(manifest, f"{i}.htm"),
            **{**FILING, "accn": f"accn-{i}", "form": form, "bytes": 100},
        )
    stats = manifest.stats()
    assert stats.by_form == {"10-K": 1, "10-Q": 2}
    assert stats.filings == 3
    assert stats.companies_with_filings == 1
    assert stats.filing_bytes == 300


def test_per_company_left_joins_so_an_empty_company_still_shows(manifest):
    """A company with zero filings is exactly what the gate needs to see."""
    manifest.upsert_company(cik="0000104169", ticker="WMT", name="Walmart", sector="retail")
    ((sector, ticker, _cik, annual, quarterly, nbytes, earliest, latest, facts),) = (
        manifest.per_company()
    )
    assert (sector, ticker) == ("retail", "WMT")
    assert (annual, quarterly, nbytes, earliest, latest, facts) == (0, 0, 0, None, None, 0)


def test_reopening_the_file_keeps_the_rows(tmp_path):
    """Resume across process boundaries is the entire point of the manifest."""
    path = tmp_path / "m.duckdb"
    with Manifest(path, tmp_path) as m:
        m.upsert_filing(path=landed(m, "a.htm"), **FILING)
    with Manifest(path, tmp_path) as m:
        assert m.has_filing(FILING["accn"])
        assert m.stats().filings == 1


def test_prune_drops_rows_for_ciks_the_universe_no_longer_declares(manifest):
    """A repointed company leaves rows under its old CIK behind.

    Pinning XOM's real filing entity changed its CIK, and without this the
    manifest would report 21 companies for a 20-company universe.
    """
    manifest.upsert_company(cik="0000034088", ticker="XOM", name="Exxon", sector="oil")
    manifest.upsert_company(cik="0002115436", ticker="XOM", name="Exxon holdco", sector="oil")
    manifest.upsert_facts(
        cik="0002115436",
        ticker="XOM",
        path=landed(manifest, "stale.json"),
        bytes=1,
        sha256="x",
        n_concepts=1,
    )
    manifest.upsert_filing(path=landed(manifest, "keep.htm"), **{**FILING, "cik": "0000034088"})

    assert manifest.prune({"0000034088"}) == 2
    assert manifest.stats().companies == 1
    assert manifest.stats().filings == 1


def test_prune_leaves_the_bytes_alone(manifest):
    """Deleting files on the strength of a YAML edit is not a trade worth making."""
    manifest.upsert_facts(
        cik="0002115436",
        ticker="XOM",
        path=landed(manifest, "stale.json"),
        bytes=1,
        sha256="x",
        n_concepts=1,
    )
    manifest.prune({"0000034088"})
    assert (manifest.data_dir / "stale.json").exists()


def test_prune_of_an_already_clean_manifest_is_a_no_op(manifest):
    """It runs on every ingest, so it has to be stable -- the gate re-runs one."""
    manifest.upsert_filing(path=landed(manifest, "a.htm"), **FILING)
    assert manifest.prune({FILING["cik"]}) == 0
    assert manifest.prune({FILING["cik"]}) == 0
    assert manifest.stats().filings == 1
