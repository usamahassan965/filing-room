"""Structured stores built *from* the corpus -- derived, rebuildable, disposable.

``data/manifest.duckdb`` is irreplaceable: losing it costs a 1.12 GB re-download
from a rate-limited government host. ``data/facts.duckdb`` is not: it is a pure
function of the JSON already on disk, and deleting it costs about a minute. The
two live in separate files so that distinction is visible in ``ls`` rather than
buried in a comment, and so a rebuild can drop and recreate everything it owns
without ever holding a write handle on the manifest.
"""

from filing.stores.facts import BuildReport, FactsStore, build_facts

__all__ = ["BuildReport", "FactsStore", "build_facts"]
