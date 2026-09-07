"""Reading the failure taxonomy back out of Phoenix.

`tracing.py` is the write side: every `agent.verify` span carries the tags the
verifier assigned. This is the read side, and it exists because M6 asks for
something stricter than "the tags are recorded" -- it asks that the failure
gallery be *a filter, not a manual read*. The difference is whether the counts
below come from a server-side query or from downloading every span and
grepping it locally, and only one of those is a gallery.

The path syntax cost an hour and is the whole reason this module is not two
lines inline in the CLI. Phoenix stores an OTel attribute named
``filing.failure.kinds`` un-flattened, as nested JSON under ``filing``, and its
filter language resolves *only* the bracket-chained form::

    attributes["filing"]["failure"]["kinds"]     # 70 spans
    attributes["filing.failure.kinds"]           # 0
    attributes.filing.failure.kinds              # 0

The two wrong forms do not raise. They parse, run, and return an empty result,
which reads exactly like "nothing ever failed" -- the most flattering possible
way for an observability query to be broken. That is why `counts` cross-checks
against a tag-agnostic total and `filing failures` prints both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from filing.agent.verify import TAXONOMY
from filing.config import Settings, settings

log = logging.getLogger(__name__)

#: The attribute the filters index into. Bracket-chained -- see the module note.
KINDS = 'attributes["filing"]["failure"]["kinds"]'
JOINED = 'attributes["filing"]["failure"]["kind"]'

#: The span the verifier writes to. Everything here is scoped to it.
SPAN = "agent.verify"


def filter_for(kind: str) -> str:
    """The expression to paste into the Phoenix UI's filter box for one tag."""
    return f"{kind!r} in {KINDS}"


#: Every question the gallery can be asked, as the filter that answers it.
QUERIES: dict[str, str] = {
    "verified": f'name == {SPAN!r} and attributes["filing"]["verify"]["enabled"] == True',
    "flagged": 'attributes["filing"]["verify"]["ok"] == False',
    "blocked": 'attributes["filing"]["verify"]["blocked"] == True',
    "tagged": f"{JOINED} != ''",
    **{kind: filter_for(kind) for kind in TAXONOMY},
}


@dataclass
class Gallery:
    """What a filter run found, plus enough to tell empty from broken."""

    project: str
    endpoint: str
    counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def consistent(self) -> bool:
        """Do the per-tag filters account for every span the corpus tagged?

        A span can carry several tags, so the per-tag counts sum to at least
        the tagged total, never less. Fewer means a filter silently matched
        nothing -- the failure mode this module exists to make visible.
        """
        tagged = self.counts.get("tagged", 0)
        return sum(self.counts.get(k, 0) for k in TAXONOMY) >= tagged > 0


def _client(cfg: Settings):  # noqa: ANN202
    from phoenix.client import Client

    return Client(base_url=cfg.phoenix_endpoint.rstrip("/"))


def count(expr: str, *, cfg: Settings | None = None, timeout: int = 180) -> int:
    """How many spans match one filter expression, counted by the server."""
    cfg = cfg or settings()
    from phoenix.trace.dsl import SpanQuery

    df = _client(cfg).spans.get_spans_dataframe(
        project_identifier=cfg.phoenix_project,
        query=SpanQuery().where(expr),
        limit=100_000,
        timeout=timeout,
    )
    return len(df)


def examples(kind: str, *, limit: int = 5, cfg: Settings | None = None) -> list[dict[str, Any]]:
    """A few spans carrying one tag, newest first, for eyeballing a trace."""
    cfg = cfg or settings()
    from phoenix.trace.dsl import SpanQuery

    df = _client(cfg).spans.get_spans_dataframe(
        project_identifier=cfg.phoenix_project,
        query=SpanQuery().where(filter_for(kind)),
        limit=max(limit, 1),
        timeout=180,
    )
    if df.empty:
        return []
    df = df.sort_values("start_time", ascending=False).head(limit)
    out = []
    for _, row in df.iterrows():
        attrs = row.get("attributes.filing") or {}
        verify = attrs.get("verify", {}) if isinstance(attrs, dict) else {}
        out.append(
            {
                "trace_id": row.get("context.trace_id", ""),
                "start_time": row.get("start_time"),
                "kinds": (attrs.get("failure", {}) or {}).get("kinds", []),
                "reason": verify.get("reason", ""),
                "mode": verify.get("mode", ""),
            }
        )
    return out


def collect(*, cfg: Settings | None = None) -> Gallery:
    """Run every filter in QUERIES. One bad filter does not lose the others."""
    cfg = cfg or settings()
    g = Gallery(project=cfg.phoenix_project, endpoint=cfg.phoenix_endpoint)
    for label, expr in QUERIES.items():
        try:
            g.counts[label] = count(expr, cfg=cfg)
        except Exception as exc:  # noqa: BLE001 -- a dead Phoenix is not a crash
            g.errors[label] = f"{type(exc).__name__}: {exc}"
            log.debug("gallery filter failed: %s", expr, exc_info=True)
    return g


def trace_url(trace_id: str, *, cfg: Settings | None = None) -> str:
    cfg = cfg or settings()
    base = cfg.phoenix_endpoint.rstrip("/")
    return f"{base}/projects/{cfg.phoenix_project}/traces/{trace_id}"
