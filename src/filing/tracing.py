"""OpenTelemetry wiring for Phoenix.

Tracing is on from the first commit, not bolted on at M7. Every later gate --
router accuracy, the failure taxonomy, the trace exported to docs/ -- reads
spans that this module starts producing on day one.

If the collector is not running, this degrades to no-op spans and says so once.
A missing Phoenix must never fail a pipeline run.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

from filing.config import Settings, settings

log = logging.getLogger(__name__)

_configured = False
_live = False
_counter: SpanCounter | None = None
_provider = None


class SpanCounter(SpanProcessor):
    """Counts finished spans in-process.

    Exists so ``filing smoke`` can assert the M0 gate ("a trace with >= 3 spans")
    without scraping the Phoenix UI. Subclassing the SDK's SpanProcessor rather
    than duck-typing it matters: the SDK calls private hooks (``_on_ending``)
    that only the base class supplies.
    """

    def __init__(self) -> None:
        self.count = 0
        self.names: list[str] = []

    def on_end(self, span) -> None:  # noqa: ANN001
        self.count += 1
        self.names.append(span.name)


def _attach_counter(provider, counter: SpanCounter) -> None:
    """Add the counter *without* evicting the exporter.

    phoenix.otel's TracerProvider overrides add_span_processor with
    ``replace_default_processor=True`` as the default, so the obvious call
    shuts down the OTLP exporter it just installed: spans are then counted
    in-process and never reach the collector. Costly to notice, one keyword
    to fix.
    """
    try:
        provider.add_span_processor(counter, replace_default_processor=False)
    except TypeError:
        # A plain SDK TracerProvider (no Phoenix) has no such keyword.
        provider.add_span_processor(counter)


def span_count() -> int:
    return _counter.count if _counter else 0


def span_names() -> list[str]:
    return list(_counter.names) if _counter else []


def setup_tracing(cfg: Settings | None = None) -> bool:
    """Idempotent. Returns True if spans are actually being exported."""
    global _configured, _live, _counter
    if _configured:
        return _live
    _configured = True

    cfg = cfg or settings()
    if not cfg.tracing_enabled:
        log.info("tracing disabled by config")
        return False
    _counter = SpanCounter()

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register

        provider = register(
            project_name=cfg.phoenix_project,
            endpoint=f"{cfg.phoenix_endpoint.rstrip('/')}/v1/traces",
            auto_instrument=False,
            batch=True,
            set_global_tracer_provider=True,
            verbose=False,  # its banner is stdout noise on every command
        )
        _attach_counter(provider, _counter)
        global _provider
        _provider = provider
        # Instruments the OpenAI SDK, which is how chat and embed reach NIM.
        # rerank is a raw httpx call, so nvidia.py spans it by hand.
        OpenAIInstrumentor().instrument(tracer_provider=provider)
        _live = True
        log.info("tracing -> %s (project=%s)", cfg.phoenix_endpoint, cfg.phoenix_project)
    except Exception as exc:  # noqa: BLE001 - never let observability break the run
        log.warning("tracing unavailable (%s); continuing without spans", exc)
        _live = False
    return _live


def get_tracer(name: str = "filing"):
    """Tracer from *our* provider, falling back to the global one.

    Not the same thing: OpenTelemetry allows the global provider to be set only
    once per process, so anything that registers before us -- another library,
    an earlier test -- would otherwise silently swallow every span we emit.
    """
    if _provider is not None:
        return _provider.get_tracer(name)
    return trace.get_tracer(name)


def flush_tracing(timeout_ms: int = 10_000) -> bool:
    """Push batched spans before the process exits.

    Without this a short-lived command exits before the batch processor's timer
    fires and the spans are simply lost -- the trace looks like it never
    happened. Every CLI entry point calls this on the way out.
    """
    if _provider is None:
        return False
    try:
        return bool(_provider.force_flush(timeout_ms))
    except Exception as exc:  # noqa: BLE001
        log.warning("span flush failed: %s", exc)
        return False


def tracing_is_live() -> bool:
    return _live


def reset_tracing() -> None:
    """Tear down the module state so a test can configure tracing again.

    Setup is deliberately once-per-process; tests need it more than once.
    """
    global _configured, _live, _counter, _provider
    if _provider is not None:
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().uninstrument()
        except Exception:  # noqa: BLE001
            pass
        try:
            _provider.shutdown()
        except Exception:  # noqa: BLE001
            pass
    _configured = False
    _live = False
    _counter = None
    _provider = None
