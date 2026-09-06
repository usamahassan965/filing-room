"""Tracing has to reach the collector, not merely happen.

The bug these tests exist for was invisible from inside the process: spans were
created, counted, and reported as fine, while the exporter had been shut down
and nothing ever left the machine. So one test inspects the processor chain and
one watches the socket. Neither needs Docker or an API key.
"""

from __future__ import annotations

import http.server
import threading

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from filing.config import Settings
from filing.tracing import (
    SpanCounter,
    flush_tracing,
    get_tracer,
    reset_tracing,
    setup_tracing,
    span_count,
)


class _Collector(http.server.BaseHTTPRequestHandler):
    """Speaks just enough OTLP/HTTP to record what arrived."""

    posts: list[tuple[str, int]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).posts.append((self.path, len(body)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def collector():
    """A throwaway OTLP endpoint on an ephemeral port."""
    _Collector.posts = []
    _Collector.protocol_version = "HTTP/1.1"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", _Collector.posts
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture(autouse=True)
def _clean_tracing():
    reset_tracing()
    yield
    reset_tracing()


def _cfg(endpoint: str, **kw) -> Settings:
    return Settings(
        nvidia_api_key="test-key",
        phoenix_endpoint=endpoint,
        phoenix_project="filing-room-tests",
        **kw,
    )


def _emit(names: tuple[str, ...] = ("llm.chat", "llm.embed", "llm.rerank")) -> None:
    tracer = get_tracer("filing.tests")
    for name in names:
        with tracer.start_as_current_span(name) as span:
            span.set_attribute("llm.model_name", "test-stub")


def test_counter_does_not_evict_the_exporter(collector):
    """phoenix.otel's add_span_processor replaces the default unless told not to.

    Adding the span counter the obvious way shut down the OTLP exporter, which
    is silent: spans still get created and counted. Assert both processors
    survive.
    """
    endpoint, _ = collector
    assert setup_tracing(_cfg(endpoint)) is True

    from filing import tracing

    processors = tracing._provider._active_span_processor._span_processors  # noqa: SLF001
    assert any(isinstance(p, SpanCounter) for p in processors), "counter missing"
    assert any(isinstance(p, BatchSpanProcessor) for p in processors), "exporter evicted"


def test_spans_reach_the_collector_after_flush(collector):
    """The M0 gate, asserted on the wire instead of in the Phoenix UI."""
    endpoint, posts = collector
    assert setup_tracing(_cfg(endpoint)) is True
    _emit()

    assert span_count() >= 3
    assert flush_tracing() is True
    assert posts, "no OTLP request reached the collector"
    path, size = posts[0]
    assert path == "/v1/traces"
    assert size > 0, "empty export body"


def test_flush_is_what_makes_the_spans_survive(collector):
    """Justifies flush_tracing existing at all.

    A short-lived CLI process exits long before the batch timer fires, so
    without the explicit flush the spans are simply lost.
    """
    endpoint, posts = collector
    setup_tracing(_cfg(endpoint))
    _emit()
    assert posts == [], "batch processor exported without being asked to"
    flush_tracing()
    assert posts != []


def test_disabled_tracing_never_exports(collector):
    endpoint, posts = collector
    assert setup_tracing(_cfg(endpoint, tracing_enabled=False)) is False
    _emit()
    assert flush_tracing() is False
    assert posts == []


def test_unreachable_collector_does_not_break_the_run():
    """A missing Phoenix degrades to no-op spans; it must never raise."""
    assert setup_tracing(_cfg("http://127.0.0.1:9")) is True
    _emit()  # would raise if the tracer were broken rather than merely unheard
    flush_tracing(timeout_ms=1_000)
