"""Logging, tracing and metrics.

* loguru: pretty stderr + JSONL file sink (``logs/jobhunter.jsonl``) for grep/jq/ingestion.
* OpenTelemetry: spans per agent node / LLM call / scrape. OTLP export when
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (Jaeger, Tempo, Honeycomb, Langfuse...), otherwise no-op.
* LangSmith: LangGraph traces automatically when ``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY``.
* Counters (tokens, parse failures, anti-bot hits) are also persisted to SQLite so ``jobhunter stats``
  works without any external backend.
"""

from __future__ import annotations

import functools
import inspect
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from loguru import logger
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

if TYPE_CHECKING:
    from loguru import Logger

P = ParamSpec("P")
R = TypeVar("R")

# Proxy tracer/meter: they forward to the real providers once setup_telemetry() installs them.
_tracer = trace.get_tracer("jobhunter")
_meter = metrics.get_meter("jobhunter")
tokens_counter = _meter.create_counter("llm.tokens", unit="token")
llm_latency = _meter.create_histogram("llm.latency", unit="ms")
parse_failures = _meter.create_counter("llm.parse_failures")
llm_failures = _meter.create_counter("llm.failures")  # attrs: provider, model, task, kind
antibot_counter = _meter.create_counter("scraper.antibot_events")
node_latency = _meter.create_histogram("agent.node.latency", unit="ms")


def setup_telemetry(
    level: str = "INFO", otel_endpoint: str | None = None, log_dir: Path = Path("logs")
) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> "
        "<cyan>{extra[component]}</cyan> {message}",
    )
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "jobhunter.jsonl", level="DEBUG", serialize=True, rotation="20 MB", retention=10)
    logger.configure(extra={"component": "core"})

    resource = Resource.create({"service.name": "jobhunter-agent"})
    provider = TracerProvider(resource=resource)
    readers = []
    if otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            logger.info("OTLP export enabled -> {}", otel_endpoint)
        except ImportError:
            logger.warning("OTEL endpoint set but exporter missing: pip install 'jobhunter-agent[otlp]'")
    trace.set_tracer_provider(provider)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))


def get_logger(component: str) -> Logger:
    return logger.bind(component=component)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, str | int | float | bool) else str(v))
        yield s


def traced_node(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a LangGraph node: OTel span + latency histogram + structured start/end logs."""

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        log = get_logger(f"node.{name}")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                t0 = time.perf_counter()
                with span(f"node.{name}"):
                    try:
                        result: R = await fn(*args, **kwargs)
                    except Exception:
                        log.exception("node failed")
                        raise
                ms = (time.perf_counter() - t0) * 1000
                node_latency.record(ms, {"node": name})
                log.info("done in {:.0f} ms", ms)
                return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            with span(f"node.{name}"):
                result = fn(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            node_latency.record(ms, {"node": name})
            log.info("done in {:.0f} ms", ms)
            return result

        return wrapper

    return deco


class Progress:
    """Logs ``label i/N (elapsed, ~eta)`` every ``every`` steps so long stages never look frozen."""

    def __init__(self, logger: Any, label: str, total: int, every: int = 5) -> None:
        self.logger, self.label, self.total, self.every = logger, label, total, max(1, every)
        self.done = 0
        self.t0 = time.perf_counter()
        if total:
            logger.info("{}: 0/{} …", label, total)

    def step(self, detail: str = "") -> None:
        self.done += 1
        if self.done % self.every and self.done != self.total:
            return
        el = time.perf_counter() - self.t0
        eta = el / self.done * (self.total - self.done)
        self.logger.info(
            "{}: {}/{} ({:.0f}s, ~{:.0f}s left){}",
            self.label,
            self.done,
            self.total,
            el,
            eta,
            f" — {detail}" if detail else "",
        )
