"""Observability and distributed tracing for choreo-mini.

Three layers:

* **Event model** — typed dataclasses emitted at every significant framework
  boundary: agent calls, LLM HTTP requests (including retries), and episode
  steps.  Each event carries a ``trace_id`` (stable for the lifetime of a
  ``Workflow`` instance) and a ``span_id`` (unique per ``wf.send()`` call),
  enabling full call-graph reconstruction.

* **ObservabilityHook protocol** — implement ``on_event(event)`` to receive
  every event.  Hooks are synchronous and must not raise; exceptions are
  swallowed and printed to stderr so they never crash user code.

* **Built-in hooks**

  - :class:`StdoutHook` — coloured, human-readable console output.
  - :class:`JsonFileHook` — newline-delimited JSON (NDJSON) for log pipelines.
  - :class:`OTLPHook` — OpenTelemetry spans via OTLP/gRPC
    (requires ``pip install choreo-mini[otel]``).
  - :class:`CompositeHook` — fan-out to multiple hooks simultaneously.

Quick start::

    from choreo_mini.core.observability import StdoutHook, JsonFileHook, CompositeHook

    wf = MyWorkflow(observability=StdoutHook())

    # Or fan-out to two sinks:
    hook = CompositeHook(StdoutHook(), JsonFileHook("run.ndjson"))
    wf = MyWorkflow(observability=hook)

    # OpenTelemetry (Jaeger, Tempo, Honeycomb, …):
    from choreo_mini.core.observability import OTLPHook
    wf = MyWorkflow(observability=OTLPHook("my-service", endpoint="http://localhost:4317"))
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def new_trace_id() -> str:
    """Generate a 32-hex-char trace ID (128-bit, OTEL-compatible)."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """Generate a 16-hex-char span ID (64-bit, OTEL-compatible)."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

@dataclass
class ObservabilityEvent:
    """Base class for all choreo-mini observability events.

    Every event carries:

    * ``event_type`` — string discriminator (e.g. ``"agent_call_start"``).
    * ``timestamp`` — Unix time in seconds at the moment of emission.
    * ``trace_id`` — stable for the lifetime of a :class:`~choreo_mini.core.workflow.Workflow`
      instance; groups all spans belonging to one workflow run.
    * ``span_id`` — unique per :meth:`~choreo_mini.core.workflow.Workflow.send` call;
      matches the ``call_id`` on the returned :class:`~choreo_mini.core.llm.Message`.
    * ``parent_span_id`` — set when a call is nested inside an :class:`~choreo_mini.core.episode.Episode`
      step, enabling parent-child span relationships in OTEL.
    """

    event_type: str
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""


# -- Agent call events -------------------------------------------------------

@dataclass
class AgentCallStart(ObservabilityEvent):
    """Emitted immediately before an agent LLM call."""

    event_type: str = "agent_call_start"
    workflow_name: str = ""
    agent_name: str = ""
    prompt_preview: str = ""   # first 200 chars of the user turn


@dataclass
class AgentCallEnd(ObservabilityEvent):
    """Emitted after a successful agent LLM call."""

    event_type: str = "agent_call_end"
    workflow_name: str = ""
    agent_name: str = ""
    latency_s: float = 0.0
    memory_bytes: float = 0.0
    response_preview: str = ""  # first 200 chars of the response


@dataclass
class AgentCallError(ObservabilityEvent):
    """Emitted when an agent LLM call raises an exception."""

    event_type: str = "agent_call_error"
    workflow_name: str = ""
    agent_name: str = ""
    latency_s: float = 0.0
    error_type: str = ""
    error_message: str = ""


# -- LLM HTTP events ---------------------------------------------------------

@dataclass
class LLMRequestStart(ObservabilityEvent):
    """Emitted before each HTTP attempt to the LLM endpoint."""

    event_type: str = "llm_request_start"
    endpoint: str = ""
    model: str = ""
    attempt: int = 0           # 0-based; 0 = first attempt


@dataclass
class LLMRequestEnd(ObservabilityEvent):
    """Emitted after a successful (2xx) LLM HTTP response."""

    event_type: str = "llm_request_end"
    endpoint: str = ""
    model: str = ""
    attempt: int = 0
    status_code: int = 200
    latency_s: float = 0.0


@dataclass
class LLMRetry(ObservabilityEvent):
    """Emitted each time the LLM client backs off before retrying."""

    event_type: str = "llm_retry"
    endpoint: str = ""
    model: str = ""
    attempt: int = 0           # attempt that just failed (0-based)
    status_code: int = 0       # 0 for connection / timeout errors
    delay_s: float = 0.0
    reason: str = ""


# -- LLM pool events ---------------------------------------------------------

@dataclass
class LLMPoolRoute(ObservabilityEvent):
    """Emitted when an LLMPool selects a candidate to handle a call."""

    event_type: str = "llm_pool_route"
    pool_name: str = ""
    candidate_name: str = ""
    policy: str = ""
    candidate_index: int = 0        # position in the sorted/rotated order
    cost_per_1k_tokens: float = 0.0


@dataclass
class LLMPoolFallback(ObservabilityEvent):
    """Emitted when an LLMPool falls back from a failed candidate to the next."""

    event_type: str = "llm_pool_fallback"
    pool_name: str = ""
    failed_candidate: str = ""
    next_candidate: str = ""
    error_type: str = ""
    error_message: str = ""


# -- Episode events ----------------------------------------------------------

@dataclass
class EpisodeStepStart(ObservabilityEvent):
    """Emitted at the start of each episode round."""

    event_type: str = "episode_step_start"
    episode_id: str = ""
    round_number: int = 0
    agent_names: List[str] = field(default_factory=list)


@dataclass
class EpisodeStepEnd(ObservabilityEvent):
    """Emitted at the end of each episode round."""

    event_type: str = "episode_step_end"
    episode_id: str = ""
    round_number: int = 0
    actions: Dict[str, str] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    latency_s: float = 0.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ObservabilityHook(Protocol):
    """Implement this protocol to receive framework events.

    ``on_event`` is called synchronously on the thread that triggered the
    event.  Implementations must not raise — exceptions are caught by the
    framework and printed to stderr so they never crash user code.

    Example — minimal custom hook::

        class MyHook:
            def on_event(self, event):
                print(event.event_type, event.span_id[:8])
    """

    def on_event(self, event: ObservabilityEvent) -> None:
        ...


def _safe_emit(hook: Any, event: ObservabilityEvent) -> None:
    """Emit *event* to *hook*, swallowing any exception the hook raises."""
    try:
        hook.on_event(event)
    except Exception:  # noqa: BLE001
        print(
            f"[choreo-mini] ObservabilityHook.on_event raised an exception "
            f"(event={event.event_type!r}):",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------

class CompositeHook:
    """Fan-out events to multiple hooks simultaneously.

    Usage::

        hook = CompositeHook(StdoutHook(), JsonFileHook("run.ndjson"))
        wf = MyWorkflow(observability=hook)
    """

    def __init__(self, *hooks: Any) -> None:
        self._hooks = list(hooks)

    def add(self, hook: Any) -> None:
        """Append an additional hook at runtime."""
        self._hooks.append(hook)

    def on_event(self, event: ObservabilityEvent) -> None:
        for hook in self._hooks:
            _safe_emit(hook, event)


# -- Colour helpers for StdoutHook -------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"


def _c(text: str, *codes: str, color: bool = True) -> str:
    if not color:
        return text
    return "".join(codes) + text + _RESET


class StdoutHook:
    """Human-readable, optionally coloured output to stdout.

    Usage::

        wf = MyWorkflow(observability=StdoutHook())
        wf = MyWorkflow(observability=StdoutHook(color=False))  # CI-friendly
    """

    def __init__(self, color: bool = True, show_previews: bool = True) -> None:
        self.color = color
        self.show_previews = show_previews

    def on_event(self, event: ObservabilityEvent) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        tid = event.trace_id[:8] if event.trace_id else "--------"
        sid = event.span_id[:8] if event.span_id else "--------"
        prefix = _c(f"[{ts}]", _DIM, color=self.color) + f" trace={tid} span={sid}"

        if isinstance(event, AgentCallStart):
            line = (
                f"{prefix} "
                + _c("→", _GREEN + _BOLD, color=self.color)
                + f" {_c(event.workflow_name, _CYAN, color=self.color)}"
                + f".{_c(event.agent_name, _BOLD, color=self.color)}"
            )
            if self.show_previews and event.prompt_preview:
                line += _c(f"  ❝{event.prompt_preview[:80]}❞", _DIM, color=self.color)

        elif isinstance(event, AgentCallEnd):
            line = (
                f"{prefix} "
                + _c("←", _BLUE + _BOLD, color=self.color)
                + f" {_c(event.workflow_name, _CYAN, color=self.color)}"
                + f".{_c(event.agent_name, _BOLD, color=self.color)}"
                + f"  latency={event.latency_s:.3f}s"
                + f"  mem={_fmt_bytes(event.memory_bytes)}"
            )
            if self.show_previews and event.response_preview:
                line += _c(f"  ❝{event.response_preview[:80]}❞", _DIM, color=self.color)

        elif isinstance(event, AgentCallError):
            line = (
                f"{prefix} "
                + _c("✗", _RED + _BOLD, color=self.color)
                + f" {_c(event.workflow_name, _CYAN, color=self.color)}"
                + f".{_c(event.agent_name, _BOLD, color=self.color)}"
                + f"  {_c(event.error_type, _RED, color=self.color)}: {event.error_message}"
            )

        elif isinstance(event, LLMRetry):
            line = (
                f"{prefix} "
                + _c("⟳ retry", _YELLOW + _BOLD, color=self.color)
                + f"  attempt={event.attempt + 1}"
                + (f"  status={event.status_code}" if event.status_code else "")
                + f"  delay={event.delay_s:.1f}s"
                + (f"  reason={event.reason}" if event.reason else "")
            )

        elif isinstance(event, LLMRequestStart):
            line = (
                f"{prefix} "
                + _c("⇡ llm", _DIM, color=self.color)
                + f"  {event.model}"
                + (f"  attempt={event.attempt}" if event.attempt else "")
            )

        elif isinstance(event, LLMRequestEnd):
            line = (
                f"{prefix} "
                + _c("⇣ llm", _DIM, color=self.color)
                + f"  {event.model}"
                + f"  status={event.status_code}"
                + f"  latency={event.latency_s:.3f}s"
            )

        elif isinstance(event, LLMPoolRoute):
            line = (
                f"{prefix} "
                + _c("⇢ pool", _CYAN, color=self.color)
                + f"  {_c(event.pool_name, _BOLD, color=self.color)}"
                + f"  → {_c(event.candidate_name, _GREEN, color=self.color)}"
                + f"  policy={event.policy}"
                + (f"  cost=${event.cost_per_1k_tokens:.4f}/1k" if event.cost_per_1k_tokens else "")
            )

        elif isinstance(event, LLMPoolFallback):
            line = (
                f"{prefix} "
                + _c("⇢ fallback", _YELLOW + _BOLD, color=self.color)
                + f"  {event.pool_name}"
                + f"  {_c(event.failed_candidate, _RED, color=self.color)}"
                + f" → {_c(event.next_candidate, _GREEN, color=self.color)}"
                + f"  {event.error_type}: {event.error_message[:60]}"
            )

        elif isinstance(event, EpisodeStepStart):
            line = (
                f"{prefix} "
                + _c(f"▶ episode step {event.round_number}", _MAGENTA + _BOLD, color=self.color)
                + f"  agents={event.agent_names}"
            )

        elif isinstance(event, EpisodeStepEnd):
            rewards_str = "  ".join(f"{k}={v:.4f}" for k, v in event.rewards.items())
            line = (
                f"{prefix} "
                + _c(f"■ episode step {event.round_number}", _MAGENTA, color=self.color)
                + f"  latency={event.latency_s:.3f}s"
                + (f"  {rewards_str}" if rewards_str else "")
                + (_c("  DONE", _YELLOW + _BOLD, color=self.color) if event.done else "")
            )

        else:
            line = f"{prefix} {event.event_type}"

        print(line)


def _fmt_bytes(n: float) -> str:
    if abs(n) < 1024:
        return f"{n:.0f}B"
    if abs(n) < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    return f"{n / 1024 ** 2:.1f}MB"


class JsonFileHook:
    """Newline-delimited JSON (NDJSON) log file.

    Each event is serialised as one JSON object per line.  Suitable for
    ingestion into Elasticsearch, Loki, BigQuery, Splunk, or any log pipeline
    that understands NDJSON.

    Usage::

        wf = MyWorkflow(observability=JsonFileHook("run.ndjson"))
        wf = MyWorkflow(observability=JsonFileHook("run.ndjson", append=False))
    """

    def __init__(self, path: str, append: bool = True) -> None:
        self._path = path
        self._mode = "a" if append else "w"

    def on_event(self, event: ObservabilityEvent) -> None:
        record = dataclasses.asdict(event)
        with open(self._path, self._mode, encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self._mode = "a"   # always append after the first write


class OTLPHook:
    """OpenTelemetry spans exported via OTLP/gRPC.

    Creates one span per ``wf.send()`` call and one span per episode step.
    All spans belonging to one ``Workflow`` instance share the same ``trace_id``
    so you can reconstruct the full call graph in Jaeger, Grafana Tempo,
    Honeycomb, or any OTLP-compatible backend.

    Requires the optional ``otel`` extras::

        pip install choreo-mini[otel]

    Usage::

        hook = OTLPHook("my-service", endpoint="http://localhost:4317")
        wf = MyWorkflow(observability=hook)

    Parameters
    ----------
    service_name:
        Service name tag applied to all exported spans.
    endpoint:
        OTLP/gRPC collector endpoint.  Defaults to ``http://localhost:4317``
        (the standard OpenTelemetry Collector default).
    """

    def __init__(
        self,
        service_name: str = "choreo-mini",
        endpoint: str = "http://localhost:4317",
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise ImportError(
                "OTLPHook requires the 'otel' extra: pip install 'choreo-mini[otel]'"
            ) from exc

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        self._tracer = trace.get_tracer("choreo-mini")
        self._trace_mod = trace
        self._open_spans: Dict[str, Any] = {}   # span_id → span + ctx_token

    def on_event(self, event: ObservabilityEvent) -> None:
        trace = self._trace_mod

        if isinstance(event, (AgentCallStart, EpisodeStepStart)):
            span_name = (
                f"{event.workflow_name}.{event.agent_name}"
                if isinstance(event, AgentCallStart)
                else f"episode.step.{event.round_number}"
            )
            ctx = None
            if event.parent_span_id and event.parent_span_id in self._open_spans:
                parent_entry = self._open_spans[event.parent_span_id]
                ctx = trace.set_span_in_context(parent_entry["span"])

            span = self._tracer.start_span(span_name, context=ctx)
            token = trace.use_span(span, end_on_exit=False)  # manual lifecycle
            self._open_spans[event.span_id] = {"span": span, "token": token}

            span.set_attribute("trace_id", event.trace_id)
            span.set_attribute("span_id", event.span_id)
            if isinstance(event, AgentCallStart):
                span.set_attribute("agent.name", event.agent_name)
                span.set_attribute("workflow.name", event.workflow_name)
                if event.prompt_preview:
                    span.set_attribute("prompt.preview", event.prompt_preview)
            else:
                span.set_attribute("episode.id", event.episode_id)
                span.set_attribute("round", event.round_number)

        elif isinstance(event, (AgentCallEnd, EpisodeStepEnd)):
            entry = self._open_spans.pop(event.span_id, None)
            if entry:
                span = entry["span"]
                span.set_attribute("latency_s", event.latency_s)
                if isinstance(event, AgentCallEnd):
                    span.set_attribute("memory_bytes", event.memory_bytes)
                    if event.response_preview:
                        span.set_attribute("response.preview", event.response_preview)
                elif isinstance(event, EpisodeStepEnd):
                    for k, v in event.rewards.items():
                        span.set_attribute(f"reward.{k}", v)
                    span.set_attribute("done", event.done)
                span.end()

        elif isinstance(event, AgentCallError):
            entry = self._open_spans.pop(event.span_id, None)
            if entry:
                from opentelemetry.trace import StatusCode
                span = entry["span"]
                span.set_status(StatusCode.ERROR, event.error_message)
                span.set_attribute("error.type", event.error_type)
                span.set_attribute("error.message", event.error_message)
                span.end()

        elif isinstance(event, LLMRetry):
            # Add as an event on the parent agent-call span if open.
            parent = self._open_spans.get(event.span_id)
            if parent:
                parent["span"].add_event(
                    "llm_retry",
                    attributes={
                        "attempt": event.attempt,
                        "status_code": event.status_code,
                        "delay_s": event.delay_s,
                        "reason": event.reason,
                    },
                )
