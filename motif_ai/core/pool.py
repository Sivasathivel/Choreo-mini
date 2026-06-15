"""Cost- and constraint-aware LLM pool for motif-ai.

An :class:`LLMPool` holds multiple :class:`LLMCandidate` backends and selects
which one to call on each request according to a pluggable routing *policy*.
When a candidate fails and ``fallback=True`` the pool automatically tries the
next candidate in policy order, so transient failures in a primary model do not
bubble up to the caller.

Because :class:`LLMPool` exposes the same ``generate`` / ``chat`` / ``stream``
interface as :class:`~motif_ai.core.llm.LLM` and
:class:`~motif_ai.core.llm.CustomLLM`, it can be passed directly to any
:class:`~motif_ai.core.nodes.AgentNode`::

    from motif_ai.core.pool import LLMCandidate, LLMPool

    fast   = LLMCandidate(llm=gpt4o_mini, name="gpt-4o-mini", cost_per_1k_tokens=0.15)
    strong = LLMCandidate(llm=gpt4o,      name="gpt-4o",      cost_per_1k_tokens=2.50)

    pool = LLMPool([fast, strong], policy="cost_first", fallback=True)

    class MyWorkflow(Workflow):
        def __init__(self):
            super().__init__("my_wf")
            self.agent = AgentNode(self, "Worker", role="...", llm=pool)

Routing policies
----------------
``"cost_first"``
    Sort candidates by :attr:`LLMCandidate.cost_per_1k_tokens` ascending.
    Cheapest model is tried first; on failure the pool tries the next cheapest.
    Candidates with ``cost_per_1k_tokens=0`` are sorted first (treated as
    free / unpriced) — useful for local models.

``"priority"``
    Sort candidates by :attr:`LLMCandidate.priority` ascending (0 = highest).
    Lets you express an explicit preference order independent of cost.

``"round_robin"``
    Rotate through candidates in registration order.  Each call moves the
    cursor forward by one.  Useful for spreading load across equivalent models.

``"reliability"``
    Prefer the candidate with the highest runtime success rate.  Falls back to
    ``"priority"`` order when no calls have been made yet (cold start).  A
    candidate that fails repeatedly drops in the ranking automatically.

Observability
-------------
Every routing decision emits an :class:`~motif_ai.core.observability.LLMPoolRoute`
event; every fallback emits an :class:`~motif_ai.core.observability.LLMPoolFallback`
event.  Pass an :class:`~motif_ai.core.observability.ObservabilityHook` to the
pool constructor to capture these.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

from motif_ai.core.llm import Message, ToolCallMessage, ToolSchema
from motif_ai.core.observability import (
    LLMPoolFallback,
    LLMPoolRoute,
    ObservabilityHook,
    _safe_emit,
    new_span_id,
    new_trace_id,
)

_VALID_POLICIES = {"cost_first", "priority", "round_robin", "reliability"}


# ---------------------------------------------------------------------------
# Candidate descriptor
# ---------------------------------------------------------------------------

@dataclass
class LLMCandidate:
    """Wraps an LLM backend with metadata used by routing policies.

    Parameters
    ----------
    llm:
        Any object with a ``generate`` / ``chat`` / ``stream`` / ``chat_async``
        method — i.e. an :class:`~motif_ai.core.llm.LLM` or
        :class:`~motif_ai.core.llm.CustomLLM` instance.
    name:
        Human-readable identifier shown in observability events.  Defaults to
        ``f"candidate_{id}"`` when not set.
    cost_per_1k_tokens:
        Estimated USD cost per 1 000 tokens.  Used by ``"cost_first"`` policy.
        Set to ``0.0`` for local / unpriced models (they are sorted first).
    priority:
        Explicit ordering for ``"priority"`` policy.  Lower numbers are tried
        first.  Default ``0`` — candidates share priority and are tried in
        registration order within the same priority level.
    max_latency_s:
        Soft latency budget in seconds.  ``0.0`` means no constraint.
        Future versions will skip candidates whose ``avg_latency_s`` already
        exceeds this budget.
    """

    llm: Any
    name: str = ""
    cost_per_1k_tokens: float = 0.0
    priority: int = 0
    max_latency_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"candidate_{id(self.llm)}"


# ---------------------------------------------------------------------------
# Runtime stats (one per candidate, owned by the pool)
# ---------------------------------------------------------------------------

@dataclass
class _CandidateStats:
    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_s: float = 0.0

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.successes if self.successes else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 1.0   # optimistic prior

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "avg_latency_s": round(self.avg_latency_s, 6),
            "success_rate": round(self.success_rate, 4),
        }


# ---------------------------------------------------------------------------
# LLMPool
# ---------------------------------------------------------------------------

class LLMPool:
    """Cost- and constraint-aware pool of LLM backends.

    Parameters
    ----------
    candidates:
        Ordered list of :class:`LLMCandidate` objects.  At least one is required.
    policy:
        Routing policy.  One of ``"cost_first"``, ``"priority"``,
        ``"round_robin"``, or ``"reliability"``.  Default ``"cost_first"``.
    fallback:
        When ``True`` (default), the pool catches exceptions from a candidate
        and retries with the next one in policy order, emitting an
        :class:`~motif_ai.core.observability.LLMPoolFallback` event.
        When ``False``, the first failure propagates immediately.
    name:
        Pool identifier shown in observability events.
    observability:
        An :class:`~motif_ai.core.observability.ObservabilityHook` instance
        to receive ``llm_pool_route`` and ``llm_pool_fallback`` events.

    Attributes
    ----------
    stats:
        Per-candidate runtime statistics as a list of dicts.  Inspect after
        calls to tune policy or monitor failure rates.
    """

    def __init__(
        self,
        candidates: List[LLMCandidate],
        policy: str = "cost_first",
        fallback: bool = True,
        name: str = "pool",
        observability: Optional[ObservabilityHook] = None,
    ) -> None:
        if not candidates:
            raise ValueError("LLMPool requires at least one candidate.")
        if policy not in _VALID_POLICIES:
            raise ValueError(
                f"Unknown policy {policy!r}. "
                f"Valid options: {sorted(_VALID_POLICIES)}"
            )

        self.candidates = list(candidates)
        self.policy = policy
        self.fallback = fallback
        self.name = name
        self._observability = observability
        self._trace_id = new_trace_id()

        # Runtime stats — one entry per candidate, keyed by candidate name.
        self._stats: Dict[str, _CandidateStats] = {
            c.name: _CandidateStats(name=c.name) for c in self.candidates
        }

        # Mutable cursor for round_robin
        self._rr_index: int = 0

    # ------------------------------------------------------------------
    # Policy: ordered list of candidates to try on this call
    # ------------------------------------------------------------------

    def _ordered_candidates(self) -> List[LLMCandidate]:
        """Return candidates in the order dictated by the current policy."""
        if self.policy == "cost_first":
            return sorted(self.candidates, key=lambda c: c.cost_per_1k_tokens)

        if self.policy == "priority":
            return sorted(self.candidates, key=lambda c: c.priority)

        if self.policy == "round_robin":
            # Rotate so the next candidate in line is first.
            n = len(self.candidates)
            start = self._rr_index % n
            self._rr_index = (self._rr_index + 1) % n
            return self.candidates[start:] + self.candidates[:start]

        if self.policy == "reliability":
            # Sort by success_rate descending; tie-break by priority.
            return sorted(
                self.candidates,
                key=lambda c: (-self._stats[c.name].success_rate, c.priority),
            )

        return list(self.candidates)  # unreachable after __init__ validation

    # ------------------------------------------------------------------
    # Core dispatch — shared by generate / chat
    # ------------------------------------------------------------------

    def _call_with_fallback(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Try each candidate in policy order, falling back on failure."""
        ordered = self._ordered_candidates()
        span_id = new_span_id()
        last_exc: Exception = RuntimeError("No candidates available.")

        for idx, candidate in enumerate(ordered):
            stats = self._stats[candidate.name]
            stats.calls += 1

            if self._observability:
                _safe_emit(self._observability, LLMPoolRoute(
                    trace_id=self._trace_id,
                    span_id=span_id,
                    pool_name=self.name,
                    candidate_name=candidate.name,
                    policy=self.policy,
                    candidate_index=idx,
                    cost_per_1k_tokens=candidate.cost_per_1k_tokens,
                ))

            t0 = time.time()
            try:
                fn = getattr(candidate.llm, method)
                result = fn(*args, **kwargs)
                latency = time.time() - t0
                stats.successes += 1
                stats.total_latency_s += latency
                return result
            except Exception as exc:
                latency = time.time() - t0
                stats.failures += 1
                last_exc = exc

                if not self.fallback or idx == len(ordered) - 1:
                    raise

                next_candidate = ordered[idx + 1]
                if self._observability:
                    _safe_emit(self._observability, LLMPoolFallback(
                        trace_id=self._trace_id,
                        span_id=span_id,
                        pool_name=self.name,
                        failed_candidate=candidate.name,
                        next_candidate=next_candidate.name,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:200],
                    ))

        raise last_exc

    # ------------------------------------------------------------------
    # LLM-compatible public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response, routing through the pool's policy.

        Compatible with :meth:`~motif_ai.core.llm.LLM.generate` and
        :meth:`~motif_ai.core.llm.CustomLLM.generate`.
        """
        return self._call_with_fallback("generate", prompt, context=context, **kwargs)

    def stream(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream response tokens from the selected candidate.

        Falls back to a single-chunk yield when streaming is not supported.
        """
        # stream() returns a generator so we can't use _call_with_fallback directly
        # — we need to pick the candidate first, then iterate.
        ordered = self._ordered_candidates()
        span_id = new_span_id()
        last_exc: Exception = RuntimeError("No candidates available.")

        for idx, candidate in enumerate(ordered):
            stats = self._stats[candidate.name]
            stats.calls += 1

            if self._observability:
                _safe_emit(self._observability, LLMPoolRoute(
                    trace_id=self._trace_id,
                    span_id=span_id,
                    pool_name=self.name,
                    candidate_name=candidate.name,
                    policy=self.policy,
                    candidate_index=idx,
                    cost_per_1k_tokens=candidate.cost_per_1k_tokens,
                ))

            t0 = time.time()
            try:
                chunks: List[str] = []
                for chunk in candidate.llm.stream(prompt, context=context, **kwargs):
                    chunks.append(chunk)
                    yield chunk
                latency = time.time() - t0
                stats.successes += 1
                stats.total_latency_s += latency
                return
            except Exception as exc:
                stats.failures += 1
                last_exc = exc

                if not self.fallback or idx == len(ordered) - 1:
                    raise

                next_candidate = ordered[idx + 1]
                if self._observability:
                    _safe_emit(self._observability, LLMPoolFallback(
                        trace_id=self._trace_id,
                        span_id=span_id,
                        pool_name=self.name,
                        failed_candidate=candidate.name,
                        next_candidate=next_candidate.name,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:200],
                    ))

        raise last_exc

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Union[Message, ToolCallMessage]:
        """Send a full conversation history through the pool.

        Compatible with :meth:`~motif_ai.core.llm.LLM.chat`.
        """
        return self._call_with_fallback("chat", messages, tools, **kwargs)

    async def chat_async(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Union[Message, ToolCallMessage]:
        """Async variant of :meth:`chat`."""
        return await asyncio.to_thread(self.chat, messages, tools, **kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> List[Dict[str, Any]]:
        """Per-candidate runtime statistics.

        Returns a list of dicts (one per candidate) with keys:
        ``name``, ``calls``, ``successes``, ``failures``,
        ``avg_latency_s``, ``success_rate``.
        """
        return [self._stats[c.name].to_dict() for c in self.candidates]

    def reset_stats(self) -> None:
        """Reset all runtime counters to zero."""
        for name in self._stats:
            self._stats[name] = _CandidateStats(name=name)
