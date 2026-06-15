"""Tests for Phase 3: LLMPool cost/constraint-aware scheduling."""
from __future__ import annotations

import pytest

from motif_ai.core.llm import CustomLLM, Message
from motif_ai.core.observability import LLMPoolFallback, LLMPoolRoute
from motif_ai.core.pool import LLMCandidate, LLMPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(reply: str) -> CustomLLM:
    return CustomLLM(lambda p, context=None, **kw: reply)


def _failing_llm(exc: Exception) -> CustomLLM:
    def _raise(p, context=None, **kw):
        raise exc
    return CustomLLM(_raise)


def _capture_hook():
    """Returns (hook, events_list)."""
    events = []

    class _Hook:
        def on_event(self, event):
            events.append(event)

    return _Hook(), events


# ---------------------------------------------------------------------------
# LLMCandidate
# ---------------------------------------------------------------------------

class TestLLMCandidate:
    def test_name_defaults_to_id(self):
        llm = _llm("hi")
        c = LLMCandidate(llm=llm)
        assert c.name.startswith("candidate_")

    def test_explicit_name(self):
        c = LLMCandidate(llm=_llm("hi"), name="gpt-4o-mini")
        assert c.name == "gpt-4o-mini"

    def test_defaults(self):
        c = LLMCandidate(llm=_llm("hi"), name="x")
        assert c.cost_per_1k_tokens == 0.0
        assert c.priority == 0
        assert c.max_latency_s == 0.0


# ---------------------------------------------------------------------------
# LLMPool construction
# ---------------------------------------------------------------------------

class TestLLMPoolConstruction:
    def test_requires_at_least_one_candidate(self):
        with pytest.raises(ValueError, match="at least one"):
            LLMPool([])

    def test_rejects_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            LLMPool([LLMCandidate(_llm("x"), name="a")], policy="bogus")

    def test_valid_policies_accepted(self):
        c = LLMCandidate(_llm("x"), name="a")
        for policy in ("cost_first", "priority", "round_robin", "reliability"):
            pool = LLMPool([c], policy=policy)
            assert pool.policy == policy

    def test_default_policy_is_cost_first(self):
        pool = LLMPool([LLMCandidate(_llm("x"), name="a")])
        assert pool.policy == "cost_first"

    def test_default_fallback_true(self):
        pool = LLMPool([LLMCandidate(_llm("x"), name="a")])
        assert pool.fallback is True


# ---------------------------------------------------------------------------
# Policy: cost_first
# ---------------------------------------------------------------------------

class TestCostFirstPolicy:
    def _pool(self, *name_cost_pairs, **kwargs):
        candidates = [
            LLMCandidate(_llm(f"reply-{name}"), name=name, cost_per_1k_tokens=cost)
            for name, cost in name_cost_pairs
        ]
        return LLMPool(candidates, policy="cost_first", **kwargs)

    def test_picks_cheapest_first(self):
        pool = self._pool(("expensive", 2.50), ("cheap", 0.15), ("free", 0.0))
        result = pool.generate("hi")
        assert result == "reply-free"

    def test_fallback_to_next_on_failure(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("down")), name="free", cost_per_1k_tokens=0.0),
            LLMCandidate(_llm("reply-cheap"), name="cheap", cost_per_1k_tokens=0.15),
        ]
        pool = LLMPool(candidates, policy="cost_first", fallback=True)
        result = pool.generate("hi")
        assert result == "reply-cheap"

    def test_no_fallback_raises_immediately(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("down")), name="only", cost_per_1k_tokens=0.0),
        ]
        pool = LLMPool(candidates, policy="cost_first", fallback=False)
        with pytest.raises(RuntimeError, match="down"):
            pool.generate("hi")

    def test_all_fail_raises(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("a")), name="a", cost_per_1k_tokens=0.0),
            LLMCandidate(_failing_llm(RuntimeError("b")), name="b", cost_per_1k_tokens=0.15),
        ]
        pool = LLMPool(candidates, policy="cost_first", fallback=True)
        with pytest.raises(RuntimeError):
            pool.generate("hi")


# ---------------------------------------------------------------------------
# Policy: priority
# ---------------------------------------------------------------------------

class TestPriorityPolicy:
    def test_picks_lowest_priority_number_first(self):
        candidates = [
            LLMCandidate(_llm("low-prio"), name="low", priority=10),
            LLMCandidate(_llm("high-prio"), name="high", priority=1),
        ]
        pool = LLMPool(candidates, policy="priority")
        assert pool.generate("hi") == "high-prio"

    def test_fallback_within_priority_order(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("fail")), name="p1", priority=1),
            LLMCandidate(_llm("p2-reply"), name="p2", priority=2),
        ]
        pool = LLMPool(candidates, policy="priority", fallback=True)
        assert pool.generate("hi") == "p2-reply"


# ---------------------------------------------------------------------------
# Policy: round_robin
# ---------------------------------------------------------------------------

class TestRoundRobinPolicy:
    def test_rotates_across_calls(self):
        candidates = [
            LLMCandidate(_llm("a"), name="a"),
            LLMCandidate(_llm("b"), name="b"),
            LLMCandidate(_llm("c"), name="c"),
        ]
        pool = LLMPool(candidates, policy="round_robin")
        results = [pool.generate("hi") for _ in range(6)]
        assert results == ["a", "b", "c", "a", "b", "c"]

    def test_fallback_in_rotated_order(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("down")), name="a"),
            LLMCandidate(_llm("b-ok"), name="b"),
        ]
        pool = LLMPool(candidates, policy="round_robin", fallback=True)
        # First call: a fails → falls back to b
        assert pool.generate("hi") == "b-ok"


# ---------------------------------------------------------------------------
# Policy: reliability
# ---------------------------------------------------------------------------

class TestReliabilityPolicy:
    def test_cold_start_uses_registration_order(self):
        candidates = [
            LLMCandidate(_llm("first"), name="first"),
            LLMCandidate(_llm("second"), name="second"),
        ]
        pool = LLMPool(candidates, policy="reliability")
        assert pool.generate("hi") == "first"

    def test_reliability_drops_after_failure(self):
        fail_flag = [True]
        def _flaky(p, context=None, **kw):
            if fail_flag[0]:
                fail_flag[0] = False
                raise RuntimeError("flaky")
            return "flaky-ok"

        candidates = [
            LLMCandidate(CustomLLM(_flaky), name="flaky"),
            LLMCandidate(_llm("stable"), name="stable"),
        ]
        pool = LLMPool(candidates, policy="reliability", fallback=True)
        # First call: flaky fails → stable is used
        result = pool.generate("hi")
        assert result == "stable"
        # After failure, flaky has success_rate=0 → stable (100%) should be preferred
        result2 = pool.generate("hi")
        assert result2 == "stable"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_initial_structure(self):
        pool = LLMPool([
            LLMCandidate(_llm("a"), name="alpha"),
            LLMCandidate(_llm("b"), name="beta"),
        ])
        stats = pool.stats
        assert len(stats) == 2
        names = {s["name"] for s in stats}
        assert names == {"alpha", "beta"}
        for s in stats:
            assert s["calls"] == 0
            assert s["successes"] == 0
            assert s["failures"] == 0

    def test_stats_incremented_on_success(self):
        pool = LLMPool([LLMCandidate(_llm("ok"), name="main")], policy="cost_first")
        pool.generate("hi")
        pool.generate("hi")
        s = pool.stats[0]
        assert s["calls"] == 2
        assert s["successes"] == 2
        assert s["failures"] == 0
        assert s["success_rate"] == 1.0

    def test_stats_incremented_on_failure_and_fallback(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("down")), name="bad"),
            LLMCandidate(_llm("ok"), name="good"),
        ]
        pool = LLMPool(candidates, policy="cost_first", fallback=True)
        pool.generate("hi")
        bad_stats = next(s for s in pool.stats if s["name"] == "bad")
        good_stats = next(s for s in pool.stats if s["name"] == "good")
        assert bad_stats["calls"] == 1
        assert bad_stats["failures"] == 1
        assert good_stats["calls"] == 1
        assert good_stats["successes"] == 1

    def test_reset_stats(self):
        pool = LLMPool([LLMCandidate(_llm("ok"), name="main")])
        pool.generate("hi")
        pool.reset_stats()
        assert pool.stats[0]["calls"] == 0


# ---------------------------------------------------------------------------
# Observability events
# ---------------------------------------------------------------------------

class TestObservabilityEvents:
    def test_route_event_emitted_on_success(self):
        hook, events = _capture_hook()
        pool = LLMPool(
            [LLMCandidate(_llm("hi"), name="m", cost_per_1k_tokens=0.5)],
            observability=hook,
        )
        pool.generate("hello")
        route_events = [e for e in events if isinstance(e, LLMPoolRoute)]
        assert len(route_events) == 1
        ev = route_events[0]
        assert ev.candidate_name == "m"
        assert ev.policy == "cost_first"
        assert ev.cost_per_1k_tokens == 0.5

    def test_fallback_event_emitted_on_failure(self):
        hook, events = _capture_hook()
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("boom")), name="bad"),
            LLMCandidate(_llm("ok"), name="good"),
        ]
        pool = LLMPool(candidates, policy="cost_first", fallback=True, observability=hook)
        pool.generate("hi")
        fallback_events = [e for e in events if isinstance(e, LLMPoolFallback)]
        assert len(fallback_events) == 1
        ev = fallback_events[0]
        assert ev.failed_candidate == "bad"
        assert ev.next_candidate == "good"
        assert "boom" in ev.error_message

    def test_no_fallback_event_when_success(self):
        hook, events = _capture_hook()
        pool = LLMPool([LLMCandidate(_llm("hi"), name="m")], observability=hook)
        pool.generate("hello")
        fallback_events = [e for e in events if isinstance(e, LLMPoolFallback)]
        assert fallback_events == []

    def test_route_event_carries_pool_name(self):
        hook, events = _capture_hook()
        pool = LLMPool(
            [LLMCandidate(_llm("hi"), name="m")],
            name="my_pool",
            observability=hook,
        )
        pool.generate("hello")
        ev = events[0]
        assert ev.pool_name == "my_pool"


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

class TestChatInterface:
    def test_chat_returns_message(self):
        pool = LLMPool([LLMCandidate(_llm("response"), name="m")])
        msgs = [Message(role="user", content="hello")]
        result = pool.chat(msgs)
        assert isinstance(result, Message)
        assert result.content == "response"

    def test_chat_fallback(self):
        candidates = [
            LLMCandidate(_failing_llm(RuntimeError("fail")), name="bad"),
            LLMCandidate(_llm("ok"), name="good"),
        ]
        pool = LLMPool(candidates, policy="cost_first", fallback=True)
        msgs = [Message(role="user", content="hi")]
        result = pool.chat(msgs)
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# Public __init__ exports
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_pool_exported_from_package(self):
        import motif_ai
        assert hasattr(motif_ai, "LLMPool")
        assert hasattr(motif_ai, "LLMCandidate")
        assert hasattr(motif_ai, "LLMPoolRoute")
        assert hasattr(motif_ai, "LLMPoolFallback")

    def test_pool_usable_from_package(self):
        from motif_ai import LLMCandidate, LLMPool
        pool = LLMPool([LLMCandidate(_llm("hi"), name="m")])
        assert pool.generate("hello") == "hi"
