"""Tests for the observability layer.

Covers:
* Event dataclass fields (trace_id, span_id, event_type)
* Workflow.send() emits AgentCallStart / AgentCallEnd / AgentCallError
* Workflow.trace_id is stable; each send() gets a unique span_id
* Message.call_id matches the span_id on the End event
* Episode.step() emits EpisodeStepStart / EpisodeStepEnd
* LLM._post() emits LLMRequestStart / LLMRequestEnd / LLMRetry
* CompositeHook fans out to multiple hooks
* JsonFileHook writes NDJSON (one JSON object per line)
* StdoutHook.on_event does not raise on any event type
* _safe_emit swallows hook exceptions without crashing the caller
* OTLPHook raises ImportError when opentelemetry is not installed
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from motif_ai.core.llm import CustomLLM, LLM, Message
from motif_ai.core.nodes import AgentNode
from motif_ai.core.workflow import Workflow
from motif_ai.core.episode import Episode
from motif_ai.core.observability import (
    AgentCallEnd,
    AgentCallError,
    AgentCallStart,
    CompositeHook,
    EpisodeStepEnd,
    EpisodeStepStart,
    JsonFileHook,
    LLMRequestEnd,
    LLMRequestStart,
    LLMRetry,
    ObservabilityEvent,
    OTLPHook,
    StdoutHook,
    _safe_emit,
    new_span_id,
    new_trace_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CapturingHook:
    """Collects all events for assertion."""

    def __init__(self):
        self.events: list[ObservabilityEvent] = []

    def on_event(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def of_type(self, cls):
        return [e for e in self.events if isinstance(e, cls)]

    def first(self, cls):
        matches = self.of_type(cls)
        assert matches, f"No event of type {cls.__name__} recorded"
        return matches[0]


def _simple_workflow(hook=None) -> tuple[Workflow, CapturingHook]:
    """Return a minimal workflow with an echo LLM and an optional hook."""
    hook = hook or CapturingHook()
    llm = CustomLLM(lambda prompt, context=None, **kw: f"reply:{prompt[:20]}")

    class W(Workflow):
        def __init__(self):
            super().__init__("test_wf", observability=hook)
            self.agent = AgentNode(self, "Bot", role="assistant", llm=llm)

    return W(), hook


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

class TestIDHelpers:
    def test_trace_id_is_32_hex(self):
        tid = new_trace_id()
        assert len(tid) == 32
        assert all(c in "0123456789abcdef" for c in tid)

    def test_span_id_is_16_hex(self):
        sid = new_span_id()
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)

    def test_ids_are_unique(self):
        ids = {new_trace_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# Workflow-level events
# ---------------------------------------------------------------------------

class TestWorkflowObservability:
    def test_trace_id_stable_across_calls(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "hello")
        wf.send("Bot", "world")
        starts = hook.of_type(AgentCallStart)
        assert len(starts) == 2
        assert starts[0].trace_id == starts[1].trace_id
        assert starts[0].trace_id == wf.trace_id

    def test_span_id_unique_per_call(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "a")
        wf.send("Bot", "b")
        span_ids = [e.span_id for e in hook.of_type(AgentCallStart)]
        assert span_ids[0] != span_ids[1]

    def test_start_and_end_share_span_id(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "hi")
        start = hook.first(AgentCallStart)
        end = hook.first(AgentCallEnd)
        assert start.span_id == end.span_id

    def test_message_call_id_matches_span(self):
        wf, hook = _simple_workflow()
        msg = wf.send("Bot", "hello")
        end = hook.first(AgentCallEnd)
        assert msg.call_id == end.span_id

    def test_start_carries_workflow_and_agent_name(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "test prompt")
        start = hook.first(AgentCallStart)
        assert start.workflow_name == "test_wf"
        assert start.agent_name == "Bot"

    def test_start_carries_prompt_preview(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "this is the prompt text")
        start = hook.first(AgentCallStart)
        assert "this is the prompt text" in start.prompt_preview

    def test_end_carries_latency(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "hi")
        end = hook.first(AgentCallEnd)
        assert end.latency_s >= 0.0

    def test_end_carries_response_preview(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "hi")
        end = hook.first(AgentCallEnd)
        assert "reply:" in end.response_preview

    def test_error_event_emitted_on_exception(self):
        hook = CapturingHook()
        failing_llm = CustomLLM(lambda p, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

        class W(Workflow):
            def __init__(self):
                super().__init__("err_wf", observability=hook)
                self.agent = AgentNode(self, "Boom", role="x", llm=failing_llm)

        wf = W()
        with pytest.raises(RuntimeError, match="boom"):
            wf.send("Boom", "trigger")

        err = hook.first(AgentCallError)
        assert err.error_type == "RuntimeError"
        assert err.error_message == "boom"
        assert err.agent_name == "Boom"

    def test_error_event_span_id_matches_start(self):
        hook = CapturingHook()
        failing_llm = CustomLLM(lambda p, **kw: (_ for _ in ()).throw(ValueError("oops")))

        class W(Workflow):
            def __init__(self):
                super().__init__("err_wf2", observability=hook)
                self.agent = AgentNode(self, "X", role="x", llm=failing_llm)

        wf = W()
        with pytest.raises(ValueError):
            wf.send("X", "go")

        start = hook.first(AgentCallStart)
        err = hook.first(AgentCallError)
        assert start.span_id == err.span_id

    def test_no_hook_does_not_break_workflow(self):
        llm = CustomLLM(lambda p, **kw: "ok")

        class W(Workflow):
            def __init__(self):
                super().__init__("plain_wf")
                self.agent = AgentNode(self, "A", role="x", llm=llm)

        wf = W()
        result = wf.send("A", "hello")
        assert result.content == "ok"

    def test_event_order_start_before_end(self):
        wf, hook = _simple_workflow()
        wf.send("Bot", "hi")
        types_seen = [type(e).__name__ for e in hook.events]
        assert types_seen.index("AgentCallStart") < types_seen.index("AgentCallEnd")


# ---------------------------------------------------------------------------
# Episode-level events
# ---------------------------------------------------------------------------

class TestEpisodeObservability:
    def _make_episode(self, hook):
        agents = {
            "A": lambda env, r: "action_a",
            "B": lambda env, r: "action_b",
        }
        reward_fn = lambda env, actions, r: {"A": 1.0, "B": 0.5}
        return Episode(agents=agents, env={}, reward_fn=reward_fn, observability=hook)

    def test_step_emits_start_and_end(self):
        hook = CapturingHook()
        ep = self._make_episode(hook)
        ep.step()
        assert hook.of_type(EpisodeStepStart)
        assert hook.of_type(EpisodeStepEnd)

    def test_step_events_share_span_id(self):
        hook = CapturingHook()
        ep = self._make_episode(hook)
        ep.step()
        start = hook.first(EpisodeStepStart)
        end = hook.first(EpisodeStepEnd)
        assert start.span_id == end.span_id

    def test_episode_id_stable_across_steps(self):
        hook = CapturingHook()
        ep = self._make_episode(hook)
        ep.step()
        ep.step()
        starts = hook.of_type(EpisodeStepStart)
        assert starts[0].episode_id == starts[1].episode_id == ep.episode_id

    def test_step_end_carries_rewards(self):
        hook = CapturingHook()
        ep = self._make_episode(hook)
        ep.step()
        end = hook.first(EpisodeStepEnd)
        assert end.rewards == {"A": 1.0, "B": 0.5}

    def test_step_end_done_flag(self):
        hook = CapturingHook()
        ep = self._make_episode(hook)
        ep.run()   # runs to max_rounds (100) or termination
        ends = hook.of_type(EpisodeStepEnd)
        # At least the last end event should have done=True
        assert ends[-1].done is True

    def test_step_carries_agent_names(self):
        hook = CapturingHook()
        ep = self._make_episode(hook)
        ep.step()
        start = hook.first(EpisodeStepStart)
        assert "A" in start.agent_names
        assert "B" in start.agent_names

    def test_no_hook_does_not_break_episode(self):
        agents = {"X": lambda env, r: "x"}
        ep = Episode(agents=agents, env={}, reward_fn=lambda e, a, r: {"X": 1.0})
        step = ep.step()
        assert step.round == 1


# ---------------------------------------------------------------------------
# LLM-level events
# ---------------------------------------------------------------------------

def _ok_resp(content="ok"):
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.headers = {}
    resp.json.return_value = {
        "choices": [{"message": {"content": content, "role": "assistant"},
                     "finish_reason": "stop"}]
    }
    return resp


def _err_resp(status):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = False
    resp.headers = {}
    resp.json.return_value = {"error": {"message": f"error {status}"}}
    resp.text = f"error {status}"
    return resp


class TestLLMObservability:
    def test_request_start_and_end_emitted(self):
        hook = CapturingHook()
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m",
                  max_retries=0, observability=hook)
        with patch("requests.post", return_value=_ok_resp()):
            llm.chat([Message(role="user", content="hi")])
        assert hook.of_type(LLMRequestStart)
        assert hook.of_type(LLMRequestEnd)

    def test_retry_event_emitted_on_503(self):
        hook = CapturingHook()
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m",
                  max_retries=1, retry_base_delay=0.0, observability=hook)
        with patch("requests.post", side_effect=[_err_resp(503), _ok_resp()]):
            with patch("time.sleep"):
                llm.chat([Message(role="user", content="hi")])
        retries = hook.of_type(LLMRetry)
        assert len(retries) == 1
        assert retries[0].status_code == 503

    def test_retry_event_carries_delay(self):
        hook = CapturingHook()
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m",
                  max_retries=1, retry_base_delay=2.0, observability=hook)
        err = _err_resp(429)
        err.headers = {"retry-after": "7"}
        with patch("requests.post", side_effect=[err, _ok_resp()]):
            with patch("time.sleep"):
                llm.chat([Message(role="user", content="hi")])
        retry = hook.first(LLMRetry)
        assert retry.delay_s == 7.0

    def test_request_end_carries_latency(self):
        hook = CapturingHook()
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m",
                  max_retries=0, observability=hook)
        with patch("requests.post", return_value=_ok_resp()):
            llm.chat([Message(role="user", content="hi")])
        end = hook.first(LLMRequestEnd)
        assert end.latency_s >= 0.0

    def test_no_events_without_hook(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m",
                  max_retries=0)
        with patch("requests.post", return_value=_ok_resp()):
            result = llm.chat([Message(role="user", content="hi")])
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# CompositeHook
# ---------------------------------------------------------------------------

class TestCompositeHook:
    def test_fans_out_to_all_hooks(self):
        h1, h2 = CapturingHook(), CapturingHook()
        composite = CompositeHook(h1, h2)
        event = AgentCallStart(trace_id="t", span_id="s",
                               workflow_name="wf", agent_name="A")
        composite.on_event(event)
        assert len(h1.events) == 1
        assert len(h2.events) == 1

    def test_add_hook_at_runtime(self):
        h1 = CapturingHook()
        composite = CompositeHook(h1)
        h2 = CapturingHook()
        composite.add(h2)
        composite.on_event(AgentCallStart(trace_id="t", span_id="s",
                                          workflow_name="w", agent_name="A"))
        assert len(h2.events) == 1

    def test_one_hook_failing_does_not_block_others(self):
        class BrokenHook:
            def on_event(self, event):
                raise RuntimeError("I am broken")

        good = CapturingHook()
        composite = CompositeHook(BrokenHook(), good)
        # CompositeHook calls _safe_emit for each hook
        composite.on_event(AgentCallStart(trace_id="t", span_id="s",
                                          workflow_name="w", agent_name="A"))
        assert len(good.events) == 1


# ---------------------------------------------------------------------------
# _safe_emit
# ---------------------------------------------------------------------------

class TestSafeEmit:
    def test_swallows_hook_exception(self, capsys):
        class BrokenHook:
            def on_event(self, event):
                raise ValueError("hook exploded")

        event = AgentCallStart(trace_id="t", span_id="s",
                               workflow_name="w", agent_name="A")
        # Must not raise
        _safe_emit(BrokenHook(), event)
        captured = capsys.readouterr()
        assert "hook exploded" in captured.err

    def test_good_hook_called_normally(self):
        hook = CapturingHook()
        event = AgentCallEnd(trace_id="t", span_id="s",
                             workflow_name="w", agent_name="A")
        _safe_emit(hook, event)
        assert len(hook.events) == 1


# ---------------------------------------------------------------------------
# JsonFileHook
# ---------------------------------------------------------------------------

class TestJsonFileHook:
    def test_writes_ndjson(self, tmp_path):
        path = tmp_path / "run.ndjson"
        hook = JsonFileHook(str(path))
        hook.on_event(AgentCallStart(trace_id="abc", span_id="def",
                                     workflow_name="wf", agent_name="A",
                                     prompt_preview="hello"))
        hook.on_event(AgentCallEnd(trace_id="abc", span_id="def",
                                   workflow_name="wf", agent_name="A",
                                   latency_s=0.1))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        obj = json.loads(lines[0])
        assert obj["event_type"] == "agent_call_start"
        assert obj["workflow_name"] == "wf"
        assert obj["trace_id"] == "abc"

    def test_appends_by_default(self, tmp_path):
        path = tmp_path / "log.ndjson"
        hook = JsonFileHook(str(path))
        hook.on_event(AgentCallStart(trace_id="t", span_id="s",
                                     workflow_name="w", agent_name="A"))
        hook.on_event(AgentCallStart(trace_id="t", span_id="s2",
                                     workflow_name="w", agent_name="B"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_overwrite_mode(self, tmp_path):
        path = tmp_path / "log.ndjson"
        path.write_text('{"existing": true}\n')
        hook = JsonFileHook(str(path), append=False)
        hook.on_event(AgentCallStart(trace_id="t", span_id="s",
                                     workflow_name="w", agent_name="A"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert "agent_call_start" in lines[0]


# ---------------------------------------------------------------------------
# StdoutHook
# ---------------------------------------------------------------------------

class TestStdoutHook:
    """Smoke tests — just ensure StdoutHook doesn't raise on any event type."""

    def _all_events(self):
        return [
            AgentCallStart(trace_id="t", span_id="s", workflow_name="wf", agent_name="A",
                           prompt_preview="hello"),
            AgentCallEnd(trace_id="t", span_id="s", workflow_name="wf", agent_name="A",
                         latency_s=0.1, memory_bytes=1024, response_preview="ok"),
            AgentCallError(trace_id="t", span_id="s", workflow_name="wf", agent_name="A",
                           latency_s=0.05, error_type="ValueError", error_message="bad"),
            LLMRequestStart(trace_id="t", span_id="s", endpoint="https://api.openai.com",
                            model="gpt-4o", attempt=0),
            LLMRequestEnd(trace_id="t", span_id="s", endpoint="https://api.openai.com",
                          model="gpt-4o", attempt=0, status_code=200, latency_s=0.2),
            LLMRetry(trace_id="t", span_id="s", endpoint="https://api.openai.com",
                     model="gpt-4o", attempt=0, status_code=429, delay_s=1.0,
                     reason="rate limit"),
            EpisodeStepStart(trace_id="t", span_id="s", episode_id="ep1",
                             round_number=3, agent_names=["A", "B"]),
            EpisodeStepEnd(trace_id="t", span_id="s", episode_id="ep1",
                           round_number=3, rewards={"A": 0.8, "B": 0.6},
                           done=False, latency_s=0.05),
        ]

    def test_no_exception_with_color(self, capsys):
        hook = StdoutHook(color=True)
        for event in self._all_events():
            hook.on_event(event)   # must not raise

    def test_no_exception_without_color(self, capsys):
        hook = StdoutHook(color=False)
        for event in self._all_events():
            hook.on_event(event)

    def test_output_contains_agent_name(self, capsys):
        hook = StdoutHook(color=False)
        hook.on_event(AgentCallStart(trace_id="t", span_id="s",
                                     workflow_name="wf", agent_name="MyAgent"))
        captured = capsys.readouterr()
        assert "MyAgent" in captured.out

    def test_output_contains_latency(self, capsys):
        hook = StdoutHook(color=False)
        hook.on_event(AgentCallEnd(trace_id="t", span_id="s",
                                   workflow_name="wf", agent_name="A",
                                   latency_s=1.234))
        captured = capsys.readouterr()
        assert "1.234s" in captured.out


# ---------------------------------------------------------------------------
# OTLPHook — import guard
# ---------------------------------------------------------------------------

class TestOTLPHook:
    def test_raises_import_error_when_otel_missing(self):
        """When opentelemetry is not installed, OTLPHook must fail with a clear message."""
        # Hide opentelemetry from sys.modules
        saved = {k: v for k, v in sys.modules.items() if "opentelemetry" in k}
        for k in list(sys.modules):
            if "opentelemetry" in k:
                del sys.modules[k]
        try:
            with pytest.raises(ImportError, match="otel"):
                OTLPHook("my-service")
        finally:
            sys.modules.update(saved)
