"""Tests for Phase 2 Diagnostics: structured exceptions, eval error surfacing,
workflow dump(), and EpisodeError transitions.
"""
from __future__ import annotations

import json
import pytest

from choreo_mini.core.exceptions import (
    ChoreoError,
    WorkflowError,
    AgentNotFoundError,
    AgentRegistrationError,
    EpisodeError,
    LLMError,
    ConversionError,
)
from choreo_mini.core.workflow import Workflow
from choreo_mini.core.episode import Episode
from choreo_mini.core.llm import CustomLLM
from choreo_mini.core.nodes import AgentNode


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_all_inherit_from_choreo_error(self):
        for exc_cls in (
            WorkflowError,
            AgentNotFoundError,
            AgentRegistrationError,
            EpisodeError,
            LLMError,
            ConversionError,
        ):
            assert issubclass(exc_cls, ChoreoError)

    def test_workflow_errors_inherit_workflow_error(self):
        assert issubclass(AgentNotFoundError, WorkflowError)
        assert issubclass(AgentRegistrationError, WorkflowError)

    def test_agent_not_found_attributes(self):
        exc = AgentNotFoundError("bad_agent", "MyWorkflow", ["alpha", "beta"])
        assert exc.agent_name == "bad_agent"
        assert exc.workflow_name == "MyWorkflow"
        assert exc.registered_agents == ["alpha", "beta"]
        assert "bad_agent" in str(exc)
        assert "MyWorkflow" in str(exc)
        assert "alpha" in str(exc)

    def test_agent_registration_error_attributes(self):
        exc = AgentRegistrationError("dup", "WF")
        assert exc.agent_name == "dup"
        assert exc.workflow_name == "WF"
        assert "dup" in str(exc)
        assert "WF" in str(exc)

    def test_episode_error_attributes(self):
        exc = EpisodeError("bad transition", episode_id="abc123")
        assert exc.episode_id == "abc123"
        assert "bad transition" in str(exc)

    def test_episode_error_no_id(self):
        exc = EpisodeError("simple error")
        assert exc.episode_id is None

    def test_llm_error_attributes(self):
        exc = LLMError("timeout", endpoint="http://x", model="gpt-4", status_code=504, attempts=3)
        assert exc.endpoint == "http://x"
        assert exc.model == "gpt-4"
        assert exc.status_code == 504
        assert exc.attempts == 3
        assert "timeout" in str(exc)

    def test_llm_error_defaults(self):
        exc = LLMError("fail")
        assert exc.endpoint == ""
        assert exc.status_code == 0
        assert exc.attempts == 1

    def test_conversion_error_attributes(self):
        exc = ConversionError(
            "eval failed",
            expression="x + y",
            available_vars=["x"],
            source_hint="node_1",
        )
        assert exc.expression == "x + y"
        assert exc.available_vars == ["x"]
        assert exc.source_hint == "node_1"
        assert "eval failed" in str(exc)
        assert "x + y" in str(exc)
        assert "node_1" in str(exc)

    def test_conversion_error_defaults(self):
        exc = ConversionError("oops")
        assert exc.expression == ""
        assert exc.available_vars == []
        assert exc.source_hint == ""

    def test_catch_as_base_class(self):
        """All typed exceptions are catchable as ChoreoError."""
        for exc in (
            AgentNotFoundError("a", "w", []),
            AgentRegistrationError("a", "w"),
            EpisodeError("msg"),
            LLMError("msg"),
            ConversionError("msg"),
        ):
            with pytest.raises(ChoreoError):
                raise exc


# ---------------------------------------------------------------------------
# Workflow.send() raises AgentNotFoundError
# ---------------------------------------------------------------------------

class SimpleWorkflow(Workflow):
    def __init__(self):
        super().__init__("simple")
        self.agent = AgentNode(self, "alpha", role="helper",
                               llm=CustomLLM(lambda p, **kw: "ok"))

    def run(self, msg: str) -> str:
        return self.send("alpha", msg).content


class TestWorkflowExceptions:
    def test_send_unknown_agent_raises_agent_not_found(self):
        wf = SimpleWorkflow()
        with pytest.raises(AgentNotFoundError) as exc_info:
            wf.send("nonexistent", "hello")
        assert exc_info.value.agent_name == "nonexistent"
        assert exc_info.value.workflow_name == "simple"
        assert "alpha" in exc_info.value.registered_agents

    def test_agent_not_found_is_workflow_error(self):
        wf = SimpleWorkflow()
        with pytest.raises(WorkflowError):
            wf.send("ghost", "hi")

    def test_agent_not_found_is_choreo_error(self):
        wf = SimpleWorkflow()
        with pytest.raises(ChoreoError):
            wf.send("ghost", "hi")

    def test_duplicate_agent_raises_registration_error(self):
        wf = SimpleWorkflow()
        with pytest.raises(AgentRegistrationError) as exc_info:
            # Attempting to add a second agent with the same name
            AgentNode(wf, "alpha", role="duplicate",
                      llm=CustomLLM(lambda p, **kw: "dup"))
        assert exc_info.value.agent_name == "alpha"


# ---------------------------------------------------------------------------
# Workflow.dump()
# ---------------------------------------------------------------------------

class TestWorkflowDump:
    def test_dump_returns_dict(self):
        wf = SimpleWorkflow()
        d = wf.dump()
        assert isinstance(d, dict)

    def test_dump_keys(self):
        wf = SimpleWorkflow()
        d = wf.dump()
        assert "workflow_name" in d
        assert "trace_id" in d
        assert "state" in d
        assert "beliefs" in d
        assert "agents" in d
        assert "profiling" in d

    def test_dump_workflow_name(self):
        wf = SimpleWorkflow()
        assert wf.dump()["workflow_name"] == "simple"

    def test_dump_trace_id_format(self):
        wf = SimpleWorkflow()
        trace_id = wf.dump()["trace_id"]
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32   # 128-bit hex

    def test_dump_agents_section(self):
        wf = SimpleWorkflow()
        wf.run("hello")
        d = wf.dump()
        assert "alpha" in d["agents"]
        agent_info = d["agents"]["alpha"]
        assert "call_count" in agent_info
        assert "total_latency_s" in agent_info
        assert "history_length" in agent_info
        assert agent_info["call_count"] >= 1

    def test_dump_profiling_none_by_default(self):
        wf = SimpleWorkflow()
        assert wf.dump()["profiling"] is None

    def test_dump_profiling_when_enabled(self):
        class ProfilingWorkflow(Workflow):
            def __init__(self):
                super().__init__("prof", enable_profiling=True)
                AgentNode(self, "x", role="r",
                          llm=CustomLLM(lambda p, **kw: "r"))

        wf = ProfilingWorkflow()
        wf.send("x", "hi")
        d = wf.dump()
        # profiling key is present and not None
        assert d["profiling"] is not None

    def test_dump_is_json_serialisable(self):
        wf = SimpleWorkflow()
        wf.run("hello")
        d = wf.dump()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        roundtripped = json.loads(serialised)
        assert roundtripped["workflow_name"] == "simple"


# ---------------------------------------------------------------------------
# Episode raises EpisodeError on invalid transitions
# ---------------------------------------------------------------------------

def _make_episode(max_rounds: int = 3) -> Episode:
    llm = CustomLLM(lambda p, **kw: "action")

    class TinyWorkflow(Workflow):
        def __init__(self, name):
            super().__init__(name)
            AgentNode(self, "actor", role="r", llm=llm)

        def act(self, env, round_n):
            return self.send("actor", str(env)).content

    wf = TinyWorkflow("tiny")
    return Episode(
        agents={"tiny": wf.act},
        env={"round": 0},
        reward_fn=lambda env, acts, r: {"tiny": 1.0},
        max_rounds=max_rounds,
    )


class TestEpisodeErrors:
    def test_step_after_done_raises_episode_error(self):
        ep = _make_episode(max_rounds=1)
        ep.run()   # runs to completion
        assert ep.done
        with pytest.raises(EpisodeError) as exc_info:
            ep.step()
        assert "already done" in str(exc_info.value).lower()

    def test_episode_error_carries_episode_id(self):
        ep = _make_episode(max_rounds=1)
        ep.run()
        with pytest.raises(EpisodeError) as exc_info:
            ep.step()
        assert exc_info.value.episode_id == ep.episode_id

    def test_episode_error_is_choreo_error(self):
        ep = _make_episode(max_rounds=1)
        ep.run()
        with pytest.raises(ChoreoError):
            ep.step()

    def test_reset_allows_step_again(self):
        ep = _make_episode(max_rounds=1)
        ep.run()
        ep.reset()
        # Should not raise
        step = ep.step()
        assert step.round == 1

    def test_episode_id_is_stable(self):
        ep = _make_episode()
        eid1 = ep.episode_id
        ep.step()
        assert ep.episode_id == eid1


# ---------------------------------------------------------------------------
# Public __init__ exports
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_exceptions_exported_from_package(self):
        import choreo_mini
        for name in (
            "ChoreoError",
            "WorkflowError",
            "AgentNotFoundError",
            "AgentRegistrationError",
            "EpisodeError",
            "LLMError",
            "ConversionError",
        ):
            assert hasattr(choreo_mini, name), f"{name} not exported from choreo_mini"

    def test_exported_exceptions_are_correct_types(self):
        import choreo_mini
        assert issubclass(choreo_mini.AgentNotFoundError, choreo_mini.ChoreoError)
        assert issubclass(choreo_mini.EpisodeError, choreo_mini.ChoreoError)
