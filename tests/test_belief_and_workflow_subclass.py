"""Tests for epistemic belief state and the Workflow subclass pattern."""

from choreo_mini.core.belief import Belief, BeliefState
from choreo_mini.core.workflow import Workflow, AgentState
from choreo_mini.core.nodes import AgentNode
from choreo_mini.core.llm import CustomLLM, Message


# ---------------------------------------------------------------------------
# BeliefState unit tests
# ---------------------------------------------------------------------------

class TestBeliefState:
    def test_observe_and_query_world(self):
        bs = BeliefState()
        bs.observe("tariff", 0.15, confidence=0.9, source="observation", step=1)
        b = bs.query("tariff")
        assert b is not None
        assert b.value == 0.15
        assert b.confidence == 0.9
        assert b.source == "observation"
        assert b.step == 1

    def test_query_unknown_returns_none(self):
        bs = BeliefState()
        assert bs.query("unknown_key") is None

    def test_query_value_default(self):
        bs = BeliefState()
        assert bs.query_value("x", default=42) == 42
        bs.observe("x", 7)
        assert bs.query_value("x") == 7

    def test_observe_agent_and_query(self):
        bs = BeliefState()
        bs.observe_agent("Canada", "tariff", 0.10, confidence=0.6)
        b = bs.query_agent("Canada", "tariff")
        assert b is not None
        assert b.value == 0.10
        assert b.confidence == 0.6

    def test_query_agent_unknown_returns_none(self):
        bs = BeliefState()
        assert bs.query_agent("Mexico", "gdp") is None

    def test_query_agent_value_default(self):
        bs = BeliefState()
        assert bs.query_agent_value("Mexico", "gdp", default=0) == 0
        bs.observe_agent("Mexico", "gdp", 1.5e12)
        assert bs.query_agent_value("Mexico", "gdp") == 1.5e12

    def test_decay_reduces_confidence(self):
        bs = BeliefState()
        bs.observe("price", 100, confidence=1.0)
        bs.observe_agent("Canada", "stance", "firm", confidence=0.8)
        bs.decay(factor=0.5)
        assert bs.query("price").confidence == 0.5
        assert bs.query_agent("Canada", "stance").confidence == 0.4

    def test_decay_floors_at_zero(self):
        bs = BeliefState()
        bs.observe("x", 1, confidence=0.1)
        bs.decay(factor=0.0)
        assert bs.query("x").confidence == 0.0

    def test_decay_invalid_factor_raises(self):
        bs = BeliefState()
        try:
            bs.decay(factor=1.5)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_snapshot_structure(self):
        bs = BeliefState()
        bs.observe("tariff", 0.15, confidence=0.9, step=2)
        bs.observe_agent("Canada", "position", "defensive", confidence=0.7, step=2)
        snap = bs.snapshot()
        assert snap["world"]["tariff"]["value"] == 0.15
        assert snap["world"]["tariff"]["step"] == 2
        assert snap["agents"]["Canada"]["position"]["value"] == "defensive"

    def test_chaining(self):
        bs = BeliefState()
        result = bs.observe("a", 1).observe("b", 2).observe_agent("X", "k", 3)
        assert result is bs


# ---------------------------------------------------------------------------
# Workflow subclass pattern
# ---------------------------------------------------------------------------

class _EchoLLM(CustomLLM):
    def __init__(self):
        super().__init__(lambda prompt, context=None, **kw: f"echo: {prompt.splitlines()[-1]}")


class SimpleWorkflow(Workflow):
    """Minimal subclass — agents defined in __init__, no manual state wiring."""

    def __init__(self):
        super().__init__("simple", enable_profiling=True)
        self.agent_a = AgentNode(self, "AgentA", role="first agent", llm=_EchoLLM())
        self.agent_b = AgentNode(self, "AgentB", role="second agent", llm=_EchoLLM())

    def run(self, text: str) -> str:
        reply_a = self.send("AgentA", text)
        self.beliefs.observe("last_input", text, confidence=1.0)
        reply_b = self.send("AgentB", reply_a.content)
        return reply_b.content


class TestWorkflowSubclass:
    def test_agents_auto_registered(self):
        wf = SimpleWorkflow()
        assert "AgentA" in wf.agent_states
        assert "AgentB" in wf.agent_states

    def test_send_returns_message(self):
        wf = SimpleWorkflow()
        resp = wf.send("AgentA", "hello")
        assert isinstance(resp, Message)
        assert "hello" in resp.content

    def test_run_pipeline(self):
        wf = SimpleWorkflow()
        result = wf.run("hello world")
        assert "hello world" in result or "echo" in result

    def test_history_recorded(self):
        wf = SimpleWorkflow()
        wf.send("AgentA", "ping")
        history = wf.get_history("AgentA")
        assert len(history) == 2          # user turn + assistant reply
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_profiling_recorded(self):
        wf = SimpleWorkflow()
        wf.send("AgentA", "ping")
        profile = wf.get_profile("AgentA")
        assert profile["AgentA"]["calls"] == 1
        assert profile["AgentA"]["total_latency"] >= 0

    def test_workflow_beliefs_accessible(self):
        wf = SimpleWorkflow()
        wf.run("test input")
        b = wf.beliefs.query("last_input")
        assert b is not None
        assert b.value == "test input"
        assert b.confidence == 1.0

    def test_per_agent_belief_independent(self):
        wf = SimpleWorkflow()
        wf.get_agent_belief("AgentA").observe("stance", "aggressive", confidence=0.8)
        wf.get_agent_belief("AgentB").observe("stance", "defensive", confidence=0.6)
        assert wf.get_agent_belief("AgentA").query_value("stance") == "aggressive"
        assert wf.get_agent_belief("AgentB").query_value("stance") == "defensive"

    def test_update_agent_belief_convenience(self):
        wf = SimpleWorkflow()
        wf.update_agent_belief("AgentA", "mood", "confident", confidence=0.9, step=1)
        b = wf.get_agent_belief("AgentA").query("mood")
        assert b.value == "confident"
        assert b.confidence == 0.9
        assert b.step == 1

    def test_decay_all_beliefs(self):
        wf = SimpleWorkflow()
        wf.beliefs.observe("env", "active", confidence=1.0)
        wf.get_agent_belief("AgentA").observe("info", "x", confidence=1.0)
        wf.decay_all_beliefs(factor=0.5)
        assert wf.beliefs.query("env").confidence == 0.5
        assert wf.get_agent_belief("AgentA").query("info").confidence == 0.5

    def test_agent_state_has_belief(self):
        wf = SimpleWorkflow()
        state = wf.agent_states["AgentA"]
        assert isinstance(state.belief, BeliefState)

    def test_unknown_agent_raises(self):
        wf = SimpleWorkflow()
        try:
            wf.send("NonExistent", "hello")
            assert False, "expected KeyError"
        except KeyError as e:
            assert "NonExistent" in str(e)
