"""Workflow orchestration for motif-ai.

The intended usage pattern is to **subclass** :class:`Workflow` and define
agents and services as instance attributes inside ``__init__``.  The base
class handles all state management automatically — conversation history,
profiling metrics, and epistemic beliefs are available without any extra
wiring::

    from motif_ai.core.workflow import Workflow
    from motif_ai.core.nodes import AgentNode
    from motif_ai.core.llm import CustomLLM

    class Planner(Workflow):
        def __init__(self):
            super().__init__("planner", enable_profiling=True)
            self.analyst = AgentNode(self, "Analyst", role="analyse tasks", llm=...)
            self.executor = AgentNode(self, "Executor", role="execute plans", llm=...)

        def run(self, task: str) -> str:
            plan = self.send("Analyst", task)
            # update workflow-level belief after each step
            self.beliefs.observe("last_plan", plan.content, confidence=0.95)
            result = self.send("Executor", plan.content)
            return result.content

Each :class:`~motif_ai.core.nodes.AgentNode` created with ``self`` as the
first argument registers automatically.  The workflow exposes:

* ``self.beliefs`` — a :class:`~motif_ai.core.belief.BeliefState` shared
  across the entire workflow (environment / world observations).
* Per-agent belief states accessible via :meth:`get_agent_belief` — each
  agent independently tracks what it believes about the world and other agents.
* Conversation history via :meth:`get_history`.
* Profiling metrics via :meth:`get_profile`.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Dict, List, Optional

from motif_ai.core.belief import Belief, BeliefState
from motif_ai.core.nodes import BaseNode, AgentNode, ServiceNode
from motif_ai.core.llm import Message
from motif_ai.core.exceptions import AgentNotFoundError, AgentRegistrationError
from motif_ai.core.observability import (
    ObservabilityHook,
    AgentCallStart,
    AgentCallEnd,
    AgentCallError,
    new_trace_id,
    new_span_id,
    _safe_emit,
)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState:
    """Runtime state for a single agent managed by a :class:`Workflow`.

    Holds the full conversation history, per-call profiling metrics, and an
    independent :class:`~motif_ai.core.belief.BeliefState` for this agent.
    Users never instantiate this directly — the workflow creates and owns it.

    Attributes
    ----------
    agent:
        The :class:`~motif_ai.core.nodes.AgentNode` this state belongs to.
    history:
        Ordered list of :class:`~motif_ai.core.llm.Message` objects
        representing the full conversation for this agent.
    call_count:
        Number of times this agent has been invoked.
    total_latency:
        Cumulative wall-clock inference time in seconds.
    total_memory:
        Cumulative memory delta across all calls in bytes.
    belief:
        The agent's private :class:`~motif_ai.core.belief.BeliefState` —
        what this agent believes about the world and other agents.
    """

    def __init__(self, agent: AgentNode) -> None:
        self.agent = agent
        self.history: List[Message] = []
        self.call_count: int = 0
        self.total_latency: float = 0.0
        self.total_memory: float = 0.0
        self.belief: BeliefState = BeliefState()

    def record_response(self, response: Message, latency: float, memory: float) -> None:
        self.history.append(response)
        self.call_count += 1
        self.total_latency += latency
        self.total_memory += memory

    def clear_history(self) -> None:
        self.history.clear()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class Workflow:
    """Base class for all motif-ai agentic workflows.

    Subclass this to define your workflow.  Every
    :class:`~motif_ai.core.nodes.AgentNode` constructed with ``self`` as
    its first argument registers automatically — no manual bookkeeping needed.

    Parameters
    ----------
    name:
        Human-readable identifier for the workflow.
    enable_profiling:
        When ``True``, wall-clock latency and memory delta are recorded for
        every agent call and exposed via :meth:`get_profile`.

    Built-in state (available to all subclasses)
    --------------------------------------------
    ``self.beliefs`` : :class:`~motif_ai.core.belief.BeliefState`
        Workflow-level shared beliefs — observations about the environment
        that span all agents (e.g. current negotiation terms, round number).
    ``self.state`` : dict
        General-purpose key/value store for workflow-level variables.
    ``self.agent_states`` : dict
        Maps agent name → :class:`AgentState`.  Each entry carries its own
        :class:`~motif_ai.core.belief.BeliefState` in addition to history
        and profiling counters.

    Example
    -------
    ::

        class NegotiatorWorkflow(Workflow):
            def __init__(self, llm):
                super().__init__("negotiator")
                self.strategist = AgentNode(self, "Strategist",
                                            role="trade negotiation strategist",
                                            llm=llm)
                self.analyst = AgentNode(self, "Analyst",
                                         role="economic data analyst",
                                         llm=llm)

            def negotiate(self, proposal: str) -> str:
                analysis = self.send("Analyst", proposal)
                self.beliefs.observe("last_proposal", proposal, confidence=1.0)
                response = self.send("Strategist", analysis.content)
                return response.content
    """

    def __init__(
        self,
        name: str,
        enable_profiling: bool = False,
        observability: Optional[ObservabilityHook] = None,
    ) -> None:
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.root: Optional[BaseNode] = None
        self.state: Dict[str, Any] = {}
        self.profile_data: Dict[str, Dict[str, float]] = {}
        self.agent_states: Dict[str, AgentState] = {}

        # workflow-level shared belief state (environment / world observations)
        self.beliefs: BeliefState = BeliefState()

        self.enable_profiling = enable_profiling
        if self.enable_profiling and not tracemalloc.is_tracing():
            tracemalloc.start()

        # Observability — stable trace ID for this workflow instance's lifetime
        self._observability: Optional[ObservabilityHook] = observability
        self.trace_id: str = new_trace_id()

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def add_node(self, node: BaseNode, parent_name: Optional[str] = None) -> None:
        """Register a generic node in the workflow graph.

        Nodes created with ``workflow=self`` register automatically; call this
        method only for subclasses or dynamic construction.

        Parameters
        ----------
        node:
            The node to register.
        parent_name:
            If given, the node is appended as a child of the named parent.
            When omitted, the node becomes the root if none exists yet.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' is already registered in workflow '{self.name}'.")
        self.nodes[node.name] = node
        node.workflow = self
        if parent_name:
            parent = self.nodes.get(parent_name)
            if parent is None:
                raise ValueError(f"Parent node '{parent_name}' not found in workflow '{self.name}'.")
            parent.add_child(node)
        elif self.root is None:
            self.root = node

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def add_agent(self, agent: AgentNode) -> None:
        """Register an :class:`~motif_ai.core.nodes.AgentNode`.

        Called automatically when an ``AgentNode`` is constructed with this
        workflow.  Each agent receives its own :class:`AgentState` (including
        an independent :class:`~motif_ai.core.belief.BeliefState`).

        Raises
        ------
        AgentRegistrationError
            If an agent with the same name is already registered.
        """
        if agent.name in self.agent_states:
            raise AgentRegistrationError(agent.name, self.name)
        self.agent_states[agent.name] = AgentState(agent)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(
        self,
        agent_name: str,
        user_input: str,
        _parent_span_id: str = "",
    ) -> Message:
        """Send a message to a named agent and return the reply.

        Conversation history is maintained automatically.

        Parameters
        ----------
        agent_name:
            Name of a registered :class:`~motif_ai.core.nodes.AgentNode`.
        user_input:
            The user-turn text to send.
        """
        state = self._get_agent_state(agent_name)
        span_id = new_span_id()

        if self._observability:
            _safe_emit(self._observability, AgentCallStart(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=_parent_span_id,
                workflow_name=self.name,
                agent_name=agent_name,
                prompt_preview=user_input[:200],
            ))

        state.history.append(Message(role="user", content=user_input))
        context = state.history.copy()

        snap_before = tracemalloc.take_snapshot() if self.enable_profiling else None
        start = time.time()
        try:
            response = state.agent.execute(context=context)
        except Exception as exc:
            latency = time.time() - start
            if self._observability:
                _safe_emit(self._observability, AgentCallError(
                    trace_id=self.trace_id,
                    span_id=span_id,
                    parent_span_id=_parent_span_id,
                    workflow_name=self.name,
                    agent_name=agent_name,
                    latency_s=latency,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ))
            raise

        latency = time.time() - start

        memory_used = 0.0
        if snap_before is not None:
            snap_after = tracemalloc.take_snapshot()
            memory_used = sum(s.size_diff for s in snap_after.compare_to(snap_before, "lineno"))

        self._record(agent_name, state, response, latency, memory_used)

        # Attach call_id so callers can correlate the response to its span.
        response.call_id = span_id

        if self._observability:
            _safe_emit(self._observability, AgentCallEnd(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=_parent_span_id,
                workflow_name=self.name,
                agent_name=agent_name,
                latency_s=latency,
                memory_bytes=memory_used,
                response_preview=(response.content or "")[:200],
            ))

        return response

    async def send_async(
        self,
        agent_name: str,
        user_input: str,
        _parent_span_id: str = "",
    ) -> Message:
        """Async variant of :meth:`send` with full tool-use loop support.

        Identical to :meth:`send` except that
        :meth:`~motif_ai.core.nodes.AgentNode.execute_async` is called,
        which resolves tool calls when the agent has a ``toolset`` configured.
        """
        state = self._get_agent_state(agent_name)
        span_id = new_span_id()

        if self._observability:
            _safe_emit(self._observability, AgentCallStart(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=_parent_span_id,
                workflow_name=self.name,
                agent_name=agent_name,
                prompt_preview=user_input[:200],
            ))

        state.history.append(Message(role="user", content=user_input))
        context = state.history.copy()

        snap_before = tracemalloc.take_snapshot() if self.enable_profiling else None
        start = time.time()
        try:
            response = await state.agent.execute_async(context=context)
        except Exception as exc:
            latency = time.time() - start
            if self._observability:
                _safe_emit(self._observability, AgentCallError(
                    trace_id=self.trace_id,
                    span_id=span_id,
                    parent_span_id=_parent_span_id,
                    workflow_name=self.name,
                    agent_name=agent_name,
                    latency_s=latency,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ))
            raise

        latency = time.time() - start

        memory_used = 0.0
        if snap_before is not None:
            snap_after = tracemalloc.take_snapshot()
            memory_used = sum(s.size_diff for s in snap_after.compare_to(snap_before, "lineno"))

        self._record(agent_name, state, response, latency, memory_used)

        response.call_id = span_id

        if self._observability:
            _safe_emit(self._observability, AgentCallEnd(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=_parent_span_id,
                workflow_name=self.name,
                agent_name=agent_name,
                latency_s=latency,
                memory_bytes=memory_used,
                response_preview=(response.content or "")[:200],
            ))

        return response

    # ------------------------------------------------------------------
    # Epistemic belief helpers
    # ------------------------------------------------------------------

    def get_agent_belief(self, agent_name: str) -> BeliefState:
        """Return the private :class:`~motif_ai.core.belief.BeliefState`
        for the named agent.

        Use this to read or update what a specific agent believes — distinct
        from ``self.beliefs`` which holds workflow-wide shared beliefs.
        """
        return self._get_agent_state(agent_name).belief

    def update_agent_belief(
        self,
        agent_name: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "observation",
        step: int = 0,
    ) -> None:
        """Convenience wrapper: update a world-belief for a named agent."""
        self._get_agent_state(agent_name).belief.observe(
            key, value, confidence=confidence, source=source, step=step
        )

    def decay_all_beliefs(self, factor: float = 0.95) -> None:
        """Decay confidence in all beliefs — both workflow-level and per-agent.

        Call this at the end of each episode step to model the passage of time
        and force re-observation of stale information.
        """
        self.beliefs.decay(factor)
        for state in self.agent_states.values():
            state.belief.decay(factor)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all tool-client connections held by registered agents."""
        for agent_state in self.agent_states.values():
            await agent_state.agent.close()
        if self.enable_profiling and tracemalloc.is_tracing():
            tracemalloc.stop()

    def close_sync(self) -> None:
        """Synchronous convenience wrapper around :meth:`close`.

        Use this when running a purely synchronous workflow that has no
        surrounding event loop::

            wf = MyWorkflow(llm=llm)
            wf.run("task")
            wf.close_sync()
        """
        import asyncio
        asyncio.run(self.close())

    # ------------------------------------------------------------------
    # History & profiling
    # ------------------------------------------------------------------

    def get_history(self, agent_name: str) -> List[Message]:
        """Return a copy of the conversation history for the named agent."""
        return list(self._get_agent_state(agent_name).history)

    def clear_history(self, agent_name: str) -> None:
        """Clear the conversation history for the named agent."""
        self._get_agent_state(agent_name).clear_history()

    def dump(self) -> Dict[str, Any]:
        """Return a complete state snapshot for debugging, logging, or checkpointing.

        The returned dict is JSON-serialisable (all values are plain Python
        primitives) and captures everything needed to understand the current
        state of a running or completed workflow::

            import json
            print(json.dumps(wf.dump(), indent=2))

        Returns
        -------
        dict with keys:

        * ``workflow_name`` — workflow identifier.
        * ``trace_id`` — observability trace ID for this instance.
        * ``state`` — current ``wf.state`` key/value store.
        * ``beliefs`` — workflow-level belief snapshot (confidence-weighted).
        * ``agents`` — per-agent call counts, latency, history length, beliefs.
        * ``profiling`` — per-agent profiling counters, or ``None`` when
          ``enable_profiling=False``.
        """
        agents_snapshot: Dict[str, Any] = {}
        for name, s in self.agent_states.items():
            agents_snapshot[name] = {
                "call_count": s.call_count,
                "total_latency_s": round(s.total_latency, 6),
                "total_memory_bytes": s.total_memory,
                "history_length": len(s.history),
                "beliefs": s.belief.snapshot(),
            }
        return {
            "workflow_name": self.name,
            "trace_id": self.trace_id,
            "state": dict(self.state),
            "beliefs": self.beliefs.snapshot(),
            "agents": agents_snapshot,
            "profiling": dict(self.profile_data) if self.enable_profiling else None,
        }

    def get_profile(self, agent_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Return profiling data collected since ``enable_profiling=True``.

        Parameters
        ----------
        agent_name:
            When provided, returns data for that agent only.  Otherwise
            returns the full ``profile_data`` dict for all agents.
        """
        if agent_name:
            data = self.profile_data.get(agent_name)
            if data is None:
                # Profiling may be disabled or the agent hasn't been called yet.
                return {agent_name: {"calls": 0, "total_latency": 0.0, "total_memory": 0.0}}
            return {agent_name: data}
        return dict(self.profile_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_agent_state(self, agent_name: str) -> AgentState:
        state = self.agent_states.get(agent_name)
        if state is None:
            raise AgentNotFoundError(
                agent_name=agent_name,
                workflow_name=self.name,
                registered_agents=list(self.agent_states),
            )
        return state

    def _record(
        self,
        agent_name: str,
        state: AgentState,
        response: Message,
        latency: float,
        memory_used: float,
    ) -> None:
        state.record_response(response, latency, memory_used)
        if self.enable_profiling:
            agg = self.profile_data.setdefault(
                agent_name, {"calls": 0, "total_latency": 0.0, "total_memory": 0.0}
            )
            agg["calls"] += 1
            agg["total_latency"] += latency
            agg["total_memory"] += memory_used
