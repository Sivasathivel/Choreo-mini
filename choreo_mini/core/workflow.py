"""Workflow orchestration for choreo-mini.

A :class:`Workflow` owns a set of nodes and manages the runtime state,
conversation history, and profiling metrics for every agent it contains.
It is the primary entry-point for sending messages, collecting histories,
and inspecting performance.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Dict, List, Optional

from choreo_mini.core.nodes import BaseNode, AgentNode, ServiceNode
from choreo_mini.core.llm import Message


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState:
    """Runtime state for a single agent managed by a :class:`Workflow`."""

    def __init__(self, agent: AgentNode) -> None:
        self.agent = agent
        self.history: List[Message] = []
        self.call_count: int = 0
        self.total_latency: float = 0.0
        self.total_memory: float = 0.0

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
    """Orchestrates a network of nodes and agents.

    Maintains agent states (conversation history, profiling metrics) and
    provides :meth:`send` / :meth:`send_async` as the primary interface for
    driving agent interactions.  The CLI-to-LangGraph/CrewAI/AutoGen
    conversion uses the AST parser and is independent of this runtime layer.

    Parameters
    ----------
    name:
        Human-readable identifier for the workflow.
    enable_profiling:
        When ``True``, wall-clock latency and memory delta are recorded for
        every agent call and exposed via :meth:`get_profile`.
    """

    def __init__(self, name: str, enable_profiling: bool = False) -> None:
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.root: Optional[BaseNode] = None
        self.state: Dict[str, Any] = {}
        self.profile_data: Dict[str, Dict[str, float]] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.enable_profiling = enable_profiling

        if self.enable_profiling:
            tracemalloc.start()

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def add_node(self, node: BaseNode, parent_name: Optional[str] = None) -> None:
        """Register a generic node in the workflow graph.

        Nodes created with the ``workflow`` constructor argument register
        automatically; this method exists for subclasses or dynamic
        construction.

        Parameters
        ----------
        node:
            The node to register.
        parent_name:
            If provided, the node is appended as a child of the named parent.
            When omitted, the node becomes the root if no root exists yet.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' is already registered in this workflow.")
        self.nodes[node.name] = node
        node.workflow = self
        if parent_name:
            parent = self.nodes.get(parent_name)
            if parent is None:
                raise ValueError(f"Parent node '{parent_name}' not found.")
            parent.add_child(node)
        elif self.root is None:
            self.root = node

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def add_agent(self, agent: AgentNode) -> None:
        """Register an :class:`~choreo_mini.core.nodes.AgentNode`.

        Called automatically when an ``AgentNode`` is constructed with this
        workflow.  Agents are addressed by name in :meth:`send`.
        """
        if agent.name in self.agent_states:
            raise ValueError(f"Agent '{agent.name}' is already registered in this workflow.")
        self.agent_states[agent.name] = AgentState(agent)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(self, agent_name: str, user_input: str) -> Message:
        """Send a message to a named agent and return the reply.

        Conversation history is maintained automatically; you do not need to
        pass prior context manually.

        Parameters
        ----------
        agent_name:
            Name of a registered :class:`~choreo_mini.core.nodes.AgentNode`.
        user_input:
            The user-turn text to send.
        """
        state = self._get_agent_state(agent_name)

        state.history.append(Message(role="user", content=user_input))
        context = state.history.copy()

        snap_before = tracemalloc.take_snapshot() if self.enable_profiling else None
        start = time.time()
        response = state.agent.execute(context=context)
        latency = time.time() - start

        memory_used = 0.0
        if snap_before is not None:
            snap_after = tracemalloc.take_snapshot()
            memory_used = sum(s.size_diff for s in snap_after.compare_to(snap_before, "lineno"))

        self._record(agent_name, state, response, latency, memory_used)
        return response

    async def send_async(self, agent_name: str, user_input: str) -> Message:
        """Async variant of :meth:`send` that uses the agent's tool-capable path.

        Identical to :meth:`send` except that
        :meth:`~choreo_mini.core.nodes.AgentNode.execute_async` is called,
        which runs the full tool-use loop when the agent has a ``toolset``
        configured.
        """
        state = self._get_agent_state(agent_name)

        state.history.append(Message(role="user", content=user_input))
        context = state.history.copy()

        snap_before = tracemalloc.take_snapshot() if self.enable_profiling else None
        start = time.time()
        response = await state.agent.execute_async(context=context)
        latency = time.time() - start

        memory_used = 0.0
        if snap_before is not None:
            snap_after = tracemalloc.take_snapshot()
            memory_used = sum(s.size_diff for s in snap_after.compare_to(snap_before, "lineno"))

        self._record(agent_name, state, response, latency, memory_used)
        return response

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all tool-client connections held by registered agents."""
        for agent_state in self.agent_states.values():
            await agent_state.agent.close()

    # ------------------------------------------------------------------
    # History & profiling
    # ------------------------------------------------------------------

    def get_history(self, agent_name: str) -> List[Message]:
        """Return a copy of the conversation history for the named agent."""
        return list(self._get_agent_state(agent_name).history)

    def clear_history(self, agent_name: str) -> None:
        """Clear the conversation history for the named agent."""
        self._get_agent_state(agent_name).clear_history()

    def get_profile(self, agent_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Return profiling data.

        Parameters
        ----------
        agent_name:
            When provided, returns only that agent's data.  Otherwise returns
            the full ``profile_data`` dict for all agents.
        """
        if agent_name:
            data = self.profile_data.get(agent_name)
            if data is None:
                raise KeyError(f"No profile data for agent '{agent_name}'.")
            return {agent_name: data}
        return dict(self.profile_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_agent_state(self, agent_name: str) -> AgentState:
        state = self.agent_states.get(agent_name)
        if state is None:
            raise KeyError(f"Agent '{agent_name}' is not registered in this workflow.")
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
