from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import tracemalloc
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from choreo_mini.core.nodes import BaseNode, AgentNode, ServiceNode
from choreo_mini.core.llm import Message

class AgentState:
    """Runtime state for a single agent managed by a workflow."""

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


class Workflow:
    """Orchestrates a network of nodes and (optionally) agents.

    The original workflow class tracked generic nodes for graph generation.
    It now also maintains agent states so that conversational history,
    profiling metrics, and memory usage are handled internally.  The
    CLI-to-langgraph/crew/auto‑gen conversion relies on the AST parser and
    is independent of this runtime behaviour.
    """

    def __init__(self, name: str, enable_profiling: bool = False):
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.root: Optional[BaseNode] = None
        self.state: Dict[str, Any] = {}
        self.profile_data: Dict[str, Dict[str, float]] = {}

        # agent-related bookkeeping
        self.agent_states: Dict[str, AgentState] = {}
        self.enable_profiling = enable_profiling
        if self.enable_profiling:
            tracemalloc.start()

    # generic node registration
    def add_node(self, node: BaseNode, parent_name: Optional[str] = None) -> None:
        """Register a generic node in the workflow graph.

        Nodes may be created with ``workflow`` argument and will also be
        registered automatically; this method exists primarily for
        subclasses or dynamic construction.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists")
        self.nodes[node.name] = node
        node.workflow = self
        if parent_name:
            parent = self.nodes.get(parent_name)
            if parent is None:
                raise ValueError(f"Parent node '{parent_name}' not found")
            parent.add_child(node)
        else:
            if self.root:
                # root already set; keep existing
                pass
            else:
                self.root = node

    # ------------------------------------------------------------------
    # agent support
    # ------------------------------------------------------------------

    def add_agent(self, agent: AgentNode) -> None:
        """Register an :class:`AgentNode` for conversational use.

        Agents are addressed by their ``name`` when calling :meth:`send`.
        """
        if agent.name in self.agent_states:
            raise ValueError(f"Agent '{agent.name}' already registered")
        self.agent_states[agent.name] = AgentState(agent)

    def send(self, agent_name: str, user_input: str) -> Message:
        """Send an utterance to the named agent, updating history and
        profiling data as configured.

        The workflow itself manages the conversation history; callers do not
        have to pass context manually unless they really want to override it.
        """
        state = self.agent_states.get(agent_name)
        if state is None:
            raise KeyError(f"Agent '{agent_name}' not found in workflow")

        # append user message to history before sending
        state.history.append(Message(role="user", content=user_input))
        context = state.history.copy()

        # profiling hooks
        if self.enable_profiling:
            snap_before = tracemalloc.take_snapshot()
        start = time.time()
        response = state.agent.execute(context=context)
        latency = time.time() - start

        memory_used = 0.0
        if self.enable_profiling:
            snap_after = tracemalloc.take_snapshot()
            diff = snap_after.compare_to(snap_before, "lineno")
            memory_used = sum(stat.size_diff for stat in diff)

        # record assistant response and update profile metrics
        state.record_response(response, latency, memory_used)
        if self.enable_profiling:
            agg = self.profile_data.setdefault(agent_name, {"calls": 0, "total_latency": 0.0, "total_memory": 0.0})
            agg["calls"] += 1
            agg["total_latency"] += latency
            agg["total_memory"] += memory_used

        return response

    def get_history(self, agent_name: str) -> List[Message]:
        state = self.agent_states.get(agent_name)
        if state is None:
            raise KeyError(f"Agent '{agent_name}' not registered")
        return list(state.history)

    def clear_history(self, agent_name: str) -> None:
        state = self.agent_states.get(agent_name)
        if state is None:
            raise KeyError(f"Agent '{agent_name}' not registered")
        state.clear_history()

    def get_profile(self, agent_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Return collected profiling information.

        If ``agent_name`` is provided, returns only that agent's data.
        Otherwise returns the entire ``profile_data`` dictionary.
        """
        if agent_name:
            data = self.profile_data.get(agent_name)
            if data is None:
                raise KeyError(f"No profile data for agent '{agent_name}'")
            return {agent_name: data}
        return dict(self.profile_data)
