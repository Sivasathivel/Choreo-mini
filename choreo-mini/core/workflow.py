from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from choreo_mini.core.nodes import BaseNode, AgentNode
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
        self.profile_data: Dict[str, Any] = {}

        # agent-related bookkeeping
        self.agent_states: Dict[str, AgentState] = {}
        self.enable_profiling = enable_profiling

    # the existing node-graph helpers can remain if you need them
    # def add_node(...):  # commented-out original implementation

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

        context = state.history.copy()
        start = time.time()
        # pass the new user text as context; previous history is prepended
        response = state.agent.execute(context=context + [Message(role="user", content=user_input)])
        latency = time.time() - start

        memory_used = 0.0  # placeholder for actual measurement

        if self.enable_profiling:
            state.record_response(response, latency, memory_used)

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
