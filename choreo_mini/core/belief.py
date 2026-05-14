"""Epistemic belief state for choreo-mini agents and workflows.

Unlike LangGraph, CrewAI, and AutoGen — which model agent state purely as
conversation history — choreo-mini gives every agent and every workflow a
structured *belief state*: a confidence-weighted map of what the agent
believes about the world and about the other agents it interacts with.

This is the foundation for genuine multi-agent reasoning:

* **First-order beliefs** — what this agent believes about the environment
  (e.g. "tariff rate is 15 %", confidence 0.8).
* **Second-order beliefs** — what this agent believes other agents believe
  (Theory of Mind), enabling cooperative or strategic behaviour.
* **Confidence decay** — beliefs can become stale over time; calling
  :meth:`BeliefState.decay` reduces confidence uniformly, forcing agents to
  re-observe rather than rely on outdated information.

Typical usage inside a :class:`~choreo_mini.core.workflow.Workflow` subclass::

    # record an observation about the environment
    self.beliefs.observe("tariff_rate", 0.15, confidence=0.9)

    # record what we believe another agent thinks
    self.beliefs.observe_agent("Canada", "tariff_rate", 0.10, confidence=0.6)

    # query a belief (returns None when unknown)
    b = self.beliefs.query("tariff_rate")
    if b and b.confidence > 0.7:
        print(f"High-confidence tariff belief: {b.value}")

    # age all beliefs after each negotiation round
    self.beliefs.decay(factor=0.9)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Belief atom
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """A single held belief with associated confidence and provenance.

    Attributes
    ----------
    value:
        The believed fact or estimate (any Python value).
    confidence:
        Certainty score in ``[0.0, 1.0]``.  1.0 means fully certain; 0.0
        means the belief is effectively unknown.
    source:
        How this belief was formed: ``"prior"``, ``"observation"``,
        ``"communication"``, or ``"inference"``.
    step:
        The episode/round index at which this belief was last updated.
        Useful for staleness checks.
    """

    value: Any
    confidence: float = 1.0
    source: str = "prior"
    step: int = 0


# ---------------------------------------------------------------------------
# Belief state container
# ---------------------------------------------------------------------------

class BeliefState:
    """Confidence-weighted belief map for a single agent or workflow.

    Two namespaces are maintained:

    * ``world`` — beliefs about the environment (key → :class:`Belief`).
    * ``agents`` — beliefs about other agents (agent_name → key → :class:`Belief`).

    All mutating methods return ``self`` to allow chaining.
    """

    def __init__(self) -> None:
        self.world: Dict[str, Belief] = {}
        self.agents: Dict[str, Dict[str, Belief]] = {}

    # ------------------------------------------------------------------
    # World beliefs
    # ------------------------------------------------------------------

    def observe(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "observation",
        step: int = 0,
    ) -> "BeliefState":
        """Record or update a belief about the environment."""
        self.world[key] = Belief(value=value, confidence=confidence, source=source, step=step)
        return self

    def query(self, key: str) -> Optional[Belief]:
        """Return the belief for ``key``, or ``None`` if unknown."""
        return self.world.get(key)

    def query_value(self, key: str, default: Any = None) -> Any:
        """Return the raw value of the belief, or ``default`` if unknown."""
        b = self.world.get(key)
        return b.value if b is not None else default

    # ------------------------------------------------------------------
    # Agent beliefs  (Theory of Mind)
    # ------------------------------------------------------------------

    def observe_agent(
        self,
        agent_name: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "communication",
        step: int = 0,
    ) -> "BeliefState":
        """Record or update a belief about what another agent believes."""
        if agent_name not in self.agents:
            self.agents[agent_name] = {}
        self.agents[agent_name][key] = Belief(
            value=value, confidence=confidence, source=source, step=step
        )
        return self

    def query_agent(self, agent_name: str, key: str) -> Optional[Belief]:
        """Return a belief about another agent, or ``None`` if unknown."""
        return self.agents.get(agent_name, {}).get(key)

    def query_agent_value(self, agent_name: str, key: str, default: Any = None) -> Any:
        """Return the raw value of a belief about another agent."""
        b = self.query_agent(agent_name, key)
        return b.value if b is not None else default

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay(self, factor: float = 0.95) -> "BeliefState":
        """Reduce the confidence of every held belief by ``factor``.

        Call this at the end of each episode step to make stale beliefs
        less influential over time.  Confidence is floored at 0.0.
        """
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"decay factor must be in [0, 1]; got {factor}")
        for belief in self.world.values():
            belief.confidence = max(0.0, belief.confidence * factor)
        for agent_beliefs in self.agents.values():
            for belief in agent_beliefs.values():
                belief.confidence = max(0.0, belief.confidence * factor)
        return self

    def snapshot(self) -> Dict[str, Any]:
        """Return a plain-dict snapshot of the full belief state.

        Useful for logging, serialisation, or passing as context to an LLM.
        """
        return {
            "world": {
                k: {"value": b.value, "confidence": b.confidence, "source": b.source, "step": b.step}
                for k, b in self.world.items()
            },
            "agents": {
                agent_name: {
                    k: {"value": b.value, "confidence": b.confidence, "source": b.source, "step": b.step}
                    for k, b in beliefs.items()
                }
                for agent_name, beliefs in self.agents.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"BeliefState(world_keys={list(self.world)}, "
            f"agent_keys={list(self.agents)})"
        )
