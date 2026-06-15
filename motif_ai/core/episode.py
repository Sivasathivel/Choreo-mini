"""Multi-agent episode loop for motif-ai.

An :class:`Episode` orchestrates a round-based MARL game across any number of
:class:`~motif_ai.core.workflow.Workflow` subclasses.  Each participating
workflow exposes an *action method* — a plain Python callable that receives the
current environment state and the round index and returns an action string.

Design principles
-----------------
* **Environment is external** — any Python object or dict; the framework never
  owns or mutates it directly.
* **Reward function is injected** — a callable you supply; the framework calls
  it after all agents act each round.
* **Policy update is external** — collect the returned trajectory and update
  your policies however you like (tabular, gradient, etc.).
* **Termination is a callable** — supply your own criterion or use the built-in
  :func:`nash_convergence_detector` helper.

Quick-start example::

    import copy
    from motif_ai.core.episode import Episode, nash_convergence_detector
    from motif_ai.core.workflow import Workflow
    from motif_ai.core.nodes import AgentNode
    from motif_ai.core.llm import CustomLLM

    # --- define workflows ---
    class CountryWorkflow(Workflow):
        def __init__(self, country, llm):
            super().__init__(country)
            self.negotiator = AgentNode(self, "Negotiator", role=f"{country} negotiator", llm=llm)

        def propose(self, env_state: dict, round: int) -> str:
            self.beliefs.observe("round", round, confidence=1.0, step=round)
            return self.send("Negotiator", str(env_state)).content

    usa = CountryWorkflow("USA", llm=CustomLLM(lambda p, **kw: "lower tariffs"))
    can = CountryWorkflow("Canada", llm=CustomLLM(lambda p, **kw: "maintain supply chain"))
    mex = CountryWorkflow("Mexico", llm=CustomLLM(lambda p, **kw: "protect agriculture"))

    # --- define environment update and reward ---
    def update_env(env_state, actions, round):
        new_state = dict(env_state)
        new_state["proposals"] = actions
        new_state["round"] = round
        return new_state

    def reward_fn(env_state, actions, round):
        # dummy: reward every country +1 per round as a placeholder
        return {name: 1.0 for name in actions}

    # --- run the episode ---
    ep = Episode(
        agents={"USA": usa.propose, "Canada": can.propose, "Mexico": mex.propose},
        env={"round": 0, "proposals": {}},
        reward_fn=reward_fn,
        env_update_fn=update_env,
        termination_fn=nash_convergence_detector(window=3, reward_threshold=0.01),
        max_rounds=20,
    )
    trajectory = ep.run()
    print(f"Converged in {len(trajectory)} rounds")
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from motif_ai.core.exceptions import EpisodeError
from motif_ai.core.observability import (
    ObservabilityHook,
    EpisodeStepStart,
    EpisodeStepEnd,
    new_trace_id,
    new_span_id,
    _safe_emit,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStep:
    """A snapshot of one completed round in the episode.

    Attributes
    ----------
    round:
        1-based round index.
    env_state:
        A shallow copy of the environment state *before* actions were taken
        this round.
    actions:
        Mapping of agent name → action string returned by each workflow.
    rewards:
        Mapping of agent name → scalar reward for this round.
    """

    round: int
    env_state: Dict[str, Any]
    actions: Dict[str, str]
    rewards: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class Episode:
    """Orchestrates a round-based MARL game across multiple workflows.

    Parameters
    ----------
    agents:
        Mapping of agent name → action callable.  Each callable must have the
        signature ``(env_state: dict, round: int) -> str``.  Typically this
        is a method on a :class:`~motif_ai.core.workflow.Workflow` subclass
        (e.g. ``usa_workflow.propose``).
    env:
        The initial environment state.  Any Python value; a shallow copy is
        taken at the start of each round so the original is never mutated by
        the framework.  Pass a dict for simplicity or a dataclass for
        structure.
    reward_fn:
        Called after all agents act each round.  Signature::

            reward_fn(env_state: dict, actions: Dict[str, str], round: int)
                -> Dict[str, float]

        Must return a reward scalar for every agent name in ``agents``.
    env_update_fn:
        Optional.  Called after ``reward_fn`` to produce the next environment
        state.  Signature::

            env_update_fn(env_state: dict, actions: Dict[str, str], round: int)
                -> new_env_state

        When omitted the environment state is not modified between rounds —
        useful when the environment object manages its own state.
    termination_fn:
        Optional callable that decides whether the episode is over.
        Signature::

            termination_fn(step: EpisodeStep, trajectory: List[EpisodeStep])
                -> bool

        Called after each completed round.  When ``None``, the episode runs
        for exactly ``max_rounds`` rounds.  Use :func:`nash_convergence_detector`
        for a ready-made Nash-equilibrium criterion.
    max_rounds:
        Hard upper bound on the number of rounds regardless of the
        ``termination_fn`` result.  Defaults to 100.

    Attributes
    ----------
    trajectory:
        List of completed :class:`EpisodeStep` objects, populated incrementally
        as :meth:`run` or :meth:`step` is called.
    round:
        Current round index (0 before the episode starts).
    done:
        ``True`` once the episode has terminated.
    """

    def __init__(
        self,
        agents: Dict[str, Callable[[Dict[str, Any], int], str]],
        env: Any,
        reward_fn: Callable[[Dict[str, Any], Dict[str, str], int], Dict[str, float]],
        env_update_fn: Optional[Callable[[Dict[str, Any], Dict[str, str], int], Any]] = None,
        termination_fn: Optional[Callable[["EpisodeStep", List["EpisodeStep"]], bool]] = None,
        max_rounds: int = 100,
        observability: Optional[ObservabilityHook] = None,
    ) -> None:
        if not agents:
            raise ValueError("Episode requires at least one agent.")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        self.agents = agents
        self._env = env
        self._initial_env = copy.deepcopy(env)   # preserved for reset()
        self.reward_fn = reward_fn
        self.env_update_fn = env_update_fn
        self.termination_fn = termination_fn
        self.max_rounds = max_rounds

        self.trajectory: List[EpisodeStep] = []
        self.round: int = 0
        self.done: bool = False

        self._observability: Optional[ObservabilityHook] = observability
        self.episode_id: str = new_trace_id()   # stable ID for this episode

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def env(self) -> Any:
        """The current environment state."""
        return self._env

    def step(self) -> EpisodeStep:
        """Execute one round and return the completed :class:`EpisodeStep`.

        Raises
        ------
        RuntimeError
            If the episode is already done.
        """
        if self.done:
            raise EpisodeError(
                "Episode is already done. Call reset() to start a new episode.",
                episode_id=self.episode_id,
            )
        if self.round >= self.max_rounds:
            self.done = True
            raise EpisodeError(
                f"Episode reached max_rounds ({self.max_rounds}) without terminating.",
                episode_id=self.episode_id,
            )

        self.round += 1
        env_snapshot = copy.deepcopy(self._env)
        step_span_id = new_span_id()

        if self._observability:
            _safe_emit(self._observability, EpisodeStepStart(
                trace_id=self.episode_id,
                span_id=step_span_id,
                episode_id=self.episode_id,
                round_number=self.round,
                agent_names=list(self.agents),
            ))

        step_start = time.time()

        # collect actions from all agents in registration order
        actions: Dict[str, str] = {}
        for name, act_fn in self.agents.items():
            actions[name] = act_fn(env_snapshot, self.round)

        # compute rewards
        rewards = self.reward_fn(env_snapshot, actions, self.round)

        # update environment
        if self.env_update_fn is not None:
            self._env = self.env_update_fn(self._env, actions, self.round)

        episode_step = EpisodeStep(
            round=self.round,
            env_state=env_snapshot,
            actions=actions,
            rewards=rewards,
        )
        self.trajectory.append(episode_step)

        # check termination
        if self.termination_fn is not None and self.termination_fn(episode_step, self.trajectory):
            self.done = True
        elif self.round >= self.max_rounds:
            self.done = True

        if self._observability:
            _safe_emit(self._observability, EpisodeStepEnd(
                trace_id=self.episode_id,
                span_id=step_span_id,
                episode_id=self.episode_id,
                round_number=self.round,
                actions={k: str(v)[:120] for k, v in actions.items()},
                rewards=rewards,
                done=self.done,
                latency_s=time.time() - step_start,
            ))

        return episode_step

    def run(self) -> List[EpisodeStep]:
        """Run the episode to completion and return the full trajectory.

        Calls :meth:`step` in a loop until ``done`` is ``True`` or
        ``max_rounds`` is reached.
        """
        while not self.done:
            self.step()
        return self.trajectory

    def reset(self, env: Optional[Any] = None) -> None:
        """Reset the episode so it can be run again.

        Parameters
        ----------
        env:
            New initial environment state.  When omitted, the original
            ``env`` passed at construction is reused unchanged.
        """
        if env is not None:
            self._env = env
            self._initial_env = copy.deepcopy(env)
        else:
            # Restore to the state the episode was constructed with.
            self._env = copy.deepcopy(self._initial_env)
        self.trajectory = []
        self.round = 0
        self.done = False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a plain-dict summary of the completed episode.

        Includes per-agent cumulative rewards and the number of rounds played.
        """
        if not self.trajectory:
            return {"rounds": 0, "cumulative_rewards": {}}

        cumulative: Dict[str, float] = {name: 0.0 for name in self.agents}
        for step in self.trajectory:
            for name, r in step.rewards.items():
                cumulative[name] = cumulative.get(name, 0.0) + r

        return {
            "rounds": self.round,
            "done": self.done,
            "cumulative_rewards": cumulative,
            "final_actions": self.trajectory[-1].actions if self.trajectory else {},
        }


# ---------------------------------------------------------------------------
# Built-in termination helpers
# ---------------------------------------------------------------------------

def nash_convergence_detector(
    window: int = 3,
    reward_threshold: float = 0.01,
) -> Callable[[EpisodeStep, List[EpisodeStep]], bool]:
    """Return a termination function that fires when rewards have stabilised.

    The detector considers the episode converged when, over the last
    ``window`` rounds, the per-agent reward range (max − min) stays within
    ``reward_threshold`` for *all* agents.  This is a practical proxy for
    Nash equilibrium: if no agent's reward is changing, no agent has
    incentive to deviate unilaterally.

    Parameters
    ----------
    window:
        Number of recent rounds to inspect.  Must be >= 2.
    reward_threshold:
        Maximum allowed reward variance (max − min) within the window.

    Returns
    -------
    Callable
        A ``termination_fn`` compatible with :class:`Episode`.

    Example
    -------
    ::

        ep = Episode(..., termination_fn=nash_convergence_detector(window=5, reward_threshold=0.05))
    """
    if window < 2:
        raise ValueError("window must be >= 2 for Nash convergence detection.")

    def _detect(step: EpisodeStep, trajectory: List[EpisodeStep]) -> bool:
        if len(trajectory) < window:
            return False
        recent = trajectory[-window:]
        agent_names = list(recent[0].rewards)
        for name in agent_names:
            values = [s.rewards.get(name, 0.0) for s in recent]
            if max(values) - min(values) > reward_threshold:
                return False
        return True

    return _detect


def max_rounds_terminator(n: int) -> Callable[[EpisodeStep, List[EpisodeStep]], bool]:
    """Return a termination function that fires after exactly ``n`` rounds.

    Equivalent to setting ``max_rounds=n`` on the :class:`Episode`, but
    composable with other criteria via ``or``/``and`` wrappers if needed.
    """
    def _detect(_step: EpisodeStep, trajectory: List[EpisodeStep]) -> bool:
        return len(trajectory) >= n

    return _detect
