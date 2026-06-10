"""Multi-agent debate using Episode + Workflow + full observability stack.

Two agents — a Proponent and an Opponent — debate a proposition for several
rounds inside an :class:`~choreo_mini.core.episode.Episode`.  Every agent call
and every episode step is traced via :class:`~choreo_mini.core.observability.StdoutHook`
in real time.  After the debate the workflow state is serialised with
``wf.dump()`` so you can inspect the full transcript and performance metrics.

Run with no configuration needed::

    python examples/multi_agent_debate.py

The example uses ``CustomLLM`` (deterministic lambda handlers) so no API key
or network access is required.  Swap in a real :class:`~choreo_mini.core.llm.LLM`
endpoint to see genuine LLM-generated arguments.

Architecture
------------
::

    Episode
    ├── ProponentWorkflow
    │   └── AgentNode("Proponent")  ← argues in favour
    └── OpponentWorkflow
        └── AgentNode("Opponent")   ← argues against

Each round the Episode calls both ``propose()`` methods, records actions and
rewards, and checks the :func:`~choreo_mini.core.episode.nash_convergence_detector`
criterion.  When rewards stabilise (both agents keep scoring ~1.0) the debate
is declared concluded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running from the repo root or from examples/
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from choreo_mini.core.episode import Episode, nash_convergence_detector
from choreo_mini.core.llm import CustomLLM
from choreo_mini.core.nodes import AgentNode
from choreo_mini.core.observability import CompositeHook, JsonFileHook, StdoutHook
from choreo_mini.core.workflow import Workflow


# ---------------------------------------------------------------------------
# Proposition
# ---------------------------------------------------------------------------

PROPOSITION = "AI agents should have persistent memory enabled by default."

# ---------------------------------------------------------------------------
# Deterministic demo handlers
# Swap these with real LLM calls (CustomLLM(lambda ...)) for live responses.
# ---------------------------------------------------------------------------

_PROPONENT_ARGUMENTS = [
    (
        "Persistent memory lets agents accumulate context across sessions, "
        "dramatically reducing re-explanation overhead for users. "
        "Studies show 40% faster task completion when agents recall prior preferences."
    ),
    (
        "Without memory, every session is a cold start. Users repeat themselves, "
        "trust erodes, and personalisation — the key differentiator of AI assistants — "
        "becomes impossible. Memory is the foundation of continuity."
    ),
    (
        "Security concerns are addressable with encryption, scoped retention policies, "
        "and user-controlled erasure (GDPR-style). The benefits of memory far outweigh "
        "the risks when proper safeguards are in place."
    ),
    (
        "Enterprise deployments already rely on persistent memory for compliance audit "
        "trails and multi-session project tracking. Making it the default aligns the "
        "framework with real-world production needs."
    ),
]

_OPPONENT_ARGUMENTS = [
    (
        "Default-on memory violates the principle of least privilege. "
        "Users should explicitly opt in to data retention — collecting memory by default "
        "creates privacy exposure they may not even be aware of."
    ),
    (
        "Persistent memory introduces state management complexity: stale beliefs, "
        "hallucinated recall, and context poisoning. Stateless agents are far easier "
        "to reason about, test, and debug."
    ),
    (
        "Regulatory requirements (GDPR, CCPA, HIPAA) vary by jurisdiction and use case. "
        "Defaulting to persistence forces developers to implement compliance mechanisms "
        "they may not need, increasing the cost of adoption."
    ),
    (
        "Memory creates a single point of failure and a high-value attack surface. "
        "A compromised memory store leaks the entire user interaction history. "
        "Opt-in memory limits blast radius by design."
    ),
]


def _make_proponent_llm(state_store: List[str]) -> CustomLLM:
    """Return a deterministic Proponent LLM that cycles through canned arguments."""
    call_count = [0]

    def _generate(prompt: str, context=None, **kwargs) -> str:
        idx = call_count[0] % len(_PROPONENT_ARGUMENTS)
        call_count[0] += 1
        argument = _PROPONENT_ARGUMENTS[idx]
        state_store.append(f"[Proponent R{call_count[0]}] {argument}")
        return argument

    return CustomLLM(_generate)


def _make_opponent_llm(state_store: List[str]) -> CustomLLM:
    """Return a deterministic Opponent LLM that cycles through canned arguments."""
    call_count = [0]

    def _generate(prompt: str, context=None, **kwargs) -> str:
        idx = call_count[0] % len(_OPPONENT_ARGUMENTS)
        call_count[0] += 1
        argument = _OPPONENT_ARGUMENTS[idx]
        state_store.append(f"[Opponent R{call_count[0]}] {argument}")
        return argument

    return CustomLLM(_generate)


# ---------------------------------------------------------------------------
# Workflow subclasses
# ---------------------------------------------------------------------------

class ProponentWorkflow(Workflow):
    """Workflow that argues *in favour* of the proposition each round."""

    def __init__(self, observability=None):
        super().__init__("proponent", enable_profiling=True, observability=observability)
        self._transcript: List[str] = []
        self.debater = AgentNode(
            self,
            "Proponent",
            role="Argue convincingly IN FAVOUR of the proposition.",
            llm=_make_proponent_llm(self._transcript),
        )

    def propose(self, env_state: Dict[str, Any], round_n: int) -> str:
        """Action method called by the Episode each round."""
        opponent_last = env_state.get("opponent_last", "")
        prompt = (
            f"Round {round_n}. Proposition: '{PROPOSITION}'. "
            + (f"Opponent's last argument: '{opponent_last}'. " if opponent_last else "")
            + "Give your best argument in favour."
        )
        self.beliefs.observe("round", round_n, confidence=1.0, step=round_n)
        return self.send("Proponent", prompt).content


class OpponentWorkflow(Workflow):
    """Workflow that argues *against* the proposition each round."""

    def __init__(self, observability=None):
        super().__init__("opponent", enable_profiling=True, observability=observability)
        self._transcript: List[str] = []
        self.debater = AgentNode(
            self,
            "Opponent",
            role="Argue convincingly AGAINST the proposition.",
            llm=_make_opponent_llm(self._transcript),
        )

    def propose(self, env_state: Dict[str, Any], round_n: int) -> str:
        """Action method called by the Episode each round."""
        proponent_last = env_state.get("proponent_last", "")
        prompt = (
            f"Round {round_n}. Proposition: '{PROPOSITION}'. "
            + (f"Proponent's last argument: '{proponent_last}'. " if proponent_last else "")
            + "Give your best counter-argument."
        )
        self.beliefs.observe("round", round_n, confidence=1.0, step=round_n)
        return self.send("Opponent", prompt).content


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _update_env(
    env_state: Dict[str, Any],
    actions: Dict[str, str],
    round_n: int,
) -> Dict[str, Any]:
    """Carry the latest arguments forward so each workflow can see the other's last move."""
    return {
        "round": round_n,
        "proponent_last": actions.get("proponent", ""),
        "opponent_last": actions.get("opponent", ""),
        "history": env_state.get("history", []) + [
            {"round": round_n, **actions}
        ],
    }


def _reward_fn(
    env_state: Dict[str, Any],
    actions: Dict[str, str],
    round_n: int,
) -> Dict[str, float]:
    """Both debaters earn 1.0 each round — rewards stabilise immediately for demo convergence."""
    return {"proponent": 1.0, "opponent": 1.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_path = Path(project_root) / "output" / "debate_trace.ndjson"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared observability sink — colour terminal + structured NDJSON log
    hook = CompositeHook(
        StdoutHook(color=True, show_previews=True),
        JsonFileHook(str(log_path), append=False),
    )

    pro = ProponentWorkflow(observability=hook)
    opp = OpponentWorkflow(observability=hook)

    ep = Episode(
        agents={
            "proponent": pro.propose,
            "opponent":  opp.propose,
        },
        env={"round": 0, "proponent_last": "", "opponent_last": "", "history": []},
        reward_fn=_reward_fn,
        env_update_fn=_update_env,
        # Converge when both agents' rewards stay within 0.01 for 2 consecutive rounds.
        # With our deterministic reward=1.0 this fires after round 2.
        termination_fn=nash_convergence_detector(window=2, reward_threshold=0.01),
        max_rounds=4,
        observability=hook,
    )

    print("=" * 72)
    print(f"DEBATE: {PROPOSITION}")
    print("=" * 72)

    trajectory = ep.run()
    summary = ep.summary()

    print("\n" + "=" * 72)
    print("DEBATE CONCLUDED")
    print(f"  Rounds played     : {summary['rounds']}")
    print(f"  Cumulative rewards: {summary['cumulative_rewards']}")
    print("=" * 72)

    # Print the full transcript in round order
    print("\n--- Full Debate Transcript ---")
    final_env = ep.env
    for step in final_env.get("history", []):
        round_n = step["round"]
        print(f"\nRound {round_n}")
        print(f"  PROPONENT: {step.get('proponent', '')}")
        print(f"  OPPONENT : {step.get('opponent', '')}")

    # Dump workflow state snapshots (JSON-serialisable)
    print("\n--- Proponent Workflow State Snapshot ---")
    pro_dump = pro.dump()
    print(json.dumps(pro_dump, indent=2))

    print("\n--- Opponent Workflow State Snapshot ---")
    opp_dump = opp.dump()
    print(json.dumps(opp_dump, indent=2))

    # Show NDJSON log summary
    if log_path.exists():
        events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        event_types: Dict[str, int] = {}
        for ev in events:
            t = ev.get("event_type", "unknown")
            event_types[t] = event_types.get(t, 0) + 1
        print(f"\n--- NDJSON trace written to {log_path} ---")
        print(f"  Total events: {len(events)}")
        for evt_type, count in sorted(event_types.items()):
            print(f"    {evt_type}: {count}")


if __name__ == "__main__":
    main()
