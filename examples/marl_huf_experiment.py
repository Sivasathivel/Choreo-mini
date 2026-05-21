"""CUSMA/USMCA MARL experiment — maximising the Human Utility Function (HUF).

Three country-agents (USA, Canada, Mexico) negotiate five shared trade parameters
over repeated rounds.  Each agent receives a national HUF score as its reward.
The episode terminates when Nash convergence is detected.

Trade parameters (env_state)
-----------------------------
tariff_rate       float [0, 1]   Unified import tariff rate
market_access     float [0, 1]   Openness of domestic markets
labor_compliance  float [0, 1]   Labour-standard compliance score
rules_of_origin   float [0, 1]   % of product value that must be North-American
env_score         float [0, 1]   Environmental compliance score

HUF decomposition
-----------------
Each country has a different weighting over the five parameters reflecting its
negotiating interests.  The global HUF is the unweighted sum of all three.

Conflicts built into the model
-------------------------------
* USA wants high labour_compliance and rules_of_origin, low tariff_rate.
* Canada wants high market_access and env_score, low tariff_rate.
* Mexico wants low rules_of_origin, moderate labour_compliance, high market_access.

These competing interests force genuine multi-agent dynamics before convergence.

Swapping in a real LLM
-----------------------
Each CountryWorkflow takes an `llm` argument.  Replace the CustomLLM callable
with a real LLM instance to have the agents reason over the parameters in
natural language:

    from choreo_mini.core.llm import LLM
    llm = LLM(api_key="sk-...", endpoint="https://api.openai.com", model="gpt-4o")
    usa = USAWorkflow(llm=llm)
"""

from __future__ import annotations

import random
import textwrap
from typing import Any, Dict, List

from choreo_mini.core.episode import Episode, EpisodeStep, nash_convergence_detector
from choreo_mini.core.llm import CustomLLM, Message
from choreo_mini.core.nodes import AgentNode
from choreo_mini.core.workflow import Workflow


# ---------------------------------------------------------------------------
# HUF metric
# ---------------------------------------------------------------------------

# Parameter bounds — values are clipped into [0, 1] each round.
PARAMS = ["tariff_rate", "market_access", "labor_compliance", "rules_of_origin", "env_score"]


def compute_huf(env: Dict[str, float]) -> Dict[str, float]:
    """Return the national HUF score for each country given the current env."""
    t  = env["tariff_rate"]
    ma = env["market_access"]
    lc = env["labor_compliance"]
    ro = env["rules_of_origin"]
    es = env["env_score"]

    # USA: low tariffs (consumer welfare) + high labour standards + high rules-of-origin
    huf_usa = (1 - t) * 0.30 + lc * 0.40 + ro * 0.30

    # Canada: high market access (dairy/agriculture) + environmental standards + low tariffs
    huf_canada = ma * 0.40 + es * 0.35 + (1 - t) * 0.25

    # Mexico: high market access (manufacturing exports) + lenient RoO + moderate labour cost
    huf_mexico = ma * 0.50 + (1 - ro) * 0.30 + (1 - lc) * 0.20

    return {"USA": round(huf_usa, 4), "Canada": round(huf_canada, 4), "Mexico": round(huf_mexico, 4)}


def global_huf(rewards: Dict[str, float]) -> float:
    return round(sum(rewards.values()), 4)


# ---------------------------------------------------------------------------
# Country strategies (callable LLM wrappers)
# ---------------------------------------------------------------------------
#
# Each strategy receives the env state serialised as a string and returns a
# comma-separated list of signed deltas for the five parameters in PARAMS order:
#   tariff_rate, market_access, labor_compliance, rules_of_origin, env_score
#
# The workflow parses the action string back into a delta dict before handing
# it to the env_update_fn.

STEP = 0.03   # maximum per-round adjustment per parameter

def _usa_strategy(prompt: str, context: List[Message] | None = None, **_kw) -> str:
    """USA: push tariffs down, labour compliance and rules-of-origin up."""
    return f"{-STEP},{0},{STEP},{STEP},{STEP * 0.5}"


def _canada_strategy(prompt: str, context: List[Message] | None = None, **_kw) -> str:
    """Canada: push market access and env_score up, tariffs down."""
    return f"{-STEP * 0.5},{STEP},{0},{0},{STEP}"


def _mexico_strategy(prompt: str, context: List[Message] | None = None, **_kw) -> str:
    """Mexico: push market access up, resist high rules-of-origin and labour cost."""
    return f"{0},{STEP},{-STEP * 0.5},{-STEP},{0}"


def _parse_action(raw: str) -> Dict[str, float]:
    """Parse a comma-separated delta string into a {param: delta} dict."""
    values = [float(v.strip()) for v in raw.split(",")]
    return dict(zip(PARAMS, values))


# ---------------------------------------------------------------------------
# Country workflows
# ---------------------------------------------------------------------------

class CountryWorkflow(Workflow):
    """Base workflow for a single negotiating country."""

    def __init__(self, country: str, llm: Any) -> None:
        super().__init__(country)
        self.country = country
        self.negotiator = AgentNode(
            self,
            name="Negotiator",
            role=f"{country} trade negotiator. Propose parameter adjustments.",
            llm=llm,
        )
        self._last_huf: float = 0.0

    def propose(self, env: Dict[str, Any], round_: int) -> str:
        """Action method: called by Episode each round.

        Records the current HUF in beliefs, sends the env state to the LLM,
        and returns the raw action string (delta vector).
        """
        self.beliefs.observe("round", round_, confidence=1.0, step=round_)
        self.beliefs.observe("last_huf", self._last_huf, confidence=1.0, step=round_)

        # Build a compact state description for the LLM prompt
        state_summary = "  ".join(f"{k}={v:.3f}" for k, v in env.items() if k != "round")
        action = self.send("Negotiator", state_summary).content
        return action


class USAWorkflow(CountryWorkflow):
    def __init__(self, llm=None):
        super().__init__("USA", llm or CustomLLM(_usa_strategy))


class CanadaWorkflow(CountryWorkflow):
    def __init__(self, llm=None):
        super().__init__("Canada", llm or CustomLLM(_canada_strategy))


class MexicoWorkflow(CountryWorkflow):
    def __init__(self, llm=None):
        super().__init__("Mexico", llm or CustomLLM(_mexico_strategy))


# ---------------------------------------------------------------------------
# Environment functions
# ---------------------------------------------------------------------------

def env_update(env: Dict[str, Any], actions: Dict[str, str], round_: int) -> Dict[str, Any]:
    """Average the three countries' proposed deltas and apply to the env."""
    parsed = {name: _parse_action(raw) for name, raw in actions.items()}

    new_env = dict(env)
    new_env["round"] = round_

    for param in PARAMS:
        avg_delta = sum(p[param] for p in parsed.values()) / len(parsed)
        new_env[param] = round(max(0.0, min(1.0, env[param] + avg_delta)), 4)

    return new_env


def reward_fn(env: Dict[str, Any], actions: Dict[str, str], round_: int) -> Dict[str, float]:
    return compute_huf(env)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(max_rounds: int = 40, window: int = 5, threshold: float = 0.005) -> None:
    random.seed(42)

    # Initial trade parameter values (pre-negotiation baseline)
    initial_env = {
        "tariff_rate":      0.12,
        "market_access":    0.65,
        "labor_compliance": 0.55,
        "rules_of_origin":  0.62,
        "env_score":        0.58,
        "round":            0,
    }

    usa    = USAWorkflow()
    canada = CanadaWorkflow()
    mexico = MexicoWorkflow()

    ep = Episode(
        agents={
            "USA":    usa.propose,
            "Canada": canada.propose,
            "Mexico": mexico.propose,
        },
        env=initial_env,
        reward_fn=reward_fn,
        env_update_fn=env_update,
        termination_fn=nash_convergence_detector(window=window, reward_threshold=threshold),
        max_rounds=max_rounds,
    )

    print("\n" + "=" * 70)
    print("  CUSMA/USMCA MARL — HUF Maximisation Experiment")
    print("=" * 70)
    print(f"  Parameters: {', '.join(PARAMS)}")
    print(f"  Termination: Nash convergence window={window}, threshold={threshold}")
    print("=" * 70)
    print(f"  {'Round':>5}  {'USA':>7}  {'Canada':>7}  {'Mexico':>7}  {'GlobalHUF':>10}")
    print("  " + "-" * 44)

    # Print initial HUF
    init_huf = compute_huf(initial_env)
    print(f"  {'init':>5}  {init_huf['USA']:>7.4f}  {init_huf['Canada']:>7.4f}  "
          f"{init_huf['Mexico']:>7.4f}  {global_huf(init_huf):>10.4f}")

    # Run
    trajectory: List[EpisodeStep] = ep.run()

    for step in trajectory:
        r = step.rewards
        print(f"  {step.round:>5}  {r['USA']:>7.4f}  {r['Canada']:>7.4f}  "
              f"{r['Mexico']:>7.4f}  {global_huf(r):>10.4f}")

    # Final state
    final_env = ep.env
    final_huf = compute_huf(final_env)

    print("  " + "-" * 44)
    print(f"\n  Converged after {len(trajectory)} rounds  (done={ep.done})")
    print("\n  ── Final trade parameters ──────────────────────────────────────")
    for param in PARAMS:
        delta = final_env[param] - initial_env[param]
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {param:<20}  {initial_env[param]:.4f}  →  {final_env[param]:.4f}  "
              f"({arrow} {abs(delta):.4f})")

    print("\n  ── HUF at convergence ──────────────────────────────────────────")
    for country, score in final_huf.items():
        init_score = init_huf[country]
        gain = score - init_score
        print(f"  {country:<10}  {init_score:.4f}  →  {score:.4f}  "
              f"(+{gain:.4f})" if gain >= 0 else f"  {country:<10}  {init_score:.4f}  →  {score:.4f}  ({gain:.4f})")

    print(f"\n  Global HUF: {global_huf(init_huf):.4f}  →  {global_huf(final_huf):.4f}  "
          f"(+{global_huf(final_huf) - global_huf(init_huf):.4f})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_experiment()
