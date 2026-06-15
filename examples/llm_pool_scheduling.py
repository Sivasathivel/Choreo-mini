"""Cost- and constraint-aware LLM pool scheduling — end-to-end example.

Demonstrates all four routing policies of :class:`~motif_ai.core.pool.LLMPool`
inside a real :class:`~motif_ai.core.workflow.Workflow` subclass.
Every routing decision and fallback is traced live via
:class:`~motif_ai.core.observability.StdoutHook`.

Run with no configuration needed::

    python examples/llm_pool_scheduling.py

Swap ``CustomLLM`` for a real :class:`~motif_ai.core.llm.LLM` instance to
route actual API calls across multiple providers.

Scenario
--------
A research assistant workflow calls four specialist agents.  Each agent uses
a different pool configuration:

``Analyst``
    ``cost_first`` pool — cheap local model first, expensive GPT-4 only as fallback.

``Summariser``
    ``priority`` pool — a primary model preferred, secondary on failure.

``FactChecker``
    ``round_robin`` pool — two equivalent models share the load evenly.

``Editor``
    ``reliability`` pool — adapts at runtime; if the primary model starts
    failing its success rate drops and the stable backup takes over.

After the run the script prints per-pool stats (calls, success rate, avg latency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from motif_ai.core.llm import CustomLLM
from motif_ai.core.nodes import AgentNode
from motif_ai.core.observability import CompositeHook, JsonFileHook, StdoutHook
from motif_ai.core.pool import LLMCandidate, LLMPool
from motif_ai.core.workflow import Workflow


# ---------------------------------------------------------------------------
# Demo LLM backends (replace with real LLM() instances for live calls)
# ---------------------------------------------------------------------------

def _llm(tag: str) -> CustomLLM:
    """Return a deterministic CustomLLM that labels its reply with a backend tag."""
    return CustomLLM(lambda p, context=None, **kw: f"[{tag}] {p[:60]}")


def _llm_flaky(tag: str, fail_first_n: int = 2) -> CustomLLM:
    """Return a CustomLLM that raises for the first N calls, then succeeds."""
    call_count = [0]

    def _generate(p, context=None, **kw):
        call_count[0] += 1
        if call_count[0] <= fail_first_n:
            raise RuntimeError(f"{tag} unavailable (simulated outage, call {call_count[0]})")
        return f"[{tag}-recovered] {p[:60]}"

    return CustomLLM(_generate)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class ResearchAssistantWorkflow(Workflow):
    """Research assistant with four agents, each using a different pool policy.

    ``Analyst``     — cost_first  (cheapest first, expensive fallback)
    ``Summariser``  — priority    (explicit primary/secondary order)
    ``FactChecker`` — round_robin (load-spread across equivalents)
    ``Editor``      — reliability (adapts to runtime failure rates)
    """

    def __init__(self, observability=None):
        super().__init__("research_assistant", enable_profiling=True,
                         observability=observability)

        # -- Analyst: cost_first pool ------------------------------------------
        analyst_pool = LLMPool(
            candidates=[
                LLMCandidate(
                    llm=_llm("local-llama"),
                    name="local-llama",
                    cost_per_1k_tokens=0.0,     # free local model
                ),
                LLMCandidate(
                    llm=_llm("gpt-4o-mini"),
                    name="gpt-4o-mini",
                    cost_per_1k_tokens=0.15,
                ),
                LLMCandidate(
                    llm=_llm("gpt-4o"),
                    name="gpt-4o",
                    cost_per_1k_tokens=2.50,    # expensive — last resort
                ),
            ],
            policy="cost_first",
            fallback=True,
            name="analyst_pool",
            observability=observability,
        )
        self.analyst_pool = analyst_pool
        self.analyst = AgentNode(
            self, "Analyst",
            role="Analyse the research question and identify key sub-questions.",
            llm=analyst_pool,
        )

        # -- Summariser: priority pool -----------------------------------------
        summariser_pool = LLMPool(
            candidates=[
                LLMCandidate(
                    llm=_llm("claude-haiku"),
                    name="claude-haiku",
                    priority=1,             # preferred
                ),
                LLMCandidate(
                    llm=_llm("gpt-4o-mini"),
                    name="gpt-4o-mini-backup",
                    priority=2,             # secondary
                ),
            ],
            policy="priority",
            fallback=True,
            name="summariser_pool",
            observability=observability,
        )
        self.summariser_pool = summariser_pool
        self.summariser = AgentNode(
            self, "Summariser",
            role="Summarise findings into a concise paragraph.",
            llm=summariser_pool,
        )

        # -- FactChecker: round_robin pool -------------------------------------
        fact_pool = LLMPool(
            candidates=[
                LLMCandidate(llm=_llm("checker-A"), name="checker-A"),
                LLMCandidate(llm=_llm("checker-B"), name="checker-B"),
            ],
            policy="round_robin",
            name="fact_pool",
            observability=observability,
        )
        self.fact_pool = fact_pool
        self.fact_checker = AgentNode(
            self, "FactChecker",
            role="Verify the factual accuracy of the summary.",
            llm=fact_pool,
        )

        # -- Editor: reliability pool ------------------------------------------
        # editor-primary fails its first 2 calls → editor-stable takes over
        # and reliability policy learns to prefer it.
        editor_pool = LLMPool(
            candidates=[
                LLMCandidate(
                    llm=_llm_flaky("editor-primary", fail_first_n=2),
                    name="editor-primary",
                    priority=1,
                ),
                LLMCandidate(
                    llm=_llm("editor-stable"),
                    name="editor-stable",
                    priority=2,
                ),
            ],
            policy="reliability",
            fallback=True,
            name="editor_pool",
            observability=observability,
        )
        self.editor_pool = editor_pool
        self.editor = AgentNode(
            self, "Editor",
            role="Polish the fact-checked summary into publication-ready prose.",
            llm=editor_pool,
        )

    def research(self, question: str) -> str:
        """Run the full research pipeline for one question."""
        analysis = self.send("Analyst", f"Research question: {question}").content
        summary = self.send("Summariser", f"Summarise: {analysis}").content
        verified = self.send("FactChecker", f"Verify: {summary}").content
        final = self.send("Editor", f"Polish: {verified}").content
        return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What are the key trade-offs between stateful and stateless AI agents?",
    "How does persistent memory affect LLM agent trust calibration?",
    "What observability patterns work best for multi-agent MARL systems?",
]


def _section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print("─" * 70)


def main() -> None:
    log_path = Path(project_root) / "output" / "pool_trace.ndjson"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    hook = CompositeHook(
        StdoutHook(color=True, show_previews=False),
        JsonFileHook(str(log_path), append=False),
    )

    wf = ResearchAssistantWorkflow(observability=hook)

    _section("Research Assistant — LLMPool Scheduling Demo")
    print("Policies in use:")
    print("  Analyst     → cost_first  (local-llama → gpt-4o-mini → gpt-4o)")
    print("  Summariser  → priority    (claude-haiku → gpt-4o-mini-backup)")
    print("  FactChecker → round_robin (checker-A ↔ checker-B)")
    print("  Editor      → reliability (editor-primary fails 2× → editor-stable takes over)")

    for i, question in enumerate(QUESTIONS, start=1):
        _section(f"Question {i}: {question}")
        result = wf.research(question)
        print(f"\nFinal output:\n  {result}")

    # -- Pool stats summary --
    _section("Pool Routing Stats")
    for pool_name, pool in [
        ("analyst_pool",    wf.analyst_pool),
        ("summariser_pool", wf.summariser_pool),
        ("fact_pool",       wf.fact_pool),
        ("editor_pool",     wf.editor_pool),
    ]:
        print(f"\n{pool_name} (policy={pool.policy}):")
        for s in pool.stats:
            sr = s["success_rate"]
            bar = "█" * int(sr * 10) + "░" * (10 - int(sr * 10))
            print(
                f"  {s['name']:<22}  calls={s['calls']}  "
                f"ok={s['successes']}  fail={s['failures']}  "
                f"success_rate={sr:.0%} {bar}  "
                f"avg_latency={s['avg_latency_s']:.4f}s"
            )

    # -- Workflow dump --
    _section("Workflow State Snapshot")
    print(json.dumps(wf.dump(), indent=2))

    # -- NDJSON event summary --
    _section(f"NDJSON trace → {log_path}")
    events = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    counts: dict = {}
    for ev in events:
        t = ev.get("event_type", "?")
        counts[t] = counts.get(t, 0) + 1
    print(f"Total events: {len(events)}")
    for etype, count in sorted(counts.items()):
        print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
