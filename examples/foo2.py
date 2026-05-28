"""Complex smoke test for choreo-mini → backend conversion.

Demonstrates the *subclass pattern*: each ``Workflow`` subclass maps to a
LangGraph subgraph (or an equivalent unit in CrewAI / AutoGen).

The ``TicketTriageWorkflow`` class owns all agents and service nodes for
the ticket-triage domain.  Its ``process_batch`` method drives the execution
logic: load tickets, classify each one, route to the right specialist, review,
and return final responses.

Run directly::

    python examples/foo2.py
"""

import sys
from pathlib import Path

# Allow running directly from the examples/ folder.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from choreo_mini.core.llm import CustomLLM
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.workflow import Workflow


# ---------------------------------------------------------------------------
# Stub LLM callables (no real API needed for tests / local runs)
# ---------------------------------------------------------------------------

def _classifier_response(prompt: str, context=None, **kwargs) -> str:
    text = prompt.lower().splitlines()[-1] if prompt else ""
    if any(w in text for w in ("payment", "invoice", "refund", "billing")):
        return "billing"
    if any(w in text for w in ("error", "bug", "crash", "timeout")):
        return "technical"
    return "general"


def _billing_response(prompt: str, context=None, **kwargs) -> str:
    return f"Billing action plan: {prompt}"


def _technical_response(prompt: str, context=None, **kwargs) -> str:
    return f"Technical debug plan: {prompt}"


def _general_response(prompt: str, context=None, **kwargs) -> str:
    return f"General support response: {prompt}"


def _review_response(prompt: str, context=None, **kwargs) -> str:
    if "urgent" in prompt.lower():
        return f"Priority review approved: {prompt}"
    return f"Review approved: {prompt}"


def split_tickets(raw_batch: str):
    """Split a semicolon-delimited batch string into individual ticket strings."""
    return [part.strip() for part in raw_batch.split(";") if part.strip()]


# ---------------------------------------------------------------------------
# Workflow subclass — one subgraph per class
# ---------------------------------------------------------------------------

class TicketTriageWorkflow(Workflow):
    """Triage support tickets: classify, route to a specialist, then review.

    This class is the unit of conversion: the choreo-mini CLI converts it to a
    LangGraph StateGraph (or equivalent CrewAI / AutoGen structure) where each
    AgentNode / ServiceNode becomes a graph node and the execution logic of
    ``process_batch`` drives the edges.
    """

    def __init__(self):
        super().__init__("ticket_triage", enable_profiling=True)

        # Agents — each becomes a graph node
        self.classifier = AgentNode(
            self, "Classifier", role="triage classifier", llm=CustomLLM(_classifier_response)
        )
        self.billing_specialist = AgentNode(
            self, "BillingSpecialist", role="billing specialist", llm=CustomLLM(_billing_response)
        )
        self.tech_specialist = AgentNode(
            self, "TechSpecialist", role="technical specialist", llm=CustomLLM(_technical_response)
        )
        self.generalist = AgentNode(
            self, "Generalist", role="general support", llm=CustomLLM(_general_response)
        )
        self.reviewer = AgentNode(
            self, "Reviewer", role="quality reviewer", llm=CustomLLM(_review_response)
        )

        # Service node — wraps a pure Python callable
        self.ticket_loader = ServiceNode(self, "TicketLoader", split_tickets)

        # Workflow-level state
        self.state["round"] = 0
        self.state["last_batch"] = []

    def process_batch(self, raw_batch: str) -> list:
        """Load, classify, route, and review a semicolon-separated batch of tickets."""
        tickets = self.ticket_loader.execute(self, raw_batch)
        self.state["last_batch"] = tickets
        self.state["round"] = self.state["round"] + 1

        results = []
        for ticket in tickets:
            route = self.send("Classifier", ticket).content.strip().lower()

            if route == "billing":
                draft = self.send("BillingSpecialist", ticket).content
            elif route == "technical":
                draft = self.send("TechSpecialist", ticket).content
            else:
                draft = self.send("Generalist", ticket).content

            final = self.send("Reviewer", draft).content
            results.append(final)

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wf = TicketTriageWorkflow()
    print("Enter semicolon-separated tickets (empty line or 'quit' to stop):")
    while True:
        try:
            raw = input("Batch> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not raw.strip() or raw.strip().lower() == "quit":
            break
        results = wf.process_batch(raw)
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result}")
    if wf.enable_profiling:
        print("\nProfiling summary:")
        for agent, stats in wf.profile_data.items():
            print(f"  {agent}: calls={stats['calls']} latency={stats['total_latency']:.3f}s")
