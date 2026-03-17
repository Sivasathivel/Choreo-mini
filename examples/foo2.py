"""Complex smoke test for choreo-mini conversion to LangGraph.

Toy problem: triage batches of support tickets, route each ticket to the right
specialist, and run review before printing the final response.
"""

import sys
from pathlib import Path

# Ensure local package imports work when run from examples/.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from choreo_mini.core.llm import CustomLLM
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.workflow import Workflow


def _classifier_response(prompt: str, context=None, **kwargs) -> str:
    text = prompt.lower().splitlines()[-1] if prompt else ""
    if any(word in text for word in ("payment", "invoice", "refund", "billing")):
        return "billing"
    if any(word in text for word in ("error", "bug", "crash", "timeout")):
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
    return [part.strip() for part in raw_batch.split(";") if part.strip()]


def main():
    wf = Workflow("ticket_triage", enable_profiling=True)
    wf.state["round"] = 0
    wf.state["last_batch"] = []

    classifier = AgentNode(wf, "Classifier", role="triage", llm=CustomLLM(_classifier_response))
    billing_specialist = AgentNode(wf, "BillingSpecialist", role="billing", llm=CustomLLM(_billing_response))
    tech_specialist = AgentNode(wf, "TechSpecialist", role="technical", llm=CustomLLM(_technical_response))
    generalist = AgentNode(wf, "Generalist", role="general", llm=CustomLLM(_general_response))
    reviewer = AgentNode(wf, "Reviewer", role="review", llm=CustomLLM(_review_response))

    ticket_loader = ServiceNode(wf, "TicketLoader", split_tickets)

    print("Enter semicolon-separated tickets (empty line or 'quit' to stop):")
    while True:
        try:
            raw = input("Batch> ")
        except (EOFError, KeyboardInterrupt):
            break

        if raw.strip().lower() == "quit":
            break
        if not raw.strip():
            break

        tickets = ticket_loader.execute(wf, raw)
        wf.state["last_batch"] = tickets
        wf.state["round"] += 1

        for index, ticket in enumerate(tickets, start=1):
            classification = wf.send("Classifier", ticket).content

            if classification == "billing":
                owner = "BillingSpecialist"
            elif classification == "technical":
                owner = "TechSpecialist"
            else:
                owner = "Generalist"

            draft = wf.send(owner, f"ticket#{index}: {ticket}").content

            if "urgent" in ticket.lower():
                final = wf.send("Reviewer", f"urgent review for {owner}: {draft}").content
            else:
                final = wf.send("Reviewer", f"review for {owner}: {draft}").content

            print(f"[{owner}] {final}")

    if wf.enable_profiling:
        print("\nProfiling summary:")
        for agent_name, stats in wf.profile_data.items():
            print(
                f"{agent_name}: calls={stats['calls']} "
                f"latency={stats['total_latency']:.4f}s memory={stats['total_memory']}"
            )


if __name__ == "__main__":
    main()
