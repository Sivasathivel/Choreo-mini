"""Customer-ops pipeline with full observability — end-to-end sanity check.

Runs :class:`CustomerOpsWorkflow` in demo mode (no API key needed) while
routing every framework event to three sinks simultaneously via
:class:`~choreo_mini.core.observability.CompositeHook`:

1. **StdoutHook** — colour-coded, timestamped live output in the terminal.
2. **JsonFileHook** — NDJSON trace file (``output/customer_ops_trace.ndjson``).
3. A custom inline hook that counts events by type for the summary report.

At the end the script prints:

* Per-case routed responses.
* Workflow escalation and follow-up lists.
* Full ``wf.dump()`` snapshot — call counts, latency, beliefs, profiling.
* NDJSON event-type histogram from the log file.

Run with no configuration::

    python examples/customer_ops_observability.py

To run against a real LLM endpoint set environment variables and re-run::

    export CHOREO_LLM_URL=https://api.openai.com/v1/responses
    export CHOREO_API_TOKEN=sk-...
    export CHOREO_LLM_MODEL=gpt-4o-mini
    python examples/customer_ops_observability.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Allow running from the repo root or from examples/
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from choreo_mini.core.observability import (
    CompositeHook,
    JsonFileHook,
    ObservabilityEvent,
    ObservabilityHook,
    StdoutHook,
)

# Import the workflow and helpers from the sibling example module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from customer_ops_url import (
    EXAMPLE_BATCH,
    CustomerOpsWorkflow,
    RemoteLLMConfig,
    _normalize_endpoint,
    prompt_for_remote_config,
)


# ---------------------------------------------------------------------------
# Custom counting hook (demonstrates the ObservabilityHook protocol)
# ---------------------------------------------------------------------------

class EventCounterHook:
    """Lightweight hook that tallies events by type — useful for metrics dashboards."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.total_llm_latency_s: float = 0.0
        self.agent_calls: int = 0

    def on_event(self, event: ObservabilityEvent) -> None:
        etype = event.event_type
        self.counts[etype] = self.counts.get(etype, 0) + 1

        # Accumulate LLM latency from LLMRequestEnd events
        if etype == "llm_request_end":
            latency = getattr(event, "latency_s", 0.0) or 0.0
            self.total_llm_latency_s += latency

        if etype == "agent_call_end":
            self.agent_calls += 1

    def report(self) -> str:
        lines = ["Event-type counts:"]
        for etype, count in sorted(self.counts.items()):
            lines.append(f"  {etype}: {count}")
        lines.append(f"Total agent calls  : {self.agent_calls}")
        lines.append(f"Total LLM latency  : {self.total_llm_latency_s:.4f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def _load_ndjson(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    demo_mode = not os.getenv("CHOREO_LLM_URL", "").strip()
    log_path = Path(project_root) / "output" / "customer_ops_trace.ndjson"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- build the composite hook ---
    counter = EventCounterHook()
    hook = CompositeHook(
        StdoutHook(color=True, show_previews=True),
        JsonFileHook(str(log_path), append=False),
        counter,
    )

    _section("Setup")
    if demo_mode:
        print("CHOREO_LLM_URL not set — running in demo mode (no network calls).")
        wf = CustomerOpsWorkflow(demo_mode=True, observability=hook)
    else:
        print("Remote LLM mode — reading credentials from environment.")
        cfg = prompt_for_remote_config()
        wf = CustomerOpsWorkflow(client_config=cfg, observability=hook)

    print(f"Workflow trace_id : {wf.trace_id}")
    print(f"Log file          : {log_path}")

    # --- run the pipeline ---
    _section("Processing batch")
    print("Input batch:")
    for i, segment in enumerate(EXAMPLE_BATCH.split(";"), start=1):
        print(f"  [{i}] {segment.strip()}")
    print()

    t0 = time.time()
    results = wf.process_batch(EXAMPLE_BATCH)
    elapsed = time.time() - t0

    # --- per-case results ---
    _section("Results")
    cases = wf.state["last_batch"]
    for case, response in zip(cases, results):
        print(f"  [{case['case_id']}] priority={case['priority']}  route={case.get('route', '?')}")
        print(f"      → {response}")
        print()

    print(f"  Escalations : {wf.state['escalations']}")
    print(f"  Follow-ups  : {wf.state['follow_ups']}")
    print(f"  Wall time   : {elapsed:.3f}s")

    # --- dump() snapshot ---
    _section("Workflow state snapshot (wf.dump())")
    snapshot = wf.dump()
    print(json.dumps(snapshot, indent=2))

    # --- in-memory event counter report ---
    _section("In-memory event counter")
    print(counter.report())

    # --- NDJSON log summary ---
    _section(f"NDJSON trace — {log_path}")
    events = _load_ndjson(log_path)
    if events:
        type_counts: Dict[str, int] = {}
        for ev in events:
            t = ev.get("event_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Total events written: {len(events)}")
        for etype, count in sorted(type_counts.items()):
            print(f"  {etype}: {count}")
        print(f"\nFirst event sample:")
        print(json.dumps(events[0], indent=2))
    else:
        print("(no events — log file empty or missing)")


if __name__ == "__main__":
    main()
