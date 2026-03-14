"""Quick smoke test for the choreo_mini framework.

Run with ``python examples/foo.py`` from the project root (or install the
package in editable mode) to ensure imports work.
"""

import sys
from pathlib import Path

# Ensure this script can import the local package when run from the examples/ folder.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from choreo_mini.core.workflow import Workflow
from choreo_mini.core.nodes import AgentNode
from choreo_mini.core.llm import CustomLLM


def main():
    wf = Workflow("demo", enable_profiling=True)

    # simple CLI loop with a trivial LLM that echoes input
    echo_llm = CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")
    greeter = AgentNode(wf, "Greeter", role="greeter", llm=echo_llm)

    print("Type something (empty line to quit):")
    while True:
        try:
            text = input("You> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        resp = wf.send("Greeter", text)
        print("Bot>", resp.content)

    if wf.enable_profiling:
        print("\nProfiling summary:")
        for agent, stats in wf.profile_data.items():
            print(f"Agent {agent}: calls={stats['calls']} latency={stats['total_latency']:.3f}s memory={stats['total_memory']} bytes")


if __name__ == "__main__":
    main()
