# Choreo-Mini

**A lightweight Python meta-framework for building, experimenting with, and orchestrating LLM-based agents.**

Modern agent frameworks — LangGraph, CrewAI, AutoGen — each solve similar orchestration problems but introduce fragmented abstractions and steep learning curves. Choreo-Mini provides a Python-native developer experience that allows engineers to prototype agent workflows quickly while retaining the flexibility to run them on different orchestration runtimes.

Instead of forcing developers to commit to a single framework, Choreo-Mini acts as an orchestration meta-layer: you express your workflow once in plain Python, and it compiles it to your target runtime, if needed.

---

## How it works

```
Your Python Workflow
        │
        ▼
    AST Parser  ── extracts agents, services, control flow, state mutations
        │
        ▼
  Intermediate Representation
        │
        ▼
  Jinja2 Template Compiler
        │
    ┌───┴────────────────┐
    ▼         ▼          ▼
LangGraph  CrewAI    AutoGen
```

---

## Status

Choreo-Mini is an actively developed prototype. The **LangGraph** backend is the most mature path, supporting branch-aware conditional routing, loop budgets, and service node dispatch. **CrewAI** and **AutoGen** backends produce structurally correct scaffolding and are being extended for deeper runtime fidelity.

---

## Roadmap

Near-term next steps:

- Build **MCP server support** so choreo-mini workflows can be exposed as tools/resources/prompts via Model Context Protocol.
- Add **A2A (agent-to-agent) support** for explicit handoffs, delegation, and structured cross-agent messaging.
- Introduce first-class **tool usage orchestration** (typed tool schemas, argument validation, retries/timeouts, and tool-call tracing).
- Bring **CrewAI** and **AutoGen** behavior closer to LangGraph parity for routing, loop handling, and workflow state transitions.
- Replace provider stubs with production-ready adapters for OpenAI, Anthropic, and Gemini in `LLM.generate()`.
- Expand AST parser coverage for more real-world Python patterns (deeper branching, loop variants, and service composition).
- Add backend snapshot tests and regression fixtures around `examples/foo2.py` plus additional realistic workflows.
- Improve CLI ergonomics with clearer conversion diagnostics and optional inspection output for parsed workflow data.
- Publish reference demos and backend comparison benchmarks built from the same source workflow.

---

## Installation

```bash
pip install choreo-mini
```

---

## Quick Start

### 1) Define your workflow once in plain Python

```python
# my_workflow.py
from choreo_mini.core.workflow import Workflow
from choreo_mini.core.nodes import AgentNode, ServiceNode

wf = Workflow("support", enable_profiling=True)

# agents auto-register with the workflow on creation
classifier = AgentNode(wf, "Classifier", role="ticket triage")
specialist  = AgentNode(wf, "Specialist",  role="issue resolver")


def main():
    while True:
        ticket = input("Ticket> ")
        if not ticket.strip():
            break
        category = wf.send("Classifier", ticket)
        response = wf.send("Specialist", f"{category.content}: {ticket}")
        print(response.content)
```

### 2) Convert your workflow to each backend

```bash
# to LangGraph
choreo_mini -f my_workflow.py -b langgraph -o output/langgraph_output.py

# to CrewAI
choreo_mini -f my_workflow.py -b crewai -o output/crewai_output.py

# to AutoGen
choreo_mini -f my_workflow.py -b autogen -o output/autogen_output.py
```

### 3) Convert the bundled complete example (recommended)

```bash
# examples/foo2.py includes routing, specialist selection, review flow, and looped batch handling
choreo_mini -f examples/foo2.py -b langgraph -o output/foo2_langgraph_output.py
choreo_mini -f examples/foo2.py -b crewai -o output/foo2_crewai_output.py
choreo_mini -f examples/foo2.py -b autogen -o output/foo2_autogen_output.py
```

Optional minimal smoke test (`examples/foo.py`):

```bash
choreo_mini -f examples/foo.py -b langgraph -o output/foo_langgraph_output.py
```

### 4) Run a generated LangGraph app directly

```python
from output.langgraph_output import app
from choreo_mini import Workflow, AgentNode
from choreo_mini.core.llm import LLM

wf = Workflow("support", enable_profiling=True)
AgentNode(wf, "Classifier", role="triage", llm=LLM.create("openai", api_key="..."))
AgentNode(wf, "Specialist",  role="resolver", llm=LLM.create("openai", api_key="..."))

result = app.invoke({"wf": wf, "input": "login broken", "messages": [], "loop_budget": 1})
print(result["last_response"])
```

---

## Python API

```python
from choreo_mini import Workflow, AgentNode, ServiceNode, LLM, CustomLLM

# create a workflow — enable_profiling tracks latency and memory per agent
wf = Workflow("myflow", enable_profiling=True)

# attach agents with a real or custom LLM
A1 = AgentNode(wf, "Greeter",   role="greeter",   llm=LLM.create("openai",    api_key="..."))
A2 = AgentNode(wf, "Responder", role="responder", llm=LLM.create("anthropic", api_key="..."))

# CustomLLM wraps any callable — great for local models or mocks
A3 = AgentNode(wf, "Fallback", role="fallback",
               llm=CustomLLM(lambda prompt, **kw: f"Fallback: {prompt}"))

# service nodes wrap arbitrary data functions
loader = ServiceNode(wf, "Loader", service_fn=lambda wf, path: open(path).read())

# send a message to an agent — history and profiling handled automatically
response = wf.send("Greeter", "Hello")
print(response.content)

# inspect profiling
print(wf.get_profile("Greeter"))  # {"calls": 1, "total_latency": ..., "total_memory": ...}
```

---

## Observability

When `enable_profiling=True`, Choreo-Mini instruments every agent call automatically:

| Metric | Description |
|--------|-------------|
| `call_count` | Number of times the agent was invoked |
| `total_latency` | Cumulative wall-clock inference time (seconds) |
| `total_memory` | Cumulative memory delta across calls (bytes) |
| `history` | Full conversation history per agent |

This makes it straightforward to compare runtimes or detect bottlenecks before committing to a specific framework.

---

## Supported LLM Providers

| Provider | Class | Notes |
|----------|-------|-------|
| OpenAI | `LLM.create("openai", api_key=...)` | Stub — wire real `openai` client in `generate()` |
| Anthropic | `LLM.create("anthropic", api_key=...)` | Stub — wire real `anthropic` client |
| Gemini | `LLM.create("gemini", api_key=...)` | Stub — wire real Google client |
| Custom | `CustomLLM(fn)` | Wraps any `(prompt, **kw) -> str` callable |

The provider stubs make scaffolding easy; replace `generate()` with the real SDK call for production.

---

## Development

```bash
git clone https://github.com/Sivasathivel/Choreo-mini
cd choreo-mini
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

CI runs the full regression suite against Python 3.10, 3.11, and 3.12 on every push.

---

## Author

**Sivasathivel Kandasamy** — [LinkedIn](https://www.linkedin.com/in/sivasathivelkandasamy/)

---

## License

This project is released under the [Choreo-Mini Source License](LICENSE).

**What is allowed:**
- Use choreo-mini as a library or dependency inside any project, including
    commercial applications and internal enterprise deployments — no restriction.
- Modify the source and contribute back.
- Keep your larger application closed source when it only depends on
    choreo-mini and is not itself a derivative of choreo-mini.
- Ship a proprietary larger product that uses unmodified choreo-mini as a
    component, with license notices preserved.

**What is not allowed:**
- Building and selling a product, plugin, extension, or SaaS where
    choreo-mini is the core value being offered by a third party.
- Distributing or hosting a modified derivative of choreo-mini without
    releasing the derivative source under the same license.
- Selling paid access to choreo-mini or a derivative API/service, even when
    bundled with other paid features.

**Other terms:** citation is required in public materials and user-facing
interfaces (for example docs, demos, public repos, benchmark reports, websites,
or service UI); contributors grant the author relicensing rights; the author
reserves the right to publish enterprise/commercial editions. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.
