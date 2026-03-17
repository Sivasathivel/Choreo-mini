# Choreo-Mini

**A lightweight Python meta-framework for building, experimenting with, and orchestrating LLM-based agents.**

Modern agent frameworks — LangGraph, CrewAI, AutoGen — each solve similar orchestration problems but introduce fragmented abstractions and steep learning curves. Choreo-Mini provides a Python-native developer experience that allows engineers to prototype agent workflows quickly while retaining the flexibility to run them on different orchestration runtimes.

Instead of forcing developers to commit to a single framework, Choreo-Mini acts as an orchestration meta-layer: you express your workflow once in plain Python, and it compiles it to your target runtime.

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

## Installation

```bash
pip install choreo-mini
```

---

## Quick Start

**Define your workflow once in plain Python:**

```python
# examples/my_workflow.py
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

**Compile to any supported runtime:**

```bash
# to LangGraph
choreo_mini -f examples/my_workflow.py -b langgraph -o output/langgraph_output.py

# to CrewAI
choreo_mini -f examples/my_workflow.py -b crewai  -o output/crewai_output.py

# to AutoGen
choreo_mini -f examples/my_workflow.py -b autogen -o output/autogen_output.py
```

**Run the generated LangGraph app directly:**

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

This project is released under the [Choreo-Mini Community License](LICENSE).

Key terms: derivatives must remain open source; contributors waive ownership claims;
the author reserves the right to relicense future versions.
