# Choreo-Mini

[![CI](https://github.com/Sivasathivel/Choreo-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivasathivel/Choreo-mini/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/choreo-mini)](https://pypi.org/project/choreo-mini/)
[![Python versions](https://img.shields.io/pypi/pyversions/choreo-mini)](https://pypi.org/project/choreo-mini/)
[![License](https://img.shields.io/badge/license-Choreo--Mini--1.2-blue)](LICENSE)

**A lightweight Python meta-framework for building, experimenting with, and orchestrating LLM-based agents.**

Modern agent frameworks — LangGraph, CrewAI, AutoGen — each solve similar orchestration problems but introduce fragmented abstractions and steep learning curves. Choreo-Mini provides a Python-native developer experience that lets you prototype agent workflows quickly, run genuine multi-agent experiments, and compile to any target runtime when you're ready.

Instead of forcing you to commit to a single framework, Choreo-Mini acts as an orchestration meta-layer: express your workflow once in plain Python, and compile it to your target runtime if needed.

---

## What makes Choreo-Mini different

| Feature | LangGraph | CrewAI | AutoGen | **Choreo-Mini** |
|---------|-----------|--------|---------|-----------------|
| Subclass-based workflow definition | ✗ | ✗ | ✗ | ✓ |
| Epistemic belief state per agent | ✗ | ✗ | ✗ | ✓ |
| Built-in MARL episode loop | ✗ | ✗ | ✗ | ✓ |
| Nash convergence detection | ✗ | ✗ | ✗ | ✓ |
| Export to other frameworks | ✗ | ✗ | ✗ | ✓ |
| Any OpenAI-compatible endpoint | ✓ | ✓ | ✓ | ✓ |

---

## How it works

```
Your Python Workflow  (subclass Workflow, define AgentNodes)
        │
        ├── Run directly with any OpenAI-compatible endpoint
        │
        ├── Run MARL experiments (Episode loop, HUF/reward, Nash convergence)
        │
        └── Compile to another runtime (optional)
                    │
              AST Parser + Jinja2
                    │
           ┌────────┼────────┐
           ▼        ▼        ▼
       LangGraph  CrewAI  AutoGen
```

---

## Core concepts

### Workflow subclass pattern

Define agents as instance attributes — no manual state wiring:

```python
from choreo_mini import Workflow, AgentNode, LLM

class TradingAdvisor(Workflow):
    def __init__(self, llm):
        super().__init__("trading-advisor")
        self.analyst  = AgentNode(self, "Analyst",  role="Trade data analyst", llm=llm)
        self.advisor  = AgentNode(self, "Advisor",  role="Senior policy advisor", llm=llm)

    def advise(self, question: str) -> str:
        analysis = self.send("Analyst", question)
        self.beliefs.observe("last_question", question, confidence=1.0)
        reply = self.send("Advisor", analysis.content)
        return reply.content
```

### Epistemic belief state

Every agent and workflow has a confidence-weighted belief map — a capability not found in LangGraph, CrewAI, or AutoGen:

```python
# workflow-level world beliefs
wf.beliefs.observe("penalty", 0.12, confidence=0.9, step=round_)

# per-agent theory-of-mind beliefs
wf.get_agent_belief("Analyst").observe_agent("agent2", "stance", "defensive", confidence=0.7)

# decay beliefs at end of each round (model passage of time)
wf.decay_all_beliefs(factor=0.95)

# snapshot for logging or passing as context to the LLM
print(wf.beliefs.snapshot())
```

### Any OpenAI-compatible endpoint

One class, any provider — no SDK lock-in:

```python
from choreo_mini import LLM, CustomLLM

# OpenAI
llm = LLM(api_key="sk-...", endpoint="https://api.openai.com", model="gpt-4o")

# Anthropic (via OpenAI-compat headers)
llm = LLM(
    endpoint="https://api.anthropic.com",
    model="claude-opus-4-5",
    headers={"x-api-key": "sk-ant-...", "anthropic-version": "2023-06-01"},
)

# Local Ollama (no auth)
llm = LLM(endpoint="http://localhost:11434", model="llama3.2")

# Groq, Together, vLLM, LM Studio, ...
llm = LLM(api_key="...", endpoint="https://api.groq.com/openai", model="llama-3.3-70b-versatile")

# Any callable — great for tests, local models, or rule-based stubs
llm = CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")
```

The `LLM` class normalises endpoint URLs (strips accidental `/v1` suffixes), omits the `Authorization` header when no API key is set, enforces a configurable timeout (default 60 s), and surfaces the full API error body on failures.

### MARL episode loop

Run multi-agent reinforcement learning experiments with reward functions and Nash convergence detection baked in:

```python
from choreo_mini import Episode, nash_convergence_detector

ep = Episode(
    agents={
        "agent1":    agent1_workflow.propose,
        "agent2": agent2_workflow.propose,
        "agent3": Agent3_workflow.propose,
    },
    env={"penalty": 0.12, "coverage": 0.65, ...},
    reward_fn=huf_reward,           # your Human Utility Function
    env_update_fn=negotiation_step, # how proposals combine into new state
    termination_fn=nash_convergence_detector(window=5, reward_threshold=0.005),
    max_rounds=40,
)

trajectory = ep.run()
print(ep.summary())
```

Each `EpisodeStep` records the env snapshot, every agent's action, and every agent's reward — ready to feed a policy-update step.

---

## MARL experiment — Benefit maximisation

`examples/marl_huf_experiment.py` ships a ready-to-run multi-agent experiment where agent1, agent2, and Agent3 negotiate five trade parameters to maximise their national Human Utility Function scores:

| Parameter | Agent 1 | Agent 2 | Agent 3 |
|-----------|---------------|-------------------|-------------------|
| `penalty` | ↓ | ↓ | neutral |
| `coverage` | neutral | ↑ | ↑ |
| `safety` | ↑ | neutral | ↓ |
| `source` | ↑ | neutral | ↓ |
| `env_score` | ↑ | ↑ | neutral |

```bash
python examples/marl_huf_experiment.py
```

Sample output:
```
======================================================================
   MARL — Benefit Maximisation Experiment
======================================================================
  Round      Agent1   Agent2   Agent3   GlobalHUF
  --------------------------------------------
   init   0.6700   0.6830   0.5290      1.8820
      1   0.6700   0.6830   0.5290      1.8820
      ...
     40   0.7840   1.0000   0.6650      2.4490
  --------------------------------------------
  Converged after 40 rounds

  Final trade parameters
  penalty           0.1200  →  0.0000  (▼ 0.1200)
  coverage          0.6500  →  1.0000  (▲ 0.3500)
  safety            0.5500  →  0.7500  (▲ 0.2000)
  source            0.6200  →  0.6200  (─ 0.0000)
  env_score         0.5800  →  1.0000  (▲ 0.4200)

  Global Benefit Score: 1.8820  →  2.4500  (+0.5680)
```

Swap in a real LLM to have agents reason over the parameters in natural language:

```python
llm = LLM(api_key="sk-...", endpoint="https://api.openai.com", model="gpt-4o")
agent1 = agent1Workflow(llm=llm)
```

---

## Installation

```bash
pip install choreo-mini
```

Or from source:

```bash
git clone https://github.com/Sivasathivel/Choreo-mini
cd choreo-mini
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## Quick start — single agent

```python
from choreo_mini import Workflow, AgentNode, LLM

class Advisor(Workflow):
    def __init__(self, llm):
        super().__init__("advisor")
        self.agent = AgentNode(self, "Advisor", role="Senior trade policy analyst", llm=llm)

    def ask(self, question: str) -> str:
        return self.send("Advisor", question).content

llm = LLM(api_key="sk-...", endpoint="https://api.openai.com", model="gpt-4o-mini")
wf  = Advisor(llm)
print(wf.ask("What are the three main pillars of CUSMA/USMCA?"))
```

---

## Compile to another runtime (optional)

When you're ready to migrate to LangGraph, CrewAI, or AutoGen, compile your workflow with the CLI:

```bash
choreo_mini -f examples/foo2.py -b langgraph -o output/foo2_langgraph.py
choreo_mini -f examples/foo2.py -b crewai    -o output/foo2_crewai.py
choreo_mini -f examples/foo2.py -b autogen   -o output/foo2_autogen.py
```

Full customer-operations example (four-way routing, risk escalation, QA review):

```bash
choreo_mini -f examples/customer_ops_url.py -b langgraph \
            -o output/customer_ops_langgraph.py
```

---

## Conversion guide — which pattern to use

Choreo-Mini supports two authoring styles.  Choose the right one for your use case:

| | Flat / functional | **Subclass (recommended)** |
|---|---|---|
| **How** | `wf = Workflow(...)` inside `main()` | `class X(Workflow)` with `__init__` + methods |
| **Run directly** | ✓ | ✓ |
| **Compile to LangGraph / CrewAI / AutoGen** | ✗ | ✓ |
| **Best for** | Quick scripts, one-off experiments | Shareable, reusable, production workflows |

### Subclass pattern (correct conversion target)

```python
class TicketTriageWorkflow(Workflow):
    def __init__(self):
        super().__init__("ticket_triage", enable_profiling=True)
        # Declare all agents in __init__
        self.classifier = AgentNode(self, "Classifier", role="triage classifier", llm=...)
        self.reviewer   = AgentNode(self, "Reviewer",   role="quality reviewer",  llm=...)

    def process_batch(self, raw_batch: str) -> list:
        # Execution logic in a method — this is what gets compiled
        tickets = raw_batch.split(";")
        results = []
        for ticket in tickets:
            route = self.send("Classifier", ticket).content.strip().lower()
            final = self.send("Reviewer", route).content
            results.append(final)
        return results

# Run directly
wf = TicketTriageWorkflow()
wf.process_batch("billing question; app crash")

# Or compile to LangGraph
# choreo_mini -f myfile.py -b langgraph -o output/graph.py
```

### Flat pattern (run directly only)

```python
def main():
    wf = Workflow("my_workflow")
    agent = AgentNode(wf, "Agent", role="...", llm=...)

    response = wf.send("Agent", "hello")
    print(response.content)

# python myfile.py  ← works fine
# choreo_mini -f myfile.py -b langgraph  ← NOT supported
```

> **Rule of thumb**: if you ever plan to compile your workflow, use the subclass pattern from the start.  It runs identically to the flat pattern at development time and unlocks compilation later.

---

## Observability

When `enable_profiling=True`, every agent call is instrumented automatically:

| Metric | Description |
|--------|-------------|
| `calls` | Number of times the agent was invoked |
| `total_latency` | Cumulative wall-clock inference time (seconds) |
| `total_memory` | Cumulative memory delta across calls (bytes) |
| `history` | Full conversation history per agent |

```python
wf = MyWorkflow(enable_profiling=True, llm=llm)
wf.run("some input")
print(wf.get_profile("Analyst"))
# {"Analyst": {"calls": 3, "total_latency": 1.42, "total_memory": 8192}}
```

---

## MCP server — expose your workflow to any MCP client

`WorkflowMCPServer` wraps any `Workflow` and makes it instantly accessible from
Claude Desktop, other choreo-mini workflows, or any MCP-compatible client —
with zero extra configuration.

```python
from choreo_mini import WorkflowMCPServer

server = WorkflowMCPServer(my_workflow, host="0.0.0.0", port=8000)

server.serve_sse()          # HTTP SSE — accessible over a network
server.serve_stdio()        # stdio — for Claude Desktop / subprocess
app = server.sse_app()      # raw Starlette ASGI app for custom uvicorn
```

What gets registered automatically:

| MCP concept | What it is |
|-------------|-----------|
| **Tool** per `AgentNode` | Sends a message to that agent and returns its reply |
| **Resource** `workflow://{name}/beliefs` | Live JSON snapshot of workflow belief state |
| **Resource** `workflow://{name}/agents/{agent}/history` | Full conversation history |

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "my-workflow": {
      "command": "python",
      "args": ["-m", "my_package.mcp_entrypoint"]
    }
  }
}
```

---

## Status

Choreo-Mini is an actively developed prototype.

- **Core runtime** (`Workflow`, `AgentNode`, `LLM`, `BeliefState`, `Episode`) — stable, 165 tests passing across Python 3.10–3.13
- **MCP client + server** — consume external MCP servers and expose workflows as MCP servers
- **LangGraph backend** — most mature; supports branching, loop budgets, service node dispatch
- **CrewAI / AutoGen backends** — structurally correct scaffolding; runtime fidelity in progress

---

## Roadmap

- **Developer documentation** — full API reference, tutorials, and architecture guide
- **LLM-driven MARL strategies** — replace fixed-delta strategies with agents that reason over belief states to propose actions
- **Policy update hooks** — plug gradient or tabular RL updates directly into the `Episode` trajectory
- **Expanded test coverage** — integration tests against real MCP servers, end-to-end conversion pipeline tests
- **A2A (agent-to-agent)** — explicit handoffs, delegation, and structured cross-agent messaging
- **CrewAI / AutoGen parity** — deeper runtime fidelity for routing, loops, and state transitions
- **CLI improvements** — conversion diagnostics, IR inspection output

---

## Author

**Sivasathivel Kandasamy** — [LinkedIn](https://www.linkedin.com/in/sivasathivelkandasamy/)

---

## License

This project is released under the [Choreo-Mini Source License](LICENSE).

**What is allowed:**
- Use choreo-mini as a library or dependency inside any project, including commercial applications and internal enterprise deployments.
- Modify the source and contribute back.
- Keep your larger application closed source when it only depends on choreo-mini and is not itself a derivative of choreo-mini.
- Ship a proprietary larger product that uses unmodified choreo-mini as a component, with license notices preserved.

**What is not allowed:**
- Building and selling a product, plugin, extension, or SaaS where choreo-mini is the core value being offered by a third party.
- Distributing or hosting a modified derivative of choreo-mini without releasing the derivative source under the same license.
- Selling paid access to choreo-mini or a derivative API/service, even when bundled with other paid features.

**Other terms:** citation is required in public materials and user-facing interfaces (docs, demos, public repos, benchmark reports, websites, or service UI); contributors grant the author relicensing rights; the author reserves the right to publish enterprise/commercial editions. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.
