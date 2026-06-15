# MotifAI

[![CI](https://github.com/Sivasathivel/MotifAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivasathivel/MotifAI/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/motif-ai)](https://pypi.org/project/motif-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/motif-ai)](https://pypi.org/project/motif-ai/)
[![License](https://img.shields.io/badge/license-Choreo--Mini--1.2-blue)](LICENSE)

**Lightweight Python framework for multi-agent LLM workflows** — MARL episode loop, cost-aware LLM pool routing, OTEL observability, and optional compile-to-any-runtime.

---

## What MotifAI does that others don't

| Feature | LangGraph | CrewAI | AutoGen | **MotifAI** |
|---------|-----------|--------|---------|-----------------|
| Write once, compile to any runtime | ✗ | ✗ | ✗ | ✓ |
| Epistemic belief state per agent | ✗ | ✗ | ✗ | ✓ |
| MARL episode loop + custom termination | ✗ | ✗ | ✗ | ✓ |
| Cost-aware LLM pool with fallback | ✗ | ✗ | ✗ | ✓ |
| OTEL-compatible tracing out of the box | ✗ | ✗ | ✗ | ✓ |
| MCP server exposure (zero config) | ✗ | ✗ | ✗ | ✓ |
| Any OpenAI-compatible endpoint | ✓ | ✓ | ✓ | ✓ |

---

## Headline demo — CUSMA / USMCA multi-party negotiation

Three AI agents — USA, Canada, Mexico — negotiate a trade agreement in real time. Each country runs its own `Workflow`, calls a real LLM, and signals when it has achieved its goals. The episode ends asymmetrically: **USA stops when it has secured the lion's share; Canada and Mexico stop when core interests are protected OR trade diversification makes further concessions irrelevant.**

```bash
pip install motif-ai openai
export OPENAI_API_KEY=sk-...
python examples/cusma_negotiation.py
```

Sample terminal output (3 rounds, live):

```
╔══════════════════════════════════════════════════════════════════════╗
║           CUSMA / USMCA  —  Multi-Agent Trade Negotiation            ║
║    Powered by motif-ai  ·  Multi-Agent Reinforcement Learning     ║
╚══════════════════════════════════════════════════════════════════════╝

── Round 1 of 8 ──────────────────────────────────────────────────────

🇺🇸  USA
  Automotive tariffs: Demand immediate zero tariffs. ...
  READY_TO_STOP: NO
  STOP_REASON: We have not yet secured dairy access or energy pricing.

🇨🇦  Canada
  Supply management is constitutionally protected and politically untouchable...
  READY_TO_STOP: NO

🇲🇽  Mexico
  A flat $16/hr floor is economically incoherent in Mexico. GAGI-adjusted
  productivity benchmarks by sector are the only fair metric...
  READY_TO_STOP: NO

── Round 3 of 8 ──────────────────────────────────────────────────────

🇺🇸  USA   ✓ DONE  "The United States has secured the lion's share —
                    dairy access, zero tariffs, wage floors, and digital
                    openness — America First has delivered."

🇨🇦  Canada ✓ DONE  "Core interests protected, trade diversification makes
                     further CUSMA concessions unnecessary."

🇲🇽  Mexico ✓ DONE  "Agricultural sovereignty protected, CATL battery
                     ruling secured. CUSMA is now one of three major trade
                     corridors rather than a dependency."

◈ All parties satisfied — negotiation concluded after round 3
```

Each party stops for a **different reason**. No Nash equilibrium required.

---

## Installation

```bash
pip install motif-ai
```

Optional extras:

```bash
pip install "motif-ai[otel]"      # OpenTelemetry export
pip install "motif-ai[langgraph]" # compile to LangGraph
pip install "motif-ai[crewai]"    # compile to CrewAI
pip install "motif-ai[autogen]"   # compile to AutoGen
```

---

## Core concepts

### Workflow subclass

Define agents as instance attributes. No manual state wiring:

```python
from motif_ai import Workflow, AgentNode, LLM

class TradingAdvisor(Workflow):
    def __init__(self, llm):
        super().__init__("trading-advisor")
        self.analyst = AgentNode(self, "Analyst", role="Trade data analyst", llm=llm)
        self.advisor = AgentNode(self, "Advisor", role="Senior policy advisor", llm=llm)

    def advise(self, question: str) -> str:
        analysis = self.send("Analyst", question).content
        return self.send("Advisor", analysis).content
```

### Any OpenAI-compatible endpoint

```python
from motif_ai import LLM, CustomLLM

# OpenAI
llm = LLM(api_key="sk-...", endpoint="https://api.openai.com", model="gpt-4o")

# Anthropic (via OpenAI-compat headers)
llm = LLM(
    endpoint="https://api.anthropic.com",
    model="claude-opus-4-5",
    headers={"x-api-key": "sk-ant-...", "anthropic-version": "2023-06-01"},
)

# Local Ollama
llm = LLM(endpoint="http://localhost:11434", model="llama3.2")

# Groq, Together, vLLM, LM Studio, ...
llm = LLM(api_key="...", endpoint="https://api.groq.com/openai", model="llama-3.3-70b-versatile")

# Any callable — great for tests or rule-based stubs
llm = CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")
```

### Epistemic belief state

Every workflow and agent carries a confidence-weighted belief map:

```python
# record what the workflow has observed
wf.beliefs.observe("canada_dairy_stance", "defensive", confidence=0.9, step=round_)

# per-agent theory-of-mind beliefs
wf.get_agent_belief("Analyst").observe_agent("Advisor", "risk_tolerance", "low", confidence=0.7)

# decay beliefs over time (model information aging)
wf.decay_all_beliefs(factor=0.95)

# snapshot for logging or LLM context injection
print(wf.beliefs.snapshot())
```

### MARL episode loop

Run multi-party experiments where agents observe each other's actions, receive rewards, and update their proposals each round:

```python
from motif_ai import Episode, max_rounds_terminator

def huf_reward(agent_id, action, env, step, trajectory):
    """Your Human Utility Function — score each party's outcome."""
    ...

ep = Episode(
    agents={
        "USA":    usa_wf.propose,
        "Canada": canada_wf.propose,
        "Mexico": mexico_wf.propose,
    },
    reward_fn=huf_reward,
    termination_fn=_all_parties_satisfied,   # custom — each stops for its own reason
    max_rounds=8,
)

trajectory = ep.run()
```

The episode records every round's env snapshot, agent actions, and rewards — ready for policy-update steps or downstream analysis.

### LLM pool — cost-aware routing with fallback

`LLMPool` is a drop-in replacement for `llm=` on any `AgentNode`. It routes across multiple backends according to a policy and falls back transparently on failure:

```python
from motif_ai import LLMPool, LLMCandidate

pool = LLMPool(
    candidates=[
        LLMCandidate(llm=local_llm,  name="local-llama",  cost_per_1k_tokens=0.0),
        LLMCandidate(llm=gpt4o_mini, name="gpt-4o-mini",  cost_per_1k_tokens=0.15),
        LLMCandidate(llm=gpt4o,      name="gpt-4o",       cost_per_1k_tokens=2.50),
    ],
    policy="cost_first",   # cost_first | priority | round_robin | reliability
    fallback=True,
)

self.analyst = AgentNode(self, "Analyst", role="...", llm=pool)
```

Four routing policies:

| Policy | Behaviour |
|--------|-----------|
| `cost_first` | Cheapest first; escalate on failure |
| `priority` | Explicit preference order independent of cost |
| `round_robin` | Spread load evenly across equivalent models |
| `reliability` | Adapts at runtime — candidates that fail more are deprioritised |

```python
# inspect routing decisions after a run
for s in pool.stats:
    print(f"{s['name']}  calls={s['calls']}  success_rate={s['success_rate']:.0%}")
```

---

## Observability and tracing

Every framework boundary emits a typed event. Pass a hook to any workflow, episode, or pool:

```python
from motif_ai import StdoutHook, JsonFileHook, CompositeHook, OTLPHook

hook = CompositeHook(
    StdoutHook(color=True),                      # coloured terminal output
    JsonFileHook("output/trace.ndjson"),          # NDJSON for offline analysis
    OTLPHook(endpoint="http://localhost:4317"),   # OpenTelemetry Collector
)

wf = MyWorkflow(observability=hook)
ep = Episode(..., observability=hook)
pool = LLMPool(..., observability=hook)
```

Built-in events (all carry `trace_id` / `span_id` in OTEL-compatible hex format):

| Event | When |
|-------|------|
| `AgentCallStart` / `AgentCallEnd` | Agent receives / returns a message |
| `LLMRequestStart` / `LLMRequestEnd` | HTTP call to the LLM endpoint |
| `LLMRetry` | Automatic retry after a transient failure |
| `EpisodeStepStart` / `EpisodeStepEnd` | Each MARL round (with per-agent rewards) |
| `LLMPoolRoute` | Which candidate was selected and why |
| `LLMPoolFallback` | Primary failed; which candidate is next |

Write your own hook in one method:

```python
class MetricsHook:
    def on_event(self, event):
        if isinstance(event, AgentCallEnd):
            metrics.histogram("agent.latency", event.latency_s, tags={"agent": event.agent_name})
```

### Full workflow state snapshot

```python
import json
print(json.dumps(wf.dump(), indent=2))
# {
#   "name": "usa_workflow",
#   "trace_id": "ba2d02fb2955...",
#   "agents": {
#     "USANegotiator": {"calls": 3, "latency_s": 1.42, "history_len": 6}
#   },
#   "beliefs": { ... },
#   "profiling": { ... }
# }
```

---

## Structured exceptions

MotifAI raises typed exceptions so you can handle failures precisely:

```python
from motif_ai import (
    AgentNotFoundError,   # wf.send() called with an unregistered agent name
    AgentRegistrationError,  # duplicate agent added to a workflow
    EpisodeError,         # episode misuse (step after done, etc.)
    LLMError,             # HTTP / API failure with status code and attempt count
    ConversionError,      # compile-time eval failure with in-scope variable dump
)
```

---

## Compile to another runtime (optional)

When you're ready to migrate, compile your `Workflow` subclass with the CLI:

```bash
motif_ai -f examples/my_workflow.py -b langgraph -o output/graph.py
motif_ai -f examples/my_workflow.py -b crewai    -o output/crewai_crew.py
motif_ai -f examples/my_workflow.py -b autogen   -o output/autogen_agents.py
```

Only the **subclass pattern** is a valid compilation target:

```python
class TicketTriageWorkflow(Workflow):   # ← subclass: compiles ✓
    def __init__(self):
        super().__init__("triage")
        self.classifier = AgentNode(self, "Classifier", role="...", llm=llm)
        self.reviewer   = AgentNode(self, "Reviewer",   role="...", llm=llm)

    def process(self, ticket: str) -> str:
        route = self.send("Classifier", ticket).content
        return self.send("Reviewer", route).content
```

A flat `wf = Workflow(...)` inside `main()` runs fine but cannot be compiled — use the subclass pattern from the start if you plan to migrate later.

---

## MCP server — expose your workflow to any MCP client

```python
from motif_ai import WorkflowMCPServer

server = WorkflowMCPServer(my_workflow)
server.serve_sse()    # HTTP SSE for network clients
server.serve_stdio()  # stdio for Claude Desktop
```

What gets registered automatically:

| MCP concept | What it is |
|-------------|-----------|
| **Tool** per `AgentNode` | Send a message to that agent and get its reply |
| **Resource** `workflow://{name}/beliefs` | Live belief-state snapshot |
| **Resource** `workflow://{name}/agents/{agent}/history` | Full conversation history |

---

## Quick-start from source

```bash
git clone https://github.com/Sivasathivel/MotifAI
cd motif-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest          # 272 tests, all green

# Run the CUSMA negotiation demo (demo mode — no API key needed)
python examples/cusma_negotiation.py

# Run the LLM pool scheduling demo
python examples/llm_pool_scheduling.py
```

---

## Examples

| File | What it shows |
|------|--------------|
| `examples/cusma_negotiation.py` | Three-party trade negotiation with asymmetric termination and live terminal output |
| `examples/llm_pool_scheduling.py` | Four routing policies — cost_first, priority, round_robin, reliability — with live fallback |
| `examples/customer_ops_observability.py` | `CompositeHook` with custom `EventCounterHook` and NDJSON logging |
| `examples/multi_agent_debate.py` | Proponent vs. Opponent in an `Episode` loop with `wf.dump()` snapshot |
| `examples/customer_ops_url.py` | Four-agent customer-ops triage workflow, compile-ready subclass |

---

## Status

- **Core runtime** (`Workflow`, `AgentNode`, `LLM`, `BeliefState`, `Episode`) — stable, **272 tests passing** across Python 3.10–3.13
- **LLMPool** — production-ready, 4 routing policies, stats, observability events
- **Observability / OTEL** — full hook system, NDJSON, OTLP export, composite fan-out
- **MCP client + server** — consume external MCP servers and expose workflows as MCP servers
- **LangGraph backend** — most mature; supports branching, loop budgets, service node dispatch
- **CrewAI / AutoGen backends** — structurally correct scaffolding; runtime fidelity in progress

---

## Roadmap

- **Developer documentation** — full API reference, tutorials, architecture guide
- **Policy update hooks** — plug gradient or tabular RL updates into the `Episode` trajectory
- **A2A (agent-to-agent)** — explicit handoffs, delegation, structured cross-agent messaging
- **CrewAI / AutoGen parity** — deeper runtime fidelity for routing, loops, state transitions
- **Async-native episode** — `async def step()` for concurrent agent calls per round

---

## Author

**Sivasathivel Kandasamy** — [LinkedIn](https://www.linkedin.com/in/sivasathivelkandasamy/)

---

## License

This project is released under the [MotifAI Source License](LICENSE).

**Allowed:**
- Use as a library or dependency in any project, including commercial applications
- Modify and contribute back
- Keep your larger application closed-source when it only depends on motif-ai

**Not allowed:**
- Building and selling a product or SaaS where motif-ai is the core value being offered
- Distributing a modified derivative without releasing the derivative source under the same license
- Selling paid access to motif-ai or a derivative API/service

Citation is required in public materials. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.
