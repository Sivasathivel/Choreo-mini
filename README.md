# Choreo-Mini

Choreo-Mini is a lightweight Python meta-framework designed to simplify the
development and experimentation of LLM agent workflows.

It allows developers to define agent workflows using a Python-native
programming model and compile them into modern orchestration runtimes such as:

- LangGraph
- CrewAI
- AutoGen

The goal is to remove the framework-specific learning curve while enabling
developers to experiment with agent orchestration concepts in real execution
environments.

---

## Status

⚠️ Choreo-Mini is currently an experimental prototype and work in progress.
Some components may contain bugs or incomplete functionality.

---

## Key Idea

Instead of writing orchestration code directly for a specific framework,
developers define workflows using simple Python classes.

Choreo-Mini then compiles the workflow into framework-specific code.

Architecture overview:

Developer Python Workflow
        ↓
AST Parser
        ↓
Intermediate Workflow Representation
        ↓
Template Compiler (Jinja2)
        ↓
Target Runtime (LangGraph / CrewAI / AutoGen)

---

## Features

### Pythonic Agent Development

Developers define agents and services using native Python abstractions rather
than framework-specific DSLs.

### Framework Portability

Workflows can be compiled into different orchestration runtimes, enabling
experimentation across ecosystems.

### Workflow Observability Layer

Choreo-Mini includes runtime instrumentation that tracks:

- latency across workflow nodes
- memory utilization
- execution loops

This enables debugging and performance optimization of agent pipelines.

The observability layer can be extended to track token usage and cost metrics
for LLM providers.

---
## Installation

```bash
pip install choreo-mini
```
---
## Usage

```python
import choreo_mini as choreo
from choreo_mini.core.llm import LLM
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.workflow import Workflow

# example usage

# create workflow; profiling can be toggled via CLI flag
wf = Workflow("myflow", enable_profiling=True)

# agents automatically register when given the workflow
A1 = AgentNode(wf, "Greeter", role="greeter", tasks=["say hello"],
               properties={"provider":"openai","api_key":"..."})
A2 = AgentNode(wf, "Responder", role="responder")
A3 = AgentNode(wf, "Fallback", role="fallback")

# service node example
S1 = ServiceNode(wf, "Loader", lambda path: open(path).read())

input_data = S1.execute(wf, "data.csv")

output = []
for row in input_data:
    resp = A1.execute(row)
    if resp and "foo" in resp.content:
        output.append(A2.execute(resp))
    else:
        output.append(A3.execute(resp))
```
---
## Development

1. Clone the repo
2. Create a virtual environment
3. Install dependencies
---
## License

This project is licensed under the MIT License. See the LICENSE file for details.
