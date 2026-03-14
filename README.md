# Choreo-Mini

Choreo-Mini is a lightweight python MetaFramework created to simplify agent creation and porting among the different modern LLM orchestration runtimes (LangGraph/CrewAI/AutoGen), making it easy to develop and demo LLM Agent orchestration concepts in a “real” runtime.

## Features

- Enables the developer to developer to create agents without the learning curve of any modern frameworks (LangGraph/CrewAI/Autogen). 
- Enables Agent Development in a more pythonic way
- Enables portability by allowing the user to convert the code into any of the frameworks of their choice
- Enables workflow observability that tracks latency, memory utilization, and execution loops across agent nodes, enabling debugging and performance optimization

## Installation

```bash
pip install choreo
```

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

## Development

1. Clone the repo
2. Create a virtual environment
3. Install dependencies

## License

This project is licensed under the MIT License. See the LICENSE file for details.
# Choreo-mini
