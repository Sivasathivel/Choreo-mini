import importlib.util
from pathlib import Path

import jinja2

from choreo_mini.core.ast_parser import parse_workflow_code
from choreo_mini.core.llm import CustomLLM
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.workflow import Workflow


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "choreo_mini" / "templates" / "langgraph"


def _contains_logic_type(entries, kind):
    for entry in entries:
        if entry.get("type") == kind:
            return True

        body = entry.get("body")
        if isinstance(body, list) and _contains_logic_type(body, kind):
            return True

        orelse = entry.get("orelse")
        if isinstance(orelse, list) and _contains_logic_type(orelse, kind):
            return True

    return False


def _render_langgraph(example_name: str, output_name: str) -> Path:
    code = (ROOT / "examples" / example_name).read_text()
    workflow_data = parse_workflow_code(code)

    render_data = dict(workflow_data)
    all_nodes = [node for node in workflow_data.get("nodes", []) if node.get("var_name")]
    agent_nodes = [node for node in all_nodes if node.get("type") == "AgentNode"]
    render_data["all_nodes"] = all_nodes
    render_data["agent_nodes"] = agent_nodes
    render_data["execution_logic_literal"] = repr(workflow_data.get("execution_logic", []))
    render_data["has_conditionals"] = _contains_logic_type(workflow_data.get("execution_logic", []), "if")
    render_data["has_loops"] = (
        _contains_logic_type(workflow_data.get("execution_logic", []), "for_loop")
        or _contains_logic_type(workflow_data.get("execution_logic", []), "infinite_loop")
        or _contains_logic_type(workflow_data.get("execution_logic", []), "while_loop")
    )

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("workflow.j2")
    output_path = ROOT / "output" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.render(**render_data))
    return output_path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_langgraph_conversion_for_foo_smoke():
    output_path = _render_langgraph("foo.py", "test_langgraph_output_foo.py")
    generated = _load_module(output_path, "generated_foo")

    wf = Workflow("demo", enable_profiling=True)
    AgentNode(wf, "Greeter", role="greeter", llm=CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}"))

    result = generated.app.invoke({"wf": wf, "input": "hello", "messages": [], "loop_budget": 1})

    assert result["last_response"] == "echo: hello"
    assert result["last_agent"] == "Greeter"
    assert wf.agent_states["Greeter"].call_count == 1


def test_langgraph_conversion_for_foo2_branching():
    output_path = _render_langgraph("foo2.py", "test_langgraph_output_foo2.py")
    generated = _load_module(output_path, "generated_foo2")
    foo2 = _load_module(ROOT / "examples" / "foo2.py", "foo2_module")

    wf = Workflow("ticket_triage", enable_profiling=True)
    wf.state["round"] = 0
    wf.state["last_batch"] = []
    AgentNode(wf, "Classifier", role="triage", llm=CustomLLM(foo2._classifier_response))
    AgentNode(wf, "BillingSpecialist", role="billing", llm=CustomLLM(foo2._billing_response))
    AgentNode(wf, "TechSpecialist", role="technical", llm=CustomLLM(foo2._technical_response))
    AgentNode(wf, "Generalist", role="general", llm=CustomLLM(foo2._general_response))
    AgentNode(wf, "Reviewer", role="review", llm=CustomLLM(foo2._review_response))
    ServiceNode(wf, "TicketLoader", foo2.split_tickets)

    result = generated.app.invoke(
        {
            "wf": wf,
            "input": "invoice refund urgent; app crash timeout",
            "messages": [],
            "loop_budget": 1,
        }
    )

    assert result["last_agent"] == "Reviewer"
    assert wf.state["round"] == 1
    assert wf.state["last_batch"] == ["invoice refund urgent", "app crash timeout"]
    assert wf.agent_states["Classifier"].call_count == 2
    assert wf.agent_states["BillingSpecialist"].call_count == 1
    assert wf.agent_states["TechSpecialist"].call_count == 1
    assert wf.agent_states["Generalist"].call_count == 0
    assert wf.agent_states["Reviewer"].call_count == 2
    assert "Billing action plan" in result["last_response"] or "Technical debug plan" in result["last_response"]


def test_langgraph_conversion_for_foo2_loop_budget():
    output_path = _render_langgraph("foo2.py", "test_langgraph_output_foo2_loop.py")
    generated = _load_module(output_path, "generated_foo2_loop")
    foo2 = _load_module(ROOT / "examples" / "foo2.py", "foo2_module_loop")

    wf = Workflow("ticket_triage", enable_profiling=True)
    wf.state["round"] = 0
    wf.state["last_batch"] = []
    AgentNode(wf, "Classifier", role="triage", llm=CustomLLM(foo2._classifier_response))
    AgentNode(wf, "BillingSpecialist", role="billing", llm=CustomLLM(foo2._billing_response))
    AgentNode(wf, "TechSpecialist", role="technical", llm=CustomLLM(foo2._technical_response))
    AgentNode(wf, "Generalist", role="general", llm=CustomLLM(foo2._general_response))
    AgentNode(wf, "Reviewer", role="review", llm=CustomLLM(foo2._review_response))
    ServiceNode(wf, "TicketLoader", foo2.split_tickets)

    generated.app.invoke(
        {
            "wf": wf,
            "input": "invoice refund urgent; app crash timeout",
            "messages": [],
            "loop_budget": 2,
        }
    )

    assert wf.state["round"] == 2
    assert wf.agent_states["Classifier"].call_count == 4
    assert wf.agent_states["Reviewer"].call_count == 4