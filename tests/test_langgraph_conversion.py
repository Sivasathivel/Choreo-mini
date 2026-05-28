"""LangGraph conversion tests — subclass pattern.

The subclass pattern is the correct way to use the choreo-mini converter:
each ``class X(Workflow)`` becomes one LangGraph StateGraph.  The flat /
functional pattern (``wf = Workflow(...)`` inside ``main()``) is for running
choreo-mini workflows directly; it is NOT a valid conversion target.
"""

import importlib.util
from pathlib import Path

import jinja2

from choreo_mini.core.ast_parser import parse_workflow_code


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "choreo_mini" / "templates" / "langgraph"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parser tests (no LangGraph required)
# ---------------------------------------------------------------------------

def test_parser_detects_subclass():
    """The parser should prefer the subclass pattern over main()."""
    code = (ROOT / "examples" / "foo2.py").read_text()
    data = parse_workflow_code(code)

    assert data["class_name"] == "TicketTriageWorkflow"
    assert data["workflow_name"] == "ticket_triage"
    assert data["enable_profiling"] is True
    assert data["primary_method"] == "process_batch"
    assert data["primary_method_arg"] == "raw_batch"
    assert len(data["workflow_subclasses"]) == 1

    node_names = [n["var_name"] for n in data["nodes"]]
    assert "classifier" in node_names
    assert "billing_specialist" in node_names
    assert "ticket_loader" in node_names

    # Execution logic should have a for_loop (the per-ticket loop)
    assert _contains_logic_type(data["execution_logic"], "for_loop")
    assert _contains_logic_type(data["execution_logic"], "if")


# ---------------------------------------------------------------------------
# End-to-end LangGraph conversion tests (subclass pattern via foo2.py)
# ---------------------------------------------------------------------------

def test_langgraph_conversion_for_foo2_branching():
    """Convert foo2.py (subclass pattern) and run a two-ticket batch through the graph."""
    output_path = _render_langgraph("foo2.py", "test_langgraph_output_foo2.py")
    generated = _load_module(output_path, "generated_foo2")
    foo2 = _load_module(ROOT / "examples" / "foo2.py", "foo2_module")

    # The workflow is self-contained — agents are configured inside __init__.
    wf = foo2.TicketTriageWorkflow()

    result = generated.app.invoke(
        {
            "wf": wf,
            "input": "invoice refund urgent; app crash timeout",
            "messages": [],
            "loop_budget": 1,
        }
    )

    # Both tickets are processed: one billing, one technical.
    assert result["last_agent"] == "Reviewer"
    assert wf.state["round"] == 1
    assert wf.state["last_batch"] == ["invoice refund urgent", "app crash timeout"]
    assert wf.agent_states["Classifier"].call_count == 2
    assert wf.agent_states["BillingSpecialist"].call_count == 1
    assert wf.agent_states["TechSpecialist"].call_count == 1
    assert wf.agent_states["Generalist"].call_count == 0
    assert wf.agent_states["Reviewer"].call_count == 2
    assert (
        "Billing action plan" in result["last_response"]
        or "Technical debug plan" in result["last_response"]
    )


def test_langgraph_conversion_for_foo2_loop_budget():
    """loop_budget has no effect on methods without an outer while-True loop.

    process_batch processes one batch per call.  Passing loop_budget=2 does not
    repeat the method body because there is no infinite_loop entry in the
    execution logic.  The result is identical to loop_budget=1.
    """
    output_path = _render_langgraph("foo2.py", "test_langgraph_output_foo2_loop.py")
    generated = _load_module(output_path, "generated_foo2_loop")
    foo2 = _load_module(ROOT / "examples" / "foo2.py", "foo2_module_loop")

    wf = foo2.TicketTriageWorkflow()

    generated.app.invoke(
        {
            "wf": wf,
            "input": "invoice refund urgent; app crash timeout",
            "messages": [],
            "loop_budget": 2,
        }
    )

    # No infinite loop in the method → only one batch is processed regardless
    # of loop_budget.
    assert wf.state["round"] == 1
    assert wf.agent_states["Classifier"].call_count == 2
    assert wf.agent_states["Reviewer"].call_count == 2
