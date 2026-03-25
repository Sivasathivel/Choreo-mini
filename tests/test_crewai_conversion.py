import importlib.util
import sys
import types
from pathlib import Path

import jinja2

from choreo_mini.cli import _build_render_data
from choreo_mini.core.ast_parser import parse_workflow_code
from choreo_mini.core.llm import CustomLLM
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.workflow import Workflow


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "choreo_mini" / "templates" / "crewai"


class FakeAgent:
    def __init__(self, role, goal, backstory, allow_delegation=False, verbose=False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.allow_delegation = allow_delegation
        self.verbose = verbose


class FakeTask:
    def __init__(self, description, agent, expected_output):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output


class FakeCrew:
    def __init__(self, agents, tasks, verbose=False):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose


def _install_fake_crewai() -> None:
    module = types.ModuleType("crewai")
    module.Agent = FakeAgent
    module.Task = FakeTask
    module.Crew = FakeCrew
    sys.modules["crewai"] = module


def _render_crewai(example_name: str, output_name: str) -> Path:
    code = (ROOT / "examples" / example_name).read_text()
    workflow_data = parse_workflow_code(code)
    render_data = _build_render_data(workflow_data, "crewai")

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


def _build_foo2_workflow():
    foo2 = _load_module(ROOT / "examples" / "foo2.py", "foo2_runtime")

    wf = Workflow("ticket_triage", enable_profiling=True)
    wf.state["round"] = 0
    wf.state["last_batch"] = []
    AgentNode(wf, "Classifier", role="triage", llm=CustomLLM(foo2._classifier_response))
    AgentNode(wf, "BillingSpecialist", role="billing", llm=CustomLLM(foo2._billing_response))
    AgentNode(wf, "TechSpecialist", role="technical", llm=CustomLLM(foo2._technical_response))
    AgentNode(wf, "Generalist", role="general", llm=CustomLLM(foo2._general_response))
    AgentNode(wf, "Reviewer", role="review", llm=CustomLLM(foo2._review_response))
    ServiceNode(wf, "TicketLoader", foo2.split_tickets)
    return wf


def test_crewai_conversion_for_foo2_branching_runtime():
    _install_fake_crewai()
    output_path = _render_crewai("foo2.py", "test_crewai_output_foo2.py")
    generated = _load_module(output_path, "generated_crewai_foo2")

    wf = _build_foo2_workflow()

    result = generated.kickoff(
        inputs={"input": "invoice refund urgent; app crash timeout"},
        wf=wf,
        loop_budget=1,
    )

    assert isinstance(result["crew"], FakeCrew)
    assert wf.state["round"] == 1
    assert wf.state["last_batch"] == ["invoice refund urgent", "app crash timeout"]
    assert wf.agent_states["Classifier"].call_count == 2
    assert wf.agent_states["BillingSpecialist"].call_count == 1
    assert wf.agent_states["TechSpecialist"].call_count == 1
    assert wf.agent_states["Generalist"].call_count == 0
    assert wf.agent_states["Reviewer"].call_count == 2
    assert result["last_agent"] == "Reviewer"
    assert len(result["tasks"]) == 6


def test_crewai_conversion_for_foo2_multiple_input_iterations():
    _install_fake_crewai()
    output_path = _render_crewai("foo2.py", "test_crewai_output_foo2_loop.py")
    generated = _load_module(output_path, "generated_crewai_foo2_loop")

    wf = _build_foo2_workflow()

    generated.kickoff(
        inputs={
            "input_queue": [
                "invoice refund urgent; app crash timeout",
                "general question; billing dispute",
            ]
        },
        wf=wf,
        loop_budget=2,
    )

    assert wf.state["round"] == 2
    assert wf.agent_states["Classifier"].call_count == 4
    assert wf.agent_states["Reviewer"].call_count == 4