import importlib.util
import sys
import types
from pathlib import Path

import jinja2

from motif_ai.cli import _build_render_data
from motif_ai.core.ast_parser import parse_workflow_code


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "customer_ops_url.py"


class FakeAssistantAgent:
    def __init__(self, name, system_message=None, llm_config=None, **kwargs):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config


class FakeUserProxyAgent:
    def __init__(self, name, human_input_mode=None, max_consecutive_auto_reply=None,
                 is_termination_msg=None, code_execution_config=None, **kwargs):
        self.name = name
        self.human_input_mode = human_input_mode


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


def _install_fake_autogen() -> None:
    module = types.ModuleType("autogen")
    module.AssistantAgent = FakeAssistantAgent
    module.UserProxyAgent = FakeUserProxyAgent
    sys.modules["autogen"] = module


def _install_fake_crewai() -> None:
    module = types.ModuleType("crewai")
    module.Agent = FakeAgent
    module.Task = FakeTask
    module.Crew = FakeCrew
    sys.modules["crewai"] = module


def _render_backend(backend: str, output_name: str) -> Path:
    code = EXAMPLE_PATH.read_text()
    workflow_data = parse_workflow_code(code)
    render_data = _build_render_data(workflow_data, backend)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(ROOT / "motif_ai" / "templates" / backend)
    )
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


def _build_demo_workflow():
    example = _load_module(EXAMPLE_PATH, "customer_ops_url_runtime")
    wf = example.CustomerOpsWorkflow(demo_mode=True)
    return wf, example.EXAMPLE_BATCH


def test_endpoint_normalization_without_scheme_and_chat_detection():
    example = _load_module(EXAMPLE_PATH, "customer_ops_url_endpoint_normalize")

    endpoint, api_style = example._normalize_endpoint("api.openai.com/v1")
    assert endpoint == "https://api.openai.com/v1/responses"
    assert api_style == "responses"

    endpoint, api_style = example._normalize_endpoint("https://api.openai.com/v1/chat/completions")
    assert endpoint == "https://api.openai.com/v1/chat/completions"
    assert api_style == "chat_completions"


def test_langgraph_conversion_for_customer_ops_url():
    output_path = _render_backend("langgraph", "test_langgraph_customer_ops_url.py")
    generated = _load_module(output_path, "generated_langgraph_customer_ops_url")
    wf, example_batch = _build_demo_workflow()

    result = generated.app.invoke({
        "wf": wf,
        "input": example_batch,
        "messages": [],
        "loop_budget": 1,
    })

    assert result["last_agent"] == "QAReviewer"
    assert wf.state["batch_number"] == 1
    assert len(wf.state["last_batch"]) == 4
    assert wf.agent_states["Router"].call_count == 4
    assert wf.agent_states["BillingDesk"].call_count == 1
    assert wf.agent_states["TechnicalDesk"].call_count == 1
    assert wf.agent_states["LogisticsDesk"].call_count == 1
    assert wf.agent_states["RetentionDesk"].call_count == 1
    assert wf.agent_states["RiskLead"].call_count == 1
    assert wf.agent_states["QAReviewer"].call_count == 4


def test_crewai_conversion_for_customer_ops_url():
    _install_fake_crewai()
    output_path = _render_backend("crewai", "test_crewai_customer_ops_url.py")
    generated = _load_module(output_path, "generated_crewai_customer_ops_url")
    wf, example_batch = _build_demo_workflow()

    result = generated.kickoff(
        inputs={"input": example_batch},
        wf=wf,
        loop_budget=1,
    )

    assert isinstance(result["crew"], FakeCrew)
    assert result["last_agent"] == "QAReviewer"
    assert wf.state["batch_number"] == 1
    assert len(wf.state["last_batch"]) == 4
    assert wf.agent_states["Router"].call_count == 4
    assert wf.agent_states["RiskLead"].call_count == 1
    assert wf.agent_states["QAReviewer"].call_count == 4


def test_autogen_conversion_for_customer_ops_url():
    _install_fake_autogen()
    output_path = _render_backend("autogen", "test_autogen_customer_ops_url.py")
    generated = _load_module(output_path, "generated_autogen_customer_ops_url")
    wf, example_batch = _build_demo_workflow()

    result = generated.kickoff(
        inputs={"input": example_batch},
        wf=wf,
        loop_budget=1,
    )

    assert result["last_agent"] == "QAReviewer"
    assert wf.state["batch_number"] == 1
    assert len(wf.state["last_batch"]) == 4
    assert wf.agent_states["Router"].call_count == 4
    assert wf.agent_states["RiskLead"].call_count == 1
    assert wf.agent_states["QAReviewer"].call_count == 4