import ast
from pathlib import Path

import jinja2

from choreo_mini.cli import _build_render_data
from choreo_mini.core.ast_parser import parse_workflow_code


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = ROOT / "choreo_mini" / "templates"


def _render_backend(example_name: str, backend: str) -> tuple[str, dict]:
    code = (ROOT / "examples" / example_name).read_text()
    workflow_data = parse_workflow_code(code)
    render_data = _build_render_data(workflow_data, backend)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_ROOT / backend))
    template = env.get_template("workflow.j2")
    rendered = template.render(**render_data)
    return rendered, render_data


def test_autogen_template_renders_for_complex_workflow():
    rendered, render_data = _render_backend("foo2.py", "autogen")

    assert "EXECUTION_LOGIC = [" in rendered
    assert "def kickoff(" in rendered
    assert len(render_data["send_calls"]) >= 4
    assert "{{" not in rendered

    # Syntax-level validation for generated Python.
    ast.parse(rendered)


def test_crewai_template_renders_for_complex_workflow():
    rendered, render_data = _render_backend("foo2.py", "crewai")

    assert "EXECUTION_LOGIC = [" in rendered
    assert "def kickoff(" in rendered
    assert len(render_data["send_calls"]) >= 4
    assert "{{" not in rendered

    # Syntax-level validation for generated Python.
    ast.parse(rendered)


def test_backend_render_data_collects_nested_calls():
    code = (ROOT / "examples" / "foo2.py").read_text()
    workflow_data = parse_workflow_code(code)

    autogen_data = _build_render_data(workflow_data, "autogen")
    crewai_data = _build_render_data(workflow_data, "crewai")

    assert len(autogen_data["send_calls"]) >= 4
    assert len(crewai_data["send_calls"]) >= 4


# ---------------------------------------------------------------------------
# Toolset rendering tests
# ---------------------------------------------------------------------------

_TOOLSET_SNIPPET = """
toolset=[
    {
        'url': 'http://localhost:8000/sse',
        'name': 'calculator',
        'description': 'arithmetic',
        'type': 'mcp',
        'subtype': 'sse',
    }
]
"""

_TOOLSET_SOURCE = f"""\
from choreo_mini.core.workflow import Workflow
from choreo_mini.core.nodes import AgentNode
from choreo_mini.core.llm import CustomLLM

wf = Workflow("demo")
bot = AgentNode(wf, "Bot", role="assistant", llm=CustomLLM(lambda p, **kw: "ok"),
                {_TOOLSET_SNIPPET.strip()})
result = wf.send("Bot", "hello")
"""


def _render_from_source(source: str, backend: str) -> str:
    from choreo_mini.cli import _build_render_data
    workflow_data = parse_workflow_code(source)
    render_data = _build_render_data(workflow_data, backend)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_ROOT / backend))
    template = env.get_template("workflow.j2")
    return template.render(**render_data)


def test_langgraph_toolset_rendered():
    rendered = _render_from_source(_TOOLSET_SOURCE, "langgraph")
    assert "AGENT_TOOLSETS" in rendered
    assert "'calculator'" in rendered or "\"calculator\"" in rendered
    assert "_run_async" in rendered
    assert "wf.send_async" in rendered
    # Must be valid Python
    ast.parse(rendered)


def test_langgraph_no_toolset_no_async_send():
    # foo2.py has no toolsets — AGENT_TOOLSETS is present but empty at runtime
    rendered, _ = _render_backend("foo2.py", "langgraph")
    assert "AGENT_TOOLSETS" in rendered
    ast.parse(rendered)


def test_crewai_toolset_rendered():
    rendered = _render_from_source(_TOOLSET_SOURCE, "crewai")
    assert "AGENT_TOOLSETS" in rendered
    assert "_run_async" in rendered
    assert "wf.send_async" in rendered
    ast.parse(rendered)


def test_autogen_toolset_rendered():
    rendered = _render_from_source(_TOOLSET_SOURCE, "autogen")
    assert "AGENT_TOOLSETS" in rendered
    assert "_run_async" in rendered
    assert "wf.send_async" in rendered
    ast.parse(rendered)