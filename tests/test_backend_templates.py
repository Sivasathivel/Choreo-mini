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

    assert "SEND_CALLS = [" in rendered
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