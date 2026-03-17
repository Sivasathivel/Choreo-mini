"""CLI entry point for choreo-mini framework.

Provides commands to convert Python agent workflows into other frameworks
like LangGraph, CrewAI, or AutoGen using AST parsing and Jinja2 templates.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import jinja2

from choreo_mini.core.ast_parser import parse_workflow_code


def _contains_logic_type(entries: List[Dict[str, Any]], kind: str) -> bool:
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


def _collect_calls(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("type") == "call":
            calls.append(entry)

        body = entry.get("body")
        if isinstance(body, list):
            calls.extend(_collect_calls(body))

        orelse = entry.get("orelse")
        if isinstance(orelse, list):
            calls.extend(_collect_calls(orelse))

    return calls


def _build_render_data(workflow_data: Dict[str, Any], backend: str) -> Dict[str, Any]:
    render_data = dict(workflow_data)

    all_nodes = [
        node for node in workflow_data.get("nodes", [])
        if node.get("var_name")
    ]
    agent_nodes = [
        node for node in all_nodes
        if node.get("type") == "AgentNode"
    ]
    flattened_calls = _collect_calls(workflow_data.get("execution_logic", []))

    render_data["all_nodes"] = all_nodes
    render_data["agent_nodes"] = agent_nodes

    if backend == "langgraph":
        render_data["execution_logic_literal"] = repr(workflow_data.get("execution_logic", []))
        render_data["has_conditionals"] = _contains_logic_type(workflow_data.get("execution_logic", []), "if")
        render_data["has_loops"] = (
            _contains_logic_type(workflow_data.get("execution_logic", []), "for_loop")
            or _contains_logic_type(workflow_data.get("execution_logic", []), "infinite_loop")
            or _contains_logic_type(workflow_data.get("execution_logic", []), "while_loop")
        )

    if backend in ("autogen", "crewai"):
        send_calls: List[Dict[str, str]] = []
        execute_calls: List[Dict[str, str]] = []
        for call_entry in flattened_calls:
            call = call_entry.get("call", {})
            func_name = call.get("func", "")
            args = call.get("args", [])

            if func_name.endswith(".send"):
                agent_expr = args[0] if args else "'assistant'"
                message_expr = args[1] if len(args) > 1 else "'Hello'"
                send_calls.append(
                    {
                        "func": func_name,
                        "agent_expr": agent_expr,
                        "message_expr": message_expr,
                        "agent_expr_literal": repr(agent_expr),
                        "message_expr_literal": repr(message_expr),
                    }
                )
            elif func_name.endswith(".execute"):
                message_expr = args[0] if args else "'Execute task'"
                node_expr = func_name.rsplit(".", 1)[0]
                execute_calls.append(
                    {
                        "func": func_name,
                        "node_expr": node_expr,
                        "node_expr_literal": repr(node_expr),
                        "message_expr": message_expr,
                        "message_expr_literal": repr(message_expr),
                    }
                )

        render_data["send_calls"] = send_calls
        render_data["execute_calls"] = execute_calls

    return render_data


def main():
    parser = argparse.ArgumentParser(description="Convert choreo-mini workflows to other frameworks")
    parser.add_argument("-f", "--input", required=True, help="Input Python file (e.g., foo.py)")
    parser.add_argument("-b", "--backend", required=True, choices=["langgraph", "crewai", "autogen"], help="Target framework")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling in generated code")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Parse the input code
    with open(input_path, "r") as f:
        code = f.read()

    try:
        workflow_data = parse_workflow_code(code, enable_profiling=args.enable_profiling)
    except Exception as e:
        print(f"Error parsing {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Load and render template
    template_dir = Path(__file__).parent / "templates" / args.backend
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template("workflow.j2")

    render_data = _build_render_data(workflow_data, args.backend)

    output_code = template.render(**render_data)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output_code)

    print(f"Generated {args.backend} code in {output_path}")


if __name__ == "__main__":
    main()
