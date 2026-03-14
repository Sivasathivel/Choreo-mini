"""CLI entry point for choreo-mini framework.

Provides commands to convert Python agent workflows into other frameworks
like LangGraph, CrewAI, or AutoGen using AST parsing and Jinja2 templates.
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, Any

import jinja2

from choreo_mini.core.ast_parser import parse_workflow_code


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

    output_code = template.render(**workflow_data)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output_code)

    print(f"Generated {args.backend} code in {output_path}")


if __name__ == "__main__":
    main()
