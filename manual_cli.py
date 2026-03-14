#!/usr/bin/env python3
"""Manual CLI simulation for choreo-mini."""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from choreo_mini.core.ast_parser import parse_workflow_code
import jinja2

def convert_file(input_file, backend, output_file, enable_profiling=False):
    """Convert a choreo-mini file to the specified backend."""

    # Read input
    with open(input_file, 'r') as f:
        code = f.read()

    # Parse
    workflow_data = parse_workflow_code(code, enable_profiling=enable_profiling)

    # Load template
    template_dir = project_root / "choreo_mini" / "templates" / backend
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template("workflow.j2")

    # Render
    output_code = template.render(**workflow_data)

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output_code)

    print(f"Generated {backend} code in {output_path}")

if __name__ == "__main__":
    # Test foo.py conversion
    convert_file("examples/foo.py", "langgraph", "output/langgraph_output.py", enable_profiling=True)
    convert_file("examples/foo.py", "crewai", "output/crewai_output.py", enable_profiling=True)
    convert_file("examples/foo.py", "autogen", "output/autogen_output.py", enable_profiling=True)