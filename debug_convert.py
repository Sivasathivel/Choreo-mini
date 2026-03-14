#!/usr/bin/env python3
"""Manual conversion script for debugging."""

import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from choreo_mini.core.ast_parser import parse_workflow_code
import jinja2

def main():
    # Parse foo.py
    with open('examples/foo.py', 'r') as f:
        code = f.read()

    result = parse_workflow_code(code)
    print('Parsed data:')
    for key, value in result.items():
        print(f'{key}: {value}')

    # Render template
    template_dir = Path('choreo_mini/templates/langgraph')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template('workflow.j2')
    output = template.render(**result)

    # Write output
    output_path = Path('output/langgraph_output.py')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output)

    print(f'Generated output at {output_path}')

if __name__ == '__main__':
    main()