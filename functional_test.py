#!/usr/bin/env python3
"""Functional test script for choreo-mini framework."""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_ast_parser():
    """Test the AST parser on foo.py."""
    print("=== Testing AST Parser ===")

    try:
        from choreo_mini.core.ast_parser import parse_workflow_code

        with open('examples/foo.py', 'r') as f:
            code = f.read()

        result = parse_workflow_code(code, enable_profiling=True)

        print("✅ AST parsing successful!")
        print(f"Workflow name: {result.get('workflow_name')}")
        print(f"Enable profiling: {result.get('enable_profiling')}")
        print(f"Number of nodes: {len(result.get('nodes', []))}")
        print(f"Number of execution logic items: {len(result.get('execution_logic', []))}")
        print(f"Imports found: {len(result.get('imports', []))}")

        # Show node details
        for node in result.get('nodes', []):
            print(f"  Node: {node.get('var_name')} ({node.get('type')})")

        return result

    except Exception as e:
        print(f"❌ AST parsing failed: {e}")
        return None

def test_template_rendering(workflow_data):
    """Test template rendering for different backends."""
    print("\n=== Testing Template Rendering ===")

    try:
        import jinja2

        backends = ['langgraph', 'crewai', 'autogen']

        for backend in backends:
            print(f"\nTesting {backend} template...")

            template_dir = Path('choreo_mini/templates') / backend
            if not template_dir.exists():
                print(f"❌ Template directory {template_dir} not found")
                continue

            env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
            template = env.get_template('workflow.j2')

            output = template.render(**workflow_data)
            print(f"✅ {backend} template rendered successfully ({len(output)} chars)")

            # Save the output
            output_file = Path('output') / f'functional_test_{backend}.py'
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(output)
            print(f"✅ Output saved to {output_file}")

    except Exception as e:
        print(f"❌ Template rendering failed: {e}")

def test_cli_logic():
    """Test the CLI logic without argparse."""
    print("\n=== Testing CLI Logic ===")

    try:
        from choreo_mini.core.ast_parser import parse_workflow_code
        import jinja2
        from pathlib import Path

        # Simulate CLI arguments
        input_file = 'examples/foo.py'
        backend = 'langgraph'
        output_file = 'output/cli_test_langgraph.py'
        enable_profiling = True

        # Read input
        with open(input_file, 'r') as f:
            code = f.read()

        # Parse
        workflow_data = parse_workflow_code(code, enable_profiling=enable_profiling)

        # Render template
        template_dir = Path('choreo_mini/templates') / backend
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template('workflow.j2')
        output_code = template.render(**workflow_data)

        # Write output
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(output_code)

        print(f"✅ CLI logic test successful! Output: {output_file}")

    except Exception as e:
        print(f"❌ CLI logic test failed: {e}")

def main():
    """Run all functional tests."""
    print("Starting functional tests for choreo-mini...")

    # Test AST parser
    workflow_data = test_ast_parser()
    if workflow_data is None:
        return

    # Test template rendering
    test_template_rendering(workflow_data)

    # Test CLI logic
    test_cli_logic()

    print("\n=== Functional Testing Complete ===")

if __name__ == '__main__':
    main()