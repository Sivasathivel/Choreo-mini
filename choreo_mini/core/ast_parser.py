"""AST parser for choreo-mini workflow code.

Analyzes Python code to extract workflow definitions, nodes, and execution
logic for conversion to other frameworks.
"""

import ast
from typing import Dict, Any, List, Optional


class WorkflowVisitor(ast.NodeVisitor):
    """AST visitor to extract workflow components from choreo-mini code."""

    def __init__(self):
        self.workflow_name: Optional[str] = None
        self.enable_profiling: bool = False
        self.nodes: List[Dict[str, Any]] = []
        self.execution_logic: List[Dict[str, Any]] = []
        self.imports: List[str] = []
        self.assignments: List[Dict[str, str]] = []  # preserve top-level assignments (e.g., wf, llm) for reconstruction

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.imports.append(f"from {module} import {', '.join(names)}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        var_name = None
        # Look for Workflow instantiation
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "Workflow":
                    # Extract workflow name and profiling flag
                    args = node.value.args
                    kwargs = {kw.arg: kw.value for kw in node.value.keywords if kw.arg}

                    self.workflow_name = var_name
                    if "enable_profiling" in kwargs:
                        if isinstance(kwargs["enable_profiling"], ast.Constant):
                            self.enable_profiling = kwargs["enable_profiling"].value

        # Preserve top-level assignments (e.g., wf = Workflow(...), echo_llm = CustomLLM(...))
        if var_name and isinstance(node.value, ast.Call):
            # Record top-level assignment expressions for reconstruction
            self.assignments.append({
                "target": var_name,
                "expr": self._format_expr(node.value),
            })

        # Look for AgentNode/ServiceNode instantiations
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            if func_name in ("AgentNode", "ServiceNode"):
                node_data = {
                    "var_name": var_name,
                    "type": func_name,
                    "args": [],
                    "kwargs": {}
                }

                # Extract arguments
                for arg in node.value.args:
                    node_data["args"].append(self._format_expr(arg))

                for kw in node.value.keywords:
                    if isinstance(kw.value, ast.Dict):
                        # Handle dict arguments like properties
                        node_data["kwargs"][kw.arg] = self._extract_dict(kw.value)
                    else:
                        node_data["kwargs"][kw.arg] = self._format_expr(kw.value)

                self.nodes.append(node_data)

        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        # Look for main execution loop
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # Infinite loop
            self.execution_logic.append({"type": "infinite_loop", "body": self._extract_body(node.body)})
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        # Only capture loops that contain workflow-relevant calls (e.g., wf.send / agent.execute)
        body_calls = [stmt for stmt in node.body if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)]

        def _contains_workflow_call(stmts: List[ast.stmt]) -> bool:
            for stmt in stmts:
                call = stmt.value
                if isinstance(call, ast.Call):
                    func_name = self._get_full_name(call.func) if isinstance(call.func, ast.Attribute) else (call.func.id if isinstance(call.func, ast.Name) else "")
                    if func_name in ("wf.send", "agent.execute"):
                        return True
            return False

        if _contains_workflow_call(body_calls):
            iter_var = None
            if isinstance(node.target, ast.Name):
                iter_var = node.target.id
            elif isinstance(node.target, ast.Attribute):
                iter_var = self._get_full_name(node.target)
            else:
                iter_var = str(node.target)

            self.execution_logic.append({
                "type": "for_loop",
                "iter_var": iter_var,
                "iter_expr": self._extract_call(node.iter),
                "body": self._extract_body(node.body)
            })

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        # Only capture conditionals that contain workflow-relevant calls (e.g., wf.send / agent.execute).
        body_calls = [stmt for stmt in node.body if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)]
        orelse_calls = [stmt for stmt in node.orelse if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)]

        def _contains_workflow_call(stmts: List[ast.stmt]) -> bool:
            for stmt in stmts:
                call = stmt.value
                if isinstance(call, ast.Call):
                    func_name = self._get_full_name(call.func) if isinstance(call.func, ast.Attribute) else (call.func.id if isinstance(call.func, ast.Name) else "")
                    if func_name in ("wf.send", "agent.execute"):
                        return True
            return False

        if _contains_workflow_call(body_calls) or _contains_workflow_call(orelse_calls):
            condition = self._extract_expr(node.test)
            self.execution_logic.append({
                "type": "if",
                "condition": condition,
                "body": self._extract_body(node.body),
                "orelse": self._extract_body(node.orelse) if node.orelse else []
            })

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            call_info = self._extract_call(node.value)
            if call_info.get("func") in ("wf.send", "agent.execute"):
                self.execution_logic.append({"type": "call", "call": call_info})
        self.generic_visit(node)

    def _extract_dict(self, node: ast.Dict) -> Dict[str, Any]:
        result = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                result[key.value] = value.value
        return result

    def _get_full_name(self, node: ast.AST) -> str:
        """Return a dot-separated name for Name/Attribute AST nodes."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._get_full_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _format_expr(self, node: ast.AST) -> str:
        """Format an AST expression as source code."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        # ast.unparse is available in Python 3.9+
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        return str(node)

    def _extract_call(self, node: ast.Call) -> Dict[str, Any]:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = self._get_full_name(node.func)

        args = [self._format_expr(arg) for arg in node.args]

        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = self._format_expr(kw.value)

        return {"func": func_name, "args": args, "kwargs": kwargs}

    def _extract_expr(self, node: ast.AST) -> str:
        # Simple expression extraction - could be enhanced
        if isinstance(node, ast.Compare):
            return f"{self._extract_expr(node.left)} {node.ops[0].__class__.__name__.lower()} {self._extract_expr(node.comparators[0])}"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)

    def _extract_body(self, body: List[ast.stmt]) -> List[Dict[str, Any]]:
        # Extract key statements from a block
        result = []
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                result.append({"type": "call", "call": self._extract_call(stmt.value)})
            elif isinstance(stmt, ast.If):
                result.append({
                    "type": "if",
                    "condition": self._extract_expr(stmt.test),
                    "body": self._extract_body(stmt.body)
                })
        return result


def parse_workflow_code(code: str, enable_profiling: bool = False) -> Dict[str, Any]:
    """Parse choreo-mini workflow code and extract components.

    Returns a dictionary with workflow data suitable for template rendering.
    """
    tree = ast.parse(code)
    visitor = WorkflowVisitor()
    visitor.visit(tree)

    # Override profiling if specified via CLI
    if enable_profiling:
        visitor.enable_profiling = True

    return {
        "workflow_name": visitor.workflow_name,
        "enable_profiling": visitor.enable_profiling,
        "nodes": visitor.nodes,
        "execution_logic": visitor.execution_logic,
        "imports": visitor.imports,
        "assignments": visitor.assignments,
    }
