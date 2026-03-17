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
        self._scope_depth: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

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
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

        # Look for Workflow instantiation
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "Workflow" and var_name:
                kwargs = {kw.arg: kw.value for kw in node.value.keywords if kw.arg}
                self.workflow_name = var_name
                if "enable_profiling" in kwargs and isinstance(kwargs["enable_profiling"], ast.Constant):
                    self.enable_profiling = kwargs["enable_profiling"].value

        # Preserve only module-level assignments for optional reconstruction templates.
        if self._scope_depth == 0 and var_name and isinstance(node.value, ast.Call):
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
                    "kwargs": {},
                    "runtime_name_expr": self._format_expr(node.value.args[1]) if len(node.value.args) >= 2 else repr(var_name or "node"),
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

        # Capture assigned call results, e.g., resp = wf.send("Greeter", text)
        # and wrapped variants like resp = wf.send(...).content
        assigned_call = self._extract_call_from_expr(node.value)
        if assigned_call is not None and self._is_workflow_relevant_call(assigned_call):
            call_entry: Dict[str, Any] = {
                "type": "call",
                "call": self._extract_call(assigned_call),
            }
            if var_name:
                call_entry["assign_to"] = var_name
            self.execution_logic.append(call_entry)

        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        # Look for main execution loop
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # Infinite loop
            self.execution_logic.append({"type": "infinite_loop", "body": self._extract_body(node.body)})
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        body_calls = self._extract_calls_from_stmts(node.body)

        if self._contains_workflow_call(body_calls):
            iter_var = None
            if isinstance(node.target, ast.Name):
                iter_var = node.target.id
            elif isinstance(node.target, ast.Attribute):
                iter_var = self._get_full_name(node.target)
            else:
                iter_var = self._format_expr(node.target)

            self.execution_logic.append({
                "type": "for_loop",
                "iter_var": iter_var,
                "iter_expr": self._format_expr(node.iter),
                "body": self._extract_body(node.body)
            })

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        body_calls = self._extract_calls_from_stmts(node.body)
        orelse_calls = self._extract_calls_from_stmts(node.orelse)

        if self._contains_workflow_call(body_calls) or self._contains_workflow_call(orelse_calls):
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
            if self._is_workflow_relevant_call(node.value):
                self.execution_logic.append({"type": "call", "call": call_info})
        self.generic_visit(node)

    def _call_from_stmt(self, stmt: ast.stmt) -> Optional[ast.Call]:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.Assign):
            return self._extract_call_from_expr(stmt.value)
        return None

    def _extract_call_from_expr(self, node: ast.AST) -> Optional[ast.Call]:
        """Unwrap expressions like call().attr and return the underlying call."""
        current = node
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        if isinstance(current, ast.Call):
            return current
        return None

    def _extract_calls_from_stmts(self, statements: List[ast.stmt]) -> List[ast.Call]:
        calls: List[ast.Call] = []
        for stmt in statements:
            call_node = self._call_from_stmt(stmt)
            if call_node is not None:
                calls.append(call_node)
        return calls

    def _contains_workflow_call(self, calls: List[ast.Call]) -> bool:
        for call in calls:
            if self._is_workflow_relevant_call(call):
                return True
        return False

    def _is_workflow_relevant_call(self, call: ast.Call) -> bool:
        call_info = self._extract_call(call)
        func = call_info.get("func", "")
        if not func:
            return False

        if func == "input":
            return True

        if self.workflow_name and func == f"{self.workflow_name}.send":
            return True
        if func.endswith(".send"):
            return True

        if func.endswith(".execute"):
            base = func.rsplit(".", 1)[0]
            for node_data in self.nodes:
                if node_data.get("var_name") == base:
                    return True
            return True

        return False

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
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)

    def _extract_result_accessor(self, node: ast.AST) -> List[Dict[str, str]]:
        accessor: List[Dict[str, str]] = []
        current = node

        while not isinstance(current, ast.Call):
            if isinstance(current, ast.Attribute):
                accessor.append({"kind": "attr", "value": current.attr})
                current = current.value
                continue

            if isinstance(current, ast.Subscript):
                accessor.append({"kind": "subscript", "value": self._format_expr(current.slice)})
                current = current.value
                continue

            return []

        accessor.reverse()
        return accessor

    def _extract_body(self, body: List[ast.stmt], capture_assignments: bool = False) -> List[Dict[str, Any]]:
        # Extract key statements from a block.
        result = []
        for stmt in body:
            call_node = self._call_from_stmt(stmt)
            if call_node is not None and self._is_workflow_relevant_call(call_node):
                call_entry: Dict[str, Any] = {"type": "call", "call": self._extract_call(call_node)}
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        call_entry["assign_to"] = target.id
                    else:
                        call_entry["assign_target_expr"] = self._format_expr(target)

                    accessor = self._extract_result_accessor(stmt.value)
                    if accessor:
                        call_entry["result_accessor"] = accessor

                result.append(call_entry)
            elif isinstance(stmt, ast.Assign):
                if capture_assignments and len(stmt.targets) == 1:
                    result.append({
                        "type": "assign",
                        "target_expr": self._format_expr(stmt.targets[0]),
                        "expr": self._format_expr(stmt.value),
                    })
            elif isinstance(stmt, ast.AugAssign):
                if capture_assignments:
                    result.append({
                        "type": "augassign",
                        "target_expr": self._format_expr(stmt.target),
                        "op": stmt.op.__class__.__name__,
                        "expr": self._format_expr(stmt.value),
                    })
            elif isinstance(stmt, ast.If):
                body_logic = self._extract_body(stmt.body, capture_assignments=True)
                orelse_logic = self._extract_body(stmt.orelse, capture_assignments=True) if stmt.orelse else []
                if body_logic or orelse_logic:
                    result.append({
                        "type": "if",
                        "condition": self._extract_expr(stmt.test),
                        "body": body_logic,
                        "orelse": orelse_logic,
                    })
            elif isinstance(stmt, ast.For):
                body_logic = self._extract_body(stmt.body, capture_assignments=True)
                orelse_logic = self._extract_body(stmt.orelse, capture_assignments=True) if stmt.orelse else []
                if body_logic or orelse_logic:
                    result.append({
                        "type": "for_loop",
                        "iter_var": self._format_expr(stmt.target),
                        "iter_expr": self._format_expr(stmt.iter),
                        "body": body_logic,
                        "orelse": orelse_logic,
                    })
            elif isinstance(stmt, ast.While):
                body_logic = self._extract_body(stmt.body, capture_assignments=True)
                orelse_logic = self._extract_body(stmt.orelse, capture_assignments=True) if stmt.orelse else []
                if body_logic or orelse_logic:
                    if isinstance(stmt.test, ast.Constant) and stmt.test.value is True:
                        result.append({
                            "type": "infinite_loop",
                            "body": body_logic,
                            "orelse": orelse_logic,
                        })
                    else:
                        result.append({
                            "type": "while_loop",
                            "condition": self._extract_expr(stmt.test),
                            "body": body_logic,
                            "orelse": orelse_logic,
                        })
            elif isinstance(stmt, ast.Try):
                result.extend(self._extract_body(stmt.body, capture_assignments=capture_assignments))
            elif isinstance(stmt, ast.Break):
                result.append({"type": "break"})
            elif isinstance(stmt, ast.Continue):
                result.append({"type": "continue"})
        return result


def parse_workflow_code(code: str, enable_profiling: bool = False) -> Dict[str, Any]:
    """Parse choreo-mini workflow code and extract components.

    Returns a dictionary with workflow data suitable for template rendering.
    """
    tree = ast.parse(code)
    visitor = WorkflowVisitor()
    visitor.visit(tree)

    execution_body: List[ast.stmt] = tree.body
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "main":
            execution_body = stmt.body
            break

    execution_logic = visitor._extract_body(execution_body)

    # Override profiling if specified via CLI
    if enable_profiling:
        visitor.enable_profiling = True

    return {
        "workflow_name": visitor.workflow_name,
        "enable_profiling": visitor.enable_profiling,
        "nodes": visitor.nodes,
        "execution_logic": execution_logic,
        "imports": visitor.imports,
        "assignments": visitor.assignments,
    }
