"""AST parser for motif-ai workflow code.

Analyzes Python code to extract workflow definitions, nodes, and execution
logic for conversion to other frameworks.

Two source patterns are supported:

**Subclass pattern (preferred — each class = one LangGraph subgraph)**::

    class MyFlow(Workflow):
        def __init__(self):
            super().__init__("my_flow", enable_profiling=True)
            self.agent = AgentNode(self, "Agent", role="...", llm=...)

        def run(self, text: str) -> str:
            return self.send("Agent", text).content

**Flat / functional pattern (legacy fallback)**::

    def main():
        wf = Workflow("my_flow")
        agent = AgentNode(wf, "Agent", role="...", llm=...)
        resp = wf.send("Agent", input("You> "))

When a subclass is present it takes precedence; ``main()`` / module-level
scanning is only used when no subclass is found.
"""

import ast
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Low-level helpers (shared by both paths)
# ---------------------------------------------------------------------------

def _get_full_name(node: ast.AST) -> str:
    """Return a dot-separated name for Name/Attribute AST nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_full_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _format_expr(node: ast.AST) -> str:
    """Format an AST expression as source code."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    return str(node)


def _extract_expr(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return str(node)


def _extract_call(node: ast.Call) -> Dict[str, Any]:
    func_name = ""
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = _get_full_name(node.func)

    args = [_format_expr(arg) for arg in node.args]
    kwargs = {}
    for kw in node.keywords:
        kwargs[kw.arg] = _format_expr(kw.value)
    return {"func": func_name, "args": args, "kwargs": kwargs}


def _extract_call_from_expr(node: ast.AST) -> Optional[ast.Call]:
    """Unwrap expressions like call().attr and return the underlying call."""
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    if isinstance(current, ast.Call):
        return current
    return None


def _extract_result_accessor(node: ast.AST) -> List[Dict[str, str]]:
    accessor: List[Dict[str, str]] = []
    current = node
    while not isinstance(current, ast.Call):
        if isinstance(current, ast.Attribute):
            accessor.append({"kind": "attr", "value": current.attr})
            current = current.value
            continue
        if isinstance(current, ast.Subscript):
            accessor.append({"kind": "subscript", "value": _format_expr(current.slice)})
            current = current.value
            continue
        return []
    accessor.reverse()
    return accessor


def _extract_dict(node: ast.Dict) -> Dict[str, Any]:
    result = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
            result[key.value] = value.value
    return result


def _is_workflow_call(call: ast.Call, known_node_vars: set) -> bool:
    """Return True if *call* is a send/execute/input call relevant to the workflow."""
    info = _extract_call(call)
    func = info.get("func", "")
    if not func:
        return False
    if func == "input":
        return True
    if func.endswith(".send"):
        return True
    if func.endswith(".execute"):
        base = func.rsplit(".", 1)[0]
        # Match "self.ticket_loader" → strip "self." prefix for comparison
        bare = base[len("self."):] if base.startswith("self.") else base
        if bare in known_node_vars or base in known_node_vars:
            return True
        return True  # any .execute() is considered workflow-relevant
    return False


def _call_from_stmt(stmt: ast.stmt) -> Optional[ast.Call]:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    if isinstance(stmt, ast.Assign):
        return _extract_call_from_expr(stmt.value)
    return None


def _extract_body(
    body: List[ast.stmt],
    known_node_vars: set,
    capture_assignments: bool = False,
) -> List[Dict[str, Any]]:
    """Recursively extract execution-logic entries from a statement list."""
    result = []
    for stmt in body:
        call_node = _call_from_stmt(stmt)
        if call_node is not None and _is_workflow_call(call_node, known_node_vars):
            call_entry: Dict[str, Any] = {"type": "call", "call": _extract_call(call_node)}
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    call_entry["assign_to"] = target.id
                else:
                    call_entry["assign_target_expr"] = _format_expr(target)
                accessor = _extract_result_accessor(stmt.value)
                if accessor:
                    call_entry["result_accessor"] = accessor
            result.append(call_entry)

        elif isinstance(stmt, ast.Assign):
            if capture_assignments and len(stmt.targets) == 1:
                result.append({
                    "type": "assign",
                    "target_expr": _format_expr(stmt.targets[0]),
                    "expr": _format_expr(stmt.value),
                })

        elif isinstance(stmt, ast.AugAssign):
            if capture_assignments:
                result.append({
                    "type": "augassign",
                    "target_expr": _format_expr(stmt.target),
                    "op": stmt.op.__class__.__name__,
                    "expr": _format_expr(stmt.value),
                })

        elif isinstance(stmt, ast.If):
            body_logic = _extract_body(stmt.body, known_node_vars, capture_assignments=True)
            orelse_logic = _extract_body(stmt.orelse, known_node_vars, capture_assignments=True) if stmt.orelse else []
            if body_logic or orelse_logic:
                result.append({
                    "type": "if",
                    "condition": _extract_expr(stmt.test),
                    "body": body_logic,
                    "orelse": orelse_logic,
                })

        elif isinstance(stmt, ast.For):
            body_logic = _extract_body(stmt.body, known_node_vars, capture_assignments=True)
            orelse_logic = _extract_body(stmt.orelse, known_node_vars, capture_assignments=True) if stmt.orelse else []
            if body_logic or orelse_logic:
                result.append({
                    "type": "for_loop",
                    "iter_var": _format_expr(stmt.target),
                    "iter_expr": _format_expr(stmt.iter),
                    "body": body_logic,
                    "orelse": orelse_logic,
                })

        elif isinstance(stmt, ast.While):
            body_logic = _extract_body(stmt.body, known_node_vars, capture_assignments=True)
            orelse_logic = _extract_body(stmt.orelse, known_node_vars, capture_assignments=True) if stmt.orelse else []
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
                        "condition": _extract_expr(stmt.test),
                        "body": body_logic,
                        "orelse": orelse_logic,
                    })

        elif isinstance(stmt, ast.Try):
            result.extend(_extract_body(stmt.body, known_node_vars, capture_assignments))

        elif isinstance(stmt, ast.Break):
            result.append({"type": "break"})

        elif isinstance(stmt, ast.Continue):
            result.append({"type": "continue"})

    return result


# ---------------------------------------------------------------------------
# Subclass extractor
# ---------------------------------------------------------------------------

def _extract_node_from_assign(stmt: ast.Assign) -> Optional[Dict[str, Any]]:
    """Return node_data if stmt is an AgentNode/ServiceNode assignment, else None."""
    if not (len(stmt.targets) == 1 and isinstance(stmt.value, ast.Call)):
        return None

    # Both `self.x = AgentNode(...)` and `x = AgentNode(...)` forms
    target = stmt.targets[0]
    if isinstance(target, ast.Attribute):
        var_name = target.attr
    elif isinstance(target, ast.Name):
        var_name = target.id
    else:
        return None

    call = stmt.value
    func_name = ""
    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        func_name = call.func.attr

    if func_name not in ("AgentNode", "ServiceNode"):
        return None

    node_data: Dict[str, Any] = {
        "var_name": var_name,
        "type": func_name,
        "args": [],
        "kwargs": {},
        "runtime_name_expr": _format_expr(call.args[1]) if len(call.args) >= 2 else repr(var_name),
    }
    for arg in call.args:
        node_data["args"].append(_format_expr(arg))
    for kw in call.keywords:
        if isinstance(kw.value, ast.Dict):
            node_data["kwargs"][kw.arg] = _extract_dict(kw.value)
        else:
            node_data["kwargs"][kw.arg] = _format_expr(kw.value)
    return node_data


def _workflow_name_from_super_init(init_body: List[ast.stmt]) -> tuple:
    """Scan __init__ body for super().__init__(name, enable_profiling=...).

    Returns (workflow_name, enable_profiling).
    """
    for stmt in init_body:
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        # super().__init__(...)
        is_super_init = (
            isinstance(func, ast.Attribute)
            and func.attr == "__init__"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        )
        if not is_super_init:
            continue
        wf_name: Optional[str] = None
        if call.args and isinstance(call.args[0], ast.Constant):
            wf_name = call.args[0].value
        enable_prof = False
        for kw in call.keywords:
            if kw.arg == "enable_profiling" and isinstance(kw.value, ast.Constant):
                enable_prof = bool(kw.value.value)
        return wf_name, enable_prof
    return None, False


def _extract_workflow_subclass(class_node: ast.ClassDef) -> Dict[str, Any]:
    """Extract all workflow data from a single Workflow subclass definition.

    Returns a dict with:
      class_name, workflow_name, enable_profiling, nodes,
      primary_method, execution_logic, methods
    """
    nodes: List[Dict[str, Any]] = []
    workflow_name: Optional[str] = None
    enable_profiling: bool = False
    methods: Dict[str, Dict[str, Any]] = {}

    for item in class_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        method_name = item.name
        method_args = [arg.arg for arg in item.args.args if arg.arg != "self"]

        if method_name == "__init__":
            # Extract workflow name + enable_profiling from super().__init__
            wf_name, ep = _workflow_name_from_super_init(item.body)
            if wf_name:
                workflow_name = wf_name
            if ep:
                enable_profiling = ep

            # Extract AgentNode / ServiceNode assignments from __init__
            for stmt in ast.walk(item):
                if isinstance(stmt, ast.Assign):
                    nd = _extract_node_from_assign(stmt)
                    if nd is not None:
                        nodes.append(nd)
        else:
            # Non-__init__ method: extract its execution logic
            known_vars = {nd["var_name"] for nd in nodes}
            logic = _extract_body(item.body, known_vars, capture_assignments=True)
            if logic:
                methods[method_name] = {
                    "args": method_args,
                    "execution_logic": logic,
                }

    # Pick the primary method: first non-__init__ method that has execution logic
    primary_method: Optional[str] = next(iter(methods), None)
    primary_logic = methods[primary_method]["execution_logic"] if primary_method else []

    return {
        "class_name": class_node.name,
        "workflow_name": workflow_name or class_node.name,
        "enable_profiling": enable_profiling,
        "nodes": nodes,
        "primary_method": primary_method,
        "execution_logic": primary_logic,
        "methods": methods,
    }


# ---------------------------------------------------------------------------
# Legacy flat-pattern visitor (used when no subclass is found)
# ---------------------------------------------------------------------------

class WorkflowVisitor(ast.NodeVisitor):
    """AST visitor to extract workflow components from flat (non-subclass) code."""

    def __init__(self):
        self.workflow_name: Optional[str] = None
        self.enable_profiling: bool = False
        self.nodes: List[Dict[str, Any]] = []
        self.imports: List[str] = []
        self.assignments: List[Dict[str, str]] = []
        self._scope_depth: int = 0
        self._workflow_subclasses: set = {"Workflow"}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef):
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = _get_full_name(base)
            if base_name in self._workflow_subclasses:
                self._workflow_subclasses.add(node.name)
                break
        self.generic_visit(node)

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

        # Detect Workflow / subclass instantiation
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id in self._workflow_subclasses and var_name:
                kwargs = {kw.arg: kw.value for kw in node.value.keywords if kw.arg}
                self.workflow_name = var_name
                if "enable_profiling" in kwargs and isinstance(kwargs["enable_profiling"], ast.Constant):
                    self.enable_profiling = kwargs["enable_profiling"].value

        # Module-level assignments (for reconstruction templates)
        if self._scope_depth == 0 and var_name and isinstance(node.value, ast.Call):
            self.assignments.append({
                "target": var_name,
                "expr": _format_expr(node.value),
            })

        # AgentNode / ServiceNode
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            if func_name in ("AgentNode", "ServiceNode"):
                node_data = {
                    "var_name": var_name,
                    "type": func_name,
                    "args": [],
                    "kwargs": {},
                    "runtime_name_expr": _format_expr(node.value.args[1]) if len(node.value.args) >= 2 else repr(var_name or "node"),
                }
                for arg in node.value.args:
                    node_data["args"].append(_format_expr(arg))
                for kw in node.value.keywords:
                    if isinstance(kw.value, ast.Dict):
                        node_data["kwargs"][kw.arg] = _extract_dict(kw.value)
                    else:
                        node_data["kwargs"][kw.arg] = _format_expr(kw.value)
                self.nodes.append(node_data)

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_workflow_subclasses(tree: ast.Module) -> List[ast.ClassDef]:
    """Return all ClassDef nodes that (directly or transitively) inherit Workflow."""
    known: set = {"Workflow"}
    result: List[ast.ClassDef] = []
    # Two-pass to handle transitive inheritance
    for _ in range(2):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = _get_full_name(base)
                if base_name in known:
                    known.add(node.name)
                    if node not in result:
                        result.append(node)
                    break
    return result


def parse_workflow_code(code: str, enable_profiling: bool = False) -> Dict[str, Any]:
    """Parse motif-ai workflow code and extract components.

    Prefers the *subclass pattern* (``class X(Workflow): ...``).
    Falls back to the flat ``main()`` / module-level pattern for legacy code.

    Returns a dictionary suitable for Jinja2 template rendering:

    ``workflow_name``
        String name passed to ``Workflow.__init__`` / ``super().__init__``.
    ``class_name``
        Python class name (subclass pattern) or ``None`` (flat pattern).
    ``enable_profiling``
        Boolean.
    ``nodes``
        List of AgentNode / ServiceNode dicts.
    ``execution_logic``
        Structured list of call / assign / loop / if entries.
    ``workflow_subclasses``
        List of per-class dicts — one per detected ``Workflow`` subclass.
        Empty for the flat pattern.
    ``imports`` / ``assignments``
        Module-level imports and top-level assignments.
    """
    tree = ast.parse(code)

    # --- collect imports and module-level assignments via the visitor ---
    visitor = WorkflowVisitor()
    visitor.visit(tree)

    # Override profiling flag if requested via CLI
    if enable_profiling:
        visitor.enable_profiling = True

    # --- try subclass pattern first ---
    subclass_nodes = _find_workflow_subclasses(tree)
    if subclass_nodes:
        subclasses = [_extract_workflow_subclass(cn) for cn in subclass_nodes]

        # Apply CLI profiling override to each subclass
        if enable_profiling:
            for sc in subclasses:
                sc["enable_profiling"] = True

        # For single-class files, promote the first subclass as the "primary"
        primary = subclasses[0]

        # First arg of the primary method (excluding 'self') becomes the entry-point
        # variable initialized from state["input"] in the generated runtime.
        primary_method_arg: Optional[str] = None
        if primary["primary_method"] and primary["methods"].get(primary["primary_method"]):
            args = primary["methods"][primary["primary_method"]]["args"]
            primary_method_arg = args[0] if args else None

        return {
            "workflow_name": primary["workflow_name"],
            "class_name": primary["class_name"],
            "enable_profiling": primary["enable_profiling"],
            "nodes": primary["nodes"],
            "execution_logic": primary["execution_logic"],
            "primary_method": primary["primary_method"],
            "primary_method_arg": primary_method_arg,
            "workflow_subclasses": subclasses,
            "imports": visitor.imports,
            "assignments": visitor.assignments,
        }

    # --- fallback: flat / main() pattern ---
    execution_body: List[ast.stmt] = tree.body
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "main":
            execution_body = stmt.body
            break

    known_node_vars = {nd["var_name"] for nd in visitor.nodes if nd.get("var_name")}
    execution_logic = _extract_body(execution_body, known_node_vars, capture_assignments=False)

    return {
        "workflow_name": visitor.workflow_name,
        "class_name": None,
        "enable_profiling": visitor.enable_profiling,
        "nodes": visitor.nodes,
        "execution_logic": execution_logic,
        "primary_method": None,
        "primary_method_arg": None,
        "workflow_subclasses": [],
        "imports": visitor.imports,
        "assignments": visitor.assignments,
    }
