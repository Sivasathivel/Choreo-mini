# Changelog

All notable changes to MotifAI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — dev branch

### Added
- `CustomerOpsWorkflow` example: four-way routing, risk-lead escalation, QA review, and MCP-ready remote LLM support (`examples/customer_ops_url.py`)
- Conversion guide section in README explaining subclass vs flat pattern with side-by-side code examples
- `primary_method_arg` pre-population in all three generated backends — the first method argument is resolved from `state["input"]` automatically

### Changed
- `examples/customer_ops_url.py` rewritten as a `Workflow` subclass — now a valid CLI conversion target in addition to a runnable script
- `requirements.txt` replaced with setup instructions pointing to `pyproject.toml` extras

### Removed
- `examples/foo.py` (flat pattern / direct-run only — contradicted the documented subclass pattern)

---

## [0.3.0] — 2026-05-21

### Added
- **Subclass pattern as the primary conversion target**: `class X(Workflow)` is now the canonical way to define workflows that can be compiled to LangGraph, CrewAI, or AutoGen
- `TicketTriageWorkflow` example (`examples/foo2.py`): branching, for-loop, four-agent dispatch with test coverage across all three backends
- AST parser redesign: two-pass transitive subclass detection, node extraction from `__init__`, execution logic from method bodies
- `_bare_node_name()` in all templates — strips `self.` prefix before node lookup, enabling `self.node_name.execute(...)` syntax
- `"self": wf` in `_build_eval_env()` across all templates — enables `self.state[...]` and `self.send(...)` expressions in generated code
- Conversion guide section in README

### Fixed
- 16 pre-release bugs across `llm.py`, `workflow.py`, `episode.py`, `nodes.py`, `tool_clients.py`, and the CrewAI template
- `CustomLLM.chat()` now forwards system messages (`role in ("system", "user")`) to the generate lambda
- `Episode.reset()` now correctly restores initial environment state
- `tracemalloc` guard prevents double-start errors when profiling is enabled
- `get_profile()` returns zero-filled dict instead of raising `KeyError` on unseen agents
- Module-level `crew = _build_crew([])` removed from CrewAI template (triggered CrewAI init on every import)
- `assert` statements replaced with `RuntimeError` in A2A and HTTP tool clients

### Changed
- `parse_workflow_code()` now prefers the subclass path; falls back to `WorkflowVisitor` for legacy flat-pattern files
- All three Jinja2 templates updated with `_bare_node_name()` helper and `self` in eval env

---

## [0.2.0] — 2026-03-24

### Added
- MARL episode loop (`Episode`, `EpisodeStep`) with reward functions, env update hooks, and trajectory recording
- Nash convergence detector and max-rounds terminator
- Epistemic belief state (`BeliefState`, `Belief`) — confidence-weighted observation maps per agent and per workflow
- `WorkflowMCPServer` — expose any `Workflow` as an MCP server (SSE or stdio) with zero configuration
- MCP tool client and A2A client (`tool_clients.py`)
- `AgentNode` toolset support — agents can call external MCP tools mid-conversation
- `LLM` class: endpoint normalisation, configurable timeout, full error body on failure, optional auth header
- `CustomLLM` for test stubs and local/rule-based models
- MARL HUF experiment (`examples/marl_huf_experiment.py`): three country-agents negotiate five trade parameters toward Nash equilibrium
- CI pipeline: GitHub Actions matrix across Python 3.10–3.13
- PyPI publication at `v0.2.0`

### Fixed
- LLM HTTP layer: real SSE streaming, `ValueError` when endpoint is `None`
- Subprocess tool client merges with `os.environ` instead of replacing it

---

## [0.1.0] — 2026-03-10

### Added
- Initial `Workflow`, `AgentNode`, `ServiceNode` primitives
- LangGraph, CrewAI, and AutoGen Jinja2 templates
- CLI (`motif_ai -f FILE -b BACKEND -o OUTPUT`)
- AST parser (`parse_workflow_code`) — flat/functional pattern
- Basic tool use loop in `AgentNode`
- Ethical-use policy files (`PROHIBITED_USES.md`, `EXCLUDED_ENTITIES.md`)
