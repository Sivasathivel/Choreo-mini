"""MCP server exposure for motif-ai workflows.

Wraps a :class:`~motif_ai.core.workflow.Workflow` and exposes its agents
as MCP tools and its belief/history state as MCP resources — making any
motif-ai workflow consumable by Claude Desktop, another motif-ai
workflow, or any MCP-compatible client.

Architecture
------------
Each :class:`~motif_ai.core.nodes.AgentNode` becomes one MCP **tool**:

* **Tool name** — the agent name (e.g. ``"Analyst"``)
* **Tool description** — the agent's system prompt / role
* **Input** — ``{"message": str}``
* **Output** — the agent's reply string

Workflow state is exposed as MCP **resources**:

* ``workflow://{name}/beliefs`` — JSON snapshot of the workflow-level
  :class:`~motif_ai.core.belief.BeliefState`
* ``workflow://{name}/agents/{agent}/history`` — JSON list of the agent's
  conversation history (role + content)

Transports
----------
* **Stdio** — for local/subprocess use (e.g. ``claude_desktop_config.json``)
* **SSE** — for HTTP-based access over a network

Quick start::

    from motif_ai import WorkflowMCPServer
    from my_workflow import MyWorkflow

    server = WorkflowMCPServer(MyWorkflow(), host="0.0.0.0", port=8000)

    server.serve_sse()      # blocking SSE server
    server.serve_stdio()    # blocking stdio server (for subprocess / Claude Desktop)

    app = server.sse_app()  # bare Starlette ASGI app for custom uvicorn

Claude Desktop config example (stdio)::

    {
        "mcpServers": {
            "my-workflow": {
                "command": "python",
                "args": ["-m", "my_package.mcp_entrypoint"]
            }
        }
    }

where ``mcp_entrypoint.py`` contains::

    from motif_ai import WorkflowMCPServer
    from my_workflow import MyWorkflow
    WorkflowMCPServer(MyWorkflow()).serve_stdio()
"""

from __future__ import annotations

import json
from typing import Any, Optional

from motif_ai.core.workflow import Workflow


class WorkflowMCPServer:
    """Expose a :class:`~motif_ai.core.workflow.Workflow` as an MCP server.

    Parameters
    ----------
    workflow:
        The workflow whose agents will be exposed as tools.
    server_name:
        Name advertised to MCP clients.  Defaults to the workflow name.
    instructions:
        Optional free-text instructions shown to the MCP client describing
        what this server provides.
    host:
        Bind address for SSE transport (default ``"127.0.0.1"``).
    port:
        Bind port for SSE transport (default ``8000``).

    Attributes
    ----------
    mcp:
        The underlying :class:`mcp.server.fastmcp.FastMCP` instance.
        Exposed so callers can attach additional tools, resources, or
        prompts beyond what motif-ai registers automatically.
    """

    def __init__(
        self,
        workflow: Workflow,
        server_name: Optional[str] = None,
        instructions: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise ImportError(
                "mcp package is required for WorkflowMCPServer. "
                "Install it with: pip install mcp"
            ) from exc

        self.workflow = workflow
        self._host = host
        self._port = port

        self.mcp = FastMCP(
            name=server_name or workflow.name,
            instructions=instructions or self._default_instructions(),
            host=host,
            port=port,
        )

        self._register_tools()
        self._register_resources()

    # ------------------------------------------------------------------
    # Internal registration
    # ------------------------------------------------------------------

    def _default_instructions(self) -> str:
        agent_names = list(self.workflow.agent_states)
        return (
            f"This is a motif-ai workflow server named '{self.workflow.name}'. "
            f"It exposes {len(agent_names)} agent(s) as tools: {', '.join(agent_names)}. "
            "Send messages to any agent tool and receive its reply. "
            "Use the belief and history resources to inspect workflow state."
        )

    def _register_tools(self) -> None:
        """Register one MCP tool per AgentNode in the workflow."""
        for agent_name in self.workflow.agent_states:
            self._register_agent_tool(agent_name)

    def _register_agent_tool(self, agent_name: str) -> None:
        """Create and register an async MCP tool for a single agent."""
        agent_state = self.workflow.agent_states[agent_name]
        description = (
            agent_state.agent.system_prompt
            or agent_state.agent.role
            or f"Send a message to the '{agent_name}' agent."
        )

        # Build the async tool function.  The closure captures agent_name
        # so each registered function routes to the correct agent.
        async def _tool(message: str) -> str:
            reply = await self.workflow.send_async(agent_name, message)
            return reply.content

        # Give the function a meaningful name (FastMCP uses __name__ in traces)
        safe_name = agent_name.lower().replace(" ", "_").replace("-", "_")
        _tool.__name__ = safe_name
        _tool.__doc__ = description

        self.mcp.add_tool(_tool, name=agent_name, description=description)

    def _register_resources(self) -> None:
        """Register belief snapshot and per-agent history as MCP resources."""
        wf = self.workflow
        wf_name = wf.name

        # Workflow-level belief snapshot
        @self.mcp.resource(
            uri=f"workflow://{wf_name}/beliefs",
            name="beliefs",
            description=(
                f"Current belief state of the '{wf_name}' workflow. "
                "Returns a JSON snapshot of world beliefs and agent beliefs."
            ),
            mime_type="application/json",
        )
        async def _beliefs() -> str:
            return json.dumps(wf.beliefs.snapshot(), indent=2)

        # Per-agent conversation history
        for agent_name in wf.agent_states:
            self._register_agent_history_resource(agent_name)

    def _register_agent_history_resource(self, agent_name: str) -> None:
        """Register a resource for a single agent's conversation history."""
        wf = self.workflow
        wf_name = wf.name

        @self.mcp.resource(
            uri=f"workflow://{wf_name}/agents/{agent_name}/history",
            name=f"{agent_name}-history",
            description=(
                f"Conversation history for the '{agent_name}' agent — "
                "ordered list of role/content message pairs."
            ),
            mime_type="application/json",
        )
        async def _history() -> str:
            msgs = wf.get_history(agent_name)
            return json.dumps(
                [{"role": m.role, "content": m.content} for m in msgs],
                indent=2,
            )

    # ------------------------------------------------------------------
    # Public serving interface
    # ------------------------------------------------------------------

    def serve_stdio(self) -> None:
        """Start a blocking MCP stdio server.

        Use this for subprocess / Claude Desktop integration.  The process
        reads MCP messages from stdin and writes replies to stdout.

        The call blocks until stdin is closed.
        """
        self.mcp.run(transport="stdio")

    def serve_sse(self) -> None:
        """Start a blocking MCP SSE server on ``host:port``.

        The server accepts HTTP connections at ``/sse`` (event-stream) and
        ``/messages/`` (client-to-server POST).

        The call blocks until the process is interrupted.
        """
        self.mcp.run(transport="sse")

    def sse_app(self) -> Any:
        """Return the raw Starlette ASGI application for the SSE transport.

        Use this when you want to mount the MCP server inside an existing
        ASGI application or run it with a custom uvicorn configuration::

            import uvicorn
            server = WorkflowMCPServer(wf, host="0.0.0.0", port=9000)
            uvicorn.run(server.sse_app(), host="0.0.0.0", port=9000)
        """
        return self.mcp.sse_app()

    async def serve_sse_async(self) -> None:
        """Async variant of :meth:`serve_sse`.

        Awaitable; use this inside an existing async application or when
        you need to run the MCP server alongside other async tasks::

            async def main():
                server = WorkflowMCPServer(wf, port=8000)
                await asyncio.gather(
                    server.serve_sse_async(),
                    my_other_coroutine(),
                )
        """
        await self.mcp.run_sse_async()

    # ------------------------------------------------------------------
    # Dynamic tool management
    # ------------------------------------------------------------------

    def add_tool(self, fn: Any, *, name: str, description: str = "") -> None:
        """Register an additional MCP tool beyond the auto-generated agent tools.

        Parameters
        ----------
        fn:
            Any callable or async callable ``(*args, **kwargs) -> str``.
        name:
            Tool name exposed to the MCP client.
        description:
            Human-readable description of what the tool does.
        """
        self.mcp.add_tool(fn, name=name, description=description)

    def remove_tool(self, name: str) -> None:
        """Remove a tool by name (including auto-generated agent tools)."""
        self.mcp.remove_tool(name)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        agents = list(self.workflow.agent_states)
        return (
            f"WorkflowMCPServer(workflow={self.workflow.name!r}, "
            f"agents={agents}, host={self._host!r}, port={self._port})"
        )
