"""Tests for WorkflowMCPServer — MCP server exposure of motif-ai workflows.

All tests use mocked MCP internals; no real MCP client/server connection is
required.  The suite verifies:
- Tool registration (one per AgentNode)
- Resource registration (beliefs + per-agent history)
- Tool invocation routes to the correct agent via send_async
- Resource URIs embed the workflow name and agent name correctly
- serve_stdio / serve_sse / serve_sse_async delegate to FastMCP correctly
- add_tool / remove_tool pass through to the underlying FastMCP instance
- WorkflowMCPServer is importable from the top-level package
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from motif_ai.core.llm import CustomLLM, Message
from motif_ai.core.mcp_server import WorkflowMCPServer
from motif_ai.core.nodes import AgentNode
from motif_ai.core.workflow import Workflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _echo_llm():
    return CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")


class TwoAgentWorkflow(Workflow):
    def __init__(self):
        super().__init__("test-wf")
        self.analyst = AgentNode(self, "Analyst", role="Data analyst", llm=_echo_llm())
        self.advisor = AgentNode(self, "Advisor", role="Policy advisor", llm=_echo_llm())


@pytest.fixture()
def wf():
    return TwoAgentWorkflow()


@pytest.fixture()
def server(wf):
    return WorkflowMCPServer(wf, host="127.0.0.1", port=8000)


# ---------------------------------------------------------------------------
# Construction and repr
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_mcp_attribute_is_fastmcp(self, server):
        from mcp.server.fastmcp import FastMCP
        assert isinstance(server.mcp, FastMCP)

    def test_default_server_name_is_workflow_name(self, wf):
        srv = WorkflowMCPServer(wf)
        assert srv.mcp.name == "test-wf"

    def test_custom_server_name(self, wf):
        srv = WorkflowMCPServer(wf, server_name="my-mcp-server")
        assert srv.mcp.name == "my-mcp-server"

    def test_repr_contains_workflow_name(self, server):
        r = repr(server)
        assert "test-wf" in r
        assert "Analyst" in r
        assert "Advisor" in r


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_one_tool_per_agent(self, server):
        tools = await server.mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "Analyst" in tool_names
        assert "Advisor" in tool_names

    @pytest.mark.asyncio
    async def test_tool_count_matches_agent_count(self, server, wf):
        tools = await server.mcp.list_tools()
        assert len(tools) == len(wf.agent_states)

    @pytest.mark.asyncio
    async def test_tool_description_comes_from_system_prompt(self, server):
        tools = {t.name: t for t in await server.mcp.list_tools()}
        assert "Data analyst" in tools["Analyst"].description
        assert "Policy advisor" in tools["Advisor"].description

    @pytest.mark.asyncio
    async def test_tool_input_schema_has_message_param(self, server):
        tools = {t.name: t for t in await server.mcp.list_tools()}
        schema = tools["Analyst"].inputSchema
        assert "message" in schema.get("properties", {})


# ---------------------------------------------------------------------------
# Tool invocation routes to the correct agent
# ---------------------------------------------------------------------------

class TestToolInvocation:
    @pytest.mark.asyncio
    async def test_analyst_tool_calls_analyst_agent(self, server, wf):
        with patch.object(wf, "send_async", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = Message(role="assistant", content="analysis result")
            result = await server.mcp.call_tool("Analyst", {"message": "analyse GDP"})
        mock_send.assert_awaited_once_with("Analyst", "analyse GDP")
        # FastMCP wraps the return in a list of content items
        assert any("analysis result" in str(item) for item in result)

    @pytest.mark.asyncio
    async def test_advisor_tool_routes_to_advisor(self, server, wf):
        with patch.object(wf, "send_async", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = Message(role="assistant", content="policy advice")
            result = await server.mcp.call_tool("Advisor", {"message": "trade policy?"})
        mock_send.assert_awaited_once_with("Advisor", "trade policy?")
        assert any("policy advice" in str(item) for item in result)

    @pytest.mark.asyncio
    async def test_tools_are_independent(self, server, wf):
        """Calling Analyst must not accidentally call Advisor."""
        calls_log = []

        async def fake_send(name, msg):
            calls_log.append(name)
            return Message(role="assistant", content=f"{name}: ok")

        with patch.object(wf, "send_async", side_effect=fake_send):
            await server.mcp.call_tool("Analyst", {"message": "hi"})

        assert calls_log == ["Analyst"]


# ---------------------------------------------------------------------------
# Resource registration
# ---------------------------------------------------------------------------

class TestResourceRegistration:
    @pytest.mark.asyncio
    async def test_beliefs_resource_registered(self, server, wf):
        resources = await server.mcp.list_resources()
        uris = [str(r.uri) for r in resources]
        assert "workflow://test-wf/beliefs" in uris

    @pytest.mark.asyncio
    async def test_agent_history_resources_registered(self, server, wf):
        resources = await server.mcp.list_resources()
        uris = [str(r.uri) for r in resources]
        assert "workflow://test-wf/agents/Analyst/history" in uris
        assert "workflow://test-wf/agents/Advisor/history" in uris

    @pytest.mark.asyncio
    async def test_total_resource_count(self, server, wf):
        # 1 beliefs + 1 per agent
        resources = await server.mcp.list_resources()
        assert len(resources) == 1 + len(wf.agent_states)

    @pytest.mark.asyncio
    async def test_beliefs_resource_returns_json(self, server, wf):
        wf.beliefs.observe("tariff", 0.12, confidence=0.9)
        content = await server.mcp.read_resource("workflow://test-wf/beliefs")
        # content is a list of ReadResourceContents with a .content str attribute
        raw = "".join(item.content for item in content if hasattr(item, "content"))
        data = json.loads(raw)
        assert data["world"]["tariff"]["value"] == 0.12

    @pytest.mark.asyncio
    async def test_agent_history_resource_returns_json(self, server, wf):
        wf.send("Analyst", "hello")
        content = await server.mcp.read_resource(
            "workflow://test-wf/agents/Analyst/history"
        )
        raw = "".join(item.content for item in content if hasattr(item, "content"))
        data = json.loads(raw)
        assert isinstance(data, list)
        assert any(m["role"] == "user" and "hello" in m["content"] for m in data)

    @pytest.mark.asyncio
    async def test_empty_history_resource_returns_empty_list(self, server):
        content = await server.mcp.read_resource(
            "workflow://test-wf/agents/Analyst/history"
        )
        raw = "".join(item.content for item in content if hasattr(item, "content"))
        assert json.loads(raw) == []


# ---------------------------------------------------------------------------
# Serving interface delegates to FastMCP
# ---------------------------------------------------------------------------

class TestServingInterface:
    def test_serve_stdio_calls_mcp_run(self, server):
        with patch.object(server.mcp, "run") as mock_run:
            server.serve_stdio()
        mock_run.assert_called_once_with(transport="stdio")

    def test_serve_sse_calls_mcp_run(self, server):
        with patch.object(server.mcp, "run") as mock_run:
            server.serve_sse()
        mock_run.assert_called_once_with(transport="sse")

    @pytest.mark.asyncio
    async def test_serve_sse_async_calls_run_sse_async(self, server):
        with patch.object(server.mcp, "run_sse_async", new_callable=AsyncMock) as mock_async:
            await server.serve_sse_async()
        mock_async.assert_awaited_once()

    def test_sse_app_returns_starlette_app(self, server):
        from starlette.applications import Starlette
        app = server.sse_app()
        assert isinstance(app, Starlette)


# ---------------------------------------------------------------------------
# Dynamic tool management
# ---------------------------------------------------------------------------

class TestDynamicTools:
    @pytest.mark.asyncio
    async def test_add_tool_registers_new_tool(self, server):
        def extra(message: str) -> str:
            return f"extra: {message}"

        server.add_tool(extra, name="ExtraTool", description="An extra tool")
        names = {t.name for t in await server.mcp.list_tools()}
        assert "ExtraTool" in names

    @pytest.mark.asyncio
    async def test_remove_tool_unregisters_agent_tool(self, server):
        server.remove_tool("Analyst")
        names = {t.name for t in await server.mcp.list_tools()}
        assert "Analyst" not in names
        assert "Advisor" in names   # sibling unaffected


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageExport:
    def test_importable_from_top_level(self):
        import motif_ai
        assert hasattr(motif_ai, "WorkflowMCPServer")

    def test_is_correct_class(self):
        from motif_ai import WorkflowMCPServer as Cls
        from motif_ai.core.mcp_server import WorkflowMCPServer as Src
        assert Cls is Src
