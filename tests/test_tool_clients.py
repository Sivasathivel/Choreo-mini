"""Tests for choreo_mini.core.tool_clients — factory dispatch and client behaviour."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from choreo_mini.core.tool_clients import (
    A2AClient,
    BaseToolClient,
    HTTPToolClient,
    MCPSSEClient,
    MCPStdioClient,
    create_tool_client,
)
from choreo_mini.core.llm import ToolSchema


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCreateToolClient:
    def test_mcp_sse(self):
        cfg = {"type": "mcp", "subtype": "sse", "url": "http://localhost/sse", "name": "calc"}
        client = create_tool_client(cfg)
        assert isinstance(client, MCPSSEClient)
        assert client.name == "calc"
        assert client.url == "http://localhost/sse"

    def test_mcp_fastmcp(self):
        cfg = {"type": "mcp", "subtype": "fastmcp", "url": "uvx mcp-server-fetch", "name": "fetch"}
        client = create_tool_client(cfg)
        assert isinstance(client, MCPStdioClient)

    def test_mcp_stdio(self):
        cfg = {"type": "mcp", "subtype": "stdio", "url": "python server.py", "name": "srv"}
        client = create_tool_client(cfg)
        assert isinstance(client, MCPStdioClient)

    def test_a2a(self):
        cfg = {"type": "a2a", "url": "http://localhost:9000", "name": "remote_agent"}
        client = create_tool_client(cfg)
        assert isinstance(client, A2AClient)

    def test_http(self):
        cfg = {"type": "http", "url": "http://localhost:8080/tool", "name": "mycalc"}
        client = create_tool_client(cfg)
        assert isinstance(client, HTTPToolClient)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown tool type 'grpc'"):
            create_tool_client({"type": "grpc", "url": "x", "name": "x"})

    def test_unknown_subtype_raises(self):
        with pytest.raises(ValueError, match="Unknown subtype 'websocket'"):
            create_tool_client({"type": "mcp", "subtype": "websocket", "url": "x", "name": "x"})


# ---------------------------------------------------------------------------
# MCPSSEClient tests (MCP import mocked)
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str = "desc"):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = {"type": "object", "properties": {}}
    return t


def _make_content(text: str):
    c = MagicMock()
    c.text = text
    return c


class TestMCPSSEClient:
    @pytest.fixture()
    def client(self):
        return MCPSSEClient({"type": "mcp", "subtype": "sse", "url": "http://x/sse", "name": "srv"})

    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        list_result = MagicMock()
        list_result.tools = [_make_tool("add"), _make_tool("sub")]
        mock_session.list_tools = AsyncMock(return_value=list_result)

        client._connected = True
        client._session = mock_session
        schemas = await client.list_tools()

        assert len(schemas) == 2
        assert schemas[0].name == "add"
        assert isinstance(schemas[0], ToolSchema)

    @pytest.mark.asyncio
    async def test_call_tool(self, client):
        mock_session = AsyncMock()
        call_result = MagicMock()
        call_result.content = [_make_content("42")]
        mock_session.call_tool = AsyncMock(return_value=call_result)

        client._connected = True
        client._session = mock_session
        result = await client.call_tool("add", {"a": 1, "b": 2})
        assert result == "42"
        mock_session.call_tool.assert_called_once_with("add", {"a": 1, "b": 2})

    @pytest.mark.asyncio
    async def test_close_resets_state(self, client):
        mock_stack = AsyncMock()
        client._exit_stack = mock_stack
        client._connected = True
        client._session = AsyncMock()
        await client.close()
        mock_stack.aclose.assert_called_once()
        assert not client._connected
        assert client._session is None

    @pytest.mark.asyncio
    async def test_connect_raises_without_mcp(self, client):
        with patch.dict("sys.modules", {"mcp": None, "mcp.client.sse": None}):
            with pytest.raises(ImportError, match="mcp"):
                await client.connect()


# ---------------------------------------------------------------------------
# MCPStdioClient tests
# ---------------------------------------------------------------------------

class TestMCPStdioClient:
    @pytest.fixture()
    def client(self):
        return MCPStdioClient({"type": "mcp", "subtype": "fastmcp", "url": "uvx my-server", "name": "srv"})

    @pytest.mark.asyncio
    async def test_call_tool_multi_content(self, client):
        content = [_make_content("part1"), _make_content("part2")]
        call_result = MagicMock()
        call_result.content = content
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=call_result)

        client._connected = True
        client._session = mock_session
        result = await client.call_tool("fetch", {"url": "http://x"})
        assert result == "part1\npart2"

    @pytest.mark.asyncio
    async def test_connect_raises_without_mcp(self, client):
        with patch.dict("sys.modules", {"mcp": None, "mcp.client.stdio": None}):
            with pytest.raises(ImportError, match="mcp"):
                await client.connect()


# ---------------------------------------------------------------------------
# A2AClient tests
# ---------------------------------------------------------------------------

class TestA2AClient:
    @pytest.fixture()
    def client(self):
        return A2AClient({
            "type": "a2a",
            "url": "http://localhost:9000",
            "name": "agent",
            "description": "A remote agent",
        })

    @pytest.mark.asyncio
    async def test_list_tools_no_skills(self, client):
        client._connected = True
        client._agent_card = {"name": "RemoteAgent", "description": "Does stuff"}
        schemas = await client.list_tools()
        assert len(schemas) == 1
        assert schemas[0].name == "RemoteAgent"
        assert "message" in schemas[0].input_schema["properties"]

    @pytest.mark.asyncio
    async def test_list_tools_with_skills(self, client):
        client._connected = True
        client._agent_card = {
            "name": "Agent",
            "skills": [
                {"id": "s1", "name": "Skill1", "description": "First skill"},
                {"id": "s2", "name": "Skill2", "description": "Second skill"},
            ],
        }
        schemas = await client.list_tools()
        assert len(schemas) == 2
        assert schemas[0].name == "Agent__s1"
        assert schemas[1].name == "Agent__s2"

    @pytest.mark.asyncio
    async def test_call_tool_extracts_artifact_text(self, client):
        mock_http = AsyncMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "result": {
                "artifacts": [{"parts": [{"text": "hello from agent"}]}]
            }
        }
        mock_http.post = AsyncMock(return_value=response)
        client._connected = True
        client._http = mock_http

        result = await client.call_tool("Agent", {"message": "hi"})
        assert result == "hello from agent"

    @pytest.mark.asyncio
    async def test_close(self, client):
        mock_http = AsyncMock()
        client._http = mock_http
        client._connected = True
        await client.close()
        mock_http.aclose.assert_called_once()
        assert not client._connected

    @pytest.mark.asyncio
    async def test_connect_raises_without_httpx(self, client):
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                await client.connect()


# ---------------------------------------------------------------------------
# HTTPToolClient tests
# ---------------------------------------------------------------------------

class TestHTTPToolClient:
    @pytest.fixture()
    def client(self):
        return HTTPToolClient({
            "type": "http",
            "url": "http://localhost:8080/invoke",
            "name": "mycalc",
            "description": "Custom calculator",
        })

    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        client._connected = True
        schemas = await client.list_tools()
        assert schemas[0].name == "mycalc"
        assert schemas[0].description == "Custom calculator"

    @pytest.mark.asyncio
    async def test_call_tool_json_response(self, client):
        mock_http = AsyncMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"result": 42}
        mock_http.post = AsyncMock(return_value=response)
        client._connected = True
        client._http = mock_http

        result = await client.call_tool("mycalc", {"input": "1+1"})
        assert result == '{"result": 42}'

    @pytest.mark.asyncio
    async def test_call_tool_string_response(self, client):
        mock_http = AsyncMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = "pong"
        mock_http.post = AsyncMock(return_value=response)
        client._connected = True
        client._http = mock_http

        result = await client.call_tool("mycalc", {"input": "ping"})
        assert result == "pong"

    @pytest.mark.asyncio
    async def test_close(self, client):
        mock_http = AsyncMock()
        client._http = mock_http
        client._connected = True
        await client.close()
        mock_http.aclose.assert_called_once()
        assert not client._connected
