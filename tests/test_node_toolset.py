"""Tests for AgentNode toolset parameter, lazy connections, tool-use loop, and close."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from choreo_mini.core.llm import LLM, Message, ToolCallMessage, ToolCallRequest, ToolSchema
from choreo_mini.core.nodes import AgentNode
from choreo_mini.core.tool_clients import BaseToolClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EchoLLM(LLM):
    """Trivial synchronous LLM that echoes the last user message."""

    def generate(self, prompt, context=None, **kwargs):
        return f"echo: {prompt}"


class _ToolCallLLM(LLM):
    """LLM that first returns a tool-call, then returns the tool result wrapped in text."""

    def __init__(self, tool_name: str, tool_result_prefix: str = "Result") -> None:
        super().__init__()
        self._calls = 0
        self._tool_name = tool_name
        self._tool_result_prefix = tool_result_prefix

    def generate(self, prompt, context=None, **kwargs):
        return "plain"

    async def chat_async(self, messages, tools=None, **kwargs):
        self._calls += 1
        if self._calls == 1 and tools:
            # First call: request a tool
            return ToolCallMessage(
                role="assistant",
                tool_calls=[
                    ToolCallRequest(
                        tool_call_id="tc-1",
                        tool_name=self._tool_name,
                        arguments={"input": "test"},
                    )
                ],
            )
        # Subsequent calls: return plain text
        last_tool_msg = next(
            (m for m in reversed(messages) if m.role == "tool"), None
        )
        prefix = self._tool_result_prefix
        content = f"{prefix}: {last_tool_msg.content}" if last_tool_msg else "done"
        return Message(role="assistant", content=content)


def _mock_client(
    tool_names: List[str],
    call_result: str = "42",
) -> BaseToolClient:
    """Return a mock BaseToolClient that exposes *tool_names* and returns *call_result*."""
    client = AsyncMock(spec=BaseToolClient)
    client.name = tool_names[0] if tool_names else "mock"
    schemas = [
        ToolSchema(name=name, description=f"Tool {name}", input_schema={})
        for name in tool_names
    ]
    client.list_tools = AsyncMock(return_value=schemas)
    client.call_tool = AsyncMock(return_value=call_result)
    client.connect = AsyncMock()
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# AgentNode construction - toolset parameter
# ---------------------------------------------------------------------------

class TestAgentNodeToolsetParam:
    def test_no_toolset_by_default(self):
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM())
        assert agent.toolset == []
        assert agent._tool_clients == {}

    def test_toolset_stored(self):
        ts = [{"url": "http://x/sse", "name": "calc", "type": "mcp", "subtype": "sse", "description": "calc"}]
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=ts)
        assert agent.toolset == ts

    def test_toolset_none_treated_as_empty(self):
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=None)
        assert agent.toolset == []


# ---------------------------------------------------------------------------
# Lazy connection (_ensure_connected)
# ---------------------------------------------------------------------------

class TestEnsureConnected:
    @pytest.mark.asyncio
    async def test_connects_once_and_caches(self):
        ts = [{"name": "srv", "type": "mcp", "subtype": "sse", "url": "http://x/sse"}]
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=ts)

        fake_client = _mock_client(["add"])
        with patch("choreo_mini.core.nodes.create_tool_client", return_value=fake_client):
            c1 = await agent._ensure_connected(ts[0])
            c2 = await agent._ensure_connected(ts[0])

        fake_client.connect.assert_called_once()
        assert c1 is c2 is fake_client

    @pytest.mark.asyncio
    async def test_multiple_toolset_entries_each_connected(self):
        ts = [
            {"name": "a", "type": "mcp", "subtype": "sse", "url": "http://a/sse"},
            {"name": "b", "type": "http", "url": "http://b", "description": ""},
        ]
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=ts)

        fake_a = _mock_client(["tool_a"])
        fake_b = _mock_client(["tool_b"])

        with patch("choreo_mini.core.nodes.create_tool_client", side_effect=[fake_a, fake_b]):
            schemas = await agent.get_tool_schemas()

        assert len(schemas) == 2
        assert {s.name for s in schemas} == {"tool_a", "tool_b"}
        fake_a.connect.assert_called_once()
        fake_b.connect.assert_called_once()


# ---------------------------------------------------------------------------
# get_tool_schemas — aggregation and ownership tracking
# ---------------------------------------------------------------------------

class TestGetToolSchemas:
    @pytest.mark.asyncio
    async def test_ownership_tracked(self):
        ts = [{"name": "srv", "type": "mcp", "subtype": "sse", "url": "http://x/sse"}]
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=ts)

        fake_client = _mock_client(["add", "sub"])
        with patch("choreo_mini.core.nodes.create_tool_client", return_value=fake_client):
            await agent.get_tool_schemas()

        assert agent._tool_owner["add"] == "srv"
        assert agent._tool_owner["sub"] == "srv"


# ---------------------------------------------------------------------------
# invoke_tool
# ---------------------------------------------------------------------------

class TestInvokeTool:
    @pytest.mark.asyncio
    async def test_routes_to_correct_client(self):
        ts = [{"name": "srv", "type": "mcp", "subtype": "sse", "url": "http://x/sse"}]
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=ts)

        fake_client = _mock_client(["add"])
        with patch("choreo_mini.core.nodes.create_tool_client", return_value=fake_client):
            await agent.get_tool_schemas()
            result = await agent.invoke_tool("add", {"a": 1, "b": 2})

        assert result == "42"
        fake_client.call_tool.assert_called_once_with("add", {"a": 1, "b": 2})

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM())
        with pytest.raises(KeyError, match="not found"):
            await agent.invoke_tool("nonexistent", {})


# ---------------------------------------------------------------------------
# execute_async — no toolset (delegates to chat_async)
# ---------------------------------------------------------------------------

class TestExecuteAsyncNoToolset:
    @pytest.mark.asyncio
    async def test_delegates_chat_async(self):
        agent = AgentNode(None, "Bot", role="echo bot", llm=_EchoLLM())
        msg = await agent.execute_async(context="hello")
        assert isinstance(msg, Message)
        assert msg.role == "assistant"
        assert "hello" in msg.content

    @pytest.mark.asyncio
    async def test_context_list_passed_through(self):
        received: List = []

        class _CaptureLLM(LLM):
            def generate(self, prompt, context=None, **kwargs):
                return "ok"

            async def chat_async(self, messages, tools=None, **kwargs):
                received.extend(messages)
                return Message(role="assistant", content="ok")

        agent = AgentNode(None, "Bot", role="r", llm=_CaptureLLM())
        ctx = [Message(role="user", content="hi")]
        await agent.execute_async(context=ctx)
        assert any(m.content == "hi" for m in received)


# ---------------------------------------------------------------------------
# execute_async — with toolset (tool-use loop)
# ---------------------------------------------------------------------------

class TestExecuteAsyncToolUseLoop:
    @pytest.mark.asyncio
    async def test_tool_use_loop_calls_tool_and_continues(self):
        ts = [{"name": "srv", "type": "mcp", "subtype": "sse", "url": "http://x/sse"}]
        llm = _ToolCallLLM(tool_name="add", tool_result_prefix="Sum")
        agent = AgentNode(None, "Bot", role="calc", llm=llm, toolset=ts)

        fake_client = _mock_client(["add"], call_result="7")
        with patch("choreo_mini.core.nodes.create_tool_client", return_value=fake_client):
            result = await agent.execute_async(context="what is 3+4?")

        assert isinstance(result, Message)
        assert "Sum" in result.content
        fake_client.call_tool.assert_called_once_with("add", {"input": "test"})

    @pytest.mark.asyncio
    async def test_tool_call_id_propagated(self):
        ts = [{"name": "srv", "type": "mcp", "subtype": "sse", "url": "http://x/sse"}]

        collected_messages: List[Message] = []

        class _RecordingLLM(LLM):
            _call = 0

            def generate(self, prompt, context=None, **kwargs):
                return "ok"

            async def chat_async(self, messages, tools=None, **kwargs):
                self._call += 1
                if self._call == 1:
                    return ToolCallMessage(
                        role="assistant",
                        tool_calls=[ToolCallRequest(tool_call_id="id-99", tool_name="add", arguments={})],
                    )
                collected_messages.extend(messages)
                return Message(role="assistant", content="done")

        agent = AgentNode(None, "Bot", role="r", llm=_RecordingLLM(), toolset=ts)
        fake_client = _mock_client(["add"], call_result="result")
        with patch("choreo_mini.core.nodes.create_tool_client", return_value=fake_client):
            await agent.execute_async(context="go")

        tool_msgs = [m for m in collected_messages if m.role == "tool"]
        assert any(m.tool_call_id == "id-99" for m in tool_msgs)


# ---------------------------------------------------------------------------
# AgentNode.close
# ---------------------------------------------------------------------------

class TestAgentNodeClose:
    @pytest.mark.asyncio
    async def test_closes_all_clients(self):
        ts = [
            {"name": "a", "type": "mcp", "subtype": "sse", "url": "http://a/sse"},
            {"name": "b", "type": "http", "url": "http://b", "description": ""},
        ]
        agent = AgentNode(None, "Bot", role="bot", llm=_EchoLLM(), toolset=ts)

        fake_a = _mock_client(["tool_a"])
        fake_b = _mock_client(["tool_b"])

        with patch("choreo_mini.core.nodes.create_tool_client", side_effect=[fake_a, fake_b]):
            await agent.get_tool_schemas()

        await agent.close()
        fake_a.close.assert_called_once()
        fake_b.close.assert_called_once()
        assert agent._tool_clients == {}
        assert agent._tool_owner == {}


# ---------------------------------------------------------------------------
# execute() (sync) backward compat — toolset must not affect sync path
# ---------------------------------------------------------------------------

class TestExecuteSyncBackwardCompat:
    def test_execute_sync_works_without_toolset(self):
        agent = AgentNode(None, "Bot", role="greeter", llm=_EchoLLM())
        result = agent.execute(context="hi")
        assert isinstance(result, Message)
        assert result.role == "assistant"
