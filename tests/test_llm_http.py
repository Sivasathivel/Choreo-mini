"""Tests for LLM HTTP layer — no real network calls, all mocked with requests.

Covers:
* Endpoint URL normalisation (no double /v1)
* Authorization header only when api_key is set
* Timeout forwarded to requests.post
* Payload structure (model, messages, tools)
* Tool-call response parsing → ToolCallMessage
* Normal text response parsing → Message
* HTTPError surfaces API error body
* chat() fallback path when endpoint is None (includes system message)
* CustomLLM callable wrapper
* AgentNode raises when llm=None
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from choreo_mini.core.llm import LLM, CustomLLM, Message, ToolSchema, ToolCallMessage
from choreo_mini.core.nodes import AgentNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(content: str, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [
            {
                "message": {"content": content, "role": "assistant"},
                "finish_reason": finish_reason,
            }
        ]
    }
    return resp


def _mock_tool_response(tool_name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    return resp


# ---------------------------------------------------------------------------
# Endpoint normalisation
# ---------------------------------------------------------------------------

class TestEndpointNormalisation:
    def test_base_url_unchanged(self):
        llm = LLM(endpoint="https://api.openai.com", model="gpt-4o")
        assert llm.endpoint == "https://api.openai.com"

    def test_trailing_slash_stripped(self):
        llm = LLM(endpoint="https://api.openai.com/", model="gpt-4o")
        assert llm.endpoint == "https://api.openai.com"

    def test_v1_suffix_stripped(self):
        llm = LLM(endpoint="https://api.openai.com/v1", model="gpt-4o")
        assert llm.endpoint == "https://api.openai.com"

    def test_full_path_suffix_stripped(self):
        llm = LLM(endpoint="https://api.openai.com/v1/chat/completions", model="gpt-4o")
        assert llm.endpoint == "https://api.openai.com"

    def test_request_url_always_correct(self):
        """Regardless of how the endpoint was specified, the POST URL must be right."""
        for raw in [
            "https://api.openai.com",
            "https://api.openai.com/",
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
        ]:
            llm = LLM(api_key="k", endpoint=raw, model="gpt-4o")
            with patch("requests.post", return_value=_mock_response("hi")) as m:
                llm.chat([Message(role="user", content="hi")])
                assert m.call_args[0][0] == "https://api.openai.com/v1/chat/completions", raw

    def test_none_endpoint(self):
        llm = LLM()
        assert llm.endpoint is None


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------

class TestAuthHeader:
    def test_bearer_set_when_api_key_provided(self):
        llm = LLM(api_key="sk-test", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            hdrs = m.call_args[1]["headers"]
            assert hdrs["Authorization"] == "Bearer sk-test"

    def test_no_auth_header_when_api_key_none(self):
        llm = LLM(api_key=None, endpoint="http://localhost:11434", model="llama3")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            hdrs = m.call_args[1]["headers"]
            assert "Authorization" not in hdrs

    def test_custom_header_forwarded(self):
        """e.g. Anthropic uses x-api-key instead of Bearer."""
        llm = LLM(
            endpoint="https://api.anthropic.com",
            model="claude-opus-4-5",
            headers={"x-api-key": "sk-ant-xxx", "anthropic-version": "2023-06-01"},
        )
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            hdrs = m.call_args[1]["headers"]
            assert hdrs["x-api-key"] == "sk-ant-xxx"
            assert hdrs["anthropic-version"] == "2023-06-01"

    def test_content_type_always_set(self):
        llm = LLM(endpoint="http://localhost:11434", model="m")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            assert m.call_args[1]["headers"]["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_default_timeout_is_60(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            assert m.call_args[1]["timeout"] == 60

    def test_custom_timeout_respected(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m", timeout=10)
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            assert m.call_args[1]["timeout"] == 10


# ---------------------------------------------------------------------------
# Payload structure
# ---------------------------------------------------------------------------

class TestPayload:
    def test_model_in_payload(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="gpt-4o")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            payload = m.call_args[1]["json"]
            assert payload["model"] == "gpt-4o"

    def test_messages_serialised(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        msgs = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat(msgs)
            payload_msgs = m.call_args[1]["json"]["messages"]
            assert payload_msgs[0] == {"role": "system", "content": "You are helpful"}
            assert payload_msgs[1] == {"role": "user", "content": "Hello"}

    def test_tools_serialised_when_provided(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        tools = [ToolSchema(name="add", description="adds numbers", input_schema={"type": "object"})]
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")], tools=tools)
            payload = m.call_args[1]["json"]
            assert "tools" in payload
            assert payload["tools"][0]["function"]["name"] == "add"

    def test_no_tools_key_when_not_provided(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")])
            assert "tools" not in m.call_args[1]["json"]

    def test_extra_kwargs_forwarded(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_response("hi")) as m:
            llm.chat([Message(role="user", content="hi")], temperature=0.2, max_tokens=50)
            payload = m.call_args[1]["json"]
            assert payload["temperature"] == 0.2
            assert payload["max_tokens"] == 50


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_text_reply_returns_message(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_response("Hello there!")):
            result = llm.chat([Message(role="user", content="hi")])
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "Hello there!"

    def test_tool_call_returns_tool_call_message(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_tool_response("search", {"q": "test"})):
            result = llm.chat([Message(role="user", content="search for test")])
        assert isinstance(result, ToolCallMessage)
        assert result.tool_calls[0].tool_name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}

    def test_tool_call_id_preserved(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=_mock_tool_response("fn", {}, call_id="xyz")):
            result = llm.chat([Message(role="user", content="hi")])
        assert result.tool_calls[0].tool_call_id == "xyz"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def _make_error_response(self, status: int, body: dict) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body
        resp.text = str(body)
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
        return resp

    def test_api_error_message_surfaced(self):
        llm = LLM(api_key="bad", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=self._make_error_response(
            401, {"error": {"message": "Invalid API key provided"}}
        )):
            with pytest.raises(requests.HTTPError, match="Invalid API key provided"):
                llm.chat([Message(role="user", content="hi")])

    def test_rate_limit_error_message(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=self._make_error_response(
            429, {"error": {"message": "Rate limit exceeded"}}
        )):
            with pytest.raises(requests.HTTPError, match="Rate limit exceeded"):
                llm.chat([Message(role="user", content="hi")])

    def test_status_code_in_error(self):
        llm = LLM(api_key="k", endpoint="https://api.openai.com", model="m")
        with patch("requests.post", return_value=self._make_error_response(
            500, {"message": "internal error"}
        )):
            with pytest.raises(requests.HTTPError, match="500"):
                llm.chat([Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# chat() fallback path (endpoint=None)
# ---------------------------------------------------------------------------

class TestChatFallback:
    def test_fallback_calls_generate(self):
        llm = LLM()   # endpoint=None
        with patch.object(llm, "generate", return_value="pong") as m:
            result = llm.chat([Message(role="user", content="ping")])
        assert isinstance(result, Message)
        assert result.content == "pong"
        m.assert_called_once()

    def test_fallback_includes_system_message_in_prompt(self):
        llm = LLM()
        msgs = [
            Message(role="system", content="You are a pirate"),
            Message(role="user", content="say hello"),
        ]
        with patch.object(llm, "generate", return_value="ahoy") as m:
            llm.chat(msgs)
            prompt = m.call_args[0][0]
            assert "pirate" in prompt

    def test_fallback_passes_full_context(self):
        llm = LLM()
        msgs = [Message(role="user", content="hi")]
        with patch.object(llm, "generate", return_value="hello") as m:
            llm.chat(msgs)
            ctx = m.call_args[1]["context"]
            assert ctx == msgs


# ---------------------------------------------------------------------------
# CustomLLM
# ---------------------------------------------------------------------------

class TestCustomLLM:
    def test_callable_wrapper(self):
        fn = lambda prompt, context=None, **kw: f"echo: {prompt}"
        llm = CustomLLM(fn)
        result = llm.chat([Message(role="user", content="hello")])
        assert isinstance(result, Message)
        assert "hello" in result.content

    def test_generate_delegates_to_fn(self):
        fn = lambda prompt, context=None, **kw: "ok"
        llm = CustomLLM(fn)
        assert llm.generate("test") == "ok"

    def test_stream_yields_once(self):
        fn = lambda prompt, context=None, **kw: "chunk"
        llm = CustomLLM(fn)
        chunks = list(llm.stream("hi"))
        assert chunks == ["chunk"]

    @pytest.mark.asyncio
    async def test_chat_async(self):
        fn = lambda prompt, context=None, **kw: "async reply"
        llm = CustomLLM(fn)
        result = await llm.chat_async([Message(role="user", content="hi")])
        assert result.content == "async reply"


# ---------------------------------------------------------------------------
# AgentNode LLM guard
# ---------------------------------------------------------------------------

class TestAgentNodeLLMGuard:
    def test_execute_raises_when_llm_none(self):
        agent = AgentNode(None, "Bot", role="assistant")
        with pytest.raises(ValueError, match="no LLM"):
            agent.execute("hello")

    @pytest.mark.asyncio
    async def test_execute_async_raises_when_llm_none(self):
        agent = AgentNode(None, "Bot", role="assistant")
        with pytest.raises(ValueError, match="no LLM"):
            await agent.execute_async("hello")

    def test_execute_works_with_llm(self):
        llm = CustomLLM(lambda p, **kw: "answer")
        agent = AgentNode(None, "Bot", role="assistant", llm=llm)
        result = agent.execute("question")
        assert isinstance(result, Message)
        assert result.content == "answer"
