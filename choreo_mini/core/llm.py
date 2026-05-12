"""Generic LLM abstraction.

This module defines a single ``LLM`` class that talks to any OpenAI-compatible
chat-completions endpoint using ``requests``.  Bring your own API key, endpoint,
and model name; the class handles payload formatting and response parsing.

Most hosted LLMs (OpenAI, Anthropic via compat layer, Groq, Together, local
Ollama, etc.) expose an OpenAI-compatible ``/v1/chat/completions`` endpoint,
so this single class covers the common cases without any provider-specific
subclassing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

import requests


@dataclass
class Message:
    """Small container for chat messages.

    ``role`` is one of ``"system"``, ``"user"``, or ``"assistant"``.
    ``tool_call_id`` is populated on tool-result messages to correlate
    the result with the original tool-call request.
    """

    role: str
    content: Optional[str]
    tool_call_id: Optional[str] = None


@dataclass
class ToolSchema:
    """Describes a single callable tool that an LLM may invoke."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRequest:
    """A single tool invocation requested by the LLM."""

    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallMessage:
    """LLM response that requests one or more tool invocations."""

    role: str
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    content: str = ""


class LLM:
    """HTTP client for any OpenAI-compatible chat-completions endpoint.

    Parameters
    ----------
    api_key:
        Bearer token (or equivalent) sent in the ``Authorization`` header.
    endpoint:
        Base URL of the API, e.g. ``"https://api.openai.com"`` or
        ``"http://localhost:11434"`` for a local Ollama instance.
    model:
        Model identifier forwarded verbatim in the request payload.
    headers:
        Additional or override headers (e.g. ``{"x-api-key": "..."}`` for
        providers that use non-Bearer auth).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.model = model
        self._extra_headers = headers or {}

    def _serialize_tools(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _post(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Union[str, ToolCallMessage]:
        """Format the payload and POST to the chat-completions endpoint.

        Returns a plain string for normal replies, or a :class:`ToolCallMessage`
        when the model's ``finish_reason`` is ``"tool_calls"``.
        """
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, **kwargs}
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        hdrs = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        resp = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            headers=hdrs,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        msg = choice["message"]
        if choice.get("finish_reason") == "tool_calls":
            tool_calls = [
                ToolCallRequest(
                    tool_call_id=tc["id"],
                    tool_name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                for tc in msg.get("tool_calls", [])
            ]
            return ToolCallMessage(
                role="assistant",
                tool_calls=tool_calls,
                content=msg.get("content") or "",
            )
        return msg["content"]

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        """Send a prompt (with optional prior-context messages) and return the response."""
        messages = [{"role": m.role, "content": m.content} for m in (context or [])]
        messages.append({"role": "user", "content": prompt})
        return self._post(messages, **kwargs)

    def stream(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Yield the response as a single chunk (non-streaming fallback)."""
        yield self.generate(prompt, context=context, **kwargs)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Union[Message, ToolCallMessage]:
        """Send a full conversation history and return the assistant reply.

        Pass ``tools`` to enable tool-calling; the return value will be a
        :class:`ToolCallMessage` if the model decides to invoke a tool.

        When ``endpoint`` is ``None`` (e.g. in subclass mocks that only override
        ``generate``), falls back to calling ``generate`` directly.
        """
        if self.endpoint is None:
            prompt = "\n".join(m.content for m in messages if m.role == "user")
            return Message(role="assistant", content=self.generate(prompt, context=messages, **kwargs))
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        result = self._post(payload_messages, tools=tools, **kwargs)
        if isinstance(result, ToolCallMessage):
            return result
        return Message(role="assistant", content=result)

    async def chat_async(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Union[Message, ToolCallMessage]:
        """Async variant of :meth:`chat`; runs in a thread pool."""
        return await asyncio.to_thread(self.chat, messages, tools, **kwargs)


class CustomLLM:
    """Wraps any callable as an LLM-compatible backend.

    Useful for local models, test stubs, and custom logic that doesn't go
    through an HTTP endpoint::

        llm = CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")

    The callable receives ``prompt`` (str), ``context`` (List[Message] or None),
    and any extra kwargs forwarded from ``generate``/``chat``.
    """

    def __init__(self, generate_fn: Callable[[str, Any], str]) -> None:
        self._fn = generate_fn

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        return self._fn(prompt, context=context, **kwargs)

    def stream(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        yield self.generate(prompt, context=context, **kwargs)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Message:
        prompt = "\n".join(m.content for m in messages if m.role == "user")
        return Message(role="assistant", content=self._fn(prompt, context=messages, **kwargs))

    async def chat_async(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Message:
        return await asyncio.to_thread(self.chat, messages, tools, **kwargs)
