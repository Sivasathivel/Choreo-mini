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
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import requests

# Observability is imported lazily-ish to avoid a circular import:
# observability.py does not import llm.py, so this is safe.
from choreo_mini.core.observability import (
    ObservabilityHook,
    LLMRequestStart,
    LLMRequestEnd,
    LLMRetry,
    _safe_emit,
)


@dataclass
class Message:
    """Small container for chat messages.

    ``role`` is one of ``"system"``, ``"user"``, or ``"assistant"``.
    ``tool_call_id`` is populated on tool-result messages to correlate
    the result with the original tool-call request.
    ``call_id`` is populated by :meth:`~choreo_mini.core.workflow.Workflow.send`
    with the span ID of that call — correlates the response to its
    observability span.
    """

    role: str
    content: Optional[str]
    tool_call_id: Optional[str] = None
    call_id: Optional[str] = None


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
        timeout: int = 60,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        observability: Optional[ObservabilityHook] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        # Normalise: strip any /v1 or /v1/chat/completions suffix that users
        # may accidentally include so _post() never produces a double /v1 path.
        if endpoint:
            endpoint = endpoint.rstrip("/")
            for suffix in ("/v1/chat/completions", "/v1"):
                if endpoint.endswith(suffix):
                    endpoint = endpoint[: -len(suffix)]
                    break
        self.endpoint = endpoint or None
        self.model = model
        self._extra_headers = headers or {}
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self._observability: Optional[ObservabilityHook] = observability

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

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_error_detail(resp: requests.Response) -> str:
        """Pull a human-readable message out of an API error response."""
        try:
            err_body = resp.json()
            return (
                err_body.get("error", {}).get("message")
                or err_body.get("message")
                or resp.text
            )
        except Exception:
            return resp.text

    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        """Return True for transient server-side errors worth retrying."""
        return status_code in (429, 500, 502, 503, 504)

    def _retry_delay(self, attempt: int, resp: Optional[requests.Response]) -> float:
        """Compute how long to wait before the next attempt.

        Respects ``Retry-After`` when present (429); otherwise exponential
        backoff capped at 60 s.
        """
        if resp is not None:
            raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if raw is not None:
                try:
                    return max(float(raw), 0.0)
                except ValueError:
                    pass
        return min(self.retry_base_delay * (2 ** attempt), 60.0)

    # ------------------------------------------------------------------
    # Core HTTP
    # ------------------------------------------------------------------

    def _post(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolSchema]] = None,
        _trace_id: str = "",
        _span_id: str = "",
        **kwargs: Any,
    ) -> Union[str, ToolCallMessage]:
        """Format the payload and POST to the chat-completions endpoint.

        Retries up to ``max_retries`` times on transient errors (429, 5xx,
        connection resets) with exponential back-off.  Raises on permanent
        errors (4xx other than 429) immediately.

        ``_trace_id`` / ``_span_id`` are injected by the Workflow layer so
        LLM-level observability events (retries, request timing) are correlated
        to the same span as their parent ``wf.send()`` call.

        Returns a plain string for normal replies, or a :class:`ToolCallMessage`
        when the model's ``finish_reason`` is ``"tool_calls"``.
        """
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        # strip internal kwargs before forwarding to API
        payload.update({k: v for k, v in kwargs.items()
                        if not k.startswith("_")})
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        hdrs: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        hdrs.update(self._extra_headers)

        url = f"{self.endpoint}/v1/chat/completions"
        model_name = self.model or ""
        last_exc: Exception = RuntimeError("No attempts made.")

        for attempt in range(self.max_retries + 1):
            if self._observability:
                _safe_emit(self._observability, LLMRequestStart(
                    trace_id=_trace_id, span_id=_span_id,
                    endpoint=url, model=model_name, attempt=attempt,
                ))
            req_start = time.time()
            try:
                resp = requests.post(url, json=payload, headers=hdrs, timeout=self.timeout)
            except requests.ConnectionError as exc:
                delay = self._retry_delay(attempt, None)
                last_exc = RuntimeError(
                    f"Could not connect to LLM endpoint {url!r}. "
                    "Check hostname, network, VPN, or firewall settings."
                )
                if attempt < self.max_retries:
                    if self._observability:
                        _safe_emit(self._observability, LLMRetry(
                            trace_id=_trace_id, span_id=_span_id,
                            endpoint=url, model=model_name, attempt=attempt,
                            delay_s=delay, reason="ConnectionError",
                        ))
                    time.sleep(delay)
                    continue
                raise last_exc from exc
            except requests.Timeout as exc:
                delay = self._retry_delay(attempt, None)
                last_exc = RuntimeError(
                    f"LLM request timed out after {self.timeout}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})."
                )
                if attempt < self.max_retries:
                    if self._observability:
                        _safe_emit(self._observability, LLMRetry(
                            trace_id=_trace_id, span_id=_span_id,
                            endpoint=url, model=model_name, attempt=attempt,
                            delay_s=delay, reason="Timeout",
                        ))
                    time.sleep(delay)
                    continue
                raise last_exc from exc

            req_latency = time.time() - req_start

            if resp.ok:
                if self._observability:
                    _safe_emit(self._observability, LLMRequestEnd(
                        trace_id=_trace_id, span_id=_span_id,
                        endpoint=url, model=model_name, attempt=attempt,
                        status_code=resp.status_code, latency_s=req_latency,
                    ))
                break

            detail = self._extract_error_detail(resp)
            if self._is_retryable(resp.status_code) and attempt < self.max_retries:
                last_exc = requests.HTTPError(
                    f"API error {resp.status_code}: {detail}", response=resp
                )
                delay = self._retry_delay(attempt, resp)
                if self._observability:
                    _safe_emit(self._observability, LLMRetry(
                        trace_id=_trace_id, span_id=_span_id,
                        endpoint=url, model=model_name, attempt=attempt,
                        status_code=resp.status_code, delay_s=delay,
                        reason=detail[:120],
                    ))
                time.sleep(delay)
                continue

            # Permanent error — raise immediately with full detail.
            raise requests.HTTPError(
                f"API error {resp.status_code}: {detail}", response=resp
            )
        else:
            # Exhausted all retries on a retryable error.
            raise last_exc

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
        """Send a prompt (with optional prior-context messages) and return the response.

        Raises
        ------
        ValueError
            If ``endpoint`` is ``None``.  Use :class:`CustomLLM` for callable-based
            models that don't go through an HTTP endpoint.
        """
        if self.endpoint is None:
            raise ValueError(
                "LLM.generate() requires 'endpoint' to be set. "
                "Use CustomLLM for callable-based models with no HTTP endpoint."
            )
        messages = [{"role": m.role, "content": m.content} for m in (context or [])]
        messages.append({"role": "user", "content": prompt})
        return self._post(messages, **kwargs)

    def stream(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream the response token-by-token using SSE (server-sent events).

        Requires ``endpoint`` to be set.  Yields each text delta as it arrives.
        Falls back to a single chunk when the server does not support streaming
        or when ``endpoint`` is ``None``.
        """
        if self.endpoint is None:
            yield self.generate(prompt, context=context, **kwargs)
            return

        messages = [{"role": m.role, "content": m.content} for m in (context or [])]
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        hdrs: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        hdrs.update(self._extra_headers)

        url = f"{self.endpoint}/v1/chat/completions"
        last_exc: Exception = RuntimeError("No attempts made.")
        resp = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url, json=payload, headers=hdrs, timeout=self.timeout, stream=True
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt, None))
                    continue
                raise
            if resp.ok:
                break
            detail = self._extract_error_detail(resp)
            if self._is_retryable(resp.status_code) and attempt < self.max_retries:
                last_exc = requests.HTTPError(
                    f"API error {resp.status_code}: {detail}", response=resp
                )
                time.sleep(self._retry_delay(attempt, resp))
                continue
            raise requests.HTTPError(
                f"API error {resp.status_code}: {detail}", response=resp
            )
        else:
            raise last_exc

        with resp:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

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
            # No HTTP endpoint — delegate to generate().  Build the prompt from
            # all non-tool messages so the system prompt is not silently dropped.
            prompt_parts = [m.content for m in messages if m.role in ("system", "user") and m.content]
            prompt = "\n".join(prompt_parts)
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
        # Include system and user messages so the agent's role/persona is preserved.
        prompt_parts = [m.content for m in messages if m.role in ("system", "user") and m.content]
        prompt = "\n".join(prompt_parts)
        return Message(role="assistant", content=self._fn(prompt, context=messages, **kwargs))

    async def chat_async(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Message:
        return await asyncio.to_thread(self.chat, messages, tools, **kwargs)
