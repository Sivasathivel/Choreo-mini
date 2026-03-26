"""Generic LLM abstraction and implementations.

This module defines a base ``LLM`` class plus built-in providers for
popular hosted APIs (OpenAI, Anthropic, Gemini) and a ``CustomLLM``
wrapper for arbitrary backends (local LLaMA, private endpoints, etc.).

The base class implements the minimal chat protocol used throughout
MCP/A2A tooling, and a factory lets you instantiate by provider name.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Union


@dataclass
class Message:
    """Small container for chat messages.

    ``role`` should be one of ``"system"``, ``"user"``, ``"assistant"``,
    or ``"tool"``; providers generally accept the same schema.  The
    ``content`` field holds the text of the message.  ``tool_call_id`` is
    populated on tool-result messages (``role='tool'``) to correlate the
    result with the original tool-call request.
    """

    role: str
    content: str
    tool_call_id: Optional[str] = None


@dataclass
class ToolSchema:
    """Describes a single callable tool that an LLM may invoke.

    ``input_schema`` follows the JSON Schema subset used by the MCP
    protocol and the OpenAI function-calling API.
    """

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
    """LLM response that requests one or more tool invocations.

    Returned by :meth:`LLM.chat_async` when the model uses tool-calling
    instead of (or in addition to) producing plain text.  ``content`` may
    carry any partial text the model generated alongside the tool calls.
    """

    role: str
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    content: str = ""


# registry for provider name -> class
_LLM_REGISTRY: Dict[str, type[LLM]] = {}


def register_llm(name: str) -> Callable[[type], type]:
    """Decorator that registers an LLM subclass under ``name``.

    Names are case-insensitive.
    """

    def decorator(cls: type) -> type:
        _LLM_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class LLM(ABC):
    """Abstract base that supports both single-turn and chat interactions.

    Subclasses may override ``generate`` and/or ``chat``.  ``generate``
    accepts an optional ``context`` list of :class:`Message` objects that
    will be included before the prompt; this makes it trivial to maintain
    conversation history.
    """

    def __init__(self, endpoint: Optional[str] = None, **kwargs: Any) -> None:
        # store kwargs (api_key, model, endpoint, etc.) for introspection
        self.config = {**kwargs, "endpoint": endpoint}
        # convenience attribute for most providers
        self.endpoint = endpoint

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text given a prompt and optional context.

        ``context`` messages will be sent along with the prompt to the
        underlying API; providers should interpret the list as a full
        conversation history if they support it.  ``kwargs`` are passed
        through to the provider (temperature, max_tokens, etc.).
        """
        raise NotImplementedError("subclasses must implement generate")

    def stream(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Yield pieces of the generated response.

        The default simply emits the full ``generate`` result.
        """
        yield self.generate(prompt, context=context, **kwargs)

    def chat(self, messages: List[Message], **kwargs: Any) -> Message:
        """Perform a multi-turn chat operation.

        This is the canonical interface for downstream protocols (MCP/A2A,
        tool-calling, etc.).  ``kwargs`` are passed through to the provider.

        The default falls back to :meth:`generate` after flattening the
        user-submitted contents; ``context`` is the full message list so
        providers may reconstruct the conversation if desired.
        """
        prompt = "\n".join(m.content for m in messages if m.role == "user")
        return Message(role="assistant", content=self.generate(prompt, context=messages, **kwargs))

    async def chat_async(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs: Any,
    ) -> Union[Message, ToolCallMessage]:
        """Async chat call with optional tool-calling support.

        The default implementation runs :meth:`chat` in a thread pool so
        that blocking LLM providers do not stall the event loop.  Tool
        schemas are accepted but ignored by the default; subclasses that
        support tool-calling should override this method and return a
        :class:`ToolCallMessage` when the model requests tool invocations.
        """
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> "LLM":
        """Return an instance of the named provider.

        ``kwargs`` are forwarded to the provider's constructor.
        """
        provider_cls = _LLM_REGISTRY.get(provider.lower())
        if provider_cls is None:
            raise ValueError(f"Unknown LLM provider '{provider}'")
        return provider_cls(**kwargs)


@register_llm("openai")
class OpenAI(LLM):
    """Wrapper around OpenAI's completion/chat APIs.

    ``endpoint`` may be used to point at a custom base URL (e.g. if you
    are using a reverse proxy or on‑prem deployment).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, model=model, endpoint=endpoint, **kwargs)
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "https://api.openai.com"

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        # placeholder implementation
        # TODO: replace with real API call, sending ``context`` if supported
        return f"[OpenAI {self.model} at {self.endpoint} received prompt: {prompt!r} (context={context})]"


@register_llm("anthropic")
class Anthropic(LLM):
    """Stub for Anthropic's Claude family.

    Accepts an optional ``endpoint`` for custom deployments.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-2",
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, model=model, endpoint=endpoint, **kwargs)
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "https://api.anthropic.com"

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        return f"[Anthropic {self.model} at {self.endpoint} got {prompt!r} (context={context})]"


@register_llm("gemini")
class Gemini(LLM):
    """Stub for Google's Gemini models.

    ``endpoint`` can override the default Google endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.0",
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, model=model, endpoint=endpoint, **kwargs)
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "https://api.google.com/gemini"

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        return f"[Gemini {self.model} at {self.endpoint} prompts {prompt!r} (context={context})]"


@register_llm("custom")
class CustomLLM(LLM):
    """A generic wrapper for a user‑provided generation callable.

    This makes it easy to integrate arbitrary backends such as local
    fine‑tuned LLaMA models, custom REST endpoints, etc.

    Example::

        from transformers import pipeline

        gen = pipeline("text-generation", model="./models/llama-tuned")
        llm = CustomLLM(generate_fn=lambda prompt, **kw: gen(prompt, **kw)[0]["generated_text"])
    """

    def __init__(self, generate_fn: Callable[[str, Any], str], **kwargs: Any) -> None:
        super().__init__(generate_fn=generate_fn, **kwargs)
        self._generate_fn = generate_fn

    def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> str:
        # user provided function may or may not care about context; give it
        # anyway for flexibility.
        return self._generate_fn(prompt, context=context, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        # if the provided function supports streaming, you can adapt
        # this method accordingly. the default just wraps ``generate``.
        yield self.generate(prompt, **kwargs)
