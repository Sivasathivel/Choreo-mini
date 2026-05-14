"""Node types for choreo-mini workflows.

Nodes are the building blocks of a :class:`~choreo_mini.core.workflow.Workflow`.
Each node registers itself with the workflow on construction and can be
addressed by name at runtime.

Three concrete node types are provided:

* :class:`BaseNode` — generic graph node (parent of all others)
* :class:`AgentNode` — LLM-backed conversational agent with optional tool use
* :class:`ServiceNode` — wraps an arbitrary Python callable for data work
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from choreo_mini.core.llm import LLM, Message, ToolCallMessage, ToolSchema
from choreo_mini.core.tool_clients import BaseToolClient, create_tool_client


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------

class BaseNode:
    """Generic node in a workflow graph."""

    def __init__(
        self,
        name: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
        workflow: Optional["Workflow"] = None,
    ) -> None:
        self.name = name
        self.node_type = node_type
        self.properties = properties or {}
        self.children: List[BaseNode] = []
        self.workflow: Optional["Workflow"] = workflow

        # auto-register with the workflow when provided
        if workflow is not None:
            workflow.nodes[self.name] = self
            if workflow.root is None:
                workflow.root = self

    def add_child(self, child_node: BaseNode) -> None:
        self.children.append(child_node)

    def __repr__(self) -> str:
        return (
            f"BaseNode(name={self.name!r}, type={self.node_type!r}, "
            f"properties={self.properties!r})"
        )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

class AgentNode(BaseNode):
    """LLM-backed conversational agent.

    Parameters
    ----------
    workflow:
        The owning :class:`~choreo_mini.core.workflow.Workflow`; the agent
        registers itself on construction.  Pass ``None`` for standalone use.
    name:
        Unique identifier within the workflow.
    role:
        Short role description used to auto-generate a system prompt when
        ``system_prompt`` is not provided explicitly.
    tasks:
        Optional list of task descriptions included in the system prompt.
    goals:
        Optional list of goal statements (stored for reference; not yet
        injected into the system prompt automatically).
    backstory:
        Optional background narrative included in the system prompt.
    system_prompt:
        Explicit system prompt; overrides auto-generation from role/tasks/backstory.
    properties:
        Arbitrary key/value metadata forwarded to the graph layer.
    llm:
        :class:`~choreo_mini.core.llm.LLM` (or :class:`~choreo_mini.core.llm.CustomLLM`)
        instance used to generate responses.
    toolset:
        List of tool-server config dicts.  Each dict is passed to
        :func:`~choreo_mini.core.tool_clients.create_tool_client`; see that
        function for the expected schema.
    """

    def __init__(
        self,
        workflow: Optional["Workflow"],
        name: str,
        role: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        backstory: Optional[str] = None,
        system_prompt: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        llm: Optional[LLM] = None,
        toolset: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(name, "agent", properties, workflow=workflow)
        self.role: Optional[str] = role
        self.tasks: List[str] = tasks or []
        self.goals: List[str] = goals or []
        self.backstory: Optional[str] = backstory
        self.system_prompt: Optional[str] = system_prompt
        self.llm: Optional[LLM] = llm

        # toolset: list of server-config dicts
        self.toolset: List[Dict[str, Any]] = toolset or []
        # lazily populated on first use: server name -> client
        self._tool_clients: Dict[str, BaseToolClient] = {}
        # maps tool name -> owning server name
        self._tool_owner: Dict[str, str] = {}

        if workflow is not None:
            workflow.add_agent(self)

        if not self.system_prompt:
            self.system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Construct a system prompt from role/backstory/tasks."""
        parts = []
        if self.role:
            parts.append(f"Role: {self.role}")
        if self.backstory:
            parts.append(f"Backstory: {self.backstory}")
        if self.tasks:
            parts.append(f"Tasks: {', '.join(self.tasks)}")
        prompt = "\n".join(parts)
        if not prompt:
            raise ValueError(
                f"Agent '{self.name}' has no system_prompt and not enough "
                "information (role/backstory/tasks) to generate one."
            )
        return prompt

    # keep old name as an alias for backwards compatibility
    def get_system_prompt(self) -> str:
        return self.system_prompt or self._build_system_prompt()

    def set_system_prompt(self, prompt: str) -> None:
        if prompt:
            self.system_prompt = prompt

    def __repr__(self) -> str:
        return (
            f"AgentNode(name={self.name!r}, role={self.role!r}, "
            f"tasks={self.tasks!r}, backstory={self.backstory!r})"
        )

    # ------------------------------------------------------------------
    # Tool-client helpers (async)
    # ------------------------------------------------------------------

    async def _ensure_connected(self, config: Dict[str, Any]) -> BaseToolClient:
        """Lazily create and cache the client for a single toolset entry."""
        client_name = config["name"]
        if client_name not in self._tool_clients:
            client = create_tool_client(config)
            await client.connect()
            self._tool_clients[client_name] = client
        return self._tool_clients[client_name]

    async def get_tool_schemas(self) -> List[ToolSchema]:
        """Connect all toolset entries and return the aggregated tool schemas."""
        schemas: List[ToolSchema] = []
        for config in self.toolset:
            client = await self._ensure_connected(config)
            tools = await client.list_tools()
            for tool in tools:
                self._tool_owner[tool.name] = config["name"]
            schemas.extend(tools)
        return schemas

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool by name, routing to the correct client."""
        client_name = self._tool_owner.get(tool_name)
        if client_name is None:
            raise KeyError(
                f"Tool '{tool_name}' not found in any connected toolset client "
                f"for agent '{self.name}'"
            )
        return await self._tool_clients[client_name].call_tool(tool_name, arguments)

    async def close(self) -> None:
        """Close all cached tool-client connections."""
        for client in self._tool_clients.values():
            await client.close()
        self._tool_clients.clear()
        self._tool_owner.clear()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        context: Optional[Union[str, List[Message]]] = None,
        **kwargs: Any,
    ) -> Message:
        """Run the agent synchronously and return the assistant reply.

        Conversation history is normally managed by the owning
        :class:`~choreo_mini.core.workflow.Workflow`; ``context`` is filled
        automatically when called via :meth:`~choreo_mini.core.workflow.Workflow.send`.
        Pass ``context`` manually only when invoking the agent standalone.

        Parameters
        ----------
        context:
            Either a plain string (appended as a user message) or a list of
            :class:`~choreo_mini.core.llm.Message` objects representing the
            full conversation history so far.
        **kwargs:
            Forwarded to the underlying LLM (e.g. ``temperature``, ``max_tokens``).
        """
        if self.llm is None:
            raise ValueError(
                f"Agent '{self.name}' has no LLM configured. "
                "Pass llm=... when constructing AgentNode."
            )
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        if isinstance(context, str):
            messages.append(Message(role="user", content=context))
        elif context:
            messages.extend(context)
        return self.llm.chat(messages, **kwargs)

    async def execute_async(
        self,
        context: Optional[Union[str, List[Message]]] = None,
        **kwargs: Any,
    ) -> Message:
        """Run the agent asynchronously with a full tool-use loop.

        When no toolset is configured the behaviour is identical to
        :meth:`execute` but runs inside the event loop.  When a toolset is
        present, tool schemas are fetched, passed to the LLM via
        :meth:`~choreo_mini.core.llm.LLM.chat_async`, and each
        :class:`~choreo_mini.core.llm.ToolCallMessage` is resolved by invoking
        the matching tool client.  The loop runs until the LLM returns a plain
        :class:`~choreo_mini.core.llm.Message` or the iteration budget (10) is
        exceeded.
        """
        if self.llm is None:
            raise ValueError(
                f"Agent '{self.name}' has no LLM configured. "
                "Pass llm=... when constructing AgentNode."
            )
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        if isinstance(context, str):
            messages.append(Message(role="user", content=context))
        elif context:
            messages.extend(context)

        if not self.toolset:
            return await self.llm.chat_async(messages, **kwargs)

        tools = await self.get_tool_schemas()

        for _ in range(10):
            response = await self.llm.chat_async(messages, tools=tools, **kwargs)
            if not isinstance(response, ToolCallMessage):
                return response

            messages.append(Message(role="assistant", content=response.content))
            for req in response.tool_calls:
                result = await self.invoke_tool(req.tool_name, req.arguments)
                messages.append(
                    Message(role="tool", content=result, tool_call_id=req.tool_call_id)
                )

        return Message(role="assistant", content="[Tool-use loop budget exhausted]")


# ---------------------------------------------------------------------------
# Service node
# ---------------------------------------------------------------------------

class ServiceNode(BaseNode):
    """Wraps an arbitrary Python callable for data work.

    Services are intended for pre/post-processing steps such as loading data,
    parsing inputs, or formatting outputs.  They register with the workflow on
    construction and expose a simple :meth:`execute` interface.

    Parameters
    ----------
    workflow:
        The owning :class:`~choreo_mini.core.workflow.Workflow`.
    name:
        Unique identifier within the workflow.
    service_fn:
        Any callable ``(*args, **kwargs) -> Any``.
    properties:
        Arbitrary key/value metadata forwarded to the graph layer.
    """

    def __init__(
        self,
        workflow: Optional["Workflow"],
        name: str,
        service_fn: Callable[..., Any],
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, "service", properties, workflow=workflow)
        self.service_fn = service_fn

    def execute(self, workflow: "Workflow", *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped callable.

        ``workflow`` is accepted as the first positional argument so that the
        service can inspect workflow state or route results if needed, but it
        is not forwarded to ``service_fn`` — pass it explicitly inside your
        function if required.
        """
        return self.service_fn(*args, **kwargs)
