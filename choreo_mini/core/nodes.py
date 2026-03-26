from typing import Any, Callable, Dict, List, Optional, Union

# import LLM types for execution
from choreo_mini.core.llm import LLM, Message, ToolCallMessage, ToolSchema
from choreo_mini.core.tool_clients import BaseToolClient, create_tool_client

class BaseNode:
    def __init__(
        self,
        name: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
        workflow: Optional['Workflow'] = None,
    ):
        self.name = name
        self.node_type = node_type
        self.properties = properties or {}
        self.children: List['BaseNode'] = []
        self.workflow: Optional['Workflow'] = workflow

        # automatically register with workflow if provided
        if workflow is not None:
            workflow.nodes[self.name] = self
            if workflow.root is None:
                workflow.root = self


    def add_child(self, child_node: 'BaseNode'):
        self.children.append(child_node)

    def __repr__(self):
        return f"BaseNode(name={self.name}, type={self.node_type}, properties={self.properties})"

class AgentNode(BaseNode):
    def __init__(
        self,
        workflow: Optional['Workflow'],
        name: str,
        role: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        backstory: Optional[str] = None,
        system_prompt: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        llm: Optional[LLM] = None,
        toolset: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(name, "agent", properties, workflow=workflow)
        self.node_type = "agent"
        self.role: Optional[str] = role
        self.tasks: List[str] = tasks or []
        self.backstory: Optional[str] = backstory
        self.system_prompt: Optional[str] = system_prompt
        self.goals: List[str] = goals or []
        self.llm: Optional[LLM] = llm

        # toolset: list of dicts, each describing one external tool server
        self.toolset: List[Dict[str, Any]] = toolset or []
        # lazily populated on first use: config['name'] -> BaseToolClient
        self._tool_clients: Dict[str, BaseToolClient] = {}
        # tracks which client owns each tool name
        self._tool_owner: Dict[str, str] = {}

        # if a workflow was provided, register as conversational agent
        if workflow is not None:
            workflow.add_agent(self)

        if not self.system_prompt:
            self.system_prompt = self.get_system_prompt()

    def get_system_prompt(self) -> str:
        if not self.system_prompt:
            prompt_parts = []
            if self.role: prompt_parts.append(f"Role: {self.role}")
            if self.backstory: prompt_parts.append(f"Backstory: {self.backstory}")
            if self.tasks: prompt_parts.append(f"Tasks: {', '.join(self.tasks)}")
            self.system_prompt = "\n".join(prompt_parts)
        if self.system_prompt: return self.system_prompt
        else:
            raise ValueError(f"Agent '{self.name}' does not have a system prompt or enough information to generate one.")
    
    def set_system_prompt(self, prompt: str):
        if prompt:
            self.system_prompt = prompt
    
    def __repr__(self):
        return f"AgentNode(name={self.name}, role={self.role}, tasks={self.tasks}, backstory={self.backstory})"

    # ------------------------------------------------------------------
    # tool-client helpers (async)
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
        """Connect all toolset entries and return aggregated tool schemas."""
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

    async def execute_async(
        self,
        context: Optional[Union[str, List[Message]]] = None,
        **kwargs: Any,
    ) -> Message:
        """Run the agent with an optional full async tool-use loop.

        When no toolset is configured the behaviour is identical to the
        synchronous :meth:`execute`, but runs inside the event loop.
        When a toolset is present, tool schemas are fetched, passed to
        the LLM via :meth:`~choreo_mini.core.llm.LLM.chat_async`, and
        each :class:`~choreo_mini.core.llm.ToolCallMessage` is resolved
        by invoking the matching tool client.  The loop runs until the
        LLM returns a plain :class:`~choreo_mini.core.llm.Message` or the
        iteration budget (10) is exceeded.
        """
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

            # Append assistant's tool-call turn to the conversation
            messages.append(Message(role="assistant", content=response.content))

            # Execute each requested tool and feed results back
            for req in response.tool_calls:
                result = await self.invoke_tool(req.tool_name, req.arguments)
                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=req.tool_call_id,
                    )
                )

        return Message(role="assistant", content="[Tool-use loop budget exhausted]")

    async def close(self) -> None:
        """Close all cached tool-client connections for this agent."""
        for client in self._tool_clients.values():
            await client.close()
        self._tool_clients.clear()
        self._tool_owner.clear()

    def execute(
        self,
        context: Optional[Union[str, List[Message]]] = None,
        **kwds,
    ) -> Message:
        """Run the agent using an LLM.

        **Note:** conversation history is normally managed by a
        :class:`Workflow` instance.  When you invoke ``execute`` from a
        workflow, the ``context`` argument is filled automatically with the
        prior exchange.  Callers may pass ``context`` manually only if they
        wish to override or inspect the history themselves.

        ``context`` may be either a single string (appended as a user
        message) or a list of :class:`Message` objects representing a prior
        dialogue.  Additional keyword arguments are passed through to the
        model (temperature, max_tokens, etc.).

        LLM configuration is read from ``self.properties``.  At minimum
        the following keys are consulted:

        * ``provider`` – name registered with :func:`LLM.register_llm`
          (default ``"openai"``)
        * ``api_key`` – API key or credential for the service
        * ``model`` – model name (optional)
        * ``endpoint`` – base URL of the API (optional)

        The method returns the assistant message produced by the model.
        """
        # assemble provider arguments from properties (still available if needed)
        # build message list from system prompt + incoming context
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        if isinstance(context, str):
            messages.append(Message(role="user", content=context))
        elif context:
            messages.extend(context)

        # delegate to underlying LLM
        return self.llm.chat(messages, **kwds)

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    


class ServiceNode(BaseNode):
    """Node wrapping an arbitrary Python service or function.

    Services are intended for data-loading, transformation, or other
    work that precedes/follows agent interaction.  They register with a
    workflow when instantiated and provide a simple ``execute`` API.
    """

    def __init__(
        self,
        workflow: Optional['Workflow'],
        name: str,
        service_fn: Callable[..., Any],
        properties: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "service", properties, workflow=workflow)
        self.service_fn = service_fn

    def execute(self, workflow: 'Workflow', *args, **kwargs) -> Any:
        # the workflow is passed so that the service can route results, log,
        # or inspect other agent states if necessary.
        return self.service_fn(*args, **kwargs)
