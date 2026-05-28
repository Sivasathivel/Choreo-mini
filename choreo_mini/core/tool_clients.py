"""External tool/server client implementations for choreo-mini.

Each client wraps a specific transport protocol (MCP SSE, MCP Stdio,
A2A, or plain HTTP) and exposes a uniform async interface::

    client = create_tool_client(config)
    await client.connect()
    schemas = await client.list_tools()   # -> List[ToolSchema]
    result  = await client.call_tool(name, args)  # -> str
    await client.close()

Sessions are created on demand the first time an AgentNode calls
get_tool_schemas() or invoke_tool(), and persist for the agent lifetime.
No external configuration file is required.

create_tool_client(config) is the main entry point.  config is the
toolset dict entry passed to AgentNode::

    {
        'url':         'http://localhost:8000/sse',
        'name':        'calculator',
        'description': 'Arithmetic operations',
        'type':        'mcp',
        'subtype':     'sse',
        # optional -- omit when the server is unauthenticated
        'credentials': {
            'type':    'bearer',     # 'bearer' | 'basic' | 'api_key'
            'token':   'abc...',     # for 'bearer'
            # --- OR ---
            'username': 'admin',     # for 'basic'
            'password': 'secret',    # for 'basic'
            # --- OR ---
            'api_key': 'xyz...',     # for 'api_key'
            'header':  'X-API-Key',  # for 'api_key' (default: 'Authorization')
        },
    }
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import base64

if TYPE_CHECKING:
    from choreo_mini.core.llm import ToolSchema


class BaseToolClient(ABC):
    """Abstract interface for all external tool/server clients."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.name: str = config.get("name", "")
        self.url: str = config.get("url", "")
        self._connected: bool = False

    def _auth_headers(self) -> Dict[str, str]:
        """Build HTTP authorization headers from config['credentials'].

        'bearer'  -> Authorization: Bearer <token>
        'basic'   -> Authorization: Basic <base64(user:pass)>
        'api_key' -> <header>: <api_key>  (header defaults to 'Authorization')

        Returns an empty dict when credentials are absent or the type is
        unrecognised.
        """
        creds: Optional[Dict[str, Any]] = self.config.get("credentials")
        if not creds:
            return {}
        cred_type = str(creds.get("type", "")).lower()
        if cred_type == "bearer":
            return {"Authorization": f"Bearer {creds.get('token', '')}"}
        if cred_type == "basic":
            encoded = base64.b64encode(
                f"{creds.get('username', '')}:{creds.get('password', '')}".encode()
            ).decode()
            return {"Authorization": f"Basic {encoded}"}
        if cred_type == "api_key":
            header_name = creds.get("header", "Authorization")
            return {header_name: creds.get("api_key", "")}
        return {}

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection to the remote server."""

    @abstractmethod
    async def list_tools(self) -> List["ToolSchema"]:
        """Return all tool schemas exposed by this server."""

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke a named tool and return its result as a string."""

    @abstractmethod
    async def close(self) -> None:
        """Release all resources held by this client."""


# ---------------------------------------------------------------------------
# MCP SSE client
# ---------------------------------------------------------------------------

class MCPSSEClient(BaseToolClient):
    """Client for MCP servers that use the Server-Sent Events transport.

    Used for type='mcp' with subtype='sse'. The url field should be the
    full SSE endpoint URL, e.g. http://localhost:8000/sse.
    Auth headers from config['credentials'] are forwarded with every request.
    Requires the mcp package (pip install mcp).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._session: Optional[Any] = None

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for MCP SSE tool support. "
                "Install it with: pip install mcp"
            ) from exc

        auth_headers = self._auth_headers()
        if auth_headers:
            transport = await self._exit_stack.enter_async_context(
                sse_client(self.url, headers=auth_headers)
            )
        else:
            transport = await self._exit_stack.enter_async_context(
                sse_client(self.url)
            )
        read, write = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        self._connected = True

    async def list_tools(self) -> List["ToolSchema"]:
        from choreo_mini.core.llm import ToolSchema

        if not self._connected:
            await self.connect()
        result = await self._session.list_tools()
        return [
            ToolSchema(
                name=tool.name,
                description=tool.description or "",
                input_schema=getattr(tool, "inputSchema", {}),
            )
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self._connected:
            await self.connect()
        result = await self._session.call_tool(tool_name, arguments)
        parts = [
            item.text if hasattr(item, "text") else str(item)
            for item in result.content
        ]
        return "\n".join(parts)

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._connected = False
        self._session = None


# ---------------------------------------------------------------------------
# MCP Stdio / FastMCP client
# ---------------------------------------------------------------------------

class MCPStdioClient(BaseToolClient):
    """Client for MCP servers launched as local sub-processes (stdio transport).

    Used for ``subtype='stdio'`` and ``subtype='fastmcp'``.
    The ``url`` field is treated as a shell command that launches the server,
    e.g. ``'uvx mcp-server-fetch'`` or ``'python my_server.py'``.

    Requires the ``mcp`` package (``pip install mcp``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._session: Optional[Any] = None

    def _credential_env(self) -> Dict[str, str]:
        """Build subprocess env vars from credentials."""
        creds: Optional[Dict[str, Any]] = self.config.get("credentials")
        if not creds:
            return {}
        cred_type = str(creds.get("type", "")).lower()
        if cred_type == "bearer":
            return {"CHOREO_BEARER_TOKEN": creds.get("token", "")}
        if cred_type == "basic":
            return {
                "CHOREO_BASIC_CREDENTIALS": (
                    f"{creds.get('username', '')}:{creds.get('password', '')}"
                )
            }
        if cred_type == "api_key":
            return {"CHOREO_API_KEY": creds.get("api_key", "")}
        return {}

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            from mcp.client.stdio import stdio_client, StdioServerParameters
            from mcp import ClientSession
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for MCP stdio tool support. "
                "Install it with: pip install mcp"
            ) from exc

        parts = self.url.split()
        extra_env = self._credential_env()
        # Merge credential vars into the full process environment so the
        # subprocess inherits PATH, HOME, and all other inherited env vars.
        # Replacing env entirely (the previous behaviour) caused subprocess
        # launch failures because essential vars like PATH were wiped.
        subprocess_env = {**__import__("os").environ, **extra_env} if extra_env else None
        server_params = StdioServerParameters(
            command=parts[0],
            args=parts[1:],
            env=subprocess_env,
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        self._connected = True

    async def list_tools(self) -> List["ToolSchema"]:
        from choreo_mini.core.llm import ToolSchema

        if not self._connected:
            await self.connect()
        result = await self._session.list_tools()
        return [
            ToolSchema(
                name=tool.name,
                description=tool.description or "",
                input_schema=getattr(tool, "inputSchema", {}),
            )
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self._connected:
            await self.connect()
        result = await self._session.call_tool(tool_name, arguments)
        parts = [
            item.text if hasattr(item, "text") else str(item)
            for item in result.content
        ]
        return "\n".join(parts)

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._connected = False
        self._session = None


# ---------------------------------------------------------------------------
# A2A client
# ---------------------------------------------------------------------------

class A2AClient(BaseToolClient):
    """Client for Agent-to-Agent (A2A) protocol servers.

    On ``connect()``, fetches the agent card from
    ``{url}/.well-known/agent.json`` to discover skills and capabilities.
    Each skill is exposed as a separate ``ToolSchema``; when no skills are
    declared the agent itself is wrapped as a single tool.

    Invocation POSTs a task to ``{url}/tasks/send`` following the A2A
    task-send protocol.

    Requires the ``httpx`` package (``pip install httpx``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._http: Optional[Any] = None
        self._agent_card: Optional[Dict[str, Any]] = None
        self._base_url: str = config.get("url", "").rstrip("/")

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "The 'httpx' package is required for A2A tool support. "
                "Install it with: pip install httpx"
            ) from exc

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._auth_headers(),
            timeout=30.0,
        )
        response = await self._http.get("/.well-known/agent.json")
        response.raise_for_status()
        self._agent_card = response.json()
        self._connected = True

    async def list_tools(self) -> List["ToolSchema"]:
        from choreo_mini.core.llm import ToolSchema

        if not self._connected:
            await self.connect()

        card = self._agent_card or {}
        agent_name = card.get("name", self.name)
        description = card.get("description", self.config.get("description", ""))
        skills: List[Dict[str, Any]] = card.get("skills", [])

        if skills:
            return [
                ToolSchema(
                    name=f"{agent_name}__{skill.get('id', skill.get('name', str(i)))}",
                    description=skill.get("description", skill.get("name", "")),
                    input_schema={
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                )
                for i, skill in enumerate(skills)
            ]

        return [
            ToolSchema(
                name=agent_name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self._connected:
            await self.connect()

        message_text = arguments.get("message", str(arguments))
        payload: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message_text}],
            },
        }
        if self._http is None:
            raise RuntimeError(f"A2AClient '{self.name}' is not connected. Call connect() first.")
        response = await self._http.post("/tasks/send", json=payload)
        response.raise_for_status()
        data: Any = response.json()

        # Extract text from the A2A task response
        try:
            result_obj = data.get("result", data)
            artifacts = result_obj.get("artifacts", [])
            if artifacts:
                parts = artifacts[0].get("parts", [])
                if parts:
                    return parts[0].get("text", json.dumps(data))
        except (AttributeError, KeyError, IndexError):
            pass
        return json.dumps(data)

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._connected = False


# ---------------------------------------------------------------------------
# HTTP tool client
# ---------------------------------------------------------------------------

class HTTPToolClient(BaseToolClient):
    """Client for a single custom HTTP endpoint exposed as a named tool.

    Invokes the tool by POST-ing ``{'tool_name': ..., 'arguments': ...}``
    as JSON to the configured ``url``.

    Requires the ``httpx`` package (``pip install httpx``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._http: Optional[Any] = None

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "The 'httpx' package is required for HTTP tool support. "
                "Install it with: pip install httpx"
            ) from exc

        self._http = httpx.AsyncClient(
            headers=self._auth_headers(),
            timeout=30.0,
        )
        self._connected = True

    async def list_tools(self) -> List["ToolSchema"]:
        from choreo_mini.core.llm import ToolSchema

        if not self._connected:
            await self.connect()
        return [
            ToolSchema(
                name=self.name,
                description=self.config.get("description", f"HTTP tool at {self.url}"),
                input_schema={
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            )
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self._connected:
            await self.connect()
        if self._http is None:
            raise RuntimeError(f"HTTPToolClient '{self.name}' is not connected. Call connect() first.")
        response = await self._http.post(
            self.url,
            json={"tool_name": tool_name, "arguments": arguments},
        )
        response.raise_for_status()
        data: Any = response.json()
        if isinstance(data, str):
            return data
        return json.dumps(data)

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._connected = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[str, Dict[str, type]] = {
    "mcp": {
        "sse": MCPSSEClient,
        "fastmcp": MCPStdioClient,
        "stdio": MCPStdioClient,
    },
    "a2a": {
        "__default__": A2AClient,
    },
    "http": {
        "__default__": HTTPToolClient,
    },
}


def create_tool_client(config: Dict[str, Any]) -> BaseToolClient:
    """Return the correct :class:`BaseToolClient` subclass for *config*.

    *config* must contain at least ``'type'`` and ``'url'`` keys.
    ``'subtype'`` is required for ``type='mcp'``::

        create_tool_client({
            'type': 'mcp', 'subtype': 'sse',
            'url': 'http://localhost:8000/sse',
            'name': 'calc', 'description': 'Calculator',
        })

    Raises :class:`ValueError` for unknown types or subtypes.
    """
    tool_type = config.get("type", "").lower()
    subtype = config.get("subtype", "").lower()

    type_map = _TYPE_MAP.get(tool_type)
    if type_map is None:
        supported = ", ".join(sorted(_TYPE_MAP))
        raise ValueError(
            f"Unknown tool type '{tool_type}'. Supported: {supported}"
        )

    client_cls = type_map.get(subtype) or type_map.get("__default__")
    if client_cls is None:
        supported_sub = ", ".join(k for k in type_map if k != "__default__")
        raise ValueError(
            f"Unknown subtype '{subtype}' for tool type '{tool_type}'. "
            f"Supported subtypes: {supported_sub}"
        )

    return client_cls(config)
