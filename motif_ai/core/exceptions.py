"""Structured exception hierarchy for motif-ai.

All framework exceptions inherit from :class:`ChoreoError` so callers can
catch them with a single ``except ChoreoError`` clause while still being
able to distinguish specific failure modes when needed::

    from motif_ai.core.exceptions import AgentNotFoundError, ChoreoError

    try:
        wf.send("Typo", "hello")
    except AgentNotFoundError as exc:
        print(f"No agent named {exc.agent_name!r}. Did you mean one of {exc.registered_agents}?")
    except ChoreoError as exc:
        print(f"Framework error: {exc}")

Hierarchy
---------
ChoreoError
├── WorkflowError
│   ├── AgentNotFoundError
│   └── AgentRegistrationError
├── EpisodeError
├── LLMError
└── ConversionError
"""

from __future__ import annotations

from typing import Any, List, Optional


class ChoreoError(Exception):
    """Base class for all motif-ai framework exceptions."""


# ---------------------------------------------------------------------------
# Workflow exceptions
# ---------------------------------------------------------------------------

class WorkflowError(ChoreoError):
    """Raised for invalid workflow construction or state operations."""


class AgentNotFoundError(WorkflowError):
    """Raised when ``wf.send()`` targets an unregistered agent.

    Attributes
    ----------
    agent_name:
        The name that was not found.
    workflow_name:
        The workflow that was searched.
    registered_agents:
        All agent names currently registered in the workflow.
    """

    def __init__(
        self,
        agent_name: str,
        workflow_name: str,
        registered_agents: List[str],
    ) -> None:
        self.agent_name = agent_name
        self.workflow_name = workflow_name
        self.registered_agents = list(registered_agents)
        super().__init__(
            f"Agent {agent_name!r} is not registered in workflow {workflow_name!r}. "
            f"Registered agents: {self.registered_agents}. "
            f"Check for typos or ensure the AgentNode is created in __init__ before calling send()."
        )


class AgentRegistrationError(WorkflowError):
    """Raised when attempting to register an agent that is already registered.

    Attributes
    ----------
    agent_name:
        The duplicate name.
    workflow_name:
        The workflow that rejected the registration.
    """

    def __init__(self, agent_name: str, workflow_name: str) -> None:
        self.agent_name = agent_name
        self.workflow_name = workflow_name
        super().__init__(
            f"Agent {agent_name!r} is already registered in workflow {workflow_name!r}. "
            f"Each agent must have a unique name within a workflow."
        )


# ---------------------------------------------------------------------------
# Episode exceptions
# ---------------------------------------------------------------------------

class EpisodeError(ChoreoError):
    """Raised for invalid episode state transitions.

    Attributes
    ----------
    episode_id:
        The ID of the episode that raised the error (when available).
    """

    def __init__(self, message: str, episode_id: Optional[str] = None) -> None:
        self.episode_id = episode_id
        super().__init__(message)


# ---------------------------------------------------------------------------
# LLM exceptions
# ---------------------------------------------------------------------------

class LLMError(ChoreoError):
    """Raised when an LLM call fails after exhausting all retries.

    Wraps the original exception as ``__cause__`` so the full traceback is
    preserved.  Callers can inspect :attr:`status_code` to distinguish rate
    limits (429) from server errors (5xx) from connection failures (0).

    Attributes
    ----------
    endpoint:
        The URL that was called.
    model:
        The model identifier used in the request.
    status_code:
        HTTP status code of the last failed response, or 0 for
        connection / timeout errors.
    attempts:
        Number of attempts made before giving up.
    """

    def __init__(
        self,
        message: str,
        endpoint: str = "",
        model: str = "",
        status_code: int = 0,
        attempts: int = 1,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.status_code = status_code
        self.attempts = attempts
        super().__init__(message)


# ---------------------------------------------------------------------------
# Conversion exceptions
# ---------------------------------------------------------------------------

class ConversionError(ChoreoError):
    """Raised when the AST-to-runtime conversion pipeline encounters a problem.

    This can occur during:

    * Parsing (``parse_workflow_code``) — unrecognised AST patterns.
    * Template rendering — Jinja2 errors or missing render-data keys.
    * Generated code execution — ``eval()`` fails on a captured expression.

    Attributes
    ----------
    expression:
        The source expression (or fragment) that could not be processed.
    available_vars:
        Variable names in scope at the point of failure (for eval errors).
    source_hint:
        File name or method name where the error occurred.
    """

    def __init__(
        self,
        message: str,
        expression: str = "",
        available_vars: Optional[List[str]] = None,
        source_hint: str = "",
    ) -> None:
        self.expression = expression
        self.available_vars = available_vars or []
        self.source_hint = source_hint

        detail_parts = [message]
        if expression:
            detail_parts.append(f"  Expression : {expression!r}")
        if available_vars is not None:
            detail_parts.append(f"  In scope   : {available_vars}")
        if source_hint:
            detail_parts.append(f"  Location   : {source_hint}")

        super().__init__("\n".join(detail_parts))
