"""MotifAI — Python-native LLM agent workflow orchestration."""

from motif_ai.core.belief import Belief, BeliefState
from motif_ai.core.workflow import Workflow, AgentState
from motif_ai.core.nodes import AgentNode, ServiceNode
from motif_ai.core.llm import LLM, CustomLLM, Message, ToolSchema, ToolCallRequest, ToolCallMessage
from motif_ai.core.episode import Episode, EpisodeStep, nash_convergence_detector, max_rounds_terminator
from motif_ai.core.pool import LLMCandidate, LLMPool
from motif_ai.core.mcp_server import WorkflowMCPServer
from motif_ai.core.exceptions import (
    ChoreoError,
    WorkflowError,
    AgentNotFoundError,
    AgentRegistrationError,
    EpisodeError,
    LLMError,
    ConversionError,
)
from motif_ai.core.observability import (
    ObservabilityHook,
    ObservabilityEvent,
    AgentCallStart,
    AgentCallEnd,
    AgentCallError,
    LLMRequestStart,
    LLMRequestEnd,
    LLMRetry,
    EpisodeStepStart,
    EpisodeStepEnd,
    LLMPoolRoute,
    LLMPoolFallback,
    StdoutHook,
    JsonFileHook,
    OTLPHook,
    CompositeHook,
)

__version__ = "0.3.0"
__author__ = "Sivasathivel Kandasamy"

__all__ = [
    # Workflow
    "Workflow",
    "AgentState",
    # Nodes
    "AgentNode",
    "ServiceNode",
    # LLM
    "LLM",
    "CustomLLM",
    "Message",
    "ToolSchema",
    "ToolCallRequest",
    "ToolCallMessage",
    # Belief
    "Belief",
    "BeliefState",
    # Episode / MARL
    "Episode",
    "EpisodeStep",
    "nash_convergence_detector",
    "max_rounds_terminator",
    # LLM pool
    "LLMCandidate",
    "LLMPool",
    # MCP server
    "WorkflowMCPServer",
    # Exceptions
    "ChoreoError",
    "WorkflowError",
    "AgentNotFoundError",
    "AgentRegistrationError",
    "EpisodeError",
    "LLMError",
    "ConversionError",
    # Observability
    "ObservabilityHook",
    "ObservabilityEvent",
    "AgentCallStart",
    "AgentCallEnd",
    "AgentCallError",
    "LLMRequestStart",
    "LLMRequestEnd",
    "LLMRetry",
    "EpisodeStepStart",
    "EpisodeStepEnd",
    "LLMPoolRoute",
    "LLMPoolFallback",
    "StdoutHook",
    "JsonFileHook",
    "OTLPHook",
    "CompositeHook",
]
