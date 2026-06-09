"""Choreo-Mini — Python-native LLM agent workflow orchestration."""

from choreo_mini.core.belief import Belief, BeliefState
from choreo_mini.core.workflow import Workflow, AgentState
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.llm import LLM, CustomLLM, Message, ToolSchema, ToolCallRequest, ToolCallMessage
from choreo_mini.core.episode import Episode, EpisodeStep, nash_convergence_detector, max_rounds_terminator
from choreo_mini.core.mcp_server import WorkflowMCPServer
from choreo_mini.core.observability import (
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
    # MCP server
    "WorkflowMCPServer",
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
    "StdoutHook",
    "JsonFileHook",
    "OTLPHook",
    "CompositeHook",
]
