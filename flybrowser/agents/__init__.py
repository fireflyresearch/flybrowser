# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlyBrowser ReAct Agentic Framework.

This module provides a state-of-the-art ReAct (Reasoning and Acting) framework
for browser automation with explicit reasoning, type-safe tools, and sophisticated
safety mechanisms.

Core Components:
    - ReActAgent: The core agent implementing the ReAct paradigm
    - AgentOrchestrator: High-level orchestrator with approval workflows
    - ToolRegistry: Type-safe tool registration and management
    - AgentMemory: Context and memory management system

Example Usage:
    ```python
    from flybrowser.agents import (
        ReActAgent,
        AgentOrchestrator,
        ToolRegistry,
        ExecutionMode,
    )
    
    # Create tool registry and register browser tools
    registry = ToolRegistry()
    registry.register_defaults()
    
    # Create agent with LLM backend
    agent = ReActAgent(llm=llm, tools=registry)
    
    # Create orchestrator with supervised execution
    orchestrator = AgentOrchestrator(
        agent=agent,
        mode=ExecutionMode.SUPERVISED,
    )
    
    # Execute a task
    result = await orchestrator.execute(
        task="Navigate to example.com and extract the page title",
        browser_context=context,
    )
    ```
"""

# Core types
from flybrowser.agents.types import (
    Action,
    ToolResult,
    Observation,
    ReActStep,
    ExecutionState,
    SafetyLevel,
    MemoryEntry,
    ReasoningStrategy,
)

# Configuration
from flybrowser.agents.config import AgentConfig

# Tool system
from flybrowser.agents.tools import (
    BaseTool,
    ToolMetadata,
    ToolParameter,
    ToolRegistry,
    # Navigation tools
    NavigateTool,
    GoBackTool,
    GoForwardTool,
    RefreshTool,
    # Interaction tools
    ClickTool,
    TypeTool,
    ScrollTool,
    HoverTool,
    PressKeyTool,
    # Extraction tools
    ExtractTextTool,
    ScreenshotTool,
    GetPageStateTool,
    # System tools
    CompleteTool,
    FailTool,
    WaitTool,
    AskUserTool,
)

# Core agent
from flybrowser.agents.react_agent import ReActAgent

# Orchestrator
from flybrowser.agents.orchestrator import (
    AgentOrchestrator,
    ExecutionMode,
    SafetyConfig,
    StopReason,
    ProgressTracker,
    CircuitBreaker,
    CircuitBreakerState,
)

# Parser
from flybrowser.agents.parser import ReActParser

# Memory
from flybrowser.agents.memory import AgentMemory, WorkingMemory

# Strategy Selection
from flybrowser.agents.strategy_selector import StrategySelector

# Planning System
from flybrowser.agents.planner import (
    TaskPlanner,
    ExecutionPlan,
    Phase,
    Goal,
    PhaseStatus,
    GoalStatus,
)

# SDK Integration
from flybrowser.agents.sdk_integration import ReActBrowserAgent, create_react_agent_for_sdk

# Response Models
from flybrowser.agents.response import (
    AgentRequestResponse,
    create_response,
    LLMUsageInfo,
    ExecutionInfo,
)

# Context System
from flybrowser.agents.context import (
    ContextType,
    ActionContext,
    FileUploadSpec,
    ContextBuilder,
    ContextValidator,
    create_form_context,
    create_upload_context,
    create_filter_context,
)

__all__ = [
    # Core types
    "Action",
    "ToolResult",
    "Observation",
    "ReActStep",
    "ExecutionState",
    "SafetyLevel",
    "MemoryEntry",
    "AgentConfig",
    "ReasoningStrategy",
    # Tool system
    "BaseTool",
    "ToolMetadata",
    "ToolParameter",
    "ToolRegistry",
    # Navigation tools
    "NavigateTool",
    "GoBackTool",
    "GoForwardTool",
    "RefreshTool",
    # Interaction tools
    "ClickTool",
    "TypeTool",
    "ScrollTool",
    "HoverTool",
    "PressKeyTool",
    # Extraction tools
    "ExtractTextTool",
    "ScreenshotTool",
    "GetPageStateTool",
    # System tools
    "CompleteTool",
    "FailTool",
    "WaitTool",
    "AskUserTool",
    # Core agent
    "ReActAgent",
    # Orchestrator
    "AgentOrchestrator",
    "ExecutionMode",
    "SafetyConfig",
    "StopReason",
    "ProgressTracker",
    "CircuitBreaker",
    "CircuitBreakerState",
    # Parser
    "ReActParser",
    # Memory
    "AgentMemory",
    "WorkingMemory",
    # Strategy Selection
    "StrategySelector",
    # Planning System
    "TaskPlanner",
    "ExecutionPlan",
    "Phase",
    "Goal",
    "PhaseStatus",
    "GoalStatus",
    # SDK Integration
    "ReActBrowserAgent",
    "create_react_agent_for_sdk",
    # Response Models
    "AgentRequestResponse",
    "create_response",
    "LLMUsageInfo",
    "ExecutionInfo",
    # Context System
    "ContextType",
    "ActionContext",
    "FileUploadSpec",
    "ContextBuilder",
    "ContextValidator",
    "create_form_context",
    "create_upload_context",
    "create_filter_context",
]

