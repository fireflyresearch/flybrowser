# Copyright 2026 Firefly Software Solutions Inc.
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
System and control tools for the ReAct agent.

This module provides tools for agent control flow including
task completion, failure signaling, waiting, and user interaction.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from flybrowser.agents.types import SafetyLevel, ToolCategory, ToolResult
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter


class CompleteTool(BaseTool):
    """Signal successful task completion."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="complete",
            description="Signal that the task has been completed successfully. Use this when you have accomplished the goal.",
            category=ToolCategory.SYSTEM,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="summary",
                    type="string",
                    description="A summary of what was accomplished",
                    required=True,
                ),
                ToolParameter(
                    name="result",
                    type="object",
                    description="Optional structured result data",
                    required=False,
                ),
            ],
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Signal task completion."""
        summary = kwargs.get("summary", "Task completed")
        result_data = kwargs.get("result", {})
        
        return ToolResult.success_result(
            data={
                "completed": True,
                "summary": summary,
                "result": result_data,
            },
            is_terminal=True,
        )


class FailTool(BaseTool):
    """Signal task failure."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="fail",
            description="Signal that the task cannot be completed. Use this when you encounter an unrecoverable error.",
            category=ToolCategory.SYSTEM,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="reason",
                    type="string",
                    description="The reason why the task failed",
                    required=True,
                ),
                ToolParameter(
                    name="recoverable",
                    type="boolean",
                    description="Whether the failure might be recoverable with different approach",
                    required=False,
                ),
            ],
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Signal task failure."""
        reason = kwargs.get("reason", "Unknown failure")
        recoverable = kwargs.get("recoverable", False)
        
        # Create error result with metadata
        result = ToolResult.error_result(
            error=reason,
            error_code="TASK_FAILED",
        )
        result.metadata["recoverable"] = recoverable
        result.metadata["is_terminal"] = True
        return result


class WaitTool(BaseTool):
    """Wait for a specified duration or condition."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="wait",
            description="Wait for a specified number of seconds. Use this when you need to wait for page loading or animations.",
            category=ToolCategory.SYSTEM,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="seconds",
                    type="number",
                    description="Number of seconds to wait (max 30)",
                    required=True,
                ),
                ToolParameter(
                    name="reason",
                    type="string",
                    description="Reason for waiting",
                    required=False,
                ),
            ],
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute wait."""
        seconds = kwargs.get("seconds", 1)
        reason = kwargs.get("reason", "")
        
        # Cap at 30 seconds for safety
        seconds = min(max(0, float(seconds)), 30)
        
        await asyncio.sleep(seconds)
        
        return ToolResult.success_result(
            data={"waited_seconds": seconds, "reason": reason},
            message=f"Waited {seconds} seconds",
        )


class AskUserTool(BaseTool):
    """Request input or clarification from the user."""
    
    def __init__(self, page_controller: Optional[Any] = None, user_input_callback: Optional[Any] = None) -> None:
        super().__init__(page_controller)
        self._callback = user_input_callback
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="ask_user",
            description="Ask the user for input or clarification. Use this when you need information that wasn't provided.",
            category=ToolCategory.SYSTEM,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="question",
                    type="string",
                    description="The question to ask the user",
                    required=True,
                ),
            ],
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Request user input."""
        question = kwargs.get("question")
        
        if not question:
            return ToolResult.error_result("Question is required")
        
        # If a callback is provided, use it
        if self._callback:
            try:
                response = await self._callback(question)
                return ToolResult.success_result(
                    data={"question": question, "response": response},
                    message=f"User responded: {response}",
                )
            except Exception as e:
                return ToolResult.error_result(f"Failed to get user input: {str(e)}")
        
        # Otherwise, return a pending state
        result = ToolResult.success_result(
            data={"question": question, "awaiting_response": True},
        )
        result.metadata["requires_user_input"] = True
        return result

