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
Agent response models for FlyBrowser autonomous operations.

Provides structured response objects with execution metadata, timing information,
LLM usage tracking, and formatted output capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMUsageInfo:
    """LLM usage and cost tracking information."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    calls_count: int = 0
    cached_calls: int = 0
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LLMUsageInfo":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            model=data.get("model", ""),
            calls_count=data.get("calls_count", 0),
            cached_calls=data.get("cached_calls", 0),
        )
    
    def __str__(self) -> str:
        return (
            f"LLM Usage: {self.total_tokens} tokens "
            f"({self.calls_count} calls, ${self.cost_usd:.4f})"
        )


@dataclass
class ExecutionInfo:
    """Execution metadata and progress tracking."""
    
    iterations: int = 0
    max_iterations: int = 0
    duration_seconds: float = 0.0
    pages_scraped: int = 0
    actions_taken: int = 0
    success: bool = False
    summary: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: Optional[str] = None
    current_goal: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExecutionInfo":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            iterations=data.get("iterations", 0),
            max_iterations=data.get("max_iterations", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            pages_scraped=data.get("pages_scraped", 0),
            actions_taken=data.get("actions_taken", 0),
            success=data.get("success", False),
            summary=data.get("summary", ""),
            history=data.get("history", []),
            current_phase=data.get("current_phase"),
            current_goal=data.get("current_goal"),
            plan=data.get("plan"),
        )
    
    def progress_bar(self, width: int = 40) -> str:
        """Generate a text-based progress bar."""
        if self.max_iterations == 0:
            return "[N/A]"
        
        percentage = min(100, int((self.iterations / self.max_iterations) * 100))
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage}%"
    
    def __str__(self) -> str:
        status = "[ok] Success" if self.success else "[fail] Failed"
        return (
            f"{status} | "
            f"{self.iterations}/{self.max_iterations} iterations | "
            f"{self.duration_seconds:.2f}s | "
            f"{self.actions_taken} actions"
        )


@dataclass
class AgentRequestResponse:
    """
    Structured response from agent operations with comprehensive metadata.
    
    Attributes:
        success: Whether the operation succeeded
        data: Extracted or returned data from the operation
        error: Error message if operation failed
        operation: Type of operation performed (e.g., "auto", "extract", "act")
        query: Original user query/goal
        execution: Execution metadata and progress information
        llm_usage: LLM usage and cost tracking
        metadata: Additional metadata dictionary
    """
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    operation: str = "unknown"
    query: str = ""
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)
    llm_usage: LLMUsageInfo = field(default_factory=LLMUsageInfo)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def pprint(self, max_width: int = 100, show_full_data: bool = True, show_history: bool = False) -> None:
        """
        Pretty print the response with rich formatting.
        
        Args:
            max_width: Maximum width for output lines
            show_full_data: If True, show complete data without truncation
            show_history: If True, show execution history/steps
        """
        separator = "=" * max_width
        thin_sep = "-" * max_width
        
        print(f"\n{separator}")
        print(f"  AGENT RESPONSE | Operation: {self.operation.upper()}")
        print(separator)
        
        # Status and Query
        status_icon = "[ok]" if self.success else "[fail]"
        status = f"{status_icon} SUCCESS" if self.success else f"{status_icon} FAILED"
        print(f"\n  Status: {status}")
        if self.query:
            # Show full query, wrapping if needed
            print(f"  Query:  {self.query}")
        
        # Execution Summary
        print(f"\n{thin_sep}")
        print("  EXECUTION SUMMARY")
        print(thin_sep)
        
        duration = self.execution.duration_seconds
        if duration >= 60:
            time_str = f"{int(duration // 60)}m {duration % 60:.1f}s"
        elif duration > 0:
            time_str = f"{duration:.2f}s"
        else:
            time_str = "N/A"
        
        print(f"  Duration:       {time_str}")
        
        # Show iterations with progress indicator
        iter_max = self.execution.max_iterations or "?"
        if self.execution.iterations > 0 or self.execution.max_iterations > 0:
            print(f"  Iterations:     {self.execution.iterations}/{iter_max}")
        else:
            print(f"  Iterations:     N/A")
        
        if self.execution.pages_scraped > 0:
            print(f"  Pages Scraped:  {self.execution.pages_scraped}")
        if self.execution.actions_taken > 0:
            print(f"  Actions Taken:  {self.execution.actions_taken}")
        
        if self.execution.current_phase:
            print(f"  Last Phase:     {self.execution.current_phase}")
        if self.execution.current_goal:
            print(f"  Last Goal:      {self.execution.current_goal}")
        if self.execution.summary:
            print(f"  Summary:        {self.execution.summary}")
        
        # LLM Usage - always show if we have any data
        has_llm_data = (
            self.llm_usage.total_tokens > 0 or 
            self.llm_usage.calls_count > 0 or
            self.llm_usage.model
        )
        # Also check metadata for llm_usage in case it wasn't properly parsed
        metadata_llm = self.metadata.get("llm_usage", {})
        if not has_llm_data and metadata_llm:
            has_llm_data = True
        
        if has_llm_data:
            print(f"\n{thin_sep}")
            print("  LLM USAGE")
            print(thin_sep)
            
            # Use direct values or fall back to metadata
            model = self.llm_usage.model or metadata_llm.get("model", "N/A")
            calls_count = self.llm_usage.calls_count or metadata_llm.get("calls_count", 0)
            cached_calls = self.llm_usage.cached_calls or metadata_llm.get("cached_calls", 0)
            total_tokens = self.llm_usage.total_tokens or metadata_llm.get("total_tokens", 0)
            prompt_tokens = self.llm_usage.prompt_tokens or metadata_llm.get("prompt_tokens", 0)
            completion_tokens = self.llm_usage.completion_tokens or metadata_llm.get("completion_tokens", 0)
            cost_usd = self.llm_usage.cost_usd or metadata_llm.get("cost_usd", 0.0)
            
            print(f"  Model:          {model}")
            print(f"  API Calls:      {calls_count}" + 
                  (f" ({cached_calls} cached)" if cached_calls else ""))
            if total_tokens > 0:
                print(f"  Tokens:         {total_tokens:,} total ({prompt_tokens:,} prompt / {completion_tokens:,} completion)")
            if cost_usd > 0:
                print(f"  Estimated Cost: ${cost_usd:.4f}")
        
        # Data - this is the main output!
        if self.data is not None:
            print(f"\n{thin_sep}")
            print("  RESPONSE DATA")
            print(thin_sep)
            self._print_data(self.data, indent=2, max_width=max_width - 4, show_full=show_full_data)
        
        # Execution History (optional)
        if show_history and self.execution.history:
            print(f"\n{thin_sep}")
            print(f"  EXECUTION HISTORY ({len(self.execution.history)} steps)")
            print(thin_sep)
            for i, step in enumerate(self.execution.history, 1):
                if isinstance(step, dict):
                    thought = step.get("thought", "")
                    action = step.get("action", {})
                    tool_name = action.get("tool_name", "unknown") if isinstance(action, dict) else str(action)
                    result = step.get("result", "")
                    print(f"\n  [{i}] {tool_name}")
                    if thought:
                        thought_short = thought[:100] + "..." if len(thought) > 100 else thought
                        print(f"      Thought: {thought_short}")
                    if result:
                        result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                        print(f"      Result:  {result_str}")
                else:
                    print(f"  [{i}] {step}")
        
        # Metadata summary (only interesting fields)
        if self.metadata:
            # Filter out fields already shown in execution summary
            skip_keys = {"success", "result", "error", "steps", "total_iterations", 
                        "execution_time_ms", "final_state", "execution_plan",
                        "max_iterations", "llm_usage", "cost_tracking"}
            interesting_metadata = {k: v for k, v in self.metadata.items() if k not in skip_keys}
            
            if interesting_metadata:
                print(f"\n{thin_sep}")
                print("  ADDITIONAL INFO")
                print(thin_sep)
                # Use the same _print_data method to show full content
                self._print_data(interesting_metadata, indent=2, max_width=max_width - 4, show_full=show_full_data)
        
        # Error
        if self.error:
            print(f"\n{thin_sep}")
            print("  ERROR")
            print(thin_sep)
            # Show full error message
            for line in self.error.split("\n"):
                print(f"  {line}")
        
        print(f"\n{separator}\n")
    
    def _print_data(self, data: Any, indent: int = 0, max_width: int = 96, show_full: bool = True, depth: int = 0) -> None:
        """
        Recursively print data with nice formatting.
        
        Args:
            data: Data to print
            indent: Current indentation level
            max_width: Maximum width for lines
            show_full: If True, show all data without truncation
            depth: Current recursion depth
        """
        prefix = " " * indent
        max_depth = 10  # Prevent infinite recursion
        
        if depth > max_depth:
            print(f"{prefix}[... nested data ...]")
            return
        
        if data is None:
            print(f"{prefix}None")
            return
        
        if isinstance(data, dict):
            if not data:
                print(f"{prefix}{{}}")
                return
            
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    self._print_data(value, indent + 2, max_width, show_full, depth + 1)
                elif isinstance(value, list):
                    if not value:
                        print(f"{prefix}{key}: []")
                    elif isinstance(value[0], dict):
                        print(f"{prefix}{key}: [{len(value)} items]")
                        # Show all items if show_full, otherwise first 5
                        max_items = len(value) if show_full else min(5, len(value))
                        for i, item in enumerate(value[:max_items]):
                            print(f"{prefix}  [{i+1}]")
                            self._print_data(item, indent + 4, max_width, show_full, depth + 1)
                        if not show_full and len(value) > 5:
                            print(f"{prefix}  ... and {len(value) - 5} more items")
                    else:
                        # List of primitives
                        if show_full or len(value) <= 10:
                            items_str = ", ".join(str(v) for v in value)
                        else:
                            items_str = ", ".join(str(v) for v in value[:10])
                            items_str += f", ... (+{len(value) - 10} more)"
                        print(f"{prefix}{key}: [{items_str}]")
                else:
                    # Scalar value - show full content
                    value_str = str(value)
                    if show_full or len(value_str) <= 200:
                        # Show full value, potentially multi-line
                        if "\n" in value_str:
                            print(f"{prefix}{key}:")
                            for line in value_str.split("\n"):
                                print(f"{prefix}  {line}")
                        else:
                            print(f"{prefix}{key}: {value_str}")
                    else:
                        # Truncate very long values
                        print(f"{prefix}{key}: {value_str[:200]}...")
        
        elif isinstance(data, list):
            if not data:
                print(f"{prefix}[]")
                return
            
            # Show all items if show_full, otherwise first 10
            max_items = len(data) if show_full else min(10, len(data))
            for i, item in enumerate(data[:max_items]):
                if isinstance(item, dict):
                    print(f"{prefix}[{i+1}]")
                    self._print_data(item, indent + 2, max_width, show_full, depth + 1)
                elif isinstance(item, list):
                    print(f"{prefix}[{i+1}]: [...]")
                    self._print_data(item, indent + 2, max_width, show_full, depth + 1)
                else:
                    item_str = str(item)
                    if show_full or len(item_str) <= 200:
                        print(f"{prefix}[{i+1}] {item_str}")
                    else:
                        print(f"{prefix}[{i+1}] {item_str[:200]}...")
            
            if not show_full and len(data) > 10:
                print(f"{prefix}... and {len(data) - 10} more items")
        
        elif isinstance(data, str):
            if show_full or len(data) <= 500:
                # Show full string, preserving newlines
                if "\n" in data:
                    for line in data.split("\n"):
                        print(f"{prefix}{line}")
                else:
                    print(f"{prefix}{data}")
            else:
                print(f"{prefix}{data[:500]}...")
        else:
            # Other types (int, float, bool, etc)
            print(f"{prefix}{data}")
    
    def _format_dict_oneline(self, d: dict, max_len: int = 120) -> str:
        """Format a dict as a single line summary."""
        if not d:
            return "{}"
        parts = []
        for k, v in d.items():
            if isinstance(v, dict):
                parts.append(f"{k}={{...}}")
            elif isinstance(v, list):
                parts.append(f"{k}=[{len(v)} items]")
            else:
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:50] + "..."
                parts.append(f"{k}={v_str}")
        result = ", ".join(parts)
        if len(result) > max_len:
            result = result[:max_len - 3] + "..."
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "operation": self.operation,
            "query": self.query,
            "execution": {
                "iterations": self.execution.iterations,
                "max_iterations": self.execution.max_iterations,
                "duration_seconds": self.execution.duration_seconds,
                "pages_scraped": self.execution.pages_scraped,
                "actions_taken": self.execution.actions_taken,
                "success": self.execution.success,
                "summary": self.execution.summary,
                "history": self.execution.history,
                "current_phase": self.execution.current_phase,
                "current_goal": self.execution.current_goal,
                "plan": self.execution.plan,
            },
            "llm_usage": {
                "prompt_tokens": self.llm_usage.prompt_tokens,
                "completion_tokens": self.llm_usage.completion_tokens,
                "total_tokens": self.llm_usage.total_tokens,
                "cost_usd": self.llm_usage.cost_usd,
                "model": self.llm_usage.model,
                "calls_count": self.llm_usage.calls_count,
                "cached_calls": self.llm_usage.cached_calls,
            },
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        return f"AgentRequestResponse(success={self.success}, operation={self.operation})"
    
    def __repr__(self) -> str:
        return (
            f"AgentRequestResponse(success={self.success}, "
            f"operation={self.operation}, "
            f"data={self.data!r})"
        )


def create_response(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    operation: str = "unknown",
    query: str = "",
    llm_usage: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentRequestResponse:
    """
    Factory function to create AgentRequestResponse from raw data.
    
    Args:
        success: Whether operation succeeded
        data: Extracted/returned data
        error: Error message if failed
        operation: Operation type
        query: Original query/goal
        llm_usage: LLM usage dictionary
        execution: Execution info dictionary
        history: Execution history list
        metadata: Additional metadata (typically agent_result.to_dict())
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        AgentRequestResponse instance
    """
    metadata = metadata or {}
    
    # If data is None, try to extract from metadata (common when called from SDK)
    # The agent's result is often stored in metadata['result']
    if data is None:
        data = metadata.get("result")
    
    # Extract execution info from metadata if not provided directly
    # The metadata often contains the full agent result
    exec_data = execution or {}
    
    # Extract execution metrics from metadata (from agent_result.to_dict())
    if not exec_data:
        exec_data = {
            "iterations": metadata.get("total_iterations", 0),
            "max_iterations": metadata.get("max_iterations", 0),
            "duration_seconds": metadata.get("execution_time_ms", 0) / 1000.0,
            "success": metadata.get("success", success),
            "summary": metadata.get("summary", ""),
        }
        
        # Count actions and pages from steps
        steps = metadata.get("steps", [])
        if steps:
            exec_data["actions_taken"] = len(steps)
            # Count navigation actions as pages scraped
            pages_scraped = sum(
                1 for step in steps 
                if isinstance(step, dict) and step.get("action", {}).get("tool_name") in ("navigate", "goto")
            )
            exec_data["pages_scraped"] = pages_scraped
            exec_data["history"] = steps
        
        # Extract plan info if available
        plan = metadata.get("execution_plan")
        if plan:
            exec_data["plan"] = plan
            # Get current phase/goal from plan
            if isinstance(plan, dict):
                phases = plan.get("phases", [])
                for phase in phases:
                    if isinstance(phase, dict) and phase.get("status") == "in_progress":
                        exec_data["current_phase"] = phase.get("name")
                        goals = phase.get("goals", [])
                        for goal in goals:
                            if isinstance(goal, dict) and goal.get("status") == "in_progress":
                                exec_data["current_goal"] = goal.get("description")
                                break
                        break
    
    if history:
        exec_data["history"] = history
    
    exec_info = ExecutionInfo.from_dict(exec_data)
    
    # Parse LLM usage - check multiple sources
    llm_data = llm_usage
    if not llm_data:
        # Try to extract from metadata
        llm_data = metadata.get("llm_usage") or metadata.get("cost_tracking")
    
    # If still no llm_data, try to extract from nested metadata
    if not llm_data and metadata.get("metadata"):
        nested_meta = metadata.get("metadata", {})
        llm_data = nested_meta.get("llm_usage") or nested_meta.get("cost_tracking")
    
    llm_info = LLMUsageInfo.from_dict(llm_data)
    
    return AgentRequestResponse(
        success=success,
        data=data,
        error=error,
        operation=operation,
        query=query,
        execution=exec_info,
        llm_usage=llm_info,
        metadata=metadata,
    )
