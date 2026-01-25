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
Execution Logger for structured, hierarchical logging in FlyBrowser.

This module provides a unified logging system that:
- Maintains hierarchical context (Goal > Phase > Step > Action)
- Supports configurable verbosity levels
- Produces clean, readable output for humans
- Optionally outputs structured JSON for programmatic consumption

Usage:
    from flybrowser.utils.execution_logger import get_execution_logger, LogVerbosity
    
    elog = get_execution_logger()
    elog.set_verbosity(LogVerbosity.NORMAL)
    
    with elog.goal_context("Search for startup info"):
        with elog.phase_context("Navigate to Google", phase_num=1, total_phases=3):
            elog.step("Load google.com")
            elog.browser_action("Navigated to https://google.com")
            elog.step_complete(success=True, duration_ms=333)
        elog.phase_complete(success=True, duration_ms=1500)
    elog.goal_complete(success=True, duration_ms=120000)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, Generator, List, Optional, Union


class LogVerbosity(IntEnum):
    """
    Verbosity levels for execution logging.
    
    Each level includes all information from lower levels.
    """
    
    SILENT = 0    # No output at all
    MINIMAL = 1   # Goal start/end, errors only
    NORMAL = 2    # + Phase summaries (default)
    VERBOSE = 3   # + Step details, agent decisions
    DEBUG = 4     # + LLM calls, full prompts, all intermediate steps


class LogCategory(str, Enum):
    """Categories for log messages with consistent prefixes."""
    
    GOAL = "GOAL"
    PHASE = "PHASE"
    STEP = "STEP"
    SUBSTEP = "SUBSTEP"
    AGENT = "AGENT"
    LLM = "LLM"
    BROWSER = "BROWSER"
    OBSTACLE = "OBSTACLE"
    REPLAN = "REPLAN"
    ERROR = "ERROR"
    WARNING = "WARNING"
    SUCCESS = "SUCCESS"


@dataclass
class ExecutionLogContext:
    """
    Context for the current execution state.
    
    Tracks the hierarchy depth and current execution context
    for proper indentation and context-aware logging.
    """
    
    goal: Optional[str] = None
    current_phase: Optional[str] = None
    current_phase_num: Optional[int] = None
    total_phases: Optional[int] = None
    current_step: Optional[str] = None
    depth: int = 0
    start_time: Optional[float] = None
    phase_start_time: Optional[float] = None
    step_start_time: Optional[float] = None
    
    def get_indent(self) -> str:
        """Get indentation string for current depth."""
        return "  " * self.depth


@dataclass
class ExecutionLogEntry:
    """A single log entry for structured output."""
    
    timestamp: str
    level: str
    category: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "level": self.level,
            "category": self.category,
            "message": self.message,
        }
        if self.context:
            result["context"] = self.context
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.success is not None:
            result["success"] = self.success
        return result


class ExecutionLogger:
    """
    Unified execution logger with hierarchical context and verbosity control.
    
    This logger provides:
    - Consistent formatting across all FlyBrowser components
    - Hierarchical context management with indentation
    - Verbosity-aware output filtering
    - Support for both human-readable and JSON output
    
    Example:
        >>> elog = ExecutionLogger()
        >>> elog.set_verbosity(LogVerbosity.NORMAL)
        >>> 
        >>> with elog.goal_context("Search for info"):
        ...     elog.info(LogCategory.PHASE, "Starting search phase")
        ...     with elog.phase_context("Navigate", phase_num=1, total_phases=2):
        ...         elog.browser_action("Navigated to google.com")
        ...     elog.phase_complete(success=True, duration_ms=1500)
        >>> elog.goal_complete(success=True, duration_ms=5000)
    """
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "blue": "\033[34m",
        "cyan": "\033[36m",
        "magenta": "\033[35m",
        "white": "\033[37m",
    }
    
    # Category colors
    CATEGORY_COLORS = {
        LogCategory.GOAL: "bold",
        LogCategory.PHASE: "blue",
        LogCategory.STEP: "cyan",
        LogCategory.SUBSTEP: "dim",
        LogCategory.AGENT: "magenta",
        LogCategory.LLM: "yellow",
        LogCategory.BROWSER: "green",
        LogCategory.OBSTACLE: "yellow",
        LogCategory.REPLAN: "yellow",
        LogCategory.ERROR: "red",
        LogCategory.WARNING: "yellow",
        LogCategory.SUCCESS: "green",
    }
    
    def __init__(
        self,
        verbosity: LogVerbosity = LogVerbosity.NORMAL,
        use_colors: bool = True,
        json_output: bool = False,
    ) -> None:
        """
        Initialize the execution logger.
        
        Args:
            verbosity: Initial verbosity level
            use_colors: Whether to use ANSI colors in output
            json_output: Whether to output JSON instead of human-readable text
        """
        self._verbosity = verbosity
        self._use_colors = use_colors and sys.stdout.isatty()
        self._json_output = json_output
        self._context = ExecutionLogContext()
        self._entries: List[ExecutionLogEntry] = []
        self._underlying_logger = logging.getLogger("flybrowser.execution")
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_verbosity(self, verbosity: LogVerbosity) -> None:
        """Set the verbosity level."""
        self._verbosity = verbosity
    
    def get_verbosity(self) -> LogVerbosity:
        """Get the current verbosity level."""
        return self._verbosity
    
    def enable_colors(self, enabled: bool = True) -> None:
        """Enable or disable color output."""
        self._use_colors = enabled and sys.stdout.isatty()
    
    def enable_json_output(self, enabled: bool = True) -> None:
        """Enable or disable JSON output mode."""
        self._json_output = enabled
    
    # =========================================================================
    # Context Management
    # =========================================================================
    
    @contextmanager
    def goal_context(self, goal: str) -> Generator[None, None, None]:
        """
        Context manager for goal-level execution.
        
        Args:
            goal: Description of the goal being executed
            
        Yields:
            None
        """
        self._context.goal = goal
        self._context.start_time = time.time()
        self._context.depth = 0
        
        self.goal_start(goal)
        try:
            yield
        finally:
            self._context.goal = None
            self._context.start_time = None
    
    @contextmanager
    def phase_context(
        self,
        description: str,
        phase_num: int,
        total_phases: int,
    ) -> Generator[None, None, None]:
        """
        Context manager for phase-level execution.
        
        Args:
            description: Description of the phase
            phase_num: Current phase number (1-indexed)
            total_phases: Total number of phases
            
        Yields:
            None
        """
        self._context.current_phase = description
        self._context.current_phase_num = phase_num
        self._context.total_phases = total_phases
        self._context.phase_start_time = time.time()
        self._context.depth = 1
        
        if self._verbosity >= LogVerbosity.VERBOSE:
            self.phase_start(description, phase_num, total_phases)
        
        try:
            yield
        finally:
            self._context.current_phase = None
            self._context.current_phase_num = None
            self._context.phase_start_time = None
            self._context.depth = 0
    
    @contextmanager
    def step_context(self, description: str) -> Generator[None, None, None]:
        """
        Context manager for step-level execution.
        
        Args:
            description: Description of the step
            
        Yields:
            None
        """
        self._context.current_step = description
        self._context.step_start_time = time.time()
        old_depth = self._context.depth
        self._context.depth = 2
        
        if self._verbosity >= LogVerbosity.VERBOSE:
            self.step_start(description)
        
        try:
            yield
        finally:
            self._context.current_step = None
            self._context.step_start_time = None
            self._context.depth = old_depth
    
    # =========================================================================
    # Goal-Level Logging
    # =========================================================================
    
    def goal_start(self, goal: str) -> None:
        """Log goal start."""
        if self._verbosity >= LogVerbosity.MINIMAL:
            self._log(
                LogCategory.GOAL,
                f"Starting: {goal}",
                min_verbosity=LogVerbosity.MINIMAL,
            )
    
    def goal_complete(
        self,
        success: bool,
        duration_ms: Optional[float] = None,
        summary: Optional[str] = None,
    ) -> None:
        """Log goal completion."""
        if self._verbosity >= LogVerbosity.MINIMAL:
            if duration_ms is None and self._context.start_time:
                duration_ms = (time.time() - self._context.start_time) * 1000
            
            duration_str = self._format_duration(duration_ms)
            status = "Success" if success else "Failed"
            status_color = "green" if success else "red"
            
            msg = f"Completed in {duration_str} - {self._colorize(status, status_color)}"
            if summary:
                msg += f" ({summary})"
            
            self._log(
                LogCategory.GOAL,
                msg,
                min_verbosity=LogVerbosity.MINIMAL,
                success=success,
                duration_ms=duration_ms,
            )
    
    # =========================================================================
    # Phase-Level Logging
    # =========================================================================
    
    def phase_start(
        self,
        description: str,
        phase_num: int,
        total_phases: int,
    ) -> None:
        """Log phase start (VERBOSE mode)."""
        if self._verbosity >= LogVerbosity.VERBOSE:
            self._context.depth = 1
            self._log(
                LogCategory.PHASE,
                f"[{phase_num}/{total_phases}] {description}",
                min_verbosity=LogVerbosity.VERBOSE,
            )
    
    def phase_complete(
        self,
        success: bool,
        duration_ms: Optional[float] = None,
        description: Optional[str] = None,
        phase_num: Optional[int] = None,
        total_phases: Optional[int] = None,
    ) -> None:
        """Log phase completion."""
        if self._verbosity < LogVerbosity.NORMAL:
            return
        
        # Use context values if not provided
        desc = description or self._context.current_phase or "Phase"
        num = phase_num or self._context.current_phase_num or 0
        total = total_phases or self._context.total_phases or 0
        
        if duration_ms is None and self._context.phase_start_time:
            duration_ms = (time.time() - self._context.phase_start_time) * 1000
        
        duration_str = self._format_duration(duration_ms)
        status_icon = self._colorize("[ok]", "green") if success else self._colorize("[fail]", "red")
        
        self._context.depth = 1
        self._log(
            LogCategory.PHASE,
            f"[{num}/{total}] {desc} {status_icon} ({duration_str})",
            min_verbosity=LogVerbosity.NORMAL,
            success=success,
            duration_ms=duration_ms,
        )
    
    # =========================================================================
    # Step-Level Logging
    # =========================================================================
    
    def step_start(self, description: str) -> None:
        """Log step start (VERBOSE mode)."""
        if self._verbosity >= LogVerbosity.VERBOSE:
            self._context.depth = 2
            self._log(
                LogCategory.STEP,
                description,
                min_verbosity=LogVerbosity.VERBOSE,
            )
    
    def step_complete(
        self,
        success: bool,
        duration_ms: Optional[float] = None,
        result: Optional[str] = None,
    ) -> None:
        """Log step completion (VERBOSE mode)."""
        if self._verbosity >= LogVerbosity.VERBOSE:
            if duration_ms is None and self._context.step_start_time:
                duration_ms = (time.time() - self._context.step_start_time) * 1000
            
            duration_str = self._format_duration(duration_ms)
            status = self._colorize("[ok]", "green") if success else self._colorize("[fail]", "red")
            msg = f"{status} ({duration_str})"
            if result:
                msg += f" - {result[:60]}"
            
            self._context.depth = 2
            self._log(
                LogCategory.STEP,
                msg,
                min_verbosity=LogVerbosity.VERBOSE,
                success=success,
                duration_ms=duration_ms,
            )
    
    # =========================================================================
    # Agent-Level Logging
    # =========================================================================
    
    def agent_start(self, agent_name: str, operation: str) -> None:
        """Log agent operation start (DEBUG mode)."""
        if self._verbosity >= LogVerbosity.DEBUG:
            self._log(
                LogCategory.AGENT,
                f"[{agent_name.upper()}] {operation}",
                min_verbosity=LogVerbosity.DEBUG,
            )
    
    def agent_complete(
        self,
        agent_name: str,
        success: bool,
        duration_ms: float,
        tokens: int = 0,
        summary: Optional[str] = None,
    ) -> None:
        """Log agent operation completion (VERBOSE mode for summary, DEBUG for details)."""
        status_icon = self._colorize("[ok]", "green") if success else self._colorize("[fail]", "red")
        duration_str = self._format_duration(duration_ms)
        
        msg_parts = [f"[{agent_name.upper()}] {status_icon} {duration_str}"]
        if tokens and tokens > 0:
            msg_parts.append(f"{tokens:,} tokens")
        if summary:
            msg_parts.append(summary)
        
        self._log(
            LogCategory.AGENT,
            " | ".join(msg_parts),
            min_verbosity=LogVerbosity.VERBOSE,
            success=success,
            duration_ms=duration_ms,
        )
    
    # =========================================================================
    # LLM Logging
    # =========================================================================
    
    def llm_request(self, model: str, operation: str = "generate") -> None:
        """Log LLM request start (DEBUG mode)."""
        if self._verbosity >= LogVerbosity.DEBUG:
            self._log(
                LogCategory.LLM,
                f"[{operation.upper()}] {model} request...",
                min_verbosity=LogVerbosity.DEBUG,
            )
    
    def llm_response(
        self,
        model: str,
        duration_ms: float,
        tokens: Optional[int] = None,
        cached: bool = False,
    ) -> None:
        """Log LLM response (DEBUG mode)."""
        if self._verbosity >= LogVerbosity.DEBUG:
            cache_str = " (cached)" if cached else ""
            token_str = f", {tokens:,} tokens" if tokens and tokens > 0 else ""
            duration_str = self._format_duration(duration_ms)
            
            self._log(
                LogCategory.LLM,
                f"[OK] {model} {duration_str}{token_str}{cache_str}",
                min_verbosity=LogVerbosity.DEBUG,
                duration_ms=duration_ms,
            )
    
    # =========================================================================
    # Browser Action Logging
    # =========================================================================
    
    def browser_action(self, action: str, duration_ms: Optional[float] = None) -> None:
        """Log browser action (VERBOSE mode)."""
        if self._verbosity >= LogVerbosity.VERBOSE:
            msg = action
            if duration_ms is not None:
                msg += f" ({self._format_duration(duration_ms)})"
            
            self._log(
                LogCategory.BROWSER,
                msg,
                min_verbosity=LogVerbosity.VERBOSE,
                duration_ms=duration_ms,
            )
    
    def browser_navigation(self, url: str, duration_ms: Optional[float] = None) -> None:
        """Log browser navigation (VERBOSE mode)."""
        if self._verbosity >= LogVerbosity.VERBOSE:
            duration_str = f" ({self._format_duration(duration_ms)})" if duration_ms else ""
            self._log(
                LogCategory.BROWSER,
                f"Navigated to {url}{duration_str}",
                min_verbosity=LogVerbosity.VERBOSE,
                duration_ms=duration_ms,
            )
    
    # =========================================================================
    # Obstacle Logging
    # =========================================================================
    
    def obstacle_detected(
        self,
        obstacle_type: str,
        confidence: float = 1.0,
        blocking: bool = True,
    ) -> None:
        """Log obstacle detection (NORMAL mode if blocking, VERBOSE otherwise)."""
        min_level = LogVerbosity.NORMAL if blocking else LogVerbosity.VERBOSE
        
        if self._verbosity >= min_level:
            status = "blocking" if blocking else "non-blocking"
            self._log(
                LogCategory.OBSTACLE,
                f"Detected {obstacle_type} ({status}, {confidence:.0%} confidence)",
                min_verbosity=min_level,
            )
    
    def obstacle_handled(self, obstacle_type: str, success: bool) -> None:
        """Log obstacle handling result (NORMAL mode)."""
        if self._verbosity >= LogVerbosity.NORMAL:
            status = self._colorize("dismissed", "green") if success else self._colorize("failed", "red")
            self._log(
                LogCategory.OBSTACLE,
                f"{obstacle_type} {status}",
                min_verbosity=LogVerbosity.NORMAL,
                success=success,
            )
    
    # =========================================================================
    # General Logging Methods
    # =========================================================================
    
    def info(
        self,
        category: LogCategory,
        message: str,
        min_verbosity: LogVerbosity = LogVerbosity.NORMAL,
    ) -> None:
        """Log an info message with category."""
        self._log(category, message, min_verbosity=min_verbosity)
    
    def warning(self, message: str) -> None:
        """Log a warning message (always shown except SILENT)."""
        if self._verbosity >= LogVerbosity.MINIMAL:
            self._log(LogCategory.WARNING, message, min_verbosity=LogVerbosity.MINIMAL)
    
    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error message (always shown except SILENT)."""
        if self._verbosity >= LogVerbosity.MINIMAL:
            msg = message
            if exception and self._verbosity >= LogVerbosity.DEBUG:
                msg += f": {exception}"
            self._log(LogCategory.ERROR, msg, min_verbosity=LogVerbosity.MINIMAL)
    
    def debug(self, message: str) -> None:
        """Log a debug message (DEBUG mode only)."""
        if self._verbosity >= LogVerbosity.DEBUG:
            self._log(LogCategory.STEP, message, min_verbosity=LogVerbosity.DEBUG)
    
    # =========================================================================
    # Progress Summary
    # =========================================================================
    
    def progress_summary(
        self,
        phases_completed: int,
        total_phases: int,
        current_phase: Optional[str] = None,
    ) -> None:
        """Log a progress summary (VERBOSE mode)."""
        if self._verbosity >= LogVerbosity.VERBOSE:
            pct = (phases_completed / total_phases * 100) if total_phases > 0 else 0
            msg = f"Progress: {phases_completed}/{total_phases} phases ({pct:.0f}%)"
            if current_phase:
                msg += f" - Current: {current_phase}"
            
            self._log(
                LogCategory.GOAL,
                msg,
                min_verbosity=LogVerbosity.VERBOSE,
            )
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _log(
        self,
        category: LogCategory,
        message: str,
        min_verbosity: LogVerbosity = LogVerbosity.NORMAL,
        success: Optional[bool] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Internal logging method."""
        if self._verbosity < min_verbosity:
            return
        
        # Create entry for structured output
        entry = ExecutionLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level="INFO",
            category=category.value,
            message=message,
            context={
                "goal": self._context.goal,
                "phase": self._context.current_phase,
                "step": self._context.current_step,
            },
            success=success,
            duration_ms=duration_ms,
        )
        self._entries.append(entry)
        
        # Output
        if self._json_output:
            print(json.dumps(entry.to_dict()))
        else:
            self._print_human_readable(category, message)
    
    def _print_human_readable(self, category: LogCategory, message: str) -> None:
        """Print human-readable log line."""
        indent = self._context.get_indent()
        prefix = self._format_category_prefix(category)
        
        print(f"{indent}{prefix} {message}")
    
    def _format_category_prefix(self, category: LogCategory) -> str:
        """Format category prefix with optional color."""
        prefix = f"[{category.value}]"
        
        if self._use_colors:
            color_name = self.CATEGORY_COLORS.get(category, "white")
            prefix = self._colorize(prefix, color_name)
        
        return prefix
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text if colors are enabled."""
        if not self._use_colors:
            return text
        
        color_code = self.COLORS.get(color, "")
        reset = self.COLORS["reset"]
        
        if color_code:
            return f"{color_code}{text}{reset}"
        return text
    
    def _format_duration(self, duration_ms: Optional[float]) -> str:
        """Format duration for display."""
        if duration_ms is None:
            return "?"
        
        if duration_ms < 1000:
            return f"{duration_ms:.0f}ms"
        elif duration_ms < 60000:
            return f"{duration_ms / 1000:.1f}s"
        else:
            minutes = duration_ms / 60000
            return f"{minutes:.1f}m"
    
    def get_entries(self) -> List[ExecutionLogEntry]:
        """Get all log entries (for structured output)."""
        return self._entries.copy()
    
    def clear_entries(self) -> None:
        """Clear stored log entries."""
        self._entries.clear()


# Global execution logger instance
_execution_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """
    Get the global execution logger instance.
    
    Returns:
        ExecutionLogger instance
    """
    global _execution_logger
    if _execution_logger is None:
        _execution_logger = ExecutionLogger()
    return _execution_logger


def configure_execution_logger(
    verbosity: LogVerbosity = LogVerbosity.NORMAL,
    use_colors: bool = True,
    json_output: bool = False,
) -> ExecutionLogger:
    """
    Configure and return the global execution logger.
    
    Args:
        verbosity: Verbosity level
        use_colors: Whether to use ANSI colors
        json_output: Whether to output JSON
        
    Returns:
        Configured ExecutionLogger instance
    """
    global _execution_logger
    _execution_logger = ExecutionLogger(
        verbosity=verbosity,
        use_colors=use_colors,
        json_output=json_output,
    )
    return _execution_logger
