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
Timing utilities for tracking execution performance.

This module provides utilities for tracking execution times of operations
in FlyBrowser, including step-by-step timing and overall execution timing.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimingInfo:
    """
    Container for timing information.
    
    Attributes:
        total_ms: Total execution time in milliseconds
        breakdown: Dictionary of component timings
        step_timings: List of individual step timings
    """
    total_ms: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    step_timings: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "total_ms": round(self.total_ms, 2),
            "breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "step_timings": [
                {**st, "duration_ms": round(st.get("duration_ms", 0), 2)}
                for st in self.step_timings
            ],
        }


class StepTimer:
    """
    Timer for tracking individual steps in a workflow.
    
    Example:
        >>> timer = StepTimer()
        >>> timer.start_step("fetch_data")
        >>> # ... do work ...
        >>> timer.end_step("fetch_data")
        >>> print(timer.get_timings())
    """
    
    def __init__(self) -> None:
        """Initialize the step timer."""
        self._start_times: Dict[str, float] = {}
        self._step_timings: List[Dict[str, Any]] = []
        self._overall_start: Optional[float] = None
        
    def start(self) -> None:
        """Start overall timing."""
        self._overall_start = time.perf_counter()
        
    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start timing a step.
        
        Args:
            step_name: Name/identifier for the step
            metadata: Optional metadata to include with the step timing
        """
        self._start_times[step_name] = time.perf_counter()
        
    def end_step(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> float:
        """
        End timing a step and record the duration.
        
        Args:
            step_name: Name/identifier for the step
            metadata: Optional metadata to include with the step timing
            success: Whether the step succeeded
            
        Returns:
            Duration in milliseconds
        """
        if step_name not in self._start_times:
            return 0.0
            
        duration = (time.perf_counter() - self._start_times[step_name]) * 1000
        
        timing_entry = {
            "step": step_name,
            "duration_ms": duration,
            "success": success,
        }
        
        if metadata:
            timing_entry.update(metadata)
            
        self._step_timings.append(timing_entry)
        del self._start_times[step_name]
        
        return duration
        
    def get_total_ms(self) -> float:
        """Get total elapsed time since start() was called."""
        if self._overall_start is None:
            return 0.0
        return (time.perf_counter() - self._overall_start) * 1000
        
    def get_timings(self) -> TimingInfo:
        """
        Get all timing information.
        
        Returns:
            TimingInfo object with all timing data
        """
        total_ms = self.get_total_ms()
        
        # Calculate breakdown by category
        breakdown: Dict[str, float] = {}
        for step in self._step_timings:
            step_name = step["step"]
            duration = step["duration_ms"]
            
            # Categorize steps
            if "llm" in step_name.lower() or "generate" in step_name.lower():
                breakdown["llm_time_ms"] = breakdown.get("llm_time_ms", 0) + duration
            elif "browser" in step_name.lower() or "page" in step_name.lower() or "element" in step_name.lower():
                breakdown["browser_time_ms"] = breakdown.get("browser_time_ms", 0) + duration
            else:
                breakdown["other_ms"] = breakdown.get("other_ms", 0) + duration
                
        return TimingInfo(
            total_ms=total_ms,
            breakdown=breakdown,
            step_timings=self._step_timings.copy()
        )


@contextmanager
def time_operation(timer: StepTimer, step_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing an operation.
    
    Args:
        timer: StepTimer instance
        step_name: Name of the step
        metadata: Optional metadata
        
    Example:
        >>> timer = StepTimer()
        >>> timer.start()
        >>> with time_operation(timer, "fetch_data"):
        ...     # ... do work ...
        >>> print(timer.get_timings())
    """
    timer.start_step(step_name, metadata)
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        timer.end_step(step_name, metadata, success)


@asynccontextmanager
async def time_async_operation(
    timer: StepTimer,
    step_name: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Async context manager for timing an operation.
    
    Args:
        timer: StepTimer instance
        step_name: Name of the step
        metadata: Optional metadata
        
    Example:
        >>> timer = StepTimer()
        >>> timer.start()
        >>> async with time_async_operation(timer, "fetch_data"):
        ...     # ... do async work ...
        >>> print(timer.get_timings())
    """
    timer.start_step(step_name, metadata)
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        timer.end_step(step_name, metadata, success)
