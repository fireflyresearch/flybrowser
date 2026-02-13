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

"""Metrics and cost tracking via fireflyframework-genai.

This module provides a thin wrapper around the framework's
:class:`~fireflyframework_genai.observability.FireflyMetrics` and
:class:`~fireflyframework_genai.observability.UsageTracker` so that the
rest of FlyBrowser can record latency, token usage, and cost without
importing the framework directly.

Usage::

    from flybrowser.observability.metrics import (
        get_metrics,
        get_usage_tracker,
        record_operation,
        get_cost_summary,
    )

    # Record an operation
    record_operation("navigate", latency_ms=250, tokens=100, cost_usd=0.01)

    # Get cost summary
    summary = get_cost_summary()
    print(summary["total_cost_usd"])
"""

from __future__ import annotations

from typing import Optional

from fireflyframework_genai.observability import FireflyMetrics, UsageTracker

__all__ = ["get_metrics", "get_usage_tracker", "record_operation", "get_cost_summary"]

_metrics: Optional[FireflyMetrics] = None
_usage: Optional[UsageTracker] = None


def get_metrics() -> FireflyMetrics:
    """Return the module-level :class:`FireflyMetrics`, creating one lazily if needed.

    If no metrics instance has been created yet, a default one with
    ``service_name="flybrowser"`` is created automatically.

    Returns:
        The cached :class:`FireflyMetrics` instance.
    """
    global _metrics
    if _metrics is None:
        _metrics = FireflyMetrics(service_name="flybrowser")
    return _metrics


def get_usage_tracker() -> UsageTracker:
    """Return the module-level :class:`UsageTracker`, creating one lazily if needed.

    Returns:
        The cached :class:`UsageTracker` instance.
    """
    global _usage
    if _usage is None:
        _usage = UsageTracker()
    return _usage


def record_operation(
    operation: str,
    latency_ms: float = 0,
    tokens: int = 0,
    cost_usd: float = 0,
) -> None:
    """Record metrics for a single operation.

    This is a convenience function that delegates to the underlying
    :class:`FireflyMetrics` methods.

    Parameters:
        operation: Name of the operation (e.g. ``"navigate"``, ``"click"``).
        latency_ms: Wall-clock time for the operation in milliseconds.
        tokens: Total token count consumed by the operation.
        cost_usd: Estimated cost in USD for the operation.
    """
    m = get_metrics()
    if latency_ms:
        m.record_latency(latency_ms, operation=operation)
    if tokens:
        m.record_tokens(tokens)
    if cost_usd:
        m.record_cost(cost_usd)


def get_cost_summary() -> dict:
    """Return an aggregated cost and usage summary.

    Returns a plain dictionary with at least the following keys:

    - ``total_cost_usd`` -- cumulative estimated cost across all records.
    - ``total_tokens`` -- cumulative token usage across all records.

    Returns:
        A dictionary summarising cost and token usage.
    """
    tracker = get_usage_tracker()
    summary = tracker.get_summary()
    return {
        "total_cost_usd": summary.total_cost_usd,
        "total_tokens": summary.total_tokens,
    }
