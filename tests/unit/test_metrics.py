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

"""Tests for metrics and cost tracking."""

from __future__ import annotations

import pytest

from fireflyframework_genai.observability import FireflyMetrics, UsageTracker


class TestGetMetrics:
    """Tests for the get_metrics function."""

    def setup_method(self):
        """Reset module-level singletons before each test."""
        import flybrowser.observability.metrics as mod

        mod._metrics = None

    def test_get_metrics_returns_framework_instance(self):
        """get_metrics should return a FireflyMetrics instance."""
        from flybrowser.observability.metrics import get_metrics

        metrics = get_metrics()
        assert metrics is not None
        assert isinstance(metrics, FireflyMetrics)

    def test_get_metrics_creates_default(self):
        """get_metrics should lazily create a default instance when none exists."""
        import flybrowser.observability.metrics as mod
        from flybrowser.observability.metrics import get_metrics

        assert mod._metrics is None

        metrics = get_metrics()

        assert mod._metrics is metrics
        assert isinstance(metrics, FireflyMetrics)

    def test_get_metrics_returns_same_instance(self):
        """Repeated calls to get_metrics should return the same cached instance."""
        from flybrowser.observability.metrics import get_metrics

        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2


class TestGetUsageTracker:
    """Tests for the get_usage_tracker function."""

    def setup_method(self):
        """Reset module-level singletons before each test."""
        import flybrowser.observability.metrics as mod

        mod._usage = None

    def test_get_usage_tracker_returns_framework_instance(self):
        """get_usage_tracker should return a UsageTracker instance."""
        from flybrowser.observability.metrics import get_usage_tracker

        tracker = get_usage_tracker()
        assert tracker is not None
        assert isinstance(tracker, UsageTracker)

    def test_get_usage_tracker_creates_default(self):
        """get_usage_tracker should lazily create a default instance."""
        import flybrowser.observability.metrics as mod
        from flybrowser.observability.metrics import get_usage_tracker

        assert mod._usage is None

        tracker = get_usage_tracker()

        assert mod._usage is tracker
        assert isinstance(tracker, UsageTracker)

    def test_get_usage_tracker_returns_same_instance(self):
        """Repeated calls to get_usage_tracker should return the same instance."""
        from flybrowser.observability.metrics import get_usage_tracker

        t1 = get_usage_tracker()
        t2 = get_usage_tracker()
        assert t1 is t2


class TestRecordOperation:
    """Tests for the record_operation convenience function."""

    def setup_method(self):
        """Reset module-level singletons before each test."""
        import flybrowser.observability.metrics as mod

        mod._metrics = None

    def test_record_operation(self):
        """record_operation should not raise for valid arguments."""
        from flybrowser.observability.metrics import record_operation

        record_operation("navigate", latency_ms=250, tokens=100, cost_usd=0.01)

    def test_record_operation_defaults(self):
        """record_operation should accept only the operation name."""
        from flybrowser.observability.metrics import record_operation

        record_operation("click")

    def test_record_operation_partial_args(self):
        """record_operation should accept a subset of keyword arguments."""
        from flybrowser.observability.metrics import record_operation

        record_operation("extract", tokens=500)


class TestGetCostSummary:
    """Tests for the get_cost_summary convenience function."""

    def setup_method(self):
        """Reset module-level singletons before each test."""
        import flybrowser.observability.metrics as mod

        mod._usage = None

    def test_get_cost_summary(self):
        """get_cost_summary should return a dict with expected keys."""
        from flybrowser.observability.metrics import get_cost_summary

        summary = get_cost_summary()
        assert "total_cost_usd" in summary
        assert "total_tokens" in summary

    def test_get_cost_summary_initial_values(self):
        """get_cost_summary should report zero values for a fresh tracker."""
        from flybrowser.observability.metrics import get_cost_summary

        summary = get_cost_summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_tokens"] == 0
