# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SessionManager usage tracking integration."""

from unittest.mock import patch

import pytest

from flybrowser.service.session_manager import SessionManager


class TestSessionManagerUsageTracking:
    """Tests for UsageTracker integration in SessionManager."""

    def test_session_manager_has_usage_tracker(self):
        """SessionManager exposes a usage_tracker property that is not None."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            mgr = SessionManager()

        assert mgr.usage_tracker is not None

    def test_get_session_stats_includes_usage(self):
        """get_stats() includes total_cost_usd and total_tokens keys."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            mgr = SessionManager()

        stats = mgr.get_stats()

        assert "total_cost_usd" in stats
        assert "total_tokens" in stats

    def test_get_session_usage(self):
        """get_usage_summary() returns zeroed usage for a fresh manager."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            mgr = SessionManager()

        usage = mgr.get_usage_summary()

        assert usage["total_tokens"] == 0
        assert usage["total_cost_usd"] == 0.0
        assert "agent_breakdown" in usage
