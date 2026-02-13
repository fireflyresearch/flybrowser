# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for fireflyframework-genai REST exposure base.

These tests verify that the framework-provided endpoints (/health, /liveness,
/readiness, /agents) are available on the FlyBrowser app after integrating
create_genai_app() as the application factory.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    manager = MagicMock()
    manager.get_active_session_count.return_value = 0
    manager.get_stats.return_value = {
        "active_sessions": 0,
        "total_requests": 0,
        "max_sessions": 100,
    }
    manager.cleanup_all = MagicMock()
    return manager


@pytest.fixture
def test_client(mock_session_manager):
    """Create test client with mocked dependencies."""
    from flybrowser.service.app import app

    with patch("flybrowser.service.app.session_manager", mock_session_manager):
        with patch("flybrowser.service.app.start_time", 1000.0):
            client = TestClient(app)
            yield client


class TestFrameworkHealthEndpoints:
    """Tests for framework-provided health endpoints."""

    def test_health_endpoint_exists(self, test_client):
        """GET /health returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_liveness_endpoint_exists(self, test_client):
        """GET /health/live returns 200 (framework liveness check)."""
        response = test_client.get("/health/live")
        assert response.status_code == 200

    def test_readiness_endpoint_exists(self, test_client):
        """GET /health/ready returns 200 (framework readiness check)."""
        response = test_client.get("/health/ready")
        assert response.status_code == 200


class TestFrameworkAgentsEndpoint:
    """Tests for framework-provided agents endpoint."""

    def test_agents_list_endpoint(self, test_client):
        """GET /agents returns 200."""
        response = test_client.get("/agents")
        assert response.status_code == 200
