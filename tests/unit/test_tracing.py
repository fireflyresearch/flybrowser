# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenTelemetry tracing integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fireflyframework_genai.observability import FireflyTracer


class TestGetTracer:
    """Tests for the get_tracer function."""

    def setup_method(self):
        """Reset module-level _tracer before each test."""
        import flybrowser.observability.tracing as mod

        mod._tracer = None

    def test_get_tracer_returns_framework_tracer(self):
        """get_tracer should return a FireflyTracer instance (not None)."""
        from flybrowser.observability.tracing import get_tracer

        tracer = get_tracer()
        assert tracer is not None
        assert isinstance(tracer, FireflyTracer)

    def test_get_tracer_creates_default(self):
        """get_tracer should lazily create a default tracer when none is configured."""
        import flybrowser.observability.tracing as mod
        from flybrowser.observability.tracing import get_tracer

        # Ensure no tracer is set
        assert mod._tracer is None

        tracer = get_tracer()

        # Should now be cached at module level
        assert mod._tracer is tracer
        assert isinstance(tracer, FireflyTracer)

    def test_get_tracer_returns_same_instance(self):
        """Repeated calls to get_tracer should return the same cached instance."""
        from flybrowser.observability.tracing import get_tracer

        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2


class TestConfigureTracing:
    """Tests for the configure_tracing function."""

    def setup_method(self):
        """Reset module-level _tracer before each test."""
        import flybrowser.observability.tracing as mod

        mod._tracer = None

    @patch("flybrowser.observability.tracing.configure_exporters")
    def test_configure_tracing_sets_tracer(self, mock_configure):
        """configure_tracing should set the module-level _tracer."""
        import flybrowser.observability.tracing as mod
        from flybrowser.observability.tracing import configure_tracing

        assert mod._tracer is None

        configure_tracing()

        assert mod._tracer is not None
        assert isinstance(mod._tracer, FireflyTracer)
        mock_configure.assert_called_once_with(otlp_endpoint=None, console=False)

    @patch("flybrowser.observability.tracing.configure_exporters")
    def test_configure_tracing_with_service_name(self, mock_configure):
        """configure_tracing should pass the custom service_name to FireflyTracer."""
        import flybrowser.observability.tracing as mod
        from flybrowser.observability.tracing import configure_tracing

        configure_tracing(service_name="my-custom-service")

        tracer = mod._tracer
        assert tracer is not None
        assert isinstance(tracer, FireflyTracer)
        # Verify the underlying OTel proxy tracer was created with the custom name
        assert tracer._tracer._instrumenting_module_name == "my-custom-service"

    @patch("flybrowser.observability.tracing.configure_exporters")
    def test_configure_tracing_passes_otlp_endpoint(self, mock_configure):
        """configure_tracing should forward otlp_endpoint to configure_exporters."""
        from flybrowser.observability.tracing import configure_tracing

        configure_tracing(otlp_endpoint="http://localhost:4317")

        mock_configure.assert_called_once_with(
            otlp_endpoint="http://localhost:4317", console=False
        )

    @patch("flybrowser.observability.tracing.configure_exporters")
    def test_configure_tracing_passes_console_flag(self, mock_configure):
        """configure_tracing should forward console flag to configure_exporters."""
        from flybrowser.observability.tracing import configure_tracing

        configure_tracing(console=True)

        mock_configure.assert_called_once_with(otlp_endpoint=None, console=True)

    @patch("flybrowser.observability.tracing.configure_exporters")
    def test_configure_tracing_overrides_previous_tracer(self, mock_configure):
        """Calling configure_tracing again should replace the existing tracer."""
        import flybrowser.observability.tracing as mod
        from flybrowser.observability.tracing import configure_tracing

        configure_tracing(service_name="first")
        first_tracer = mod._tracer

        configure_tracing(service_name="second")
        second_tracer = mod._tracer

        assert first_tracer is not second_tracer

    def test_get_tracer_after_configure(self):
        """get_tracer should return the tracer that was set by configure_tracing."""
        import flybrowser.observability.tracing as mod
        from flybrowser.observability.tracing import configure_tracing, get_tracer

        with patch("flybrowser.observability.tracing.configure_exporters"):
            configure_tracing(service_name="configured-service")

        tracer = get_tracer()
        assert tracer is mod._tracer
        assert isinstance(tracer, FireflyTracer)
