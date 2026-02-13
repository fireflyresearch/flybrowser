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

"""OpenTelemetry tracing integration via fireflyframework-genai.

This module provides a thin wrapper around the framework's
:class:`~fireflyframework_genai.observability.FireflyTracer` and
:func:`~fireflyframework_genai.observability.configure_exporters` so that the
rest of FlyBrowser can obtain a pre-configured tracer without importing the
framework directly.

Usage::

    from flybrowser.observability.tracing import configure_tracing, get_tracer

    # Optional – call once at startup to set up exporters
    configure_tracing(otlp_endpoint="http://localhost:4317", console=True)

    # Anywhere in the codebase
    tracer = get_tracer()
    with tracer.custom_span("my-operation"):
        ...
"""

from __future__ import annotations

from typing import Optional

from fireflyframework_genai.observability import FireflyTracer
from fireflyframework_genai.observability.exporters import configure_exporters

__all__ = ["configure_tracing", "get_tracer"]

_tracer: Optional[FireflyTracer] = None


def configure_tracing(
    otlp_endpoint: Optional[str] = None,
    console: bool = False,
    service_name: str = "flybrowser",
) -> None:
    """Configure OpenTelemetry exporters and create a :class:`FireflyTracer`.

    This should be called once during application startup.  Subsequent calls
    will replace the cached tracer instance.

    Parameters:
        otlp_endpoint: gRPC OTLP collector endpoint (e.g. ``http://localhost:4317``).
            If *None*, no OTLP exporter is registered.
        console: If *True*, spans are also printed to the console (useful for
            local development).
        service_name: The OpenTelemetry ``service.name`` resource attribute.
    """
    global _tracer
    configure_exporters(otlp_endpoint=otlp_endpoint, console=console)
    _tracer = FireflyTracer(service_name=service_name)


def get_tracer() -> FireflyTracer:
    """Return the module-level :class:`FireflyTracer`, creating one lazily if needed.

    If :func:`configure_tracing` has not been called yet, a default tracer
    with ``service_name="flybrowser"`` is created automatically.

    Returns:
        The cached :class:`FireflyTracer` instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = FireflyTracer(service_name="flybrowser")
    return _tracer
