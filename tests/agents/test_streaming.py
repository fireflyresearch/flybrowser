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

"""Tests for AgentStreamEvent and SSE formatting."""

import json
import pytest
from flybrowser.agents.streaming import AgentStreamEvent, format_sse_event


class TestAgentStreamEvent:
    def test_thought_event(self):
        event = AgentStreamEvent(
            type="thought", content="Thinking...", step=1, timestamp=123.0
        )
        d = event.to_dict()
        assert d["type"] == "thought"
        assert d["content"] == "Thinking..."
        assert d["step"] == 1
        assert d["timestamp"] == 123.0
        # Optional fields should be absent
        assert "tool" not in d
        assert "args" not in d
        assert "success" not in d
        assert "usage" not in d

    def test_action_event(self):
        event = AgentStreamEvent(
            type="action",
            content="click",
            step=1,
            tool="click",
            args={"selector": "#btn"},
            timestamp=0.0,
        )
        d = event.to_dict()
        assert d["tool"] == "click"
        assert d["args"] == {"selector": "#btn"}

    def test_complete_event(self):
        event = AgentStreamEvent(
            type="complete", content="Done", step=3, success=True, timestamp=0.0
        )
        d = event.to_dict()
        assert d["success"] is True
        assert d["type"] == "complete"

    def test_error_event(self):
        event = AgentStreamEvent(
            type="error", content="Timeout", step=5, success=False, timestamp=0.0
        )
        d = event.to_dict()
        assert d["success"] is False
        assert d["type"] == "error"

    def test_observation_event_with_usage(self):
        event = AgentStreamEvent(
            type="observation",
            content="Found 3 buttons",
            step=2,
            timestamp=0.0,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        d = event.to_dict()
        assert d["usage"]["prompt_tokens"] == 100

    def test_defaults(self):
        event = AgentStreamEvent(type="thought", content="Hi")
        assert event.step == 0
        assert event.timestamp == 0.0
        assert event.tool is None
        assert event.args is None
        assert event.success is None
        assert event.usage is None


class TestFormatSSE:
    def test_format_sse(self):
        event = AgentStreamEvent(
            type="thought", content="Hi", step=1, timestamp=0.0
        )
        sse = format_sse_event(event)
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[len("data: ") : -2])
        assert payload["type"] == "thought"
        assert payload["content"] == "Hi"

    def test_format_sse_action(self):
        event = AgentStreamEvent(
            type="action",
            content="Clicking button",
            step=2,
            tool="click",
            args={"selector": "#submit"},
            timestamp=1234.5,
        )
        sse = format_sse_event(event)
        payload = json.loads(sse[len("data: ") : -2])
        assert payload["tool"] == "click"
        assert payload["args"]["selector"] == "#submit"
        assert payload["timestamp"] == 1234.5

    def test_format_sse_complete(self):
        event = AgentStreamEvent(
            type="complete",
            content="Task finished",
            step=10,
            success=True,
            timestamp=9999.0,
            usage={"total_tokens": 500},
        )
        sse = format_sse_event(event)
        payload = json.loads(sse[len("data: ") : -2])
        assert payload["success"] is True
        assert payload["usage"]["total_tokens"] == 500
