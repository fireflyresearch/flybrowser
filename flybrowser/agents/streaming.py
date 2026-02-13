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

"""Agent streaming adapter — SSE format for real-time agent reasoning."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AgentStreamEvent:
    """A single event emitted during agent execution.

    Attributes:
        type: Event type — "thought", "action", "observation", "complete", or "error".
        content: Human-readable description of what happened.
        step: The reasoning step number (0-indexed).
        timestamp: Unix timestamp of the event.
        tool: Tool name if the event is an action.
        args: Tool arguments if the event is an action.
        success: Whether the overall task succeeded (set on "complete"/"error").
        usage: Token usage statistics if available.
    """

    type: str  # "thought", "action", "observation", "complete", "error"
    content: str
    step: int = 0
    timestamp: float = 0.0
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        Only includes optional fields when they are not None to keep
        payloads compact.
        """
        d: Dict[str, Any] = {
            "type": self.type,
            "content": self.content,
            "step": self.step,
            "timestamp": self.timestamp,
        }
        if self.tool is not None:
            d["tool"] = self.tool
        if self.args is not None:
            d["args"] = self.args
        if self.success is not None:
            d["success"] = self.success
        if self.usage is not None:
            d["usage"] = self.usage
        return d


def format_sse_event(event: AgentStreamEvent) -> str:
    """Format an AgentStreamEvent as a Server-Sent Events (SSE) data line.

    Returns a string in the form ``data: {json}\\n\\n`` ready to be written
    directly to an HTTP response stream.
    """
    return f"data: {json.dumps(event.to_dict())}\n\n"
