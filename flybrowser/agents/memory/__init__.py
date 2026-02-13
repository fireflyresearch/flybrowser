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

"""Browser-specific memory extensions for fireflyframework-genai."""

# Re-export everything from the legacy memory module so that existing
# ``from flybrowser.agents.memory import AgentMemory`` style imports
# continue to work after the flat module was converted to a package.
from flybrowser.agents.memory._legacy_memory import (  # noqa: F401
    AgentMemory,
    ContextStore,
    LearnedPattern,
    LongTermMemory,
    MemoryEntry,
    ShortTermMemory,
    StateSnapshot,
    WorkingMemory,
)

# New browser-specific memory manager
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager, PageSnapshot  # noqa: F401

__all__ = [
    # Legacy
    "AgentMemory",
    "ContextStore",
    "LearnedPattern",
    "LongTermMemory",
    "MemoryEntry",
    "ShortTermMemory",
    "StateSnapshot",
    "WorkingMemory",
    # New
    "BrowserMemoryManager",
    "PageSnapshot",
]
