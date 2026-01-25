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
Prompt templates for LLM interactions.

This module provides backward-compatible prompt constants while also
integrating with the centralized prompt management system in flybrowser.prompts.

For new code, prefer using the PromptManager:
    >>> from flybrowser.prompts import PromptManager
    >>> manager = PromptManager()
    >>> prompts = manager.get_prompt("action_planning", instruction="...", url="...")

The constants below are maintained for backward compatibility.
"""

from typing import Any, Dict, Optional

# Lazy-loaded prompt manager for centralized prompts
_prompt_manager = None


def get_prompt_manager():
    """Get or create the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        from flybrowser.prompts import PromptManager
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(
    name: str,
    version: Optional[str] = None,
    **variables: Any,
) -> Dict[str, str]:
    """
    Get a rendered prompt from the centralized prompt system.

    Args:
        name: Template name (e.g., "action_planning", "data_extraction")
        version: Optional template version
        **variables: Variables to render the template

    Returns:
        Dictionary with 'system' and 'user' prompts

    Example:
        >>> prompts = get_prompt(
        ...     "action_planning",
        ...     instruction="Click the login button",
        ...     url="https://example.com",
        ...     title="Example",
        ...     visible_elements="[...]"
        ... )
        >>> print(prompts["system"])
        >>> print(prompts["user"])
    """
    manager = get_prompt_manager()
    return manager.get_prompt(name, version=version, **variables)