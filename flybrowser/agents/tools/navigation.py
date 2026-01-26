# Copyright 2026 Firefly Software Solutions Inc.
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
Navigation tools for browser control.

This module provides ReAct-compatible tools for browser navigation
including URL navigation, back/forward, and page refresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from flybrowser.agents.types import SafetyLevel, ToolCategory, ToolResult
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.tools.descriptions import get_tool_metadata
from flybrowser.agents.obstacle_detector import ObstacleDetector

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


class NavigateTool(BaseTool):
    """Navigate to a URL."""
    
    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        # Get base metadata from centralized descriptions
        base_metadata = get_tool_metadata("navigate")
        # Add context support
        base_metadata.expected_context_types = ["conditions", "constraints"]
        return base_metadata
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute navigation to URL.
        
        Supports context with:
        - conditions: Navigation conditions (e.g., requires_login, max_redirects)
        - constraints: Navigation constraints (e.g., timeout_seconds)
        """
        url = kwargs.get("url")
        wait_until = kwargs.get("wait_until", "domcontentloaded")
        
        if not url:
            return ToolResult.error_result("URL is required")
        
        # Get user context if provided
        user_context = self.get_user_context()
        conditions = {}
        constraints = {}
        
        if user_context:
            if isinstance(user_context, dict):
                conditions = user_context.get("conditions", {})
                constraints = user_context.get("constraints", {})
            else:
                # Handle ActionContext object
                from flybrowser.agents.context import ActionContext
                if isinstance(user_context, ActionContext):
                    conditions = user_context.conditions
                    constraints = user_context.constraints
        
        # Apply constraints
        timeout_seconds = constraints.get("timeout_seconds")
        if timeout_seconds:
            # Store original timeout and apply constraint
            original_timeout = self._page_controller.page.context.browser.new_context.default_timeout
            self._page_controller.page.set_default_timeout(timeout_seconds * 1000)
        
        # Check conditions before navigation
        requires_login = conditions.get("requires_login")
        if requires_login is False:
            # User explicitly indicated no login required - proceed normally
            pass
        elif requires_login is True:
            # User expects login - might want to check auth state first
            # For now, just log it
            import logging
            logging.getLogger(__name__).debug(f"Navigation to {url} requires login (as per context)")
        
        max_redirects = conditions.get("max_redirects")
        if max_redirects is not None:
            # Note: Playwright doesn't directly support max_redirects limit
            # We log it for awareness but can't enforce it directly
            import logging
            logging.getLogger(__name__).debug(f"Max redirects set to {max_redirects} (advisory only)")
        
        try:
            # Navigate to URL
            await self._page_controller.goto(url, wait_until=wait_until)
            
            # Automatically detect and handle obstacles
            llm = getattr(self, 'llm_provider', None)
            if llm:
                # Get config from agent if available
                agent_config = getattr(self, 'agent_config', None)
                obstacle_config = agent_config.obstacle_detector if agent_config else None
                
                detector = ObstacleDetector(
                    page=self._page_controller.page,
                    llm=llm,
                    config=obstacle_config
                )
                obstacle_result = await detector.detect_and_handle()
                obstacles_detected = len(obstacle_result.obstacles_found)
                obstacles_handled = obstacle_result.obstacles_dismissed
            else:
                obstacle_result = None
                obstacles_detected = 0
                obstacles_handled = 0
            
            # Get page state
            state = await self._page_controller.get_page_state()
            
            message = f"Successfully navigated to {url}"
            if obstacles_detected > 0:
                if obstacles_handled == obstacles_detected:
                    message += f" All {obstacles_handled} obstacle(s) dismissed successfully."
                elif obstacles_handled > 0:
                    message += f" Dismissed {obstacles_handled}/{obstacles_detected} obstacles. {obstacles_detected - obstacles_handled} may remain."
                else:
                    # Unhandled obstacles - provide guidance
                    message += f" {obstacles_detected} obstacle(s) remain (likely cookie banners or modals). NEXT STEPS: 1) Try 'get_page_state' to see if page is still functional, 2) If needed, use 'click' on visible accept/close buttons, 3) Continue with your task if the obstacle does not block functionality."
            
            # Include detailed obstacle handling results
            result_data = {
                "navigated_to": url,
                "current_url": state.get("url", url),
                "title": state.get("title", ""),
                "message": message,
                "obstacles_detected": obstacles_detected,
                "obstacles_handled": obstacles_handled,
                "has_unhandled_obstacles": obstacles_detected > obstacles_handled,
            }
            
            # If obstacles couldn't be handled, add actionable guidance
            if obstacles_detected > 0 and obstacles_handled == 0:
                result_data["next_action_suggestion"] = "Use 'get_page_state' to inspect the page and find clickable buttons to dismiss obstacles, OR try to continue with your task if the obstacle doesn't block all functionality."
            
            return ToolResult.success_result(data=result_data)
        except Exception as e:
            return ToolResult.error_result(f"Navigation failed: {str(e)}")


class GoBackTool(BaseTool):
    """Navigate back in browser history."""
    
    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        return get_tool_metadata("go_back")
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute back navigation."""
        try:
            await self._page_controller.page.go_back()
            state = await self._page_controller.get_page_state()
            return ToolResult.success_result(
                data={
                    "current_url": state.get("url", ""),
                    "title": state.get("title", ""),
                    "message": "Successfully navigated back",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Go back failed: {str(e)}")


class GoForwardTool(BaseTool):
    """Navigate forward in browser history."""
    
    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        return get_tool_metadata("go_forward")
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute forward navigation."""
        try:
            await self._page_controller.page.go_forward()
            state = await self._page_controller.get_page_state()
            return ToolResult.success_result(
                data={
                    "current_url": state.get("url", ""),
                    "title": state.get("title", ""),
                    "message": "Successfully navigated forward",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Go forward failed: {str(e)}")


class RefreshTool(BaseTool):
    """Refresh the current page."""
    
    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        return get_tool_metadata("refresh")
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute page refresh."""
        try:
            await self._page_controller.page.reload()
            state = await self._page_controller.get_page_state()
            return ToolResult.success_result(
                data={
                    "current_url": state.get("url", ""),
                    "title": state.get("title", ""),
                    "message": "Successfully refreshed the page",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Refresh failed: {str(e)}")

