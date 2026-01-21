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
Navigation agent for intelligent web navigation.

This module provides the NavigationAgent class which uses LLMs to understand
natural language navigation commands and intelligently navigate web pages.

The agent supports:
- URL navigation with smart waiting
- Natural language navigation commands
- Link following based on descriptions
- Breadcrumb and menu navigation
- Back/forward navigation
- Page load waiting strategies

Example:
    >>> agent = NavigationAgent(page_controller, element_detector, llm_provider)
    >>> await agent.execute("Go to the products page")
    >>> await agent.goto("https://example.com")
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from flybrowser.agents.base_agent import BaseAgent
from flybrowser.exceptions import NavigationError, ElementNotFoundError
from flybrowser.llm.prompts import NAVIGATION_PROMPT, NAVIGATION_SYSTEM
from flybrowser.utils.logger import logger


class WaitStrategy(str, Enum):
    """Strategies for waiting after navigation."""

    LOAD = "load"
    DOM_CONTENT_LOADED = "domcontentloaded"
    NETWORK_IDLE = "networkidle"
    COMMIT = "commit"
    CUSTOM = "custom"


class NavigationType(str, Enum):
    """Types of navigation actions."""

    URL = "url"
    LINK = "link"
    BACK = "back"
    FORWARD = "forward"
    REFRESH = "refresh"
    SEARCH = "search"


@dataclass
class NavigationResult:
    """
    Result of a navigation operation.

    Attributes:
        success: Whether navigation succeeded
        url: Final URL after navigation
        title: Page title after navigation
        navigation_type: Type of navigation performed
        error: Error message if failed
        wait_time: Time spent waiting for page load
    """

    success: bool
    url: str = ""
    title: str = ""
    navigation_type: NavigationType = NavigationType.URL
    error: Optional[str] = None
    wait_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class NavigationAgent(BaseAgent):
    """
    Agent specialized in intelligent web navigation.

    This agent uses LLMs to understand natural language navigation commands
    and navigate web pages intelligently. It supports various navigation
    strategies and smart waiting for page loads.

    The agent inherits from BaseAgent and has access to:
    - page_controller: For page operations
    - element_detector: For element location
    - llm: For intelligent navigation planning

    Attributes:
        default_timeout: Default navigation timeout in milliseconds
        default_wait_strategy: Default wait strategy for navigation

    Example:
        >>> agent = NavigationAgent(page_controller, element_detector, llm)
        >>>
        >>> # Direct URL navigation
        >>> await agent.goto("https://example.com")
        >>>
        >>> # Natural language navigation
        >>> await agent.execute("Go to the contact page")
        >>>
        >>> # Link following
        >>> await agent.follow_link("Learn more about our products")
    """

    def __init__(
        self,
        page_controller,
        element_detector,
        llm_provider,
        default_timeout: int = 30000,
        default_wait_strategy: WaitStrategy = WaitStrategy.DOM_CONTENT_LOADED,
        pii_handler=None,
    ) -> None:
        """
        Initialize the navigation agent.

        Args:
            page_controller: PageController instance for page operations
            element_detector: ElementDetector instance for element location
            llm_provider: BaseLLMProvider instance for LLM operations
            default_timeout: Default timeout in milliseconds (default: 30000)
            default_wait_strategy: Default wait strategy (default: DOM_CONTENT_LOADED)
            pii_handler: Optional PIIHandler for secure handling of sensitive data
        """
        super().__init__(page_controller, element_detector, llm_provider, pii_handler=pii_handler)
        self.default_timeout = default_timeout
        self.default_wait_strategy = default_wait_strategy

    async def execute(
        self,
        command: str,
        use_vision: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a natural language navigation command.

        This method uses an LLM to understand the navigation intent and
        execute the appropriate navigation action.

        Args:
            command: Natural language navigation command.
                Examples:
                - "Go to the products page"
                - "Navigate to the contact section"
                - "Click on the About Us link"
                - "Go back to the previous page"
                - "Search for 'python tutorials'"
            use_vision: Whether to use vision for link detection (default: True)
            timeout: Navigation timeout in milliseconds (default: uses default_timeout)

        Returns:
            Dictionary containing:
            - success: Whether navigation succeeded
            - url: Final URL after navigation
            - title: Page title after navigation
            - navigation_type: Type of navigation performed
            - error: Error message if failed

        Raises:
            NavigationError: If navigation fails

        Example:
            >>> result = await agent.execute("Go to the products page")
            >>> print(result["url"])
            'https://example.com/products'
        """
        try:
            # Mask command for logging to avoid exposing PII
            logger.info(f"Executing navigation: {self.mask_for_log(command)}")
            timeout = timeout or self.default_timeout

            # Analyze the command to determine navigation type
            nav_plan = await self._plan_navigation(command, use_vision)

            # Execute based on navigation type
            nav_type = NavigationType(nav_plan.get("type", "link"))

            if nav_type == NavigationType.URL:
                result = await self.goto(
                    nav_plan.get("url", ""),
                    timeout=timeout,
                )
            elif nav_type == NavigationType.BACK:
                result = await self.go_back(timeout=timeout)
            elif nav_type == NavigationType.FORWARD:
                result = await self.go_forward(timeout=timeout)
            elif nav_type == NavigationType.REFRESH:
                result = await self.refresh(timeout=timeout)
            elif nav_type == NavigationType.SEARCH:
                result = await self._perform_search(
                    nav_plan.get("query", ""),
                    use_vision=use_vision,
                    timeout=timeout,
                )
            else:  # LINK navigation
                result = await self.follow_link(
                    nav_plan.get("link_description", command),
                    use_vision=use_vision,
                    timeout=timeout,
                )

            return {
                "success": result.success,
                "url": result.url,
                "title": result.title,
                "navigation_type": result.navigation_type.value,
                "error": result.error,
                "details": result.details,
            }

        except Exception as e:
            logger.error(f"Navigation failed: {self.mask_for_log(str(e))}")
            raise NavigationError(f"Failed to execute navigation '{self.mask_for_log(command)}': {e}") from e

    async def _plan_navigation(
        self, command: str, use_vision: bool = True
    ) -> Dict[str, Any]:
        """
        Plan navigation based on natural language command.

        Args:
            command: Natural language navigation command
            use_vision: Whether to use vision for context

        Returns:
            Dictionary with navigation plan
        """
        context = await self.get_page_context()

        # Get available links for context
        links = await self._get_page_links()

        # Mask command for LLM to avoid exposing PII
        safe_command = self.mask_for_llm(command)

        prompt = NAVIGATION_PROMPT.format(
            goal=safe_command,
            url=context["url"],
            links=json.dumps(links[:30], indent=2),  # Limit links
        )

        schema = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [t.value for t in NavigationType],
                },
                "url": {"type": "string"},
                "link_description": {"type": "string"},
                "query": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["type"],
        }

        if use_vision:
            screenshot = await self.page.screenshot()
            response = await self.llm.generate_with_vision(
                prompt=prompt,
                image_data=screenshot,
                system_prompt=NAVIGATION_SYSTEM,
                temperature=0.3,
            )
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return await self.llm.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=NAVIGATION_SYSTEM,
                    temperature=0.3,
                )
        else:
            return await self.llm.generate_structured(
                prompt=prompt,
                schema=schema,
                system_prompt=NAVIGATION_SYSTEM,
                temperature=0.3,
            )


    async def goto(
        self,
        url: str,
        wait_strategy: Optional[WaitStrategy] = None,
        timeout: Optional[int] = None,
    ) -> NavigationResult:
        """
        Navigate to a URL with smart waiting.

        Args:
            url: URL to navigate to
            wait_strategy: Wait strategy to use (default: uses default_wait_strategy)
            timeout: Navigation timeout in milliseconds

        Returns:
            NavigationResult with navigation details

        Example:
            >>> result = await agent.goto("https://example.com")
            >>> result = await agent.goto(
            ...     "https://example.com",
            ...     wait_strategy=WaitStrategy.NETWORK_IDLE
            ... )
        """
        import time

        start_time = time.time()
        wait_strategy = wait_strategy or self.default_wait_strategy
        timeout = timeout or self.default_timeout

        try:
            logger.info(f"Navigating to URL: {url}")

            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            await self.page.goto(
                url,
                wait_until=wait_strategy.value,
                timeout=timeout,
            )

            # Wait for any dynamic content
            await self._wait_for_stable_page()

            final_url = await self.page.get_url()
            title = await self.page.get_title()
            wait_time = time.time() - start_time

            logger.info(f"Successfully navigated to {final_url}")

            return NavigationResult(
                success=True,
                url=final_url,
                title=title,
                navigation_type=NavigationType.URL,
                wait_time=wait_time,
            )

        except Exception as e:
            logger.error(f"Navigation to {url} failed: {e}")
            return NavigationResult(
                success=False,
                url=url,
                navigation_type=NavigationType.URL,
                error=str(e),
                wait_time=time.time() - start_time,
            )

    async def follow_link(
        self,
        description: str,
        use_vision: bool = True,
        timeout: Optional[int] = None,
    ) -> NavigationResult:
        """
        Follow a link described in natural language.

        Args:
            description: Natural language description of the link to follow
            use_vision: Whether to use vision for link detection
            timeout: Navigation timeout in milliseconds

        Returns:
            NavigationResult with navigation details

        Example:
            >>> result = await agent.follow_link("the About Us link")
            >>> result = await agent.follow_link("Learn more about pricing")
        """
        import time

        start_time = time.time()
        timeout = timeout or self.default_timeout

        try:
            logger.info(f"Following link: {description}")

            # Find the link element
            element_info = await self.detector.find_element(
                f"the link or button for: {description}",
                use_vision=use_vision,
            )

            selector = element_info["selector"]
            selector_type = element_info.get("selector_type", "css")

            # Click the link
            await self.detector.click(selector, selector_type)

            # Wait for navigation to complete
            await self._wait_for_navigation(timeout)

            final_url = await self.page.get_url()
            title = await self.page.get_title()
            wait_time = time.time() - start_time

            logger.info(f"Successfully followed link to {final_url}")

            return NavigationResult(
                success=True,
                url=final_url,
                title=title,
                navigation_type=NavigationType.LINK,
                wait_time=wait_time,
                details={"link_description": description},
            )

        except ElementNotFoundError as e:
            return NavigationResult(
                success=False,
                navigation_type=NavigationType.LINK,
                error=f"Could not find link: {description}",
                wait_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Failed to follow link '{description}': {e}")
            return NavigationResult(
                success=False,
                navigation_type=NavigationType.LINK,
                error=str(e),
                wait_time=time.time() - start_time,
            )

    async def go_back(self, timeout: Optional[int] = None) -> NavigationResult:
        """
        Navigate back in browser history.

        Args:
            timeout: Navigation timeout in milliseconds

        Returns:
            NavigationResult with navigation details
        """
        import time

        start_time = time.time()
        timeout = timeout or self.default_timeout

        try:
            logger.info("Navigating back")
            await self.page.page.go_back(timeout=timeout)
            await self._wait_for_stable_page()

            return NavigationResult(
                success=True,
                url=await self.page.get_url(),
                title=await self.page.get_title(),
                navigation_type=NavigationType.BACK,
                wait_time=time.time() - start_time,
            )
        except Exception as e:
            return NavigationResult(
                success=False,
                navigation_type=NavigationType.BACK,
                error=str(e),
                wait_time=time.time() - start_time,
            )


    async def go_forward(self, timeout: Optional[int] = None) -> NavigationResult:
        """
        Navigate forward in browser history.

        Args:
            timeout: Navigation timeout in milliseconds

        Returns:
            NavigationResult with navigation details
        """
        import time

        start_time = time.time()
        timeout = timeout or self.default_timeout

        try:
            logger.info("Navigating forward")
            await self.page.page.go_forward(timeout=timeout)
            await self._wait_for_stable_page()

            return NavigationResult(
                success=True,
                url=await self.page.get_url(),
                title=await self.page.get_title(),
                navigation_type=NavigationType.FORWARD,
                wait_time=time.time() - start_time,
            )
        except Exception as e:
            return NavigationResult(
                success=False,
                navigation_type=NavigationType.FORWARD,
                error=str(e),
                wait_time=time.time() - start_time,
            )

    async def refresh(self, timeout: Optional[int] = None) -> NavigationResult:
        """
        Refresh the current page.

        Args:
            timeout: Navigation timeout in milliseconds

        Returns:
            NavigationResult with navigation details
        """
        import time

        start_time = time.time()
        timeout = timeout or self.default_timeout

        try:
            logger.info("Refreshing page")
            await self.page.page.reload(timeout=timeout)
            await self._wait_for_stable_page()

            return NavigationResult(
                success=True,
                url=await self.page.get_url(),
                title=await self.page.get_title(),
                navigation_type=NavigationType.REFRESH,
                wait_time=time.time() - start_time,
            )
        except Exception as e:
            return NavigationResult(
                success=False,
                navigation_type=NavigationType.REFRESH,
                error=str(e),
                wait_time=time.time() - start_time,
            )

    async def _perform_search(
        self,
        query: str,
        use_vision: bool = True,
        timeout: Optional[int] = None,
    ) -> NavigationResult:
        """Perform a search on the current page."""
        import time

        start_time = time.time()

        try:
            # Find search input
            element_info = await self.detector.find_element(
                "the search input field or search box",
                use_vision=use_vision,
            )

            selector = element_info["selector"]
            selector_type = element_info.get("selector_type", "css")

            # Type search query
            await self.detector.type_text(selector, query, selector_type)

            # Press Enter to search
            await self.page.page.keyboard.press("Enter")

            # Wait for results
            await self._wait_for_navigation(timeout or self.default_timeout)

            return NavigationResult(
                success=True,
                url=await self.page.get_url(),
                title=await self.page.get_title(),
                navigation_type=NavigationType.SEARCH,
                wait_time=time.time() - start_time,
                details={"query": query},
            )
        except Exception as e:
            return NavigationResult(
                success=False,
                navigation_type=NavigationType.SEARCH,
                error=str(e),
                wait_time=time.time() - start_time,
            )

    async def _get_page_links(self) -> List[Dict[str, Any]]:
        """Get list of links on the current page."""
        script = """
        () => {
            const links = [];
            document.querySelectorAll('a[href]').forEach((el, idx) => {
                if (idx < 100) {  // Limit to 100 links
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        links.push({
                            text: (el.textContent || '').trim().substring(0, 100),
                            href: el.href,
                            title: el.title || null,
                        });
                    }
                }
            });
            return links;
        }
        """
        try:
            return await self.page.evaluate(script)
        except Exception:
            return []

    async def _wait_for_navigation(self, timeout: int) -> None:
        """Wait for navigation to complete."""
        try:
            await self.page.page.wait_for_load_state(
                self.default_wait_strategy.value,
                timeout=timeout,
            )
        except Exception:
            # Navigation might have already completed
            pass

    async def _wait_for_stable_page(self, stability_time: float = 0.5) -> None:
        """
        Wait for the page to become stable (no more DOM changes).

        Args:
            stability_time: Time in seconds to wait for stability
        """
        try:
            # Wait for network to be relatively idle
            await asyncio.sleep(stability_time)

            # Check if page is still loading
            is_loading = await self.page.evaluate(
                "() => document.readyState !== 'complete'"
            )
            if is_loading:
                await self.page.page.wait_for_load_state("load", timeout=5000)
        except Exception:
            pass

    async def get_current_location(self) -> Dict[str, Any]:
        """
        Get information about the current page location.

        Returns:
            Dictionary with URL, title, and parsed URL components
        """
        url = await self.page.get_url()
        title = await self.page.get_title()
        parsed = urlparse(url)

        return {
            "url": url,
            "title": title,
            "protocol": parsed.scheme,
            "host": parsed.netloc,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
        }

    async def wait_for_element(
        self,
        description: str,
        timeout: Optional[int] = None,
        use_vision: bool = False,
    ) -> bool:
        """
        Wait for an element to appear on the page.

        Args:
            description: Natural language description of the element
            timeout: Maximum time to wait in milliseconds
            use_vision: Whether to use vision for detection

        Returns:
            True if element appeared, False if timeout

        Example:
            >>> await agent.wait_for_element("the loading spinner to disappear")
            >>> await agent.wait_for_element("the results table")
        """
        timeout = timeout or self.default_timeout
        start_time = asyncio.get_event_loop().time()
        poll_interval = 0.5

        while (asyncio.get_event_loop().time() - start_time) * 1000 < timeout:
            try:
                await self.detector.find_element(description, use_vision=use_vision)
                return True
            except ElementNotFoundError:
                await asyncio.sleep(poll_interval)

        return False
