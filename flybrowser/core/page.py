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
Page controller for browser interactions.

This module provides the PageController class which manages page-level
operations such as navigation, content extraction, screenshots, and
page state management.

The PageController wraps Playwright's Page API with error handling
and logging for production use.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from playwright.async_api import Page

from flybrowser.exceptions import NavigationError, PageError
from flybrowser.utils.logger import logger


# =============================================================================
# Browser Action Logger - Colored logging for Playwright page actions
# =============================================================================

class _PageActionLogger:
    """
    Provides consistent colored logging for page-level browser actions.
    
    This makes it easy to see when the browser is performing navigation,
    screenshots, and other page-level operations.
    """
    
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Page action specific colors (slightly different from element actions)
    PAGE = "\033[38;5;111m"       # Light blue for page actions
    SUCCESS = "\033[38;5;82m"     # Bright green
    ERROR = "\033[38;5;196m"      # Bright red
    WARNING = "\033[38;5;220m"    # Yellow
    
    def __init__(self):
        self._action_start_time: Optional[float] = None
        self._current_action: Optional[str] = None
    
    def start_action(self, action_type: str, description: str) -> float:
        """
        Log the start of a page action.
        
        Args:
            action_type: Type of action (NAVIGATE, SCREENSHOT, WAIT, etc.)
            description: What is being done
            
        Returns:
            Start time for calculating duration
        """
        self._action_start_time = time.time()
        self._current_action = action_type
        logger.info(
            f"{self.PAGE}{self.BOLD}> [BROWSER {action_type}]{self.RESET} "
            f"{self.PAGE}{description}{self.RESET}"
        )
        return self._action_start_time
    
    def end_action(self, success: bool, details: Optional[str] = None) -> None:
        """
        Log the end of a page action.
        
        Args:
            success: Whether the action succeeded
            details: Optional additional details
        """
        duration_ms = 0.0
        if self._action_start_time:
            duration_ms = (time.time() - self._action_start_time) * 1000
        
        status_icon = "[OK]" if success else "[FAIL]"
        status_color = self.SUCCESS if success else self.ERROR
        details_str = f" -> {details}" if details else ""
        
        logger.info(
            f"{status_color}{status_icon}{self.RESET} {self.PAGE}[BROWSER]{self.RESET} "
            f"{self.DIM}{duration_ms:.0f}ms{self.RESET}{details_str}"
        )
        
        self._action_start_time = None
        self._current_action = None
    
    def log_step(self, message: str) -> None:
        """Log an intermediate step within an action."""
        logger.info(f"  {self.PAGE}|-{self.RESET} {message}")


# Global page action logger instance
_page_logger = _PageActionLogger()


@dataclass
class PageStateSnapshot:
    """
    Snapshot of page state at a point in time.
    
    Used to track state changes after actions. Includes:
    - Page info (URL, title, ready state)
    - Focus info (currently focused element)
    - Interaction elements (inputs, forms)
    - Visual state (scroll position, viewport, modals)
    - Mouse state (position, element under cursor)
    """
    url: str
    title: str
    focused_element: Optional[Dict[str, Any]] = None
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    scroll_position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    viewport: Dict[str, int] = field(default_factory=dict)
    ready_state: str = "complete"
    has_modals: bool = False
    # Mouse tracking
    mouse_position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    element_under_cursor: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def diff(self, other: "PageStateSnapshot") -> Dict[str, Any]:
        """Compare with another snapshot and return differences."""
        changes = {}
        
        if self.url != other.url:
            changes["url"] = {"before": other.url, "after": self.url}
        if self.title != other.title:
            changes["title"] = {"before": other.title, "after": self.title}
        if self.focused_element != other.focused_element:
            changes["focused_element"] = {"before": other.focused_element, "after": self.focused_element}
        if self.has_modals != other.has_modals:
            changes["has_modals"] = {"before": other.has_modals, "after": self.has_modals}
        if self.scroll_position != other.scroll_position:
            changes["scroll_position"] = {"before": other.scroll_position, "after": self.scroll_position}
        if self.mouse_position != other.mouse_position:
            changes["mouse_position"] = {"before": other.mouse_position, "after": self.mouse_position}
        if self.element_under_cursor != other.element_under_cursor:
            changes["element_under_cursor"] = {"before": other.element_under_cursor, "after": self.element_under_cursor}
        
        return changes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "focusedElement": self.focused_element,
            "inputs": self.inputs,
            "forms": self.forms,
            "scrollPosition": self.scroll_position,
            "viewport": self.viewport,
            "readyState": self.ready_state,
            "hasModals": self.has_modals,
            "mousePosition": self.mouse_position,
            "elementUnderCursor": self.element_under_cursor,
            "timestamp": self.timestamp.isoformat(),
        }


class PageController:
    """
    Controls page interactions and state management with lifecycle tracking.

    This class provides high-level methods for common page operations:
    - Navigation with configurable wait conditions
    - Screenshot capture
    - HTML content extraction
    - Page metadata retrieval
    - Scroll operations
    - State tracking and change detection
    - Page lifecycle event handling

    Attributes:
        page: The underlying Playwright Page instance
        last_state: Most recent state snapshot
        state_history: List of recent state snapshots
        event_callbacks: Registered callbacks for page events

    Example:
        >>> controller = PageController(page)
        >>> await controller.goto("https://example.com")
        >>> state = await controller.capture_state()
        >>> # After action
        >>> changes = await controller.get_state_changes()
    """

    def __init__(self, page: Page) -> None:
        """
        Initialize the page controller with lifecycle tracking.

        Args:
            page: Playwright Page instance to control

        Example:
            >>> from playwright.async_api import async_playwright
            >>> playwright = await async_playwright().start()
            >>> browser = await playwright.chromium.launch()
            >>> page = await browser.new_page()
            >>> controller = PageController(page)
        """
        self.page = page
        
        # State tracking
        self._last_state: Optional[PageStateSnapshot] = None
        self._state_history: List[PageStateSnapshot] = []
        self._max_history_size: int = 10
        
        # Event tracking
        self._event_callbacks: Dict[str, List[Callable]] = {
            "navigation": [],
            "load": [],
            "domcontentloaded": [],
            "dialog": [],
            "console": [],
            "pageerror": [],
            "request": [],
            "response": [],
            "popup": [],
        }
        self._pending_navigations: int = 0
        self._last_navigation_url: Optional[str] = None
        
        # Setup Playwright event listeners
        self._setup_event_listeners()
    
    def _setup_event_listeners(self) -> None:
        """Setup Playwright page event listeners for lifecycle tracking."""
        # Navigation events
        self.page.on("load", self._on_load)
        self.page.on("domcontentloaded", self._on_domcontentloaded)
        
        # Dialog events (alerts, confirms, prompts)
        self.page.on("dialog", self._on_dialog)
        
        # Console and error events
        self.page.on("console", self._on_console)
        self.page.on("pageerror", self._on_pageerror)
        
        # Request/response events
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)
        
        # Popup events
        self.page.on("popup", self._on_popup)
    
    def _on_load(self, *args) -> None:
        """Handle page load event."""
        logger.debug(f"Page load event: {self.page.url}")
        self._last_navigation_url = self.page.url
        for callback in self._event_callbacks.get("load", []):
            try:
                callback({"url": self.page.url, "event": "load"})
            except Exception as e:
                logger.debug(f"Load callback error: {e}")
    
    def _on_domcontentloaded(self, *args) -> None:
        """Handle DOMContentLoaded event."""
        logger.debug(f"DOMContentLoaded event: {self.page.url}")
        for callback in self._event_callbacks.get("domcontentloaded", []):
            try:
                callback({"url": self.page.url, "event": "domcontentloaded"})
            except Exception as e:
                logger.debug(f"DOMContentLoaded callback error: {e}")
    
    def _on_dialog(self, dialog) -> None:
        """Handle dialog event (alert, confirm, prompt)."""
        logger.debug(f"Dialog event: type={dialog.type}, message={dialog.message}")
        for callback in self._event_callbacks.get("dialog", []):
            try:
                callback({"type": dialog.type, "message": dialog.message, "dialog": dialog})
            except Exception as e:
                logger.debug(f"Dialog callback error: {e}")
    
    def _on_console(self, msg) -> None:
        """Handle console message."""
        # Only log errors/warnings to avoid noise
        if msg.type in ("error", "warning"):
            logger.debug(f"Console {msg.type}: {msg.text}")
        for callback in self._event_callbacks.get("console", []):
            try:
                callback({"type": msg.type, "text": msg.text})
            except Exception as e:
                logger.debug(f"Console callback error: {e}")
    
    def _on_pageerror(self, error) -> None:
        """Handle page error event."""
        logger.debug(f"Page error: {error}")
        for callback in self._event_callbacks.get("pageerror", []):
            try:
                callback({"error": str(error)})
            except Exception as e:
                logger.debug(f"Page error callback error: {e}")
    
    def _on_request(self, request) -> None:
        """Handle request event."""
        # Track navigation requests
        if request.is_navigation_request():
            self._pending_navigations += 1
        for callback in self._event_callbacks.get("request", []):
            try:
                callback({"url": request.url, "method": request.method, "is_navigation": request.is_navigation_request()})
            except Exception as e:
                logger.debug(f"Request callback error: {e}")
    
    def _on_response(self, response) -> None:
        """Handle response event."""
        # Track navigation responses
        if response.request.is_navigation_request():
            self._pending_navigations = max(0, self._pending_navigations - 1)
        for callback in self._event_callbacks.get("response", []):
            try:
                callback({"url": response.url, "status": response.status, "is_navigation": response.request.is_navigation_request()})
            except Exception as e:
                logger.debug(f"Response callback error: {e}")
    
    def _on_popup(self, popup_page) -> None:
        """Handle popup event."""
        logger.debug(f"Popup opened: {popup_page.url}")
        for callback in self._event_callbacks.get("popup", []):
            try:
                callback({"url": popup_page.url, "page": popup_page})
            except Exception as e:
                logger.debug(f"Popup callback error: {e}")
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a page event.
        
        Args:
            event: Event name (navigation, load, domcontentloaded, dialog, console, pageerror, request, response, popup)
            callback: Callback function to invoke
        """
        if event in self._event_callbacks:
            self._event_callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")
    
    def off(self, event: str, callback: Callable) -> None:
        """Remove a callback for a page event."""
        if event in self._event_callbacks and callback in self._event_callbacks[event]:
            self._event_callbacks[event].remove(callback)
    
    @property
    def last_state(self) -> Optional[PageStateSnapshot]:
        """Get the last captured state."""
        return self._last_state
    
    @property
    def state_history(self) -> List[PageStateSnapshot]:
        """Get state history (most recent last)."""
        return self._state_history.copy()
    
    @property
    def is_navigating(self) -> bool:
        """Check if navigation is in progress."""
        return self._pending_navigations > 0

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 30000) -> None:
        """
        Navigate to a URL with configurable wait conditions.

        Args:
            url: URL to navigate to (must include protocol, e.g., https://)
            wait_until: When to consider navigation succeeded. Options:
                - "load": Wait for the load event (default)
                - "domcontentloaded": Wait for DOMContentLoaded event
                - "networkidle": Wait for network to be idle (no requests for 500ms)
                - "commit": Wait for navigation to commit
            timeout: Maximum time to wait for navigation in milliseconds.
                Default: 30000 (30 seconds)

        Raises:
            NavigationError: If navigation fails or times out

        Example:
            >>> await controller.goto("https://example.com")
            >>> await controller.goto("https://example.com", wait_until="networkidle")
            >>> await controller.goto("https://example.com", timeout=60000)
        """
        # Truncate URL for display
        url_display = url[:60] + "..." if len(url) > 60 else url
        _page_logger.start_action("NAVIGATE", url_display)
        
        try:
            await self.page.goto(url, wait_until=wait_until, timeout=timeout)
            _page_logger.end_action(True, f"loaded ({wait_until})")
        except Exception as e:
            _page_logger.end_action(False, str(e)[:50])
            raise NavigationError(f"Failed to navigate to {url}: {e}") from e

    async def navigate(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 30000) -> None:
        """
        Navigate to a URL. Alias for goto().

        This method is an alias for goto() provided for API consistency.

        Args:
            url: URL to navigate to (must include protocol, e.g., https://)
            wait_until: When to consider navigation succeeded.
            timeout: Maximum time to wait for navigation in milliseconds.

        Raises:
            NavigationError: If navigation fails or times out

        Example:
            >>> await controller.navigate("https://example.com")
        """
        await self.goto(url, wait_until=wait_until, timeout=timeout)

    async def screenshot(self, full_page: bool = False) -> bytes:
        """
        Take a screenshot of the current page.

        Args:
            full_page: Whether to capture the full scrollable page.
                - True: Captures entire page including content below the fold
                - False: Captures only the visible viewport (default)

        Returns:
            Screenshot as PNG bytes

        Raises:
            PageError: If screenshot capture fails

        Example:
            >>> screenshot_bytes = await controller.screenshot()
            >>> with open("screenshot.png", "wb") as f:
            ...     f.write(screenshot_bytes)

            >>> full_screenshot = await controller.screenshot(full_page=True)
        """
        page_type = "full page" if full_page else "viewport"
        _page_logger.start_action("SCREENSHOT", page_type)
        
        try:
            result = await self.page.screenshot(full_page=full_page, type="png")
            _page_logger.end_action(True, f"{len(result) // 1024}KB")
            return result
        except Exception as e:
            _page_logger.end_action(False, str(e)[:50])
            raise PageError(f"Failed to take screenshot: {e}") from e

    async def get_html(self) -> str:
        """
        Get the complete HTML content of the page.

        Returns:
            Full HTML content as string including DOCTYPE and all elements

        Raises:
            PageError: If HTML extraction fails

        Example:
            >>> html = await controller.get_html()
            >>> print(html[:100])  # Print first 100 characters
        """
        try:
            return await self.page.content()
        except Exception as e:
            logger.error(f"Failed to get HTML: {e}")
            raise PageError(f"Failed to get page HTML: {e}") from e

    async def get_title(self) -> str:
        """
        Get the page title.

        Returns:
            Page title
        """
        try:
            return await self.page.title()
        except Exception as e:
            logger.error(f"Failed to get title: {e}")
            raise PageError(f"Failed to get page title: {e}") from e

    async def get_url(self) -> str:
        """
        Get the current page URL.

        Returns:
            Current URL
        """
        return self.page.url

    async def wait_for_selector(
        self, selector: str, timeout: int = 30000, state: str = "visible"
    ) -> None:
        """
        Wait for a selector to be in a specific state.

        Args:
            selector: CSS selector
            timeout: Timeout in milliseconds
            state: Element state to wait for (visible, hidden, attached, detached)
        """
        # Truncate selector for display
        selector_display = selector[:50] + "..." if len(selector) > 50 else selector
        _page_logger.start_action("WAIT", f"{state}: {selector_display}")
        
        try:
            await self.page.wait_for_selector(selector, timeout=timeout, state=state)
            _page_logger.end_action(True, f"element {state}")
        except Exception as e:
            _page_logger.end_action(False, str(e)[:50])
            raise PageError(f"Failed to wait for selector {selector}: {e}") from e

    async def evaluate(self, script: str) -> Any:
        """
        Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute

        Returns:
            Result of the script execution
        """
        try:
            return await self.page.evaluate(script)
        except Exception as e:
            logger.error(f"Script evaluation failed: {e}")
            raise PageError(f"Failed to evaluate script: {e}") from e

    async def get_page_state(self) -> Dict[str, Any]:
        """
        Get comprehensive page state information.

        Returns:
            Dictionary containing page state
        """
        return {
            "url": await self.get_url(),
            "title": await self.get_title(),
            "viewport": self.page.viewport_size,
        }
    
    async def get_rich_state(self) -> Dict[str, Any]:
        """
        Get rich, comprehensive state information for LLM/VLM context.
        
        This provides all the information an agent needs to understand
        the current browser state and make informed decisions.
        
        Returns:
            Dictionary containing:
            - url: Current page URL
            - title: Page title
            - viewport: Viewport dimensions
            - scroll_position: Current scroll position
            - page_dimensions: Full page dimensions
            - focused_element: Currently focused element info
            - ready_state: Document ready state
            - load_state: Page load state indicators
            - forms: List of forms on the page
            - inputs: List of input elements (for typing context)
        """
        try:
            state = await self.page.evaluate("""
                () => {
                    // Get focused element info
                    const focused = document.activeElement;
                    let focusedInfo = null;
                    if (focused && focused !== document.body) {
                        focusedInfo = {
                            tagName: focused.tagName.toLowerCase(),
                            type: focused.type || null,
                            id: focused.id || null,
                            name: focused.name || null,
                            className: focused.className || null,
                            placeholder: focused.placeholder || null,
                            value: focused.value ? focused.value.substring(0, 100) : null,
                            isEditable: focused.isContentEditable || ['input', 'textarea', 'select'].includes(focused.tagName.toLowerCase())
                        };
                    }
                    
                    // Get all input elements (relevant for typing actions)
                    const inputs = [];
                    document.querySelectorAll('input, textarea, select').forEach((el, idx) => {
                        if (idx < 20) {  // Limit for performance
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0) {
                                inputs.push({
                                    tagName: el.tagName.toLowerCase(),
                                    type: el.type || 'text',
                                    id: el.id || null,
                                    name: el.name || null,
                                    placeholder: el.placeholder || null,
                                    value: el.value ? (el.type === 'password' ? '***' : el.value.substring(0, 50)) : null,
                                    isVisible: rect.top >= 0 && rect.top <= window.innerHeight,
                                    isFocused: el === document.activeElement
                                });
                            }
                        }
                    });
                    
                    // Get forms info
                    const forms = [];
                    document.querySelectorAll('form').forEach((form, idx) => {
                        if (idx < 5) {  // Limit for performance
                            const fields = form.querySelectorAll('input, textarea, select');
                            forms.push({
                                id: form.id || null,
                                action: form.action || null,
                                method: form.method || 'GET',
                                fieldCount: fields.length,
                                hasSubmitButton: !!form.querySelector('[type="submit"], button:not([type="button"])')
                            });
                        }
                    });
                    
                    // Get interactive buttons (hamburger menus, nav toggles, etc.)
                    const buttons = [];
                    const buttonSelectors = [
                        'button',
                        '[role="button"]',
                        'a[href="#"]',  // Often used for menu toggles
                        '.menu-toggle',
                        '.nav-toggle',
                        '.hamburger',
                        '.mobile-menu',
                        '[aria-expanded]',
                        '[aria-controls]',
                    ];
                    
                    const seenButtons = new Set();
                    buttonSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach((btn) => {
                            const rect = btn.getBoundingClientRect();
                            if (rect.width === 0 || rect.height === 0) return;
                            
                            const text = btn.textContent?.trim() || '';
                            const ariaLabel = btn.getAttribute('aria-label') || '';
                            const ariaExpanded = btn.getAttribute('aria-expanded');
                            const ariaControls = btn.getAttribute('aria-controls');
                            const classes = btn.className || '';
                            
                            // Create unique identifier
                            const btnId = `${btn.tagName}-${text}-${ariaLabel}-${classes}`.substring(0, 200);
                            if (seenButtons.has(btnId)) return;
                            seenButtons.add(btnId);
                            
                            // Detect button type
                            let buttonType = 'other';
                            const isMenuButton = /menu|nav|hamburger|toggle|burger/i.test(text + ariaLabel + classes);
                            const isExpandable = ariaExpanded !== null;
                            
                            if (isMenuButton || isExpandable) {
                                buttonType = 'menu';
                            }
                            
                            const isInViewport = rect.top >= 0 && rect.top <= window.innerHeight;
                            
                            buttons.push({
                                text: text.substring(0, 100),
                                ariaLabel: ariaLabel,
                                ariaExpanded: ariaExpanded,
                                ariaControls: ariaControls,
                                type: buttonType,
                                isVisible: isInViewport,
                                selector: btn.id ? `#${btn.id}` : `${btn.tagName.toLowerCase()}`,
                                classes: classes
                            });
                        });
                    });
                    
                    // Get navigation links (important for exploration)
                    const links = [];
                    const visibleLinks = [];
                    const hiddenLinks = [];
                    const seenUrls = new Set();
                    
                    document.querySelectorAll('a[href]').forEach((link) => {
                        const href = link.href;
                        const text = link.textContent?.trim() || '';
                        
                        // Skip empty links, anchors, and javascript links
                        if (!href || href.startsWith('#') || href.startsWith('javascript:') || !text) {
                            return;
                        }
                        
                        // Deduplicate by URL
                        if (seenUrls.has(href)) {
                            return;
                        }
                        seenUrls.add(href);
                        
                        const rect = link.getBoundingClientRect();
                        const hasSize = rect.width > 0 && rect.height > 0;
                        const isInViewport = rect.top >= 0 && rect.top <= window.innerHeight;
                        
                        // Check if element or parents are hidden
                        let isHidden = false;
                        let element = link;
                        while (element && element !== document.body) {
                            const style = window.getComputedStyle(element);
                            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                                isHidden = true;
                                break;
                            }
                            element = element.parentElement;
                        }
                        
                        // Prioritize visible links in navigation areas
                        const isNav = link.closest('nav, header, [role="navigation"], .nav, .menu, .navbar, aside, footer');
                        
                        const linkData = {
                            text: text.substring(0, 100),
                            href: href,
                            isNav: !!isNav,
                            isVisible: isInViewport,
                            ariaLabel: link.getAttribute('aria-label') || null,
                        };
                        
                        if (hasSize && !isHidden && (isInViewport || isNav)) {
                            visibleLinks.push(linkData);
                        } else if (isNav) {
                            // Track hidden nav links separately
                            hiddenLinks.push(linkData);
                        }
                    });
                    
                    // Limit links to most relevant ones
                    const navLinks = visibleLinks.filter(l => l.isNav).slice(0, 30);
                    const otherLinks = visibleLinks.filter(l => !l.isNav && l.isVisible).slice(0, 20);
                    const allLinks = [...navLinks, ...otherLinks].slice(0, 40);
                    
                    // Get main content sections for context
                    const contentSections = [];
                    const contentSelectors = ['main', 'article', '[role="main"]', '#content', '.content'];
                    for (const selector of contentSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent?.trim() || '';
                            if (text.length > 100) {
                                contentSections.push({
                                    selector: selector,
                                    preview: text.substring(0, 500),
                                    textLength: text.length
                                });
                                break;  // Only get first main content area
                            }
                        }
                    }
                    
                    // Defensive checks for navigation transitions
                    const docEl = document.documentElement;
                    const pageDimensions = docEl ? {
                        width: docEl.scrollWidth || 0,
                        height: docEl.scrollHeight || 0
                    } : { width: 0, height: 0 };
                    
                    return {
                        scrollPosition: {
                            x: window.scrollX || 0,
                            y: window.scrollY || 0
                        },
                        pageDimensions: pageDimensions,
                        viewportDimensions: {
                            width: window.innerWidth || 0,
                            height: window.innerHeight || 0
                        },
                        focusedElement: focusedInfo,
                        readyState: document.readyState || 'loading',
                        isInteractive: document.readyState !== 'loading',
                        inputs: inputs,
                        forms: forms,
                        buttons: buttons.slice(0, 20),  // Limit buttons for performance
                        links: allLinks,
                        hiddenLinks: hiddenLinks.slice(0, 20),  // Track hidden nav links
                        contentSections: contentSections,
                        hasModals: !!document.querySelector('[role="dialog"], [aria-modal="true"], .modal.show')
                    };
                }
            """)
            
            # Add URL and title
            state["url"] = await self.get_url()
            state["title"] = await self.get_title()
            state["viewport"] = self.page.viewport_size
            
            # Add mouse position and element under cursor
            state["mousePosition"] = {
                "x": getattr(self, '_last_mouse_x', 0),
                "y": getattr(self, '_last_mouse_y', 0),
            }
            state["elementUnderCursor"] = getattr(self, '_last_element_under_cursor', None)
            
            return state
            
        except Exception as e:
            # This can happen during navigation transitions - not a real error
            logger.debug(f"Failed to get rich state (likely navigation in progress): {e}")
            # Fallback to basic state
            try:
                return await self.get_page_state()
            except Exception:
                # Even basic state failed - return minimal info
                return {
                    "url": "",
                    "title": "",
                    "viewport": self.page.viewport_size,
                    "readyState": "loading",
                    "isInteractive": False,
                }
    
    async def get_focused_element(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently focused element.
        
        Returns:
            Dictionary with focused element info, or None if no element is focused
        """
        try:
            return await self.page.evaluate("""
                () => {
                    const focused = document.activeElement;
                    if (!focused || focused === document.body || focused === document.documentElement) {
                        return null;
                    }
                    const rect = focused.getBoundingClientRect();
                    return {
                        tagName: focused.tagName.toLowerCase(),
                        type: focused.type || null,
                        id: focused.id || null,
                        name: focused.name || null,
                        className: focused.className || null,
                        placeholder: focused.placeholder || null,
                        isEditable: focused.isContentEditable || ['input', 'textarea', 'select'].includes(focused.tagName.toLowerCase()),
                        position: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        }
                    };
                }
            """)
        except Exception as e:
            logger.debug(f"Failed to get focused element: {e}")
            return None
    
    async def press_key(self, key: str) -> None:
        """
        Press a keyboard key.
        
        Args:
            key: Key to press (e.g., "Enter", "Escape", "Tab", "ArrowDown")
        
        Raises:
            PageError: If key press fails
        """
        try:
            await self.page.keyboard.press(key)
        except Exception as e:
            logger.error(f"Failed to press key {key}: {e}")
            raise PageError(f"Failed to press key {key}: {e}") from e
    
    async def type_text(self, text: str, delay: int = 0) -> None:
        """
        Type text into the currently focused element.
        
        Args:
            text: Text to type
            delay: Delay between key presses in milliseconds
        
        Raises:
            PageError: If typing fails
        """
        try:
            await self.page.keyboard.type(text, delay=delay)
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            raise PageError(f"Failed to type text: {e}") from e
    
    async def scroll_page(self, delta_x: int = 0, delta_y: int = 0) -> None:
        """
        Scroll the page by specified amounts.
        
        Args:
            delta_x: Horizontal scroll amount in pixels (positive = right, negative = left)
            delta_y: Vertical scroll amount in pixels (positive = down, negative = up)
        
        Raises:
            PageError: If scrolling fails
        """
        try:
            await self.page.evaluate(f"""
                () => {{
                    window.scrollBy({delta_x}, {delta_y});
                }}
            """)
        except Exception as e:
            logger.error(f"Failed to scroll page: {e}")
            raise PageError(f"Failed to scroll page: {e}") from e
    
    async def get_load_state(self) -> Dict[str, Any]:
        """
        Get detailed load state information.
        
        Returns:
            Dictionary with load state indicators
        """
        try:
            return await self.page.evaluate("""
                () => ({
                    readyState: document.readyState,
                    isComplete: document.readyState === 'complete',
                    isInteractive: document.readyState !== 'loading',
                    hasActiveRequests: window.performance.getEntriesByType('resource').some(
                        e => e.responseEnd === 0
                    )
                })
            """)
        except Exception:
            return {"readyState": "unknown", "isComplete": False, "isInteractive": True}
    
    # =========================================================================
    # STATE CAPTURE AND CHANGE TRACKING
    # =========================================================================
    
    async def capture_state(self) -> PageStateSnapshot:
        """
        Capture current page state as a snapshot.
        
        This captures all relevant state information and stores it in history.
        Use this before and after actions to track changes.
        
        Captures:
        - URL and title
        - Focused element
        - Input elements and forms
        - Scroll position and viewport
        - Modal/dialog state
        - Mouse position and element under cursor
        
        Returns:
            PageStateSnapshot with current state
        """
        rich_state = await self.get_rich_state()
        
        # Get current mouse position
        mouse_pos = await self.get_mouse_position()
        
        snapshot = PageStateSnapshot(
            url=rich_state.get("url", ""),
            title=rich_state.get("title", ""),
            focused_element=rich_state.get("focusedElement"),
            inputs=rich_state.get("inputs", []),
            forms=rich_state.get("forms", []),
            scroll_position=rich_state.get("scrollPosition", {"x": 0, "y": 0}),
            viewport=rich_state.get("viewport", {}),
            ready_state=rich_state.get("readyState", "complete"),
            has_modals=rich_state.get("hasModals", False),
            mouse_position={"x": mouse_pos.get("x", 0), "y": mouse_pos.get("y", 0)},
            element_under_cursor=mouse_pos.get("elementUnderCursor"),
        )
        
        # Update history
        self._last_state = snapshot
        self._state_history.append(snapshot)
        
        # Trim history if needed
        if len(self._state_history) > self._max_history_size:
            self._state_history = self._state_history[-self._max_history_size:]
        
        return snapshot
    
    async def get_state_changes(self, since: Optional[PageStateSnapshot] = None) -> Dict[str, Any]:
        """
        Get state changes since a previous snapshot.
        
        Args:
            since: Previous snapshot to compare against. If None, uses the previous state in history.
        
        Returns:
            Dictionary describing what changed
        """
        current = await self.capture_state()
        
        if since is None:
            # Use second-to-last state if available
            if len(self._state_history) >= 2:
                since = self._state_history[-2]
            else:
                return {"changes": {}, "current": current.to_dict()}
        
        changes = current.diff(since)
        
        return {
            "changes": changes,
            "has_changes": len(changes) > 0,
            "current": current.to_dict(),
            "previous": since.to_dict(),
        }
    
    async def wait_for_state_change(
        self, 
        check_fn: Optional[Callable[[PageStateSnapshot], bool]] = None,
        timeout_ms: int = 5000,
        poll_interval_ms: int = 100,
    ) -> Optional[PageStateSnapshot]:
        """
        Wait for a state change to occur.
        
        Args:
            check_fn: Optional function to check if desired state is reached.
                     If None, waits for any change in focused element, URL, or modals.
            timeout_ms: Maximum time to wait in milliseconds
            poll_interval_ms: How often to check for changes
        
        Returns:
            New PageStateSnapshot if change detected, None if timeout
        """
        initial_state = await self.capture_state()
        start_time = datetime.now()
        
        while True:
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms >= timeout_ms:
                return None
            
            await asyncio.sleep(poll_interval_ms / 1000)
            
            current_state = await self.capture_state()
            
            if check_fn:
                if check_fn(current_state):
                    return current_state
            else:
                # Default: check for any significant change
                if (current_state.url != initial_state.url or
                    current_state.focused_element != initial_state.focused_element or
                    current_state.has_modals != initial_state.has_modals):
                    return current_state
        
        return None
    
    # =========================================================================
    # MOUSE AND ELEMENT TRACKING
    # =========================================================================
    
    async def get_mouse_position(self) -> Dict[str, Any]:
        """
        Get current mouse position and element under cursor.
        
        Returns:
            Dictionary with:
            - x, y: Mouse coordinates
            - elementUnderCursor: Info about element at mouse position
        """
        try:
            # Note: Playwright doesn't expose raw mouse position, but we can
            # track it by storing last known position from mouse operations
            return {
                "x": getattr(self, '_last_mouse_x', 0),
                "y": getattr(self, '_last_mouse_y', 0),
                "elementUnderCursor": getattr(self, '_last_element_under_cursor', None),
            }
        except Exception as e:
            logger.debug(f"Failed to get mouse position: {e}")
            return {"x": 0, "y": 0, "elementUnderCursor": None}
    
    async def get_element_at_position(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about element at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            Dictionary with element details or None if no element found
        """
        try:
            element_info = await self.page.evaluate(f"""
                () => {{
                    const el = document.elementFromPoint({x}, {y});
                    if (!el) return null;
                    
                    const rect = el.getBoundingClientRect();
                    const styles = window.getComputedStyle(el);
                    
                    return {{
                        tagName: el.tagName.toLowerCase(),
                        id: el.id || null,
                        className: el.className || null,
                        name: el.name || null,
                        type: el.type || null,
                        text: (el.textContent || '').trim().substring(0, 100),
                        innerText: (el.innerText || '').trim().substring(0, 100),
                        value: el.value ? el.value.substring(0, 50) : null,
                        href: el.href || null,
                        src: el.src || null,
                        alt: el.alt || null,
                        title: el.title || null,
                        placeholder: el.placeholder || null,
                        ariaLabel: el.getAttribute('aria-label'),
                        role: el.getAttribute('role'),
                        isClickable: (
                            el.tagName === 'A' ||
                            el.tagName === 'BUTTON' ||
                            el.tagName === 'INPUT' ||
                            el.onclick !== null ||
                            styles.cursor === 'pointer' ||
                            el.getAttribute('role') === 'button'
                        ),
                        isEditable: (
                            el.isContentEditable ||
                            ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)
                        ),
                        isVisible: rect.width > 0 && rect.height > 0,
                        position: {{
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            top: rect.top,
                            left: rect.left,
                            bottom: rect.bottom,
                            right: rect.right,
                            centerX: rect.x + rect.width / 2,
                            centerY: rect.y + rect.height / 2
                        }},
                        computedStyle: {{
                            cursor: styles.cursor,
                            display: styles.display,
                            visibility: styles.visibility,
                            opacity: styles.opacity,
                            backgroundColor: styles.backgroundColor,
                            color: styles.color,
                            zIndex: styles.zIndex
                        }}
                    }};
                }}
            """)
            return element_info
        except Exception as e:
            logger.debug(f"Failed to get element at position: {e}")
            return None
    
    async def get_element_details(self, selector: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive details about an element by selector.
        
        Args:
            selector: CSS selector
        
        Returns:
            Dictionary with full element details
        """
        try:
            element_info = await self.page.evaluate(f"""
                (selector) => {{
                    const el = document.querySelector(selector);
                    if (!el) return null;
                    
                    const rect = el.getBoundingClientRect();
                    const styles = window.getComputedStyle(el);
                    
                    // Get all attributes
                    const attributes = {{}};
                    for (const attr of el.attributes) {{
                        attributes[attr.name] = attr.value;
                    }}
                    
                    // Check if element is in viewport
                    const inViewport = (
                        rect.top >= 0 &&
                        rect.left >= 0 &&
                        rect.bottom <= window.innerHeight &&
                        rect.right <= window.innerWidth
                    );
                    
                    return {{
                        tagName: el.tagName.toLowerCase(),
                        id: el.id || null,
                        className: el.className || null,
                        attributes: attributes,
                        text: (el.textContent || '').trim().substring(0, 200),
                        innerText: (el.innerText || '').trim().substring(0, 200),
                        innerHTML: el.innerHTML.substring(0, 500),
                        value: el.value || null,
                        checked: el.checked || null,
                        selected: el.selected || null,
                        disabled: el.disabled || false,
                        readOnly: el.readOnly || false,
                        required: el.required || false,
                        position: {{
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            centerX: rect.x + rect.width / 2,
                            centerY: rect.y + rect.height / 2,
                            inViewport: inViewport
                        }},
                        state: {{
                            isFocused: el === document.activeElement,
                            isHovered: el.matches(':hover'),
                            isClickable: (
                                el.tagName === 'A' ||
                                el.tagName === 'BUTTON' ||
                                el.onclick !== null ||
                                styles.cursor === 'pointer'
                            ),
                            isEditable: el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName),
                            isVisible: rect.width > 0 && rect.height > 0 && styles.visibility !== 'hidden' && styles.display !== 'none'
                        }},
                        style: {{
                            cursor: styles.cursor,
                            display: styles.display,
                            visibility: styles.visibility,
                            opacity: styles.opacity,
                            pointerEvents: styles.pointerEvents
                        }}
                    }};
                }}
            """, selector)
            return element_info
        except Exception as e:
            logger.debug(f"Failed to get element details: {e}")
            return None
    
    async def move_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """
        Move mouse to coordinates and get element under cursor.
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            Dictionary with mouse position and element info
        """
        try:
            # Move mouse
            await self.page.mouse.move(x, y)
            
            # Store position
            self._last_mouse_x = x
            self._last_mouse_y = y
            
            # Get element at position
            element = await self.get_element_at_position(x, y)
            self._last_element_under_cursor = element
            
            return {
                "success": True,
                "x": x,
                "y": y,
                "elementUnderCursor": element,
            }
        except Exception as e:
            logger.error(f"Move mouse failed: {e}")
            return {"success": False, "error": str(e), "x": x, "y": y}
    
    async def hover_and_track(self, selector: str, timeout: int = 30000) -> Dict[str, Any]:
        """
        Hover over an element and track the element details.
        
        Args:
            selector: CSS selector for element to hover
            timeout: Timeout in milliseconds
        
        Returns:
            Dictionary with hover result and element details
        """
        state_before = await self.capture_state()
        
        try:
            # Get element details before hover
            element_details = await self.get_element_details(selector)
            
            locator = self.page.locator(selector)
            await locator.hover(timeout=timeout)
            
            # Get bounding box to store mouse position
            box = await locator.bounding_box()
            if box:
                self._last_mouse_x = int(box['x'] + box['width'] / 2)
                self._last_mouse_y = int(box['y'] + box['height'] / 2)
                self._last_element_under_cursor = element_details
            
            # Wait for any hover effects
            await asyncio.sleep(0.1)
            
            # Get updated element details (hover state might change styles)
            element_after = await self.get_element_details(selector)
            
            state_after = await self.capture_state()
            changes = state_after.diff(state_before)
            
            return {
                "success": True,
                "element": element_details,
                "elementAfterHover": element_after,
                "mousePosition": {"x": self._last_mouse_x, "y": self._last_mouse_y},
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": changes,
            }
            
        except Exception as e:
            logger.error(f"Hover failed: {e}")
            state_after = await self.capture_state()
            return {
                "success": False,
                "error": str(e),
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
            }
    
    # =========================================================================
    # ACTION METHODS WITH STATE TRACKING
    # =========================================================================
    
    async def click_and_track(
        self, 
        selector: str, 
        wait_for_navigation: bool = False,
        timeout: int = 30000,
        button: str = "left",
        click_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Click an element with comprehensive tracking.
        
        Tracks:
        - Element details before/after click
        - Mouse position at click
        - State changes (focus, navigation, modals)
        - Click coordinates
        
        Args:
            selector: CSS selector for element to click
            wait_for_navigation: Whether to wait for navigation after click
            timeout: Timeout in milliseconds
            button: Mouse button ('left', 'right', 'middle')
            click_count: Number of clicks (1 for single, 2 for double)
        
        Returns:
            Dictionary with comprehensive click tracking info
        """
        state_before = await self.capture_state()
        
        try:
            # Get element details BEFORE click
            element_before = await self.get_element_details(selector)
            
            locator = self.page.locator(selector)
            
            # Get bounding box for click coordinates
            box = await locator.bounding_box(timeout=timeout)
            click_x = int(box['x'] + box['width'] / 2) if box else 0
            click_y = int(box['y'] + box['height'] / 2) if box else 0
            
            # Store mouse position
            self._last_mouse_x = click_x
            self._last_mouse_y = click_y
            self._last_element_under_cursor = element_before
            
            # Perform click
            await locator.click(
                timeout=timeout,
                button=button,
                click_count=click_count,
            )
            
            # Wait briefly for state to settle
            await asyncio.sleep(0.1)
            
            if wait_for_navigation:
                try:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
            
            # Get element details AFTER click (might have changed or be different element)
            element_after = await self.get_element_details(selector)
            
            # Get element currently under cursor (might be different after click)
            element_under_cursor = await self.get_element_at_position(click_x, click_y)
            
            state_after = await self.capture_state()
            changes = state_after.diff(state_before)
            
            return {
                "success": True,
                "selector": selector,
                "clickPosition": {"x": click_x, "y": click_y},
                "button": button,
                "clickCount": click_count,
                "elementClicked": element_before,
                "elementAfterClick": element_after,
                "elementUnderCursor": element_under_cursor,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": changes,
                "navigated": "url" in changes,
                "focus_changed": "focused_element" in changes,
                "modal_changed": "has_modals" in changes,
            }
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            state_after = await self.capture_state()
            return {
                "success": False,
                "error": str(e),
                "selector": selector,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": state_after.diff(state_before),
            }
    
    async def click_at_position(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Click at specific coordinates with tracking.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button
            click_count: Number of clicks
        
        Returns:
            Dictionary with click tracking info
        """
        state_before = await self.capture_state()
        
        try:
            # Get element at position before click
            element_before = await self.get_element_at_position(x, y)
            
            # Store mouse position
            self._last_mouse_x = x
            self._last_mouse_y = y
            self._last_element_under_cursor = element_before
            
            # Click at position
            await self.page.mouse.click(x, y, button=button, click_count=click_count)
            
            # Wait for state to settle
            await asyncio.sleep(0.1)
            
            # Get element after click
            element_after = await self.get_element_at_position(x, y)
            
            state_after = await self.capture_state()
            changes = state_after.diff(state_before)
            
            return {
                "success": True,
                "clickPosition": {"x": x, "y": y},
                "button": button,
                "clickCount": click_count,
                "elementAtPosition": element_before,
                "elementAfterClick": element_after,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": changes,
                "navigated": "url" in changes,
                "focus_changed": "focused_element" in changes,
            }
            
        except Exception as e:
            logger.error(f"Click at position failed: {e}")
            state_after = await self.capture_state()
            return {
                "success": False,
                "error": str(e),
                "clickPosition": {"x": x, "y": y},
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
            }
    
    async def type_and_track(
        self, 
        selector: str, 
        text: str,
        clear_first: bool = True,
        press_enter: bool = False,
        timeout: int = 30000,
    ) -> Dict[str, Any]:
        """
        Type into an element and track state changes.
        
        Args:
            selector: CSS selector for element to type into
            text: Text to type
            clear_first: Whether to clear existing text first
            press_enter: Whether to press Enter after typing
            timeout: Timeout in milliseconds
        
        Returns:
            Dictionary with state tracking info
        """
        state_before = await self.capture_state()
        
        try:
            locator = self.page.locator(selector)
            
            # Click to focus
            await locator.click(timeout=timeout)
            
            if clear_first:
                await locator.clear()
            
            await locator.type(text)
            
            if press_enter:
                await self.page.keyboard.press("Enter")
                # Wait for potential navigation
                await asyncio.sleep(0.3)
                try:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
            
            state_after = await self.capture_state()
            changes = state_after.diff(state_before)
            
            return {
                "success": True,
                "typed_text": text,
                "pressed_enter": press_enter,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": changes,
                "navigated": "url" in changes,
            }
            
        except Exception as e:
            logger.error(f"Type failed: {e}")
            state_after = await self.capture_state()
            return {
                "success": False,
                "error": str(e),
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": state_after.diff(state_before),
            }
    
    async def press_key_and_track(self, key: str) -> Dict[str, Any]:
        """
        Press a key and track state changes.
        
        Args:
            key: Key to press (e.g., "Enter", "Escape", "Tab")
        
        Returns:
            Dictionary with state tracking info
        """
        state_before = await self.capture_state()
        
        try:
            await self.page.keyboard.press(key)
            
            # Wait for potential navigation (especially for Enter)
            if key.lower() == "enter":
                await asyncio.sleep(0.3)
                try:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
            else:
                await asyncio.sleep(0.1)
            
            state_after = await self.capture_state()
            changes = state_after.diff(state_before)
            
            return {
                "success": True,
                "key": key,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": changes,
                "navigated": "url" in changes,
                "focus_changed": "focused_element" in changes,
            }
            
        except Exception as e:
            logger.error(f"Press key failed: {e}")
            state_after = await self.capture_state()
            return {
                "success": False,
                "error": str(e),
                "key": key,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": state_after.diff(state_before),
            }
    
    async def focus_and_track(self, selector: str, timeout: int = 30000) -> Dict[str, Any]:
        """
        Focus an element and track state changes.
        
        Args:
            selector: CSS selector for element to focus
            timeout: Timeout in milliseconds
        
        Returns:
            Dictionary with state tracking info
        """
        state_before = await self.capture_state()
        
        try:
            locator = self.page.locator(selector)
            await locator.focus(timeout=timeout)
            
            await asyncio.sleep(0.05)  # Brief wait for focus to settle
            
            state_after = await self.capture_state()
            changes = state_after.diff(state_before)
            
            return {
                "success": True,
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": changes,
                "focus_changed": "focused_element" in changes,
                "new_focus": state_after.focused_element,
            }
            
        except Exception as e:
            logger.error(f"Focus failed: {e}")
            state_after = await self.capture_state()
            return {
                "success": False,
                "error": str(e),
                "state_before": state_before.to_dict(),
                "state_after": state_after.to_dict(),
                "changes": state_after.diff(state_before),
            }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current state and recent history.
        
        Returns:
            Dictionary with current state summary and history stats
        """
        return {
            "has_state": self._last_state is not None,
            "last_state": self._last_state.to_dict() if self._last_state else None,
            "history_size": len(self._state_history),
            "is_navigating": self.is_navigating,
            "last_navigation_url": self._last_navigation_url,
        }

