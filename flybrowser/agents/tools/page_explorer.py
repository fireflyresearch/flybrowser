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
PageExplorer Tool - Systematic Page Exploration

This tool provides comprehensive page exploration by:
- Scrolling through entire page incrementally
- Capturing annotated screenshots at each position
- Building spatial understanding with PageMap
- Supporting configurable exploration behavior
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import TYPE_CHECKING, Any, Optional

from flybrowser.agents.types import SafetyLevel, ToolCategory, ToolResult, OperationMode
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.page_map import (
    PageMap, ViewportInfo, ScrollPosition, ScreenshotCapture, SectionType, ExplorationScope
)
from flybrowser.agents.config import PageExplorationConfig

if TYPE_CHECKING:
    from flybrowser.core.page import PageController

logger = logging.getLogger(__name__)


class PageExplorerTool(BaseTool):
    """
    Systematic page exploration tool.
    
    Scrolls through entire page, captures screenshots at intervals,
    and builds comprehensive PageMap with spatial understanding.
    
    Usage:
        explore_page() -> Returns PageMap with screenshots and positions
    """
    
    def __init__(
        self,
        page_controller: PageController,
        config: Optional[PageExplorationConfig] = None
    ) -> None:
        """
        Initialize PageExplorer tool.
        
        Args:
            page_controller: Browser page controller
            config: Page exploration configuration (uses defaults if not provided)
        """
        self._page_controller = page_controller
        self.config = config or PageExplorationConfig()
        
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="page_explorer",
            description=(
                "Systematically explore entire webpage by scrolling incrementally "
                "and capturing annotated screenshots at each position. "
                "Returns comprehensive PageMap with spatial understanding of page structure. "
                "Use this for tasks requiring full page analysis or navigation discovery."
            ),
            category=ToolCategory.EXTRACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="max_screenshots",
                    type="integer",
                    description="Maximum screenshots to capture (overrides config)",
                    required=False,
                ),
                ToolParameter(
                    name="scroll_step_px",
                    type="integer",
                    description="Pixels per scroll step (overrides config)",
                    required=False,
                ),
            ],
            returns_description="PageMap object with all captured screenshots and positions",
        )
    
    async def execute(self, operation_mode: Optional[OperationMode] = None, **kwargs: Any) -> ToolResult:
        """
        Execute page exploration with mode-specific scope.
        
        Args:
            operation_mode: Operation mode to determine exploration scope
            max_screenshots: Optional override for max screenshots
            scroll_step_px: Optional override for scroll step size
            
        Returns:
            ToolResult with PageMap data
        """
        if not self.config.enabled:
            return ToolResult.error_result(
                "Page exploration is disabled in configuration",
                error_code="FEATURE_DISABLED"
            )
        
        start_time = time.time()
        
        # Default to AUTO mode if not specified
        if operation_mode is None:
            operation_mode = OperationMode.AUTO
        
        try:
            logger.info(
                f" [PageExplorer] Starting page exploration "
                f"(mode: {operation_mode.value})..."
            )
            
            # Determine exploration scope based on operation mode
            scope = self._determine_exploration_scope(operation_mode, kwargs)
            logger.info(f"[PageExplorer] Using {scope.value} exploration scope")
            
            # Get configuration based on scope
            max_screenshots, scroll_step_px = self._get_scope_config(scope, kwargs)
            
            # Step 1: Capture viewport info
            viewport = await self._capture_viewport_info()
            logger.info(
                f"[PageExplorer] Viewport: {viewport.width}x{viewport.height}px, "
                f"DPR: {viewport.device_pixel_ratio}"
            )
            
            # Step 2: Get page dimensions
            page_width, page_height = await self._calculate_page_dimensions()
            logger.info(
                f"[PageExplorer] Page dimensions: {page_width}x{page_height}px "
                f"(scrollable area)"
            )
            
            # Validate page height
            if page_height > self.config.max_page_height_px:
                logger.warning(
                    f"[PageExplorer] Page too tall ({page_height}px > "
                    f"{self.config.max_page_height_px}px), limiting exploration"
                )
                page_height = self.config.max_page_height_px
            
            # Step 3: Get current page state
            page_state = await self._page_controller.get_page_state()
            url = page_state.get("url", "")
            title = page_state.get("title", "Untitled")
            
            # Step 4: Extract DOM data for LLM analysis (before scrolling changes state)
            dom_data = await self._extract_dom_navigation_links()
            logger.info(
                f"[PageExplorer] Extracted {len(dom_data.get('all_links', []))} links, "
                f"{len(dom_data.get('interactive_elements', []))} interactive elements from DOM"
            )
            
            # Step 5: Create PageMap
            page_map = PageMap(
                url=url,
                title=title,
                viewport=viewport,
                total_height=page_height,
                total_width=page_width,
            )
            
            # Store DOM data in PageMap for LLM analysis
            page_map.dom_navigation_links = dom_data
            
            # Step 6: Scroll and capture screenshots
            screenshots = await self._scroll_and_capture(
                viewport=viewport,
                page_height=page_height,
                scroll_step_px=scroll_step_px,
                max_screenshots=max_screenshots,
            )
            
            page_map.screenshots = screenshots
            
            elapsed_ms = (time.time() - start_time) * 1000
            coverage = page_map.get_coverage_percentage()
            
            logger.info(
                f" [PageExplorer] Exploration complete: "
                f"{len(screenshots)} screenshots, "
                f"{coverage:.1f}% coverage, "
                f"{elapsed_ms:.0f}ms"
            )
            
            # Return PageMap as structured data
            # Note: ToolResult.success_result() uses **metadata, so we unpack directly
            return ToolResult.success_result(
                data=page_map.to_dict(include_images=False),
                message=(
                    f"Explored {title} - captured {len(screenshots)} screenshots "
                    f"covering {coverage:.1f}% of page"
                ),
                page_map=page_map,  # Include full object for internal use
                screenshots_captured=len(screenshots),
                coverage_percentage=coverage,
                exploration_time_ms=elapsed_ms,
            )
        
        except asyncio.TimeoutError:
            return ToolResult.error_result(
                f"Page exploration timeout after {self.config.timeout_seconds}s",
                error_code="TIMEOUT"
            )
        except Exception as e:
            logger.exception(f"[PageExplorer] Exploration failed: {e}")
            return ToolResult.error_result(
                f"Page exploration error: {str(e)}",
                error_code="EXPLORATION_ERROR"
            )
    
    def _determine_exploration_scope(
        self,
        operation_mode: OperationMode,
        kwargs: dict[str, Any]
    ) -> ExplorationScope:
        """
        Map operation mode to exploration scope.

        Args:
            operation_mode: Current operation mode
            kwargs: Additional parameters (may contain explicit scope override)

        Returns:
            ExplorationScope enum value
        """
        # Allow explicit scope override via kwargs
        if "exploration_scope" in kwargs:
            scope_value = kwargs["exploration_scope"]
            if isinstance(scope_value, ExplorationScope):
                return scope_value
            # Try to convert string to enum
            try:
                return ExplorationScope(scope_value)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid exploration_scope '{scope_value}', using mode-based default"
                )

        # Map operation mode to exploration scope
        scope_mapping = {
            OperationMode.NAVIGATE: ExplorationScope.FULL,
            OperationMode.EXECUTE: ExplorationScope.VIEWPORT,
            OperationMode.SCRAPE: ExplorationScope.CONTENT,
            OperationMode.RESEARCH: ExplorationScope.SMART,
            OperationMode.AUTO: ExplorationScope.SMART,
        }

        return scope_mapping.get(operation_mode, ExplorationScope.SMART)

    def _get_scope_config(
        self,
        scope: ExplorationScope,
        kwargs: dict[str, Any]
    ) -> tuple[int, int]:
        """
        Get max_screenshots and scroll_step_px based on exploration scope.

        Args:
            scope: Exploration scope
            kwargs: Additional parameters (may contain explicit overrides)

        Returns:
            Tuple of (max_screenshots, scroll_step_px)
        """
        # Allow explicit overrides via kwargs
        max_screenshots_override = kwargs.get("max_screenshots")
        scroll_step_override = kwargs.get("scroll_step_px")

        # If both are explicitly provided, use them
        if max_screenshots_override is not None and scroll_step_override is not None:
            return int(max_screenshots_override), int(scroll_step_override)

        # Scope-specific defaults
        if scope == ExplorationScope.FULL:
            # Comprehensive exploration: more screenshots, smaller scroll steps
            max_screenshots = 10
            scroll_step_px = 600
        elif scope == ExplorationScope.VIEWPORT:
            # Single viewport only: 1 screenshot, no scrolling
            max_screenshots = 1
            scroll_step_px = 0
        elif scope == ExplorationScope.CONTENT:
            # Moderate coverage: focus on main content areas
            max_screenshots = 5
            scroll_step_px = 800
        else:  # SMART (default)
            # Use configuration defaults for adaptive behavior
            max_screenshots = self.config.max_screenshots_per_page
            scroll_step_px = self.config.scroll_step_px

        # Apply individual overrides if provided
        if max_screenshots_override is not None:
            max_screenshots = int(max_screenshots_override)
        if scroll_step_override is not None:
            scroll_step_px = int(scroll_step_override)

        return max_screenshots, scroll_step_px

    async def _capture_viewport_info(self) -> ViewportInfo:
        """
        Capture current viewport dimensions and device pixel ratio.

        Returns:
            ViewportInfo with viewport details
        """
        viewport_data = await self._page_controller.page.evaluate("""
            () => {
                return {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    devicePixelRatio: window.devicePixelRatio || 1.0
                };
            }
        """)

        return ViewportInfo(
            width=viewport_data["width"],
            height=viewport_data["height"],
            device_pixel_ratio=viewport_data["devicePixelRatio"]
        )
    
    async def _calculate_page_dimensions(self) -> tuple[int, int]:
        """
        Calculate total scrollable page dimensions.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        dimensions = await self._page_controller.page.evaluate("""
            () => {
                return {
                    width: Math.max(
                        document.body.scrollWidth,
                        document.documentElement.scrollWidth,
                        document.body.offsetWidth,
                        document.documentElement.offsetWidth
                    ),
                    height: Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight,
                        document.body.offsetHeight,
                        document.documentElement.offsetHeight
                    )
                };
            }
        """)
        
        return dimensions["width"], dimensions["height"]
    
    async def _scroll_and_capture(
        self,
        viewport: ViewportInfo,
        page_height: int,
        scroll_step_px: int,
        max_screenshots: int,
    ) -> list[ScreenshotCapture]:
        """
        Scroll through page incrementally and capture screenshots.
        
        Args:
            viewport: Viewport information
            page_height: Total page height
            scroll_step_px: Pixels to scroll per step
            max_screenshots: Maximum screenshots to capture
            
        Returns:
            List of ScreenshotCapture objects with positions
        """
        screenshots = []
        screenshot_index = 0
        
        # Calculate scroll positions with overlap
        overlap_px = self.config.overlap_px
        effective_step = scroll_step_px - overlap_px
        
        # Start at top (Y=0)
        scroll_positions = [0]
        
        # Calculate intermediate positions
        current_y = 0
        while current_y + viewport.height < page_height:
            current_y += effective_step
            scroll_positions.append(current_y)
            
            if len(scroll_positions) >= self.config.max_scroll_steps:
                logger.warning(
                    f"[PageExplorer] Reached max scroll steps ({self.config.max_scroll_steps})"
                )
                break
        
        # Ensure we capture bottom of page if not already covered
        if scroll_positions[-1] + viewport.height < page_height:
            scroll_positions.append(max(0, page_height - viewport.height))
        
        # Limit to max screenshots
        scroll_positions = scroll_positions[:max_screenshots]
        
        logger.info(
            f"[PageExplorer] Planned {len(scroll_positions)} scroll positions "
            f"(step: {effective_step}px, overlap: {overlap_px}px)"
        )
        
        # Capture screenshot at each position
        for scroll_y in scroll_positions:
            if screenshot_index >= max_screenshots:
                logger.info(f"[PageExplorer] Reached max screenshots ({max_screenshots})")
                break
            
            try:
                # Scroll to position
                await self._page_controller.page.evaluate(
                    f"window.scrollTo(0, {scroll_y})"
                )
                
                # Wait for content to load/settle
                await asyncio.sleep(self.config.scroll_delay_ms / 1000.0)
                
                # Get actual scroll position (may differ slightly)
                actual_position = await self._page_controller.page.evaluate("""
                    () => {
                        return {
                            x: window.pageXOffset || document.documentElement.scrollLeft,
                            y: window.pageYOffset || document.documentElement.scrollTop
                        };
                    }
                """)
                
                # Capture screenshot
                screenshot_bytes = await self._page_controller.screenshot(
                    full_page=self.config.capture_full_page
                )
                
                # Calculate visible area
                visible_area = {
                    "top": actual_position["y"],
                    "bottom": min(
                        actual_position["y"] + viewport.height,
                        page_height
                    ),
                    "left": actual_position["x"],
                    "right": actual_position["x"] + viewport.width,
                }
                
                # Create ScreenshotCapture
                screenshot = ScreenshotCapture(
                    index=screenshot_index,
                    scroll_position=ScrollPosition(
                        x=actual_position["x"],
                        y=actual_position["y"]
                    ),
                    image_data=screenshot_bytes,
                    image_size_bytes=len(screenshot_bytes),
                    visible_area=visible_area,
                )
                
                screenshots.append(screenshot)
                
                logger.debug(
                    f"[PageExplorer] Screenshot {screenshot_index}: "
                    f"Y={actual_position['y']}px, "
                    f"size={len(screenshot_bytes)//1024}KB, "
                    f"showing {visible_area['top']}-{visible_area['bottom']}px"
                )
                
                screenshot_index += 1
                
            except Exception as e:
                logger.warning(
                    f"[PageExplorer] Failed to capture screenshot at Y={scroll_y}: {e}"
                )
                continue
        
        # Scroll back to top
        await self._page_controller.page.evaluate("window.scrollTo(0, 0)")
        
        return screenshots
    
    async def _extract_dom_navigation_links(self) -> dict:
        """
        Extract ALL links from DOM without hardcoded assumptions.
        
        This captures raw link data that the LLM/VLM can then intelligently
        analyze and categorize based on context, not hardcoded selectors.
        
        Returns:
            Dictionary with all links and raw HTML context for LLM analysis
        """
        try:
            dom_data = await self._page_controller.page.evaluate("""
                () => {
                    const origin = window.location.origin;
                    
                    // Get ALL anchor elements on the page
                    const allLinks = Array.from(document.querySelectorAll('a[href]'))
                        .map(a => {
                            const rect = a.getBoundingClientRect();
                            const href = a.href || '';
                            return {
                                text: (a.innerText || a.textContent || '').trim().substring(0, 100),
                                href: href,
                                ariaLabel: a.getAttribute('aria-label') || '',
                                title: a.getAttribute('title') || '',
                                // Let LLM determine what's internal vs external
                                isInternal: href.startsWith(origin) || href.startsWith('/') || href.startsWith('#'),
                                isAnchor: href.includes('#') && !href.startsWith(origin + '/#'),
                                isVisible: a.offsetParent !== null && rect.width > 0 && rect.height > 0,
                                // Position context for LLM to understand layout
                                position: {
                                    top: Math.round(rect.top + window.scrollY),
                                    inViewport: rect.top >= 0 && rect.top < window.innerHeight
                                },
                                // Parent context helps LLM understand where link is
                                parentTag: a.parentElement?.tagName?.toLowerCase() || '',
                                grandparentTag: a.parentElement?.parentElement?.tagName?.toLowerCase() || ''
                            };
                        })
                        .filter(l => l.href && (l.text.length > 0 || l.ariaLabel.length > 0));
                    
                    // Get ALL buttons that might be menus/toggles (for LLM to analyze)
                    const interactiveElements = Array.from(document.querySelectorAll('button, [role="button"]'))
                        .map(el => {
                            const rect = el.getBoundingClientRect();
                            return {
                                text: (el.innerText || el.textContent || '').trim().substring(0, 50),
                                ariaLabel: el.getAttribute('aria-label') || '',
                                ariaExpanded: el.getAttribute('aria-expanded'),
                                ariaHaspopup: el.getAttribute('aria-haspopup'),
                                className: el.className || '',
                                isVisible: el.offsetParent !== null && rect.width > 0,
                                position: { top: Math.round(rect.top + window.scrollY) }
                            };
                        })
                        .filter(el => el.text.length > 0 || el.ariaLabel.length > 0)
                        .slice(0, 30);  // Limit for token efficiency
                    
                    // Get simplified HTML structure of header/nav areas for LLM context
                    const getSimplifiedHTML = (selector) => {
                        const el = document.querySelector(selector);
                        if (!el) return null;
                        // Get outer HTML but limit size
                        const html = el.outerHTML;
                        return html.length > 2000 ? html.substring(0, 2000) + '...' : html;
                    };
                    
                    return {
                        // Raw links for LLM to analyze and categorize
                        all_links: allLinks.slice(0, 100),  // Limit for tokens
                        // Interactive elements that might reveal hidden content
                        interactive_elements: interactiveElements,
                        // Page metadata
                        page_info: {
                            url: window.location.href,
                            origin: origin,
                            title: document.title,
                            hasHeader: !!document.querySelector('header'),
                            hasNav: !!document.querySelector('nav'),
                            hasFooter: !!document.querySelector('footer'),
                            totalLinks: document.querySelectorAll('a[href]').length
                        },
                        // Minimal HTML context for navigation areas (LLM can understand structure)
                        html_context: {
                            header: getSimplifiedHTML('header'),
                            nav: getSimplifiedHTML('nav')
                        }
                    };
                }
            """)
            return dom_data
        except Exception as e:
            logger.warning(f"[PageExplorer] Failed to extract DOM data: {e}")
            return {
                'all_links': [],
                'interactive_elements': [],
                'page_info': {},
                'html_context': {}
            }
