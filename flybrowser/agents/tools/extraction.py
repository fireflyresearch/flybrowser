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
Data extraction tools for browser control.

This module provides ReAct-compatible tools for extracting data
from web pages including text, HTML, screenshots, and page state.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Optional

from flybrowser.agents.types import SafetyLevel, ToolCategory, ToolResult
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.tools.descriptions import get_tool_metadata

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.core.element import ElementDetector
    from flybrowser.agents.tools.page_analyzer import PageAnalyzer


class ExtractTextTool(BaseTool):
    """Extract text content from an element or the entire page."""
    
    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector
    
    @property
    def metadata(self) -> ToolMetadata:
        return get_tool_metadata("extract_text")
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute text extraction.
        
        If no selector is provided, extracts all visible text from the page body.
        """
        selector = kwargs.get("selector")
        
        try:
            page = self._page_controller.page
            
            if not selector:
                # Extract all visible text from the page with structure
                result = await page.evaluate("""
                    () => {
                        // Get page metadata
                        const title = document.title;
                        const url = window.location.href;
                        
                        // Extract main content text
                        const body = document.body;
                        
                        // Get headings for structure
                        const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                            .map(h => ({
                                level: h.tagName,
                                text: h.innerText.trim()
                            }))
                            .filter(h => h.text.length > 0)
                            .slice(0, 20);  // Limit to 20 headings
                        
                        // Get navigation links
                        const navLinks = Array.from(document.querySelectorAll('nav a, header a, [role="navigation"] a'))
                            .map(a => ({
                                text: a.innerText.trim(),
                                href: a.href
                            }))
                            .filter(l => l.text.length > 0 && l.text.length < 50)
                            .slice(0, 30);  // Limit to 30 nav links
                        
                        // Get main content paragraphs
                        const mainContent = Array.from(document.querySelectorAll('main p, article p, section p, .content p, #content p'))
                            .map(p => p.innerText.trim())
                            .filter(t => t.length > 20)  // Only meaningful paragraphs
                            .slice(0, 10);  // Limit to 10 paragraphs
                        
                        // Get footer links if present
                        const footerLinks = Array.from(document.querySelectorAll('footer a'))
                            .map(a => ({
                                text: a.innerText.trim(),
                                href: a.href
                            }))
                            .filter(l => l.text.length > 0 && l.text.length < 50)
                            .slice(0, 20);  // Limit to 20 footer links
                        
                        // Get all visible text (truncated)
                        const visibleText = (body.innerText || '').substring(0, 5000);
                        
                        return {
                            title,
                            url,
                            headings,
                            navLinks,
                            mainContent,
                            footerLinks,
                            visibleText
                        };
                    }
                """)
                
                # Format a structured summary
                summary_parts = [
                    f"Page: {result['title']}",
                    f"URL: {result['url']}",
                ]
                
                if result['headings']:
                    summary_parts.append("\n## Page Structure (Headings):")
                    for h in result['headings']:
                        indent = "  " * (int(h['level'][1]) - 1)
                        summary_parts.append(f"{indent}- {h['text']}")
                
                if result['navLinks']:
                    summary_parts.append("\n## Navigation Links:")
                    for link in result['navLinks']:
                        summary_parts.append(f"- {link['text']}: {link['href']}")
                
                if result['mainContent']:
                    summary_parts.append("\n## Main Content:")
                    for para in result['mainContent']:
                        summary_parts.append(f"- {para[:200]}..." if len(para) > 200 else f"- {para}")
                
                if result['footerLinks']:
                    summary_parts.append("\n## Footer Links:")
                    for link in result['footerLinks']:
                        summary_parts.append(f"- {link['text']}: {link['href']}")
                
                structured_text = "\n".join(summary_parts)
                
                return ToolResult.success_result(
                    data={
                        "text": structured_text,
                        "raw_text": result['visibleText'][:3000],  # Truncated raw text
                        "title": result['title'],
                        "url": result['url'],
                        "headings": result['headings'],
                        "navigation_links": result['navLinks'],
                        "main_content": result['mainContent'],
                        "footer_links": result['footerLinks'],
                        "selector": "body (full page)",
                        "message": f"Extracted structured content from '{result['title']}' - {len(result['headings'])} headings, {len(result['navLinks'])} nav links, {len(result['mainContent'])} content sections",
                    },
                )
            else:
                # Extract text from specific selector
                if self._element_detector:
                    text = await self._element_detector.get_text(selector)
                else:
                    text = await page.locator(selector).text_content() or ""
                
                return ToolResult.success_result(
                    data={
                        "text": text,
                        "selector": selector,
                        "message": f"Extracted text from {selector}",
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Text extraction failed: {str(e)}")


class ScreenshotTool(BaseTool):
    """Take a screenshot of the page."""
    
    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        return get_tool_metadata("screenshot")
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute screenshot capture with detailed feedback."""
        full_page = kwargs.get("full_page", False)
        
        try:
            page = self._page_controller.page
            
            # Get current page info for context
            page_info = await page.evaluate("""
                () => ({
                    url: window.location.href,
                    title: document.title,
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    },
                    scroll: {
                        x: window.pageXOffset || document.documentElement.scrollLeft,
                        y: window.pageYOffset || document.documentElement.scrollTop
                    },
                    pageSize: {
                        width: document.documentElement.scrollWidth,
                        height: document.documentElement.scrollHeight
                    }
                })
            """)
            
            # Capture screenshot
            screenshot_bytes = await self._page_controller.screenshot(full_page=full_page)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            
            size_kb = len(screenshot_bytes) / 1024
            capture_type = "full page" if full_page else "viewport"
            
            return ToolResult.success_result(
                data={
                    "screenshot_base64": screenshot_b64,
                    "full_page": full_page,
                    "size_bytes": len(screenshot_bytes),
                    "size_kb": round(size_kb, 1),
                    "page_url": page_info["url"],
                    "page_title": page_info["title"],
                    "viewport": page_info["viewport"],
                    "scroll_position": page_info["scroll"],
                    "page_size": page_info["pageSize"],
                    "message": f"Captured {capture_type} screenshot ({size_kb:.1f}KB) of '{page_info['title']}'",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Screenshot failed: {str(e)}")


class GetPageStateTool(BaseTool):
    """Get the current page state with optional LLM analysis."""
    
    def __init__(
        self,
        page_controller: PageController,
        page_analyzer: Optional["PageAnalyzer"] = None,
    ) -> None:
        self._page_controller = page_controller
        self._page_analyzer = page_analyzer
    
    @property
    def metadata(self) -> ToolMetadata:
        return get_tool_metadata("get_page_state")
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute page state retrieval with optional LLM fallback."""
        try:
            # Get heuristic state
            state = await self._page_controller.get_rich_state()
            
            links = state.get("links", [])
            buttons = state.get("buttons", [])
            hidden_links = state.get("hiddenLinks", [])
            content_sections = state.get("contentSections", [])
            
            # Prepare links summary for the agent
            nav_links = [l for l in links if l.get("isNav")]
            other_links = [l for l in links if not l.get("isNav")]
            
            # Check if LLM fallback is needed
            total_elements = len(buttons) + len(links)
            analysis_method = "heuristic"
            llm_cost = 0.0
            llm_suggestions = []
            
            if self._page_analyzer and total_elements < 5:
                # Use LLM analyzer as fallback
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[GET_PAGE_STATE] Only {total_elements} elements found, using LLM fallback")
                
                try:
                    llm_result = await self._page_analyzer.analyze_html(
                        page_controller=self._page_controller,
                        url=state.get("url", ""),
                        title=state.get("title", ""),
                        heuristic_found_count=total_elements,
                    )
                    
                    # Merge LLM results with heuristic results
                    if llm_result.all_elements:
                        # Convert LLM elements to heuristic format
                        llm_nav_links = [
                            {
                                "text": e.text,
                                "href": e.href or "",
                                "url": e.href or "",
                                "isNav": True,
                                "ariaLabel": e.aria_label,
                            }
                            for e in llm_result.get_navigation_links()
                        ]
                        
                        # Merge with existing links (deduplicate by href)
                        existing_hrefs = {l.get("href") for l in nav_links}
                        for llm_link in llm_nav_links:
                            if llm_link["href"] not in existing_hrefs:
                                nav_links.append(llm_link)
                        
                        analysis_method = "hybrid"
                        llm_cost = llm_result.cost_usd
                        llm_suggestions = llm_result.suggestions
                        
                        logger.info(
                            f"[GET_PAGE_STATE] LLM found {len(llm_result.all_elements)} additional elements "
                            f"(cost: ${llm_cost:.4f})"
                        )
                
                except Exception as e:
                    logger.warning(f"[GET_PAGE_STATE] LLM analysis failed: {e}")
            
            return ToolResult.success_result(
                data={
                    "url": state.get("url", ""),
                    "title": state.get("title", ""),
                    "viewport": state.get("viewport", {}),
                    "scroll_position": state.get("scrollPosition", {}),
                    "focused_element": state.get("focusedElement"),
                    "ready_state": state.get("readyState", ""),
                    "forms_count": len(state.get("forms", [])),
                    "inputs_count": len(state.get("inputs", [])),
                    "buttons": buttons,
                    "navigation_links": nav_links,
                    "other_links": other_links[:10],  # Limit for context
                    "hidden_links": hidden_links,
                    "links_count": len(nav_links) + len(other_links),
                    "content_preview": content_sections[0].get("preview", "") if content_sections else "",
                    "page_dimensions": state.get("pageDimensions", {}),
                    "analysis_method": analysis_method,
                    "llm_cost_usd": llm_cost,
                    "llm_suggestions": llm_suggestions,
                    "message": f"Retrieved page state for {state.get('url', 'unknown')} with {len(nav_links)} nav links and {len(other_links)} other links (detected by: {analysis_method})",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Get page state failed: {str(e)}")

