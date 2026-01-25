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
Intelligent page analysis using LLM to detect interactive elements.

This module provides LLM-powered analysis of HTML structure to identify
buttons, links, menus, and other interactive elements that heuristics miss.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from flybrowser.types.page_analysis import (
    AnalysisMethod,
    AnalysisResult,
    ElementPurpose,
    ElementType,
    InteractiveElement,
)
from flybrowser.agents.config import PageAnalysisConfig
from flybrowser.agents.structured_llm import StructuredLLMWrapper

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.llm.base import BaseLLMProvider
    from flybrowser.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class PageAnalyzer:
    """
    Intelligent page analyzer using LLM for element detection.
    
    Uses LLM to analyze HTML structure and identify interactive elements
    that JavaScript heuristics might miss (hidden menus, custom components, etc.).
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        prompt_manager: PromptManager,
        config: Optional[PageAnalysisConfig] = None,
    ):
        """
        Initialize the page analyzer.
        
        Args:
            llm_provider: LLM provider for HTML analysis
            prompt_manager: Prompt manager for loading analysis templates
            config: Configuration for analysis behavior
        """
        self.llm = llm_provider
        self.prompt_manager = prompt_manager
        self.config = config or PageAnalysisConfig()
        self._cache: Dict[str, tuple[AnalysisResult, float]] = {}  # url -> (result, timestamp)
    
    async def analyze_html(
        self,
        page_controller: PageController,
        url: str,
        title: str,
        heuristic_found_count: Optional[int] = None,
    ) -> AnalysisResult:
        """
        Analyze HTML structure to find interactive elements.
        
        Args:
            page_controller: Page controller to get HTML from
            url: Current page URL
            title: Page title
            heuristic_found_count: Number of elements found by heuristics (for context)
            
        Returns:
            AnalysisResult with detected elements and metadata
        """
        start_time = time.time()
        
        # Check cache
        if self.config.prefer_cached_results:
            cached = self._get_cached_result(url)
            if cached:
                logger.info(f"[HTML ANALYZER] Using cached result for {url}")
                return cached
        
        try:
            # Get HTML content (focus on header/nav for performance)
            html_content = await self._extract_relevant_html(page_controller)
            
            if not html_content.strip():
                logger.warning("[HTML ANALYZER] No HTML content to analyze")
                return AnalysisResult(
                    method=AnalysisMethod.LLM_HTML,
                    warnings=["No HTML content available for analysis"]
                )
            
            # Prepare prompt
            prompt_vars = {
                "url": url,
                "title": title,
                "html_content": html_content,
            }
            
            if heuristic_found_count is not None:
                prompt_vars["heuristic_found_elements"] = heuristic_found_count
            
            # Get prompt from manager
            prompts = self.prompt_manager.get_prompt("html_analysis", **prompt_vars)
            
            # Define schema for HTML analysis response
            html_analysis_schema = {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "purpose": {"type": "string"},
                                "text": {"type": "string"},
                                "selector": {"type": "string"},
                                "confidence": {"type": "number"},
                                "reasoning": {"type": "string"},
                                "is_visible": {"type": "boolean"},
                                "href": {"type": "string"},
                                "aria_label": {"type": "string"},
                                "attributes": {"type": "object"}
                            },
                            "required": ["type", "selector"]
                        },
                        "description": "Interactive elements found on the page"
                    },
                    "analysis_summary": {
                        "type": "object",
                        "properties": {
                            "warnings": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "suggestions": {
                                "type": "array", 
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["elements"]
            }
            
            # Call LLM with structured output
            logger.info(f"[HTML ANALYZER] Analyzing HTML for {url} ({len(html_content)} chars)")
            
            if self.config.log_llm_prompts:
                logger.debug(f"[HTML ANALYZER] Prompt:\n{prompts['user'][:500]}...")
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            wrapper = StructuredLLMWrapper(self.llm, max_repair_attempts=2)
            
            try:
                data = await wrapper.generate_structured(
                    prompt=prompts["user"],
                    schema=html_analysis_schema,
                    system_prompt=prompts["system"],
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
            except ValueError as e:
                logger.error(f"[HTML ANALYZER] Structured output failed: {e}")
                return AnalysisResult(
                    method=AnalysisMethod.LLM_HTML,
                    warnings=[f"LLM analysis failed: {str(e)}"],
                    analysis_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Parse the structured response directly
            result = self._parse_structured_response(data)
            
            # Add performance metrics
            analysis_time = (time.time() - start_time) * 1000
            result.analysis_time_ms = analysis_time
            # Token tracking not available from structured wrapper - estimate from HTML size
            result.token_count = len(html_content) // 4  # Rough estimate: ~4 chars per token
            result.cost_usd = self._estimate_cost(result.token_count)
            result.method = AnalysisMethod.LLM_HTML
            result.methods_used = [AnalysisMethod.LLM_HTML]
            
            logger.info(
                f"[HTML ANALYZER] Found {len(result.all_elements)} elements "
                f"in {analysis_time:.0f}ms (${result.cost_usd:.4f})"
            )
            
            # Cache result
            if self.config.prefer_cached_results:
                self._cache_result(url, result)
            
            return result
            
        except Exception as e:
            logger.error(f"[HTML ANALYZER] Analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                method=AnalysisMethod.LLM_HTML,
                warnings=[f"LLM analysis failed: {str(e)}"],
                analysis_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def _extract_relevant_html(self, page_controller: PageController) -> str:
        """
        Extract relevant HTML sections for analysis.
        
        Focuses on header, nav, and main to reduce token count.
        """
        try:
            # Get focused HTML (header + nav + visible content)
            html = await page_controller.page.evaluate("""
                () => {
                    const parts = [];
                    
                    // Get header
                    const header = document.querySelector('header, [role="banner"]');
                    if (header) parts.push(header.outerHTML);
                    
                    // Get all nav elements
                    const navs = document.querySelectorAll('nav, [role="navigation"]');
                    navs.forEach(nav => parts.push(nav.outerHTML));
                    
                    // Get menu buttons/toggles (common selectors)
                    const menuSelectors = [
                        '.menu-toggle', '.hamburger', '.nav-toggle', '.mobile-menu',
                        '[aria-label*="menu" i]', '[aria-label*="navigation" i]',
                        'button[aria-expanded]'
                    ];
                    menuSelectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            if (!parts.some(p => p.includes(el.outerHTML))) {
                                parts.push(el.outerHTML);
                            }
                        });
                    });
                    
                    // Get main content area (first 1000 chars)
                    const main = document.querySelector('main, [role="main"], #main, .main');
                    if (main) {
                        const mainHtml = main.outerHTML;
                        parts.push(mainHtml.substring(0, 1000));
                    }
                    
                    return parts.join('\\n');
                }
            """)
            
            # Limit total size for token efficiency
            max_chars = 15000
            if len(html) > max_chars:
                logger.debug(f"[HTML ANALYZER] Truncating HTML from {len(html)} to {max_chars} chars")
                html = html[:max_chars] + "\n<!-- ... truncated ... -->"
            
            return html
            
        except Exception as e:
            logger.error(f"[HTML ANALYZER] Failed to extract HTML: {e}")
            # Fallback: get full HTML
            return await page_controller.get_html()
    
    def _parse_structured_response(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Parse structured LLM response (already validated JSON dict) into AnalysisResult.
        
        Args:
            data: Validated JSON dictionary from StructuredLLMWrapper
            
        Returns:
            AnalysisResult with parsed elements
        """
        # Parse elements
        elements = []
        for elem_data in data.get("elements", []):
            try:
                element = self._parse_element(elem_data)
                elements.append(element)
            except Exception as e:
                logger.warning(f"[HTML ANALYZER] Failed to parse element: {e}")
                continue
        
        # Categorize elements
        buttons = [e for e in elements if e.element_type in [ElementType.BUTTON, ElementType.MENU_TOGGLE]]
        links = [e for e in elements if e.element_type == ElementType.LINK]
        menu_controls = [e for e in elements if e.purpose == ElementPurpose.MENU_CONTROL]
        forms = [e for e in elements if e.element_type in [ElementType.FORM_INPUT, ElementType.FORM_SUBMIT]]
        
        # Get summary data
        summary = data.get("analysis_summary", {})
        
        result = AnalysisResult(
            buttons=buttons,
            links=links,
            menu_controls=menu_controls,
            forms=forms,
            all_elements=elements,
            overall_confidence=self._calculate_confidence(elements),
            warnings=summary.get("warnings", []) if isinstance(summary, dict) else [],
            suggestions=summary.get("suggestions", []) if isinstance(summary, dict) else [],
        )
        
        return result
    
    def _parse_llm_response(self, response_text: str) -> AnalysisResult:
        """
        Parse LLM JSON response into AnalysisResult.
        
        Handles JSON extraction and validation.
        DEPRECATED: Use _parse_structured_response with StructuredLLMWrapper instead.
        """
        try:
            # Try to extract JSON from response
            json_text = self._extract_json(response_text)
            data = json.loads(json_text)
            
            return self._parse_structured_response(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"[HTML ANALYZER] Failed to parse JSON: {e}")
            logger.debug(f"[HTML ANALYZER] Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"[HTML ANALYZER] Failed to parse response: {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other content."""
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        
        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            return text[start:end]
        
        return text
    
    def _parse_element(self, data: Dict[str, Any]) -> InteractiveElement:
        """Parse element data from LLM response."""
        # Map string types to enums
        element_type_str = data.get("type", "unknown").lower()
        type_mapping = {
            "button": ElementType.BUTTON,
            "link": ElementType.LINK,
            "menu_toggle": ElementType.MENU_TOGGLE,
            "menu": ElementType.MENU,
            "form_input": ElementType.FORM_INPUT,
            "form_submit": ElementType.FORM_SUBMIT,
        }
        element_type = type_mapping.get(element_type_str, ElementType.UNKNOWN)
        
        purpose_str = data.get("purpose", "unknown").lower()
        purpose_mapping = {
            "navigation": ElementPurpose.NAVIGATION,
            "language_switch": ElementPurpose.LANGUAGE_SWITCH,
            "menu_control": ElementPurpose.MENU_CONTROL,
            "search": ElementPurpose.SEARCH,
            "login": ElementPurpose.LOGIN,
        }
        purpose = purpose_mapping.get(purpose_str, ElementPurpose.UNKNOWN)
        
        return InteractiveElement(
            element_id=data.get("id", "unknown"),
            element_type=element_type,
            purpose=purpose,
            text=data.get("text", ""),
            aria_label=data.get("aria_label"),
            selector=data.get("selector"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            is_visible=data.get("is_visible", True),
            href=data.get("href"),
            attributes=data.get("attributes", {}),
            detected_by=AnalysisMethod.LLM_HTML,
        )
    
    def _calculate_confidence(self, elements: List[InteractiveElement]) -> float:
        """Calculate overall confidence from element confidences."""
        if not elements:
            return 0.0
        
        return sum(e.confidence for e in elements) / len(elements)
    
    def _estimate_cost(self, token_count: int) -> float:
        """Estimate cost based on token count."""
        # Rough estimate: $0.20 per 1M tokens for GPT-4o-mini
        return (token_count / 1_000_000) * 0.20
    
    def _get_cached_result(self, url: str) -> Optional[AnalysisResult]:
        """Get cached result if still valid."""
        if url not in self._cache:
            return None
        
        result, timestamp = self._cache[url]
        age = time.time() - timestamp
        
        if age > self.config.cache_ttl_seconds:
            del self._cache[url]
            return None
        
        return result
    
    def _cache_result(self, url: str, result: AnalysisResult) -> None:
        """Cache analysis result."""
        self._cache[url] = (result, time.time())
        
        # Limit cache size
        if len(self._cache) > 50:
            # Remove oldest entries
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:10]:
                del self._cache[key]
