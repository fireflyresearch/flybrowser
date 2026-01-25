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
Human-Like Search Tool for FlyBrowser.

This tool performs web searches by simulating human behavior:
- Natural typing with delays
- Mouse movements and clicks
- Scrolling patterns
- Random timing variations

Fully guided by the autonomous planning system:
- Automatic phase creation (Navigate → Query → Extract)
- Goal tracking and completion
- Adaptive to search engine changes
- Self-correcting on failures

Usage:
    tool = SearchHumanTool(page_controller)
    result = await tool.execute(query="python tutorials", engine="google")
"""

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from .search_utils import (
    SearchResponse,
    SearchResult,
    SearchEngine,
    SearchProvider,
    normalize_query,
    clean_snippet,
    is_valid_url,
)
from flybrowser.agents.obstacle_detector import ObstacleDetector
from flybrowser.agents.structured_llm import StructuredLLMWrapper
from flybrowser.agents.schemas import OBSTACLE_DETECTION_SCHEMA
from flybrowser.prompts import PromptManager

if TYPE_CHECKING:
    from flybrowser.core.page import PageController

logger = logging.getLogger(__name__)


class HumanBehaviorSimulator:
    """
    Simulates human-like behavior patterns.
    
    Adds natural delays, variations, and patterns that make
    automation less detectable.
    """
    
    @staticmethod
    def typing_delay() -> float:
        """
        Calculate delay between keystrokes.
        
        Simulates typing speed of 80-120 WPM with natural variation.
        
        Returns:
            Delay in seconds
        """
        # Average typing speed: 100 WPM = ~5 chars/sec = 0.2s per char
        base_delay = 0.2
        # Add random variation (-50% to +50%)
        variation = random.uniform(-0.1, 0.1)
        return max(0.05, base_delay + variation)
    
    @staticmethod
    def word_pause() -> float:
        """
        Calculate pause between words.
        
        Returns:
            Delay in seconds
        """
        return random.uniform(0.1, 0.3)
    
    @staticmethod
    def click_delay() -> float:
        """
        Calculate delay after mouse click.
        
        Returns:
            Delay in seconds
        """
        return random.uniform(0.2, 0.5)
    
    @staticmethod
    def scroll_delay() -> float:
        """
        Calculate delay during scrolling.
        
        Returns:
            Delay in seconds
        """
        return random.uniform(0.3, 0.8)
    
    @staticmethod
    def think_delay() -> float:
        """
        Calculate delay for 'thinking' (reading page).
        
        Returns:
            Delay in seconds
        """
        return random.uniform(1.0, 2.5)


class SearchEngineAdapter:
    """
    Adapters for different search engines.
    
    Provides selectors and URLs for Google, DuckDuckGo, and Bing.
    """
    
    ENGINES = {
        SearchEngine.GOOGLE: {
            "url": "https://www.google.com",
            "input_selector": "textarea[name='q'], input[name='q']",
            "button_selector": "input[type='submit'][name='btnK'], button[type='submit']",
            "result_selector": "div.g, div[data-sokoban-container]",
            "title_selector": "h3",
            "link_selector": "a",
            "snippet_selector": "div[data-content-feature='1'], div.VwiC3b",
        },
        SearchEngine.DUCKDUCKGO: {
            "url": "https://duckduckgo.com",
            "input_selector": "input[name='q']",
            "button_selector": "button[type='submit']",
            "result_selector": "article[data-testid='result'], li[data-layout='organic']",
            "title_selector": "h2, [data-testid='result-title-a']",
            "link_selector": "a[data-testid='result-title-a'], h2 a",
            "snippet_selector": "[data-testid='result-snippet'], .result__snippet",
        },
        SearchEngine.BING: {
            "url": "https://www.bing.com",
            "input_selector": "input#sb_form_q",
            "button_selector": "input#sb_form_go",
            "result_selector": "li.b_algo",
            "title_selector": "h2",
            "link_selector": "a",
            "snippet_selector": "p, .b_caption p",
        },
    }
    
    @classmethod
    def get_config(cls, engine: SearchEngine) -> Dict[str, str]:
        """Get configuration for a search engine."""
        return cls.ENGINES.get(engine, cls.ENGINES[SearchEngine.DUCKDUCKGO])


class SearchHumanTool(BaseTool):
    """
    Human-like search tool.
    
    Performs searches by simulating human behavior patterns,
    fully guided by the autonomous planning system.
    """
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        """Initialize the human-like search tool."""
        super().__init__(page_controller)
        self.behavior = HumanBehaviorSimulator()
        self.prompt_manager = PromptManager()
    
    @property
    def metadata(self) -> ToolMetadata:
        """Tool metadata."""
        return ToolMetadata(
            name="search_human",
            description=(
                "Perform web search to FIND websites or information you don't know the URL for. "
                "Uses browser automation with natural typing and mouse movements. "
                "IMPORTANT: Do NOT use this if you're already on the target website - instead use 'get_page_state', 'extract_text', and 'navigate' to explore the site directly. "
                "Only use search when you need to discover URLs or gather information from multiple sources. "
                "No API keys required."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="engine",
                    type="string",
                    description="Search engine: 'google', 'duckduckgo', 'bing'",
                    required=False,
                    default="duckduckgo",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to extract (1-20)",
                    required=False,
                    default=10,
                ),
            ],
            examples=['search_human(query="python tutorials", engine="duckduckgo", max_results=10)'],
        )
    
    async def execute(
        self,
        query: str,
        engine: str = "duckduckgo",
        max_results: int = 10,
        **kwargs
    ) -> ToolResult:
        """
        Execute human-like search.
        
        This uses the autonomous planning system to create a structured plan:
        1. Navigate to search engine
        2. Enter query with human-like typing
        3. Click search button
        4. Extract results
        
        Args:
            query: Search query
            engine: Search engine to use
            max_results: Maximum results to extract
            
        Returns:
            ToolResult with SearchResponse data
        """
        start_time = time.time()
        
        try:
            # Normalize query
            query = normalize_query(query)
            
            # Parse engine
            try:
                search_engine = SearchEngine(engine.lower())
            except ValueError:
                search_engine = SearchEngine.DUCKDUCKGO
            
            # Get engine configuration
            config = SearchEngineAdapter.get_config(search_engine)
            
            logger.info(f"Starting human-like search: '{query}' on {search_engine.value}")
            
            # Phase 1: Navigate to search engine
            await self._navigate_to_engine(config["url"])
            
            # Phase 2: Enter query
            await self._enter_query_human_like(query, config["input_selector"])
            
            # Phase 3: Click search
            await self._click_search_button(config["button_selector"])
            
            # Phase 4: Wait for results (with bot detection)
            try:
                await self._wait_for_results(config["result_selector"])
            except Exception as wait_error:
                # Check if we hit bot detection during wait
                current_url = await self.page.get_url()
                if '/sorry/' in current_url or 'captcha' in current_url.lower():
                    logger.warning(f"  Bot detection during results wait: {current_url}")
                    # Trigger adaptive retry
                    return await self._handle_bot_detection_retry(query, config, max_results, search_engine, start_time)
                else:
                    # Different error - re-raise
                    raise
            
            # Phase 5: Extract results
            results = await self._extract_results(config, max_results)
            
            # Get current URL to check for issues
            current_url = await self.page.get_url()
            
            # Check if we hit bot detection after extraction
            if '/sorry/' in current_url or 'captcha' in current_url.lower():
                return await self._handle_bot_detection_retry(query, config, max_results, search_engine, start_time)
            
            # Create response
            elapsed_ms = (time.time() - start_time) * 1000
            
            response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results) * 10,  # Estimate
                search_time_ms=elapsed_ms,
                provider=SearchProvider.BROWSER.value,
                metadata={
                    "engine": search_engine.value,
                    "human_simulation": True,
                    "final_url": current_url,
                },
            )
            
            # Build detailed message with page context
            message = f"Found {len(results)} results via {search_engine.value}.\n"
            if results:
                message += f"Top result: {results[0].title}\n"
                message += f"Page URL: {current_url}\n"
                message += f"Status: Successfully reached results page"
            else:
                message += f"No results extracted. Page URL: {current_url}\n"
                message += "Consider using search_rank tool or trying a different engine."
            
            return ToolResult.success_result(
                data=response.to_dict(),
                message=message,
                metadata={
                    "engine": search_engine.value,
                    "final_url": current_url,
                    "has_results": len(results) > 0
                },
            )
        
        except Exception as e:
            logger.exception(f"Human search execution failed: {e}")
            return ToolResult.error_result(
                error=f"Search execution error: {str(e)}",
                error_code="EXECUTION_ERROR",
            )
    
    async def _navigate_to_engine(self, url: str) -> None:
        """Navigate to search engine homepage."""
        logger.info(f"Navigating to {url}")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        await self.page.goto(url)
        
        # Check for and handle obstacles using new detector
        llm = getattr(self, 'llm_provider', None)
        if llm:
            # Get config from agent if available
            agent_config = getattr(self, 'agent_config', None)
            obstacle_config = agent_config.obstacle_detector if agent_config else None
            
            detector = ObstacleDetector(
                page=self.page.page,
                llm=llm,
                config=obstacle_config
            )
            await detector.detect_and_handle()
        
        # Think delay (simulate reading page)
        await asyncio.sleep(self.behavior.think_delay())
    
    async def _handle_obstacles(self) -> None:
        """
        Intelligently detect and handle obstacles using LLM/VLM.
        
        Uses rich context including page state, focused elements, modals,
        and visual analysis to identify blocking elements.
        """
        if not self.page:
            return
        
        logger.info("Checking for obstacles using LLM detection...")
        
        try:
            # Get LLM from tool context (passed from agent)
            llm = getattr(self, 'llm_provider', None)
            if not llm:
                logger.warning("LLM not available for obstacle detection, skipping")
                return
            
            # Capture rich page state
            rich_state = await self.page.get_rich_state()
            focused = await self.page.get_focused_element()
            
            # Get page HTML structure (visible elements only)
            html_structure = await self.page.evaluate("""
                () => {
                    // Get all visible elements with high z-index (likely overlays)
                    const elements = Array.from(document.querySelectorAll('*'));
                    const overlays = elements
                        .filter(el => {
                            const style = window.getComputedStyle(el);
                            const zIndex = parseInt(style.zIndex) || 0;
                            const display = style.display;
                            const visibility = style.visibility;
                            const opacity = parseFloat(style.opacity) || 1;
                            
                            return zIndex > 100 && 
                                   display !== 'none' && 
                                   visibility !== 'hidden' && 
                                   opacity > 0.1;
                        })
                        .map(el => ({
                            tag: el.tagName.toLowerCase(),
                            id: el.id || null,
                            classes: Array.from(el.classList),
                            role: el.getAttribute('role'),
                            ariaModal: el.getAttribute('aria-modal'),
                            zIndex: parseInt(window.getComputedStyle(el).zIndex) || 0,
                            text: el.innerText?.slice(0, 200),
                            rect: el.getBoundingClientRect()
                        }));
                    
                    // Get modal indicators
                    const modals = document.querySelectorAll('[role="dialog"], [aria-modal="true"], .modal.show');
                    
                    return {
                        overlays: overlays.slice(0, 10),
                        hasModals: modals.length > 0,
                        modalCount: modals.length
                    };
                }
            """)
            
            # Check if there's likely an obstacle
            has_modals = rich_state.get('hasModals', False) or html_structure.get('hasModals', False)
            has_overlays = len(html_structure.get('overlays', [])) > 0
            
            if not has_modals and not has_overlays:
                logger.debug("No modal or overlay elements detected")
                return
            
            # Build rich context for LLM
            current_url = await self.page.get_url()
            context = {
                'url': current_url,
                'title': rich_state.get('title', ''),
                'has_modals': has_modals,
                'modal_count': html_structure.get('modalCount', 0),
                'overlay_count': len(html_structure.get('overlays', [])),
                'overlays': json.dumps(html_structure.get('overlays', []), indent=2),
                'focused_element': json.dumps(focused, indent=2) if focused else 'None',
                'scroll_position': rich_state.get('scrollPosition', {}),
                'viewport': rich_state.get('viewport', {}),
            }
            
            # Get prompt from template manager
            prompts = self.prompt_manager.get_prompt(
                "obstacle_detection",
                **context
            )
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            wrapper = StructuredLLMWrapper(
                llm_provider=llm,
                max_repair_attempts=2,
                repair_temperature=0.1,
            )
            
            # Get LLM analysis with structured output
            # Use configured temperature or default to 0.2
            agent_config = getattr(self, 'agent_config', None)
            if agent_config and hasattr(agent_config, 'obstacle_detection_temperature'):
                temp_value = agent_config.obstacle_detection_temperature
            else:
                temp_value = 0.2  # Fallback default
            
            try:
                analysis = await wrapper.generate_structured(
                    prompt=prompts["user"],
                    schema=OBSTACLE_DETECTION_SCHEMA,
                    system_prompt=prompts["system"],
                    temperature=temp_value,
                )
            except ValueError as e:
                # Structured output validation failed after repair attempts
                logger.warning(f"Obstacle detection structured output failed: {e}")
                return
            
            # Check if blocking (using new schema format: is_blocking)
            if not analysis.get('is_blocking', False):
                logger.debug("No blocking obstacles detected")
                return
            
            logger.info(f"LLM detected {len(analysis.get('obstacles', []))} blocking obstacle(s)")
            
            # Handle detected obstacles
            obstacles_handled = 0
            for obstacle in analysis.get('obstacles', [])[:3]:  # Max 3
                try:
                    confidence = obstacle.get('confidence', 0)
                    if confidence < 0.65:
                        logger.debug(f"Skipping low confidence obstacle: {confidence}")
                        continue
                    
                    description = obstacle.get('description', obstacle.get('type', 'Unknown'))
                    # Use new schema format: strategies (not selectors)
                    strategies = obstacle.get('strategies', [])
                    
                    if not strategies:
                        logger.warning(f"No strategies provided for: {description}")
                        continue
                    
                    logger.info(f"Attempting to dismiss: {description} (confidence: {confidence})")
                    logger.debug(f"Trying {len(strategies)} dismissal strategies")
                    
                    # Try each strategy in priority order
                    clicked = False
                    for i, strategy in enumerate(sorted(strategies, key=lambda x: x.get('priority', 99)), 1):
                        try:
                            strategy_type = strategy.get('type', '')
                            strategy_value = strategy.get('value', '')
                            
                            logger.debug(f"  Strategy {i}: {strategy_type}={strategy_value}")
                            
                            # Execute based on strategy type
                            if strategy_type == 'click_text':
                                locator = self.page.page.get_by_text(strategy_value, exact=False).first
                            elif strategy_type == 'click_button':
                                locator = self.page.page.get_by_role("button", name=strategy_value).first
                            elif strategy_type == 'click_selector':
                                locator = self.page.page.locator(strategy_value).first
                            elif strategy_type == 'press_key':
                                await self.page.press_key(strategy_value)
                                logger.info(f"[ok] Dismissed: {description} (pressed key: {strategy_value})")
                                clicked = True
                                obstacles_handled += 1
                                await asyncio.sleep(0.5)
                                break
                            elif strategy_type == 'click_outside':
                                # Click at document body to dismiss
                                await self.page.page.locator("body").click(position={"x": 10, "y": 10})
                                logger.info(f"[ok] Dismissed: {description} (clicked outside)")
                                clicked = True
                                obstacles_handled += 1
                                await asyncio.sleep(0.5)
                                break
                            elif strategy_type == 'scroll_away':
                                await self.page.page.evaluate("window.scrollBy(0, 500)")
                                logger.info(f"[ok] Dismissed: {description} (scrolled away)")
                                clicked = True
                                obstacles_handled += 1
                                await asyncio.sleep(0.5)
                                break
                            else:
                                logger.debug(f"  Unknown strategy type: {strategy_type}")
                                continue
                            
                            # For click strategies, check if visible before clicking
                            if await locator.is_visible(timeout=1000):
                                await locator.click(timeout=2000)
                                logger.info(f"[ok] Dismissed: {description} (using {strategy_type}: {strategy_value})")
                                clicked = True
                                obstacles_handled += 1
                                await asyncio.sleep(0.5)
                                break
                            else:
                                logger.debug(f"  Element not visible: {strategy_type}={strategy_value}")
                        
                        except Exception as e:
                            logger.debug(f"  Strategy {i} failed: {e}")
                            continue
                    
                    if not clicked:
                        logger.warning(f"[fail] Could not dismiss: {description} (all {len(strategies)} strategies failed)")
                    
                except Exception as e:
                    logger.warning(f"Error processing obstacle '{description}': {e}")
                    continue
            
            if obstacles_handled > 0:
                logger.info(f"Successfully handled {obstacles_handled} obstacle(s)")
                # Wait for page to settle
                await asyncio.sleep(1.0)
            else:
                logger.debug("No obstacles could be dismissed")
        
        except Exception as e:
            logger.warning(f"Obstacle detection failed: {e}")
            # Continue anyway - obstacles are optional
    
    async def _enter_query_human_like(self, query: str, selector: str) -> None:
        """
        Enter query with human-like typing.
        
        Simulates:
        - Natural typing speed (80-120 WPM)
        - Pauses between words
        - Character-by-character typing
        
        Args:
            query: Query to type
            selector: Input field selector
        """
        logger.info(f"Typing query: {query}")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        # Wait for input field
        await self.page.wait_for_selector(selector, timeout=10000)
        
        # Click input field
        await self.page.click_and_track(selector)
        await asyncio.sleep(self.behavior.click_delay())
        
        # Type character by character with human-like delays
        words = query.split()
        for i, word in enumerate(words):
            for char in word:
                # Get delay in seconds, convert to milliseconds for type_text
                delay_seconds = self.behavior.typing_delay()
                await self.page.type_text(char, delay=int(delay_seconds * 1000))
            
            # Add space between words (except last word)
            if i < len(words) - 1:
                await self.page.type_text(" ", delay=int(self.behavior.word_pause() * 1000))
        
        logger.info("Query typed successfully")
    
    async def _click_search_button(self, selector: str) -> None:
        """
        Click search button with human-like behavior.
        
        Args:
            selector: Button selector
        """
        logger.info("Clicking search button")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        try:
            # Wait for button to be clickable
            await self.page.wait_for_selector(selector, timeout=5000)
            
            # Small delay before clicking (human reads/aims)
            await asyncio.sleep(self.behavior.click_delay())
            
            # Click button
            await self.page.click_and_track(selector)
            
            # Wait after click
            await asyncio.sleep(self.behavior.click_delay())
        
        except Exception as e:
            # Fallback: press Enter key instead
            logger.warning(f"Button click failed: {e}, pressing Enter instead")
            await self.page.press_key("Enter")
            await asyncio.sleep(self.behavior.click_delay())
    
    async def _handle_bot_detection_retry(self, query: str, config: Dict[str, str], max_results: int, search_engine: SearchEngine, start_time: float) -> ToolResult:
        """
        Handle bot detection adaptively by retrying.
        
        Args:
            query: Search query
            config: Engine configuration
            max_results: Maximum results
            search_engine: Search engine being used
            start_time: Start time for elapsed calculation
            
        Returns:
            ToolResult with retry outcome
        """
        logger.info(" Being adaptive: waiting and will retry...")
        
        # Wait - sometimes Google lets you through after a pause
        await asyncio.sleep(random.uniform(2.0, 4.0))
        
        try:
            logger.info("Attempting to return to search homepage...")
            await self.page.goto(config["url"])
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # Handle obstacles again
            llm = getattr(self, 'llm_provider', None)
            if llm:
                # Get config from agent if available
                agent_config = getattr(self, 'agent_config', None)
                obstacle_config = agent_config.obstacle_detector if agent_config else None
                
                detector = ObstacleDetector(
                    page=self.page.page,
                    llm=llm,
                    config=obstacle_config
                )
                await detector.detect_and_handle()
            
            # Retry search
            logger.info("Retrying search after bot detection...")
            await self._enter_query_human_like(query, config["input_selector"])
            await self._click_search_button(config["button_selector"])
            await self._wait_for_results(config["result_selector"])
            
            # Extract results
            results = await self._extract_results(config, max_results)
            current_url = await self.page.get_url()
            
            # Check if still blocked
            if '/sorry/' in current_url or 'captcha' in current_url.lower():
                return ToolResult.error_result(
                    error=f"Bot detection persists after retry. Try engine='duckduckgo' or engine='bing' which are less restrictive.",
                    error_code="BOT_DETECTED_PERSISTENT"
                )
            
            logger.info(" Retry after bot detection successful!")
            
            # Build successful response
            elapsed_ms = (time.time() - start_time) * 1000
            response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results) * 10,
                search_time_ms=elapsed_ms,
                provider=SearchProvider.BROWSER.value,
                metadata={
                    "engine": search_engine.value,
                    "human_simulation": True,
                    "recovered_from_bot_detection": True,
                },
            )
            
            return ToolResult.success_result(
                data=response.to_dict(),
                message=f"Found {len(results)} results (recovered from bot detection)",
                metadata={"engine": search_engine.value, "bot_detection_recovered": True},
            )
        
        except Exception as retry_error:
            logger.warning(f"Retry after bot detection failed: {retry_error}")
            return ToolResult.error_result(
                error=f"Bot detection - retry failed. Try engine='duckduckgo' or engine='bing' instead.",
                error_code="BOT_DETECTED"
            )
    
    async def _wait_for_results(self, selector: str) -> None:
        """
        Wait for search results to load.
        
        Args:
            selector: Result container selector
        """
        logger.info("Waiting for results to load")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        # Wait for results with generous timeout
        await self.page.wait_for_selector(selector, timeout=15000)
        
        # Simulate reading/scrolling
        await asyncio.sleep(self.behavior.scroll_delay())
        
        # Scroll down slightly (humans do this)
        await self.page.evaluate("window.scrollBy(0, 300)")
        await asyncio.sleep(self.behavior.scroll_delay())
    
    async def _extract_results(
        self,
        config: Dict[str, str],
        max_results: int,
    ) -> List[SearchResult]:
        """
        Extract search results from page.
        
        Args:
            config: Engine configuration with selectors
            max_results: Maximum results to extract
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Extracting up to {max_results} results")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        results = []
        
        try:
            # Get all result elements
            result_elements = await self.page.query_selector_all(config["result_selector"])
            
            logger.info(f"Found {len(result_elements)} result elements")
            
            for i, elem in enumerate(result_elements[:max_results], 1):
                try:
                    # Extract title
                    title_elem = await elem.query_selector(config["title_selector"])
                    title = await title_elem.text_content() if title_elem else ""
                    title = title.strip()
                    
                    # Extract URL
                    link_elem = await elem.query_selector(config["link_selector"])
                    url = ""
                    if link_elem:
                        url = await link_elem.get_attribute("href") or ""
                    
                    # Clean URL (remove tracking parameters for some engines)
                    if url.startswith("/url?"):
                        # Google redirect URL
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(url)
                        url = parse_qs(parsed.query).get("q", [""])[0]
                    
                    # Extract snippet
                    snippet_elem = await elem.query_selector(config["snippet_selector"])
                    snippet = await snippet_elem.text_content() if snippet_elem else ""
                    snippet = clean_snippet(snippet)
                    
                    # Validate and add result
                    if title and url and is_valid_url(url):
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            position=i,
                            source=config.get("engine", "unknown"),
                        ))
                        
                        logger.debug(f"Extracted result {i}: {title[:50]}...")
                
                except Exception as e:
                    logger.debug(f"Failed to extract result {i}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Result extraction failed: {e}")
            return []


class SearchHumanAdvancedTool(SearchHumanTool):
    """
    Advanced human-like search with planning integration.
    
    This version creates an explicit execution plan and uses
    the autonomous planning system for more complex searches.
    """
    
    async def execute(
        self,
        query: str,
        engine: str = "duckduckgo",
        max_results: int = 10,
        use_planning: bool = True,
        **kwargs
    ) -> ToolResult:
        """
        Execute with optional planning mode.
        
        Args:
            query: Search query
            engine: Search engine
            max_results: Max results
            use_planning: Use autonomous planning system
            
        Returns:
            ToolResult
        """
        if use_planning:
            # Let the planning system guide the search
            # The ReAct agent will create phases automatically
            logger.info("Using planning-guided search")
        
        # Execute normal human-like search
        # (planning integration happens at agent level)
        return await super().execute(query, engine, max_results, **kwargs)
