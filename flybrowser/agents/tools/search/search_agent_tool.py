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
SearchAgentTool - BaseTool wrapper for the SearchAgent abstraction layer.

This tool provides a unified search interface for the ReAct agent, with:
- LLM-powered search intent detection
- Multi-provider search with automatic failover
- Intelligent result ranking
- Caching and rate limiting

The tool automatically determines:
1. Whether a query requires search
2. What type of search (web, images, news, videos, etc.)
3. Optimal search parameters based on intent

Usage in ReAct:
    The agent can invoke this tool naturally:
    - "Search for Python tutorials" -> web search
    - "Find images of sunset" -> image search  
    - "What's the latest news about AI?" -> news search with freshness ranking
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolResult, ToolCategory, SafetyLevel
from flybrowser.agents.tools.search.types import (
    SearchOptions,
    SearchType,
    SafeSearchLevel,
    TimeRange,
    SearchAgentResponse,
)
from flybrowser.agents.tools.search.search_agent import SearchAgent
from flybrowser.agents.tools.search.provider_registry import ProviderRegistry
from flybrowser.agents.tools.search.providers import (
    SerperProvider,
    GoogleProvider,
    BingProvider,
)
from flybrowser.agents.tools.search.intent_detector import (
    SearchIntentDetector,
    SearchIntent,
)
from flybrowser.agents.tools.search.ranking import CompositeRanker
from flybrowser.agents.tools.search.coordinator import (
    SearchCoordinator,
    SearchCoordinatorConfig,
    SearchMethod,
    CoordinatedSearchResult,
)

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.llm.base import BaseLLMProvider
    from flybrowser.agents.config import SearchProviderConfig, AgentConfig

logger = logging.getLogger(__name__)


class SearchAgentTool(BaseTool):
    """
    Unified search tool with LLM-powered intent detection.
    
    This tool provides intelligent search capabilities for the ReAct framework:
    - Automatic search intent detection from natural language
    - Multi-provider API support (Serper, Google, Bing)
    - Browser-based fallback search (human-like navigation)
    - Intelligent result ranking (BM25, freshness, domain authority)
    - Automatic failover between methods
    
    The tool is designed to be used naturally in the ReAct framework:
    - "Search for Python tutorials" -> web search via API or browser
    - "Find images of cats" -> image search
    - "Latest news about AI" -> news search with freshness ranking
    
    Integration with ReAct:
        The tool receives `llm_provider` and `agent_config` injected by the
        ReActAgent at execution time (line 1686-1687 in react_agent.py).
        This enables LLM-powered intent detection and ranking.
    
    Example:
        ```python
        tool = SearchAgentTool.create(llm_provider, config)
        result = await tool.execute(query="Python tutorials", search_type="auto")
        ```
    """
    
    def __init__(
        self,
        page_controller: Optional["PageController"] = None,
        llm_provider: Optional["BaseLLMProvider"] = None,
        search_agent: Optional[SearchAgent] = None,
        intent_detector: Optional[SearchIntentDetector] = None,
        config: Optional["SearchProviderConfig"] = None,
        coordinator: Optional[SearchCoordinator] = None,
    ) -> None:
        """
        Initialize the search agent tool.
        
        Args:
            page_controller: Page controller for browser-based search fallback
            llm_provider: LLM provider for intent detection (also injected by ReAct)
            search_agent: Pre-configured SearchAgent (optional)
            intent_detector: Pre-configured SearchIntentDetector (optional)
            config: Search provider configuration
            coordinator: Pre-configured SearchCoordinator (optional)
        """
        super().__init__(page_controller)
        # These may be set directly or injected by ReActAgent
        self.llm_provider = llm_provider
        self.agent_config: Optional["AgentConfig"] = None  # Injected by ReActAgent
        
        self._search_agent = search_agent
        self._intent_detector = intent_detector
        self._config = config
        self._coordinator = coordinator
        self._initialized = False
    
    @property
    def metadata(self) -> ToolMetadata:
        """Tool metadata for the ReAct framework."""
        return ToolMetadata(
            name="search",
            description=(
                "PRIMARY SEARCH TOOL - Use this for ALL web searches. "
                "Fast API-based search via Serper/Google/Bing (100-300ms). "
                "Supports: web, images, news, videos, places, shopping. "
                "Auto-detects intent, optimizes query, ranks results by relevance/freshness/authority. "
                "PREFER this over 'search_human' - it's faster and more reliable."
            ),
            category=ToolCategory.EXTRACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query or natural language instruction. "
                        "Can be a direct query ('Python tutorials') or instruction "
                        "('Search for Python tutorials')."
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="search_type",
                    type="string",
                    description=(
                        "Type of search to perform. Use 'auto' for automatic detection. "
                        "Options: 'auto', 'web', 'images', 'news', 'videos', 'places', 'shopping'"
                    ),
                    required=False,
                    default="auto",
                    enum=["auto", "web", "images", "news", "videos", "places", "shopping"],
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return (1-50)",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="ranking",
                    type="string",
                    description=(
                        "Ranking preference. 'auto' uses LLM to determine best ranking. "
                        "Options: 'auto', 'balanced', 'relevance', 'freshness', 'authority'"
                    ),
                    required=False,
                    default="auto",
                    enum=["auto", "balanced", "relevance", "freshness", "authority"],
                ),
                ToolParameter(
                    name="time_range",
                    type="string",
                    description="Time range filter for results",
                    required=False,
                    default="any",
                    enum=["any", "hour", "day", "week", "month", "year"],
                ),
            ],
            returns_description=(
                "SearchAgentResponse with ranked results, including: "
                "results (list of {title, url, snippet, relevance_score}), "
                "answer_box (if available), knowledge_graph (if available), "
                "related_searches, and metadata."
            ),
            examples=[
                'search(query="Python tutorials")',
                'search(query="Find images of sunset", search_type="auto")',
                'search(query="latest AI news", ranking="freshness")',
                'search(query="best Python IDE 2024", max_results=5)',
            ],
            requires_page=False,
            timeout_seconds=30.0,
        )
    
    @classmethod
    def create(
        cls,
        llm_provider: "BaseLLMProvider",
        config: Optional["SearchProviderConfig"] = None,
        page_controller: Optional["PageController"] = None,
    ) -> "SearchAgentTool":
        """
        Factory method to create a fully configured SearchAgentTool.
        
        This is the recommended way to create the tool, as it handles
        all provider registration and configuration.
        
        Args:
            llm_provider: LLM provider for intent detection
            config: Search provider configuration (uses defaults if not provided)
            page_controller: Optional page controller
            
        Returns:
            Configured SearchAgentTool instance
        """
        # Create provider registry
        registry = ProviderRegistry()
        
        # Register providers based on available API keys
        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            registry.register(SerperProvider(api_key=serper_key))
            logger.info("Registered SerperProvider")
        
        google_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        google_cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
        if google_key and google_cx:
            registry.register(GoogleProvider(api_key=google_key, cx=google_cx))
            logger.info("Registered GoogleProvider")
        
        bing_key = os.getenv("BING_SEARCH_API_KEY")
        if bing_key:
            registry.register(BingProvider(api_key=bing_key))
            logger.info("Registered BingProvider")
        
        if not registry.list_providers():
            logger.warning(
                "No search providers registered. Set SERPER_API_KEY, "
                "GOOGLE_CUSTOM_SEARCH_API_KEY+CX, or BING_SEARCH_API_KEY."
            )
        
        # Create ranker based on config
        ranker = None
        if config and config.enable_ranking:
            ranker = CompositeRanker(weights=config.ranking_weights)
        else:
            ranker = CompositeRanker()  # Use defaults
        
        # Create search agent
        search_agent = SearchAgent(
            registry=registry,
            ranker=ranker,
            cache_ttl_seconds=config.cache_ttl_seconds if config else 300,
            max_cache_entries=config.max_cache_entries if config else 100,
        )
        
        # Create intent detector
        intent_detector = SearchIntentDetector(llm_provider)
        
        return cls(
            page_controller=page_controller,
            llm_provider=llm_provider,
            search_agent=search_agent,
            intent_detector=intent_detector,
            config=config,
        )
    
    def has_providers_configured(self) -> bool:
        """
        Check if any search providers are configured.
        
        Returns:
            True if at least one API provider is available
        """
        if self._search_agent and self._search_agent.registry:
            return bool(self._search_agent.registry.list_providers())
        
        # Check environment variables
        return bool(
            os.getenv("SERPER_API_KEY") or
            (os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY") and os.getenv("GOOGLE_CUSTOM_SEARCH_CX")) or
            os.getenv("BING_SEARCH_API_KEY")
        )
    
    def has_browser_available(self) -> bool:
        """
        Check if browser-based search is available.
        
        Returns:
            True if page controller is available for browser search
        """
        return self.page is not None
    
    def _get_coordinator(self) -> SearchCoordinator:
        """
        Get or create the search coordinator.
        
        The coordinator handles intelligent routing between API and browser search.
        """
        if self._coordinator is None:
            # Get config from agent_config if available
            provider_config = None
            if self.agent_config and hasattr(self.agent_config, 'search_providers'):
                provider_config = self.agent_config.search_providers
            elif self._config:
                provider_config = self._config
            
            # Build coordinator config
            coord_config = SearchCoordinatorConfig(
                enable_fallback=True,
                enable_intent_detection=True,
                enable_ranking=provider_config.enable_ranking if provider_config else True,
                ranking_weights=dict(provider_config.ranking_weights) if provider_config else {},
                cache_ttl_seconds=provider_config.cache_ttl_seconds if provider_config else 300,
                max_cache_entries=provider_config.max_cache_entries if provider_config else 100,
            )
            
            self._coordinator = SearchCoordinator(
                llm_provider=self.llm_provider,
                page_controller=self.page,
                config=coord_config,
                agent_config=self.agent_config,
            )
        
        return self._coordinator
    
    async def execute(
        self,
        query: str,
        search_type: str = "auto",
        max_results: int = 10,
        ranking: str = "auto",
        time_range: str = "any",
        method: str = "auto",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute a search with intelligent intent detection and method selection.
        
        This method provides unified search across API and browser-based methods:
        - API search: Fast, reliable (Serper, Google, Bing APIs)
        - Browser search: Human-like navigation when no API keys available
        
        Args:
            query: Search query or natural language instruction
            search_type: Type of search ('auto', 'web', 'images', 'news', 'videos', 'places', 'shopping')
            max_results: Maximum results to return (1-50)
            ranking: Ranking preference ('auto', 'balanced', 'relevance', 'freshness', 'authority')
            time_range: Time range filter ('any', 'hour', 'day', 'week', 'month', 'year')
            method: Search method ('auto', 'api', 'browser')
            
        Returns:
            ToolResult with search results including:
            - results: List of ranked search results
            - answer_box: Featured snippet if available
            - knowledge_graph: Knowledge panel data if available
            - related_searches: Related query suggestions
            - method_used: Whether API or browser search was used
        """
        start_time = time.time()
        
        try:
            # Check if any search method is available
            api_available = self.has_providers_configured()
            browser_available = self.has_browser_available()
            
            if not api_available and not browser_available:
                return ToolResult.error_result(
                    error=(
                        "No search methods available. Either set API keys "
                        "(SERPER_API_KEY, GOOGLE_CUSTOM_SEARCH_API_KEY+CX, or BING_SEARCH_API_KEY) "
                        "or ensure page controller is available for browser search."
                    ),
                )
            
            # Ensure we have LLM for intent detection
            if not self.llm_provider:
                logger.warning("No LLM provider - skipping intent detection")
            
            # Get or create coordinator
            coordinator = self._get_coordinator()
            
            # Map method string to enum
            search_method = SearchMethod.AUTO
            if method.lower() == "api":
                search_method = SearchMethod.API
            elif method.lower() == "browser":
                search_method = SearchMethod.BROWSER
            
            # Map search type string to enum
            search_type_enum: Optional[SearchType] = None
            if search_type != "auto":
                type_map = {
                    "web": SearchType.WEB,
                    "images": SearchType.IMAGES,
                    "news": SearchType.NEWS,
                    "videos": SearchType.VIDEOS,
                    "places": SearchType.PLACES,
                    "shopping": SearchType.SHOPPING,
                }
                search_type_enum = type_map.get(search_type.lower())
            
            # Execute coordinated search
            result = await coordinator.search(
                query=query,
                method=search_method,
                search_type=search_type_enum,
                max_results=min(max(1, max_results), 50),
                ranking_preset=ranking if ranking != "auto" else "balanced",
                detect_intent=self.llm_provider is not None,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if not result.success:
                return ToolResult.error_result(
                    error=result.error or "Search failed",
                )
            
            # Format results for tool output - include ranking signals for transparency
            result_data = {
                "query": result.query,
                "search_type": search_type_enum.value if search_type_enum else "web",
                "results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "relevance_score": round(r.relevance_score, 4),
                        "position": r.position,
                        "domain": r.domain,
                        # Include individual ranking signals so LLM understands the scoring
                        "ranking_signals": {
                            k: round(v, 3) for k, v in r.ranking_signals.items()
                        } if r.ranking_signals else {},
                    }
                    for r in result.results
                ],
                "result_count": len(result.results),
                "total_results": result.total_results,
                "method_used": result.method_used.value,
                "provider_used": result.provider_used,
                "fallback_used": result.fallback_used,
                "cached": result.cached,
                "ranking_applied": result.ranking_applied,
                "search_time_ms": round(elapsed_ms, 1),
            }
            
            # Add optional fields if present
            if result.answer_box:
                result_data["answer_box"] = result.answer_box
            if result.knowledge_graph:
                result_data["knowledge_graph"] = result.knowledge_graph
            if result.related_searches:
                result_data["related_searches"] = result.related_searches
            
            # Add intent detection metadata
            if result.detected_intent:
                result_data["intent_detection"] = {
                    "detected_type": result.detected_intent.search_type.value if result.detected_intent.search_type else None,
                    "detected_ranking": result.detected_intent.ranking_preference,
                    "confidence": result.detected_intent.confidence,
                    "reasoning": result.detected_intent.reasoning,
                }
            
            return ToolResult.success_result(
                data=result_data,
                message=(
                    f"Found {len(result.results)} results for '{result.query}' "
                    f"via {result.method_used.value}/{result.provider_used} ({elapsed_ms:.0f}ms)"
                ),
                metadata={
                    "method": result.method_used.value,
                    "provider": result.provider_used,
                    "search_type": search_type_enum.value if search_type_enum else "web",
                    "fallback_used": result.fallback_used,
                },
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
            )
    
    async def detect_intent(self, instruction: str) -> SearchIntent:
        """
        Detect search intent from an instruction.
        
        This is useful for checking if an instruction requires search
        before actually executing it.
        
        Args:
            instruction: User instruction
            
        Returns:
            SearchIntent with detection results
        """
        if not self._intent_detector:
            if not self.llm_provider:
                raise ValueError("LLM provider required for intent detection")
            self._intent_detector = SearchIntentDetector(self.llm_provider)
        
        return await self._intent_detector.detect(instruction)
