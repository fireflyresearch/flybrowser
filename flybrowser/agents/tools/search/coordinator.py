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
Search Coordinator - Unified orchestration for API and browser-based search.

This module provides intelligent coordination between:
1. API-based search (Serper, Google, Bing) - Fast, reliable, no browser needed
2. Browser-based search (human-like) - Fallback when no API keys, or for verification

The coordinator:
- Automatically selects the best search method based on availability
- Provides intelligent fallback between methods
- Unifies results format across all search methods
- Integrates with LLM for intent detection and result ranking

Architecture:
    SearchCoordinator
    ├── API Search (SearchAgent)
    │   ├── SerperProvider
    │   ├── GoogleProvider
    │   └── BingProvider
    └── Browser Search (SearchHumanTool)
        ├── Google
        ├── DuckDuckGo
        └── Bing
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flybrowser.agents.tools.search.types import (
    SearchOptions,
    SearchType,
    RankedSearchResult,
    SearchAgentResponse,
)
from flybrowser.agents.tools.search.search_agent import SearchAgent
from flybrowser.agents.tools.search.provider_registry import ProviderRegistry
from flybrowser.agents.tools.search.providers import (
    SerperProvider,
    GoogleProvider,
    BingProvider,
)
from flybrowser.agents.tools.search.ranking import CompositeRanker
from flybrowser.agents.tools.search.intent_detector import (
    SearchIntentDetector,
    SearchIntent,
)

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.llm.base import BaseLLMProvider
    from flybrowser.agents.config import SearchProviderConfig, AgentConfig


logger = logging.getLogger(__name__)


class SearchMethod(str, Enum):
    """Available search methods."""
    API = "api"  # API-based search (Serper, Google API, Bing API)
    BROWSER = "browser"  # Browser-based human-like search
    AUTO = "auto"  # Automatically select best method


@dataclass
class SearchCoordinatorConfig:
    """Configuration for the search coordinator."""
    
    # Method selection
    preferred_method: SearchMethod = SearchMethod.AUTO
    enable_fallback: bool = True  # Fall back to browser if API fails
    
    # API search settings
    api_providers_priority: List[str] = field(default_factory=lambda: ["serper", "google", "bing"])
    
    # Browser search settings
    browser_engine: str = "duckduckgo"  # Default engine for browser search
    browser_engines_fallback: List[str] = field(default_factory=lambda: ["google", "bing"])
    
    # Intent detection
    enable_intent_detection: bool = True
    
    # Ranking
    enable_ranking: bool = True
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "bm25": 0.35,
        "freshness": 0.20,
        "domain_authority": 0.15,
        "position": 0.30,
    })
    
    # Caching
    cache_ttl_seconds: int = 300
    max_cache_entries: int = 100


@dataclass
class CoordinatedSearchResult:
    """
    Result from coordinated search across methods.
    
    Provides unified format regardless of whether API or browser search was used.
    """
    success: bool
    query: str
    results: List[RankedSearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0.0
    method_used: SearchMethod = SearchMethod.API
    provider_used: str = ""
    fallback_used: bool = False
    ranking_applied: bool = False
    cached: bool = False
    
    # Optional enrichments
    answer_box: Optional[Dict[str, Any]] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    related_searches: List[str] = field(default_factory=list)
    
    # Intent detection results
    detected_intent: Optional[SearchIntent] = None
    
    # Error info
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "result_count": len(self.results),
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
            "method_used": self.method_used.value,
            "provider_used": self.provider_used,
            "fallback_used": self.fallback_used,
            "ranking_applied": self.ranking_applied,
            "cached": self.cached,
            "answer_box": self.answer_box,
            "knowledge_graph": self.knowledge_graph,
            "related_searches": self.related_searches,
            "intent_detection": self.detected_intent.to_dict() if self.detected_intent else None,
            "error": self.error,
        }


class SearchCoordinator:
    """
    Unified search coordinator for API and browser-based search.
    
    This class provides intelligent orchestration between different search methods:
    - Automatically selects the best method based on availability
    - Falls back to alternative methods on failure
    - Provides consistent result format
    - Integrates LLM for intent detection and ranking
    
    Example:
        ```python
        coordinator = SearchCoordinator(
            llm_provider=llm,
            page_controller=page,  # Optional, for browser search
            config=SearchCoordinatorConfig(),
        )
        
        result = await coordinator.search(
            query="Python tutorials",
            method=SearchMethod.AUTO,
        )
        ```
    """
    
    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        page_controller: Optional["PageController"] = None,
        config: Optional[SearchCoordinatorConfig] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """
        Initialize the search coordinator.
        
        Args:
            llm_provider: LLM provider for intent detection and ranking
            page_controller: Page controller for browser-based search (optional)
            config: Coordinator configuration
            agent_config: Full agent configuration (for search_providers settings)
        """
        self.llm = llm_provider
        self.page = page_controller
        self.config = config or SearchCoordinatorConfig()
        self.agent_config = agent_config
        
        # Initialize components lazily
        self._search_agent: Optional[SearchAgent] = None
        self._intent_detector: Optional[SearchIntentDetector] = None
        self._api_available: Optional[bool] = None
        
        logger.info("SearchCoordinator initialized")
    
    @property
    def api_available(self) -> bool:
        """Check if API-based search is available."""
        if self._api_available is None:
            self._api_available = bool(
                os.getenv("SERPER_API_KEY") or
                (os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY") and os.getenv("GOOGLE_CUSTOM_SEARCH_CX")) or
                os.getenv("BING_SEARCH_API_KEY")
            )
        return self._api_available
    
    @property
    def browser_available(self) -> bool:
        """Check if browser-based search is available."""
        return self.page is not None
    
    def _get_search_agent(self) -> SearchAgent:
        """Get or create the API search agent."""
        if self._search_agent is None:
            registry = ProviderRegistry()
            
            # Register providers in priority order
            for provider_name in self.config.api_providers_priority:
                try:
                    if provider_name == "serper":
                        key = os.getenv("SERPER_API_KEY")
                        if key:
                            registry.register(SerperProvider(api_key=key))
                    elif provider_name == "google":
                        key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
                        cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
                        if key and cx:
                            registry.register(GoogleProvider(api_key=key, cx=cx))
                    elif provider_name == "bing":
                        key = os.getenv("BING_SEARCH_API_KEY")
                        if key:
                            registry.register(BingProvider(api_key=key))
                except Exception as e:
                    logger.warning(f"Failed to register {provider_name} provider: {e}")
            
            # Create ranker
            ranker = None
            if self.config.enable_ranking:
                ranker = CompositeRanker(weights=self.config.ranking_weights)
            
            self._search_agent = SearchAgent(
                registry=registry,
                ranker=ranker,
                cache_ttl_seconds=self.config.cache_ttl_seconds,
                max_cache_entries=self.config.max_cache_entries,
            )
        
        return self._search_agent
    
    def _get_intent_detector(self) -> SearchIntentDetector:
        """Get or create the intent detector."""
        if self._intent_detector is None:
            self._intent_detector = SearchIntentDetector(self.llm)
        return self._intent_detector
    
    async def search(
        self,
        query: str,
        method: SearchMethod = SearchMethod.AUTO,
        search_type: Optional[SearchType] = None,
        max_results: int = 10,
        ranking_preset: str = "balanced",
        detect_intent: bool = True,
    ) -> CoordinatedSearchResult:
        """
        Perform a coordinated search.
        
        Args:
            query: Search query
            method: Search method (API, browser, or auto)
            search_type: Type of search (web, images, news, etc.)
            max_results: Maximum results to return
            ranking_preset: Ranking preset (balanced, freshness, authority, relevance)
            detect_intent: Whether to use LLM for intent detection
            
        Returns:
            CoordinatedSearchResult with unified format
        """
        start_time = time.time()
        
        # Detect intent if enabled
        detected_intent: Optional[SearchIntent] = None
        if detect_intent and self.config.enable_intent_detection:
            try:
                detector = self._get_intent_detector()
                detected_intent = await detector.detect(query)
                
                # Use detected search type if not explicitly specified
                if search_type is None and detected_intent.search_type:
                    search_type = detected_intent.search_type
                
                # Use detected ranking preference
                if detected_intent.ranking_preference != "balanced":
                    ranking_preset = detected_intent.ranking_preference
                
                # Use optimized query
                if detected_intent.optimized_query:
                    query = detected_intent.optimized_query
                    
            except Exception as e:
                logger.warning(f"Intent detection failed: {e}")
        
        # Default to web search
        if search_type is None:
            search_type = SearchType.WEB
        
        # Determine method to use
        actual_method = self._select_method(method)
        
        # Execute search with fallback
        result: Optional[CoordinatedSearchResult] = None
        fallback_used = False
        
        if actual_method == SearchMethod.API:
            result = await self._search_api(query, search_type, max_results, ranking_preset)
            
            # Fallback to browser if API fails and fallback is enabled
            if not result.success and self.config.enable_fallback and self.browser_available:
                logger.info("API search failed, falling back to browser search")
                result = await self._search_browser(query, max_results)
                fallback_used = True
                
        elif actual_method == SearchMethod.BROWSER:
            result = await self._search_browser(query, max_results)
            
            # Fallback to API if browser fails and API is available
            if not result.success and self.config.enable_fallback and self.api_available:
                logger.info("Browser search failed, falling back to API search")
                result = await self._search_api(query, search_type, max_results, ranking_preset)
                fallback_used = True
        
        # Ensure we have a result
        if result is None:
            result = CoordinatedSearchResult(
                success=False,
                query=query,
                error="No search method available",
            )
        
        # Update result with coordination metadata
        result.fallback_used = fallback_used
        result.detected_intent = detected_intent
        result.search_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _select_method(self, requested: SearchMethod) -> SearchMethod:
        """Select the actual search method based on availability."""
        if requested == SearchMethod.API:
            if self.api_available:
                return SearchMethod.API
            elif self.browser_available:
                logger.warning("API requested but not available, using browser")
                return SearchMethod.BROWSER
            else:
                return SearchMethod.API  # Will fail gracefully
                
        elif requested == SearchMethod.BROWSER:
            if self.browser_available:
                return SearchMethod.BROWSER
            elif self.api_available:
                logger.warning("Browser requested but not available, using API")
                return SearchMethod.API
            else:
                return SearchMethod.BROWSER  # Will fail gracefully
                
        else:  # AUTO
            # Prefer API for speed and reliability
            if self.api_available:
                return SearchMethod.API
            elif self.browser_available:
                return SearchMethod.BROWSER
            else:
                return SearchMethod.API  # Will fail gracefully
    
    async def _search_api(
        self,
        query: str,
        search_type: SearchType,
        max_results: int,
        ranking_preset: str,
    ) -> CoordinatedSearchResult:
        """Execute API-based search."""
        try:
            agent = self._get_search_agent()
            
            options = SearchOptions(
                search_type=search_type,
                max_results=max_results,
                include_answer_box=True,
                include_knowledge_graph=search_type == SearchType.WEB,
                include_related=True,
            )
            
            # Note: SearchAgent.search() doesn't accept ranking_preset
            # Ranking is applied via the ranker configured in the agent
            response = await agent.search(
                query=query,
                options=options,
            )
            
            return CoordinatedSearchResult(
                success=True,
                query=query,
                results=response.results,
                total_results=response.total_results,
                method_used=SearchMethod.API,
                provider_used=response.provider_used,
                ranking_applied=response.ranking_applied,
                cached=response.cached,
                answer_box=response.answer_box,
                knowledge_graph=response.knowledge_graph,
                related_searches=response.related_searches,
            )
            
        except Exception as e:
            logger.error(f"API search failed: {e}")
            return CoordinatedSearchResult(
                success=False,
                query=query,
                method_used=SearchMethod.API,
                error=str(e),
            )
    
    async def _search_browser(
        self,
        query: str,
        max_results: int,
    ) -> CoordinatedSearchResult:
        """Execute browser-based search."""
        if not self.page:
            return CoordinatedSearchResult(
                success=False,
                query=query,
                method_used=SearchMethod.BROWSER,
                error="Page controller not available for browser search",
            )
        
        try:
            # Import here to avoid circular imports
            from flybrowser.agents.tools.search_human import SearchHumanTool
            
            # Create and configure tool
            tool = SearchHumanTool(page_controller=self.page)
            tool.llm_provider = self.llm
            tool.agent_config = self.agent_config
            
            # Execute search
            result = await tool.execute(
                query=query,
                engine=self.config.browser_engine,
                max_results=max_results,
            )
            
            if not result.success:
                # Try fallback engines
                for engine in self.config.browser_engines_fallback:
                    if engine == self.config.browser_engine:
                        continue
                    logger.info(f"Trying fallback browser engine: {engine}")
                    result = await tool.execute(
                        query=query,
                        engine=engine,
                        max_results=max_results,
                    )
                    if result.success:
                        break
            
            if result.success:
                # Convert browser results to RankedSearchResult format
                browser_results = result.data.get("results", [])
                ranked_results = [
                    RankedSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("snippet", ""),
                        position=i + 1,
                        source=result.data.get("metadata", {}).get("engine", "browser"),
                        relevance_score=1.0 - (i * 0.05),  # Simple position-based score
                    )
                    for i, r in enumerate(browser_results)
                ]
                
                return CoordinatedSearchResult(
                    success=True,
                    query=query,
                    results=ranked_results,
                    total_results=result.data.get("total_results", len(ranked_results)),
                    method_used=SearchMethod.BROWSER,
                    provider_used=result.data.get("metadata", {}).get("engine", "browser"),
                    ranking_applied=False,  # Browser results have inherent ranking
                )
            else:
                return CoordinatedSearchResult(
                    success=False,
                    query=query,
                    method_used=SearchMethod.BROWSER,
                    error=result.error,
                )
                
        except Exception as e:
            logger.error(f"Browser search failed: {e}")
            return CoordinatedSearchResult(
                success=False,
                query=query,
                method_used=SearchMethod.BROWSER,
                error=str(e),
            )
    
    async def detect_search_intent(self, instruction: str) -> SearchIntent:
        """
        Detect search intent from an instruction.
        
        Useful for determining if a task requires search before executing.
        
        Args:
            instruction: User instruction to analyze
            
        Returns:
            SearchIntent with detection results
        """
        detector = self._get_intent_detector()
        return await detector.detect(instruction)
