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
SearchAgent - High-Level Search Orchestrator.

This module provides the SearchAgent class which orchestrates search
operations across multiple providers with intelligent failover,
caching, and result ranking.

Features:
    - Automatic provider selection based on health and capabilities
    - Failover to backup providers on failure
    - Response caching with configurable TTL
    - Intelligent result ranking using composite signals
    - Cost tracking and optimization
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from flybrowser.agents.tools.search.base_provider import (
    BaseSearchProvider,
    SearchProviderError,
)
from flybrowser.agents.tools.search.provider_registry import ProviderRegistry
from flybrowser.agents.tools.search.types import (
    RankedSearchResult,
    SearchAgentResponse,
    SearchOptions,
    SearchType,
)
from flybrowser.agents.tools.search.ranking import CompositeRanker
from flybrowser.agents.tools.search_utils import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for search results."""
    response: SearchAgentResponse
    timestamp: float
    query_hash: str


class SearchAgent:
    """
    High-level search orchestrator.
    
    Manages search operations across multiple providers with:
    - Intelligent provider selection
    - Automatic failover on failure
    - Response caching
    - Result ranking
    - Cost tracking
    
    Example:
        >>> # Create with default providers
        >>> agent = SearchAgent.create_default()
        >>> 
        >>> # Or with custom registry
        >>> registry = ProviderRegistry()
        >>> registry.register(SerperProvider(api_key="..."))
        >>> agent = SearchAgent(registry)
        >>> 
        >>> # Perform search
        >>> results = await agent.search("python tutorials")
        >>> 
        >>> # Get ranked results
        >>> for result in results.results:
        ...     print(f"{result.relevance_score:.2f}: {result.title}")
    """
    
    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        ranker: Optional[CompositeRanker] = None,
        cache_ttl_seconds: int = 300,
        max_cache_entries: int = 100,
        enable_caching: bool = True,
        enable_ranking: bool = True,
        max_retries: int = 2,
    ) -> None:
        """
        Initialize SearchAgent.
        
        Args:
            registry: Provider registry (uses default if None)
            ranker: Result ranker (uses default CompositeRanker if None)
            cache_ttl_seconds: Cache TTL in seconds
            max_cache_entries: Maximum cache entries
            enable_caching: Whether to enable response caching
            enable_ranking: Whether to enable result ranking
            max_retries: Maximum retry attempts with fallback providers
        """
        self.registry = registry or ProviderRegistry()
        self.ranker = ranker or CompositeRanker()
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_entries = max_cache_entries
        self.enable_caching = enable_caching
        self.enable_ranking = enable_ranking
        self.max_retries = max_retries
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        
        # Statistics
        self._total_searches = 0
        self._cache_hits = 0
        self._provider_usage: Dict[str, int] = {}
        self._total_cost = 0.0
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
        provider: Optional[str] = None,
    ) -> SearchAgentResponse:
        """
        Perform a search with automatic provider selection and ranking.
        
        Args:
            query: Search query
            options: Search options
            provider: Specific provider to use (or auto-select if None)
            
        Returns:
            SearchAgentResponse with ranked results
        """
        if options is None:
            options = SearchOptions()
        
        start_time = time.time()
        self._total_searches += 1
        
        # Check cache
        if self.enable_caching:
            cached = self._get_from_cache(query, options, provider)
            if cached:
                self._cache_hits += 1
                cached.cached = True
                logger.debug(f"Cache hit for query: {query}")
                return cached
        
        # Get provider(s) to try
        providers_to_try = self._get_providers_to_try(
            provider,
            options.search_type,
        )
        
        if not providers_to_try:
            return SearchAgentResponse(
                query=query,
                results=[],
                metadata={"error": "No configured providers available"},
            )
        
        # Try providers in order
        last_error = None
        response = None
        provider_used = None
        fallback_used = False
        
        for i, prov in enumerate(providers_to_try[:self.max_retries + 1]):
            try:
                logger.info(f"Searching with provider: {prov.provider_name}")
                
                raw_response = await prov.search(query, options)
                provider_used = prov.provider_name
                fallback_used = i > 0
                
                # Track usage
                self._provider_usage[provider_used] = (
                    self._provider_usage.get(provider_used, 0) + 1
                )
                self._total_cost += prov.capabilities.cost_per_request
                
                # Convert to ranked results
                response = self._convert_to_agent_response(
                    query, raw_response, provider_used, fallback_used
                )
                break
                
            except SearchProviderError as e:
                last_error = e
                logger.warning(f"Provider {prov.provider_name} failed: {e}")
                
                if not e.recoverable:
                    break  # Don't retry non-recoverable errors
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {prov.provider_name} failed: {e}")
        
        if response is None:
            # All providers failed
            return SearchAgentResponse(
                query=query,
                results=[],
                metadata={
                    "error": str(last_error) if last_error else "All providers failed",
                    "providers_tried": [p.provider_name for p in providers_to_try],
                },
            )
        
        # Apply ranking
        if self.enable_ranking and response.results:
            response = self._apply_ranking(query, response)
        
        # Calculate timing
        response.search_time_ms = (time.time() - start_time) * 1000
        
        # Cache response
        if self.enable_caching:
            self._add_to_cache(query, options, provider, response)
        
        return response
    
    async def search_multiple(
        self,
        queries: List[str],
        options: Optional[SearchOptions] = None,
        concurrent: bool = True,
    ) -> List[SearchAgentResponse]:
        """
        Search multiple queries.
        
        Args:
            queries: List of search queries
            options: Search options (applied to all)
            concurrent: Whether to search concurrently
            
        Returns:
            List of search responses
        """
        if concurrent:
            tasks = [self.search(q, options) for q in queries]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for q in queries:
                results.append(await self.search(q, options))
            return results
    
    def _get_providers_to_try(
        self,
        preferred_provider: Optional[str],
        search_type: SearchType,
    ) -> List[BaseSearchProvider]:
        """
        Get list of providers to try in order.
        
        Args:
            preferred_provider: Specific provider name if requested
            search_type: Type of search
            
        Returns:
            List of providers to try
        """
        providers = []
        
        if preferred_provider:
            # User requested specific provider
            provider = self.registry.get_provider(preferred_provider)
            if provider and provider.is_configured():
                providers.append(provider)
            # Add fallbacks
            providers.extend(
                self.registry.get_fallback_providers(
                    exclude=preferred_provider,
                    search_type=search_type,
                )
            )
        else:
            # Auto-select best provider
            best = self.registry.get_best_provider(search_type=search_type)
            if best:
                providers.append(best)
                # Add fallbacks
                providers.extend(
                    self.registry.get_fallback_providers(
                        exclude=best.provider_name,
                        search_type=search_type,
                    )
                )
        
        return providers
    
    def _convert_to_agent_response(
        self,
        query: str,
        response: SearchResponse,
        provider_used: str,
        fallback_used: bool,
    ) -> SearchAgentResponse:
        """
        Convert provider response to SearchAgentResponse.
        
        Args:
            query: Original query
            response: Raw provider response
            provider_used: Provider that returned results
            fallback_used: Whether a fallback provider was used
            
        Returns:
            SearchAgentResponse
        """
        # Convert SearchResults to RankedSearchResults
        ranked_results = []
        for result in response.results:
            ranked_results.append(RankedSearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                position=result.position,
                source=result.source,
                metadata=result.metadata,
            ))
        
        # Extract enriched data from metadata
        answer_box = response.metadata.get("answer_box")
        knowledge_graph = response.metadata.get("knowledge_graph")
        related_searches = response.metadata.get("related_searches", [])
        
        return SearchAgentResponse(
            query=query,
            results=ranked_results,
            total_results=response.total_results,
            search_time_ms=response.search_time_ms,
            provider_used=provider_used,
            fallback_used=fallback_used,
            cached=False,
            ranking_applied=False,
            answer_box=answer_box,
            knowledge_graph=knowledge_graph,
            related_searches=related_searches,
            metadata=response.metadata,
        )
    
    def _apply_ranking(
        self,
        query: str,
        response: SearchAgentResponse,
    ) -> SearchAgentResponse:
        """
        Apply ranking to search results.
        
        Args:
            query: Search query
            response: Response to rank
            
        Returns:
            Response with ranked results
        """
        try:
            ranked = self.ranker.rank(query, response.results)
            response.results = ranked
            response.ranking_applied = True
        except Exception as e:
            logger.warning(f"Ranking failed: {e}")
        
        return response
    
    def _get_cache_key(
        self,
        query: str,
        options: SearchOptions,
        provider: Optional[str],
    ) -> str:
        """Generate cache key."""
        key_parts = [
            query.lower().strip(),
            options.search_type.value,
            str(options.max_results),
            str(options.page),
            provider or "auto",
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(
        self,
        query: str,
        options: SearchOptions,
        provider: Optional[str],
    ) -> Optional[SearchAgentResponse]:
        """Get response from cache if valid."""
        key = self._get_cache_key(query, options, provider)
        
        if key in self._cache:
            entry = self._cache[key]
            age = time.time() - entry.timestamp
            
            if age < self.cache_ttl_seconds:
                return entry.response
            else:
                # Expired
                del self._cache[key]
        
        return None
    
    def _add_to_cache(
        self,
        query: str,
        options: SearchOptions,
        provider: Optional[str],
        response: SearchAgentResponse,
    ) -> None:
        """Add response to cache."""
        # Evict old entries if at capacity
        if len(self._cache) >= self.max_cache_entries:
            self._evict_oldest_cache_entry()
        
        key = self._get_cache_key(query, options, provider)
        self._cache[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            query_hash=key,
        )
    
    def _evict_oldest_cache_entry(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].timestamp
        )
        del self._cache[oldest_key]
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        logger.info("Search cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get search agent statistics.
        
        Returns:
            Dictionary with stats
        """
        cache_hit_rate = (
            self._cache_hits / self._total_searches
            if self._total_searches > 0 else 0.0
        )
        
        return {
            "total_searches": self._total_searches,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_entries": len(self._cache),
            "provider_usage": dict(self._provider_usage),
            "total_cost_usd": self._total_cost,
            "registry_stats": self.registry.get_stats(),
        }
    
    def set_ranker(
        self,
        ranker_type: str = "default",
    ) -> None:
        """
        Set the result ranker.
        
        Args:
            ranker_type: Type of ranker ("default", "news", "research", "tutorials")
        """
        if ranker_type == "news":
            self.ranker = CompositeRanker.create_for_news()
        elif ranker_type == "research":
            self.ranker = CompositeRanker.create_for_research()
        elif ranker_type == "tutorials":
            self.ranker = CompositeRanker.create_for_tutorials()
        else:
            self.ranker = CompositeRanker()
        
        logger.info(f"Ranker set to: {ranker_type}")
    
    @classmethod
    def create_default(cls) -> "SearchAgent":
        """
        Create a SearchAgent with default configuration.
        
        Uses environment variables for API keys:
        - SERPER_API_KEY
        - GOOGLE_CUSTOM_SEARCH_API_KEY + GOOGLE_CUSTOM_SEARCH_CX
        - BING_SEARCH_API_KEY
        
        Returns:
            Configured SearchAgent
        """
        registry = ProviderRegistry.create_default()
        return cls(registry=registry)
    
    def __repr__(self) -> str:
        configured = len(self.registry.list_configured_providers())
        return f"SearchAgent(providers={configured}, caching={self.enable_caching})"
