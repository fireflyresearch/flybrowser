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
Serper.dev Search Provider.

Serper.dev is a fast, affordable Google search API that provides:
- Web search with organic results
- Image search
- News search
- Places/local search
- Knowledge graph data
- Answer boxes/featured snippets
- Related searches
- People also ask

Pricing: ~$0.001 per search ($50/month for 50K searches)
Speed: ~200ms average response time

API Documentation: https://serper.dev/docs
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from flybrowser.agents.tools.search.base_provider import (
    BaseSearchProvider,
    SearchProviderError,
)
from flybrowser.agents.tools.search.types import (
    ProviderCapabilities,
    SafeSearchLevel,
    SearchOptions,
    SearchType,
    TimeRange,
)
from flybrowser.agents.tools.search_utils import (
    SearchResponse,
    SearchResult,
    SearchProvider,
    clean_snippet,
)

logger = logging.getLogger(__name__)


class SerperProvider(BaseSearchProvider):
    """
    Serper.dev search provider.
    
    A fast, affordable Google search API that provides comprehensive
    search results including organic results, knowledge graphs,
    answer boxes, and related searches.
    
    Features:
        - Web, image, news, video, places search
        - Knowledge graph data
        - Answer boxes/featured snippets
        - Related searches
        - People also ask
        - ~200ms response time
        - Cost-effective ($0.001/search)
    
    Example:
        >>> provider = SerperProvider(api_key="your-api-key")
        >>> response = await provider.search("python tutorials")
        >>> for result in response.results:
        ...     print(result.title, result.url)
    """
    
    provider_name = "serper"
    
    # API endpoints
    BASE_URL = "https://google.serper.dev"
    ENDPOINTS = {
        SearchType.WEB: "/search",
        SearchType.IMAGES: "/images",
        SearchType.NEWS: "/news",
        SearchType.VIDEOS: "/videos",
        SearchType.PLACES: "/places",
        SearchType.SHOPPING: "/shopping",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_rpm: int = 100,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Serper provider.
        
        Args:
            api_key: Serper.dev API key (from https://serper.dev)
            rate_limit_rpm: Maximum requests per minute
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(
            api_key=api_key,
            rate_limit_rpm=rate_limit_rpm,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self._session = None
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get Serper capabilities."""
        return ProviderCapabilities(
            supports_pagination=True,
            supports_images=True,
            supports_news=True,
            supports_videos=True,
            supports_places=True,
            supports_shopping=True,
            supports_safe_search=True,
            supports_time_range=True,
            supports_country=True,
            supports_language=True,
            supports_knowledge_graph=True,
            supports_answer_box=True,
            supports_related_searches=True,
            max_results_per_page=100,
            rate_limit_rpm=self.rate_limit_rpm,
            cost_per_request=0.001,  # $1 per 1000 searches
        )
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
    ) -> SearchResponse:
        """
        Perform a search query using Serper.dev API.
        
        Args:
            query: Search query string
            options: Search options and filters
            
        Returns:
            SearchResponse with results
            
        Raises:
            SearchProviderError: If search fails
        """
        if options is None:
            options = SearchOptions()
        
        # Validate configuration
        if not self.is_configured():
            raise SearchProviderError(
                "Serper API key not configured. Set SERPER_API_KEY environment variable.",
                provider=self.provider_name,
                error_code="NOT_CONFIGURED",
                recoverable=False,
            )
        
        # Acquire rate limit
        await self._acquire_rate_limit()
        
        start_time = time.time()
        
        try:
            response = await self._execute_search(query, options)
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self._record_request(latency_ms, success=True)
            
            return response
            
        except SearchProviderError:
            raise
        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self._record_request(latency_ms, success=False, error=str(e))
            
            raise SearchProviderError(
                f"Search failed: {str(e)}",
                provider=self.provider_name,
                error_code="SEARCH_ERROR",
                recoverable=True,
            )
    
    async def _execute_search(
        self,
        query: str,
        options: SearchOptions,
    ) -> SearchResponse:
        """
        Execute the actual search request.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            SearchResponse with results
        """
        try:
            import aiohttp
        except ImportError:
            raise SearchProviderError(
                "aiohttp not installed. Install with: pip install aiohttp",
                provider=self.provider_name,
                error_code="MISSING_DEPENDENCY",
                recoverable=False,
            )
        
        # Build request payload
        payload = self._build_payload(query, options)
        
        # Get endpoint for search type
        endpoint = self.ENDPOINTS.get(options.search_type, self.ENDPOINTS[SearchType.WEB])
        url = f"{self.BASE_URL}{endpoint}"
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            ) as response:
                if response.status == 401:
                    raise SearchProviderError(
                        "Invalid API key",
                        provider=self.provider_name,
                        error_code="INVALID_API_KEY",
                        recoverable=False,
                    )
                elif response.status == 429:
                    raise SearchProviderError(
                        "Rate limit exceeded",
                        provider=self.provider_name,
                        error_code="RATE_LIMITED",
                        recoverable=True,
                    )
                elif response.status != 200:
                    error_text = await response.text()
                    raise SearchProviderError(
                        f"API error {response.status}: {error_text}",
                        provider=self.provider_name,
                        error_code=f"HTTP_{response.status}",
                        recoverable=response.status >= 500,
                    )
                
                data = await response.json()
                return self._parse_response(query, data, options)
    
    def _build_payload(self, query: str, options: SearchOptions) -> Dict[str, Any]:
        """
        Build request payload for Serper API.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            Request payload dictionary
        """
        payload: Dict[str, Any] = {
            "q": query,
            "num": min(options.max_results, 100),
        }
        
        # Pagination
        if options.page > 1:
            payload["page"] = options.page
        
        # Localization
        if options.country:
            payload["gl"] = options.country.lower()
        if options.language:
            payload["hl"] = options.language.lower()
        
        # Safe search
        if options.safe_search == SafeSearchLevel.OFF:
            payload["safe"] = "off"
        elif options.safe_search == SafeSearchLevel.STRICT:
            payload["safe"] = "active"
        # MODERATE is default, no need to set
        
        # Time range
        time_range_map = {
            TimeRange.HOUR: "qdr:h",
            TimeRange.DAY: "qdr:d",
            TimeRange.WEEK: "qdr:w",
            TimeRange.MONTH: "qdr:m",
            TimeRange.YEAR: "qdr:y",
        }
        if options.time_range != TimeRange.ANY:
            tbs = time_range_map.get(options.time_range)
            if tbs:
                payload["tbs"] = tbs
        
        # Site filter
        if options.site_filter:
            payload["q"] = f"site:{options.site_filter} {query}"
        
        # File type
        if options.file_type:
            payload["q"] = f"filetype:{options.file_type} {payload['q']}"
        
        return payload
    
    def _parse_response(
        self,
        query: str,
        data: Dict[str, Any],
        options: SearchOptions,
    ) -> SearchResponse:
        """
        Parse Serper API response into SearchResponse.
        
        Args:
            query: Original query
            data: Raw API response
            options: Search options used
            
        Returns:
            SearchResponse object
        """
        results: List[SearchResult] = []
        
        # Parse based on search type
        if options.search_type == SearchType.WEB:
            results = self._parse_organic_results(data)
        elif options.search_type == SearchType.IMAGES:
            results = self._parse_image_results(data)
        elif options.search_type == SearchType.NEWS:
            results = self._parse_news_results(data)
        elif options.search_type == SearchType.VIDEOS:
            results = self._parse_video_results(data)
        elif options.search_type == SearchType.PLACES:
            results = self._parse_places_results(data)
        elif options.search_type == SearchType.SHOPPING:
            results = self._parse_shopping_results(data)
        
        # Build metadata with rich data
        metadata: Dict[str, Any] = {
            "search_type": options.search_type.value,
        }
        
        # Extract knowledge graph if present
        if "knowledgeGraph" in data:
            metadata["knowledge_graph"] = data["knowledgeGraph"]
        
        # Extract answer box if present
        if "answerBox" in data:
            metadata["answer_box"] = data["answerBox"]
        
        # Extract related searches
        if "relatedSearches" in data:
            metadata["related_searches"] = [
                item.get("query", "") for item in data.get("relatedSearches", [])
            ]
        
        # Extract "people also ask"
        if "peopleAlsoAsk" in data:
            metadata["people_also_ask"] = data["peopleAlsoAsk"]
        
        # Get search information
        search_info = data.get("searchParameters", {})
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results) * 10,  # Estimate
            search_time_ms=0,  # Will be set by caller
            provider=SearchProvider.SERP_API.value,
            metadata=metadata,
        )
    
    def _parse_organic_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse organic web search results."""
        results = []
        
        for i, item in enumerate(data.get("organic", []), 1):
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = clean_snippet(item.get("snippet", ""))
            
            if not title or not url:
                continue
            
            metadata = {}
            
            # Extract rich data
            if "sitelinks" in item:
                metadata["sitelinks"] = item["sitelinks"]
            if "date" in item:
                metadata["date"] = item["date"]
            if "position" in item:
                metadata["original_position"] = item["position"]
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                position=i,
                source=self.provider_name,
                metadata=metadata,
            ))
        
        return results
    
    def _parse_image_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse image search results."""
        results = []
        
        for i, item in enumerate(data.get("images", []), 1):
            title = item.get("title", "")
            url = item.get("link", "")
            
            if not url:
                continue
            
            results.append(SearchResult(
                title=title or "Image",
                url=url,
                snippet=item.get("source", ""),
                position=i,
                source=self.provider_name,
                metadata={
                    "image_url": item.get("imageUrl", ""),
                    "thumbnail_url": item.get("thumbnailUrl", ""),
                    "width": item.get("imageWidth"),
                    "height": item.get("imageHeight"),
                },
            ))
        
        return results
    
    def _parse_news_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse news search results."""
        results = []
        
        for i, item in enumerate(data.get("news", []), 1):
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = clean_snippet(item.get("snippet", ""))
            
            if not title or not url:
                continue
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                position=i,
                source=self.provider_name,
                metadata={
                    "date": item.get("date", ""),
                    "source": item.get("source", ""),
                    "image_url": item.get("imageUrl", ""),
                },
            ))
        
        return results
    
    def _parse_video_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse video search results."""
        results = []
        
        for i, item in enumerate(data.get("videos", []), 1):
            title = item.get("title", "")
            url = item.get("link", "")
            
            if not title or not url:
                continue
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=item.get("snippet", ""),
                position=i,
                source=self.provider_name,
                metadata={
                    "duration": item.get("duration", ""),
                    "channel": item.get("channel", ""),
                    "date": item.get("date", ""),
                    "thumbnail_url": item.get("thumbnailUrl", ""),
                    "platform": item.get("platform", ""),
                },
            ))
        
        return results
    
    def _parse_places_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse places/local search results."""
        results = []
        
        for i, item in enumerate(data.get("places", []), 1):
            title = item.get("title", "")
            address = item.get("address", "")
            
            if not title:
                continue
            
            # Use Google Maps link if available
            url = item.get("link", f"https://www.google.com/maps/search/{title}")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=address,
                position=i,
                source=self.provider_name,
                metadata={
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "phone": item.get("phone", ""),
                    "type": item.get("type", ""),
                    "hours": item.get("hours", ""),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "cid": item.get("cid"),
                },
            ))
        
        return results
    
    def _parse_shopping_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse shopping search results."""
        results = []
        
        for i, item in enumerate(data.get("shopping", []), 1):
            title = item.get("title", "")
            url = item.get("link", "")
            
            if not title or not url:
                continue
            
            price = item.get("price", "")
            source = item.get("source", "")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=f"{price} - {source}" if price else source,
                position=i,
                source=self.provider_name,
                metadata={
                    "price": price,
                    "currency": item.get("currency", ""),
                    "merchant": source,
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "image_url": item.get("imageUrl", ""),
                    "delivery": item.get("delivery", ""),
                },
            ))
        
        return results
