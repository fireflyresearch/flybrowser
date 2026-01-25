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
Bing Web Search API Provider.

Bing Web Search API provides:
- Web search with organic results
- Image search
- News search
- Video search
- Safe search filtering
- Country and language localization

Pricing: $3 per 1000 transactions (various tiers available)
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
"""

from __future__ import annotations

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


class BingProvider(BaseSearchProvider):
    """
    Bing Web Search API provider.
    
    Provides access to Microsoft Bing search through the Cognitive Services API.
    
    Example:
        >>> provider = BingProvider(api_key="your-api-key")
        >>> response = await provider.search("python tutorials")
    """
    
    provider_name = "bing"
    
    # API endpoints
    BASE_URL = "https://api.bing.microsoft.com/v7.0"
    ENDPOINTS = {
        SearchType.WEB: "/search",
        SearchType.IMAGES: "/images/search",
        SearchType.NEWS: "/news/search",
        SearchType.VIDEOS: "/videos/search",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_rpm: int = 100,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Bing Web Search provider.
        
        Args:
            api_key: Bing Search API key (Ocp-Apim-Subscription-Key)
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
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get Bing capabilities."""
        return ProviderCapabilities(
            supports_pagination=True,
            supports_images=True,
            supports_news=True,
            supports_videos=True,
            supports_places=False,
            supports_shopping=False,
            supports_safe_search=True,
            supports_time_range=True,
            supports_country=True,
            supports_language=True,
            supports_knowledge_graph=False,
            supports_answer_box=False,
            supports_related_searches=True,
            max_results_per_page=50,  # Bing max is 50
            rate_limit_rpm=self.rate_limit_rpm,
            cost_per_request=0.003,  # $3 per 1000 queries (basic tier)
        )
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
    ) -> SearchResponse:
        """
        Perform a search query using Bing Web Search API.
        
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
        
        if not self.is_configured():
            raise SearchProviderError(
                "Bing Search not configured. Set BING_SEARCH_API_KEY environment variable.",
                provider=self.provider_name,
                error_code="NOT_CONFIGURED",
                recoverable=False,
            )
        
        await self._acquire_rate_limit()
        
        start_time = time.time()
        
        try:
            response = await self._execute_search(query, options)
            latency_ms = (time.time() - start_time) * 1000
            self._record_request(latency_ms, success=True)
            return response
            
        except SearchProviderError:
            raise
        except Exception as e:
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
        """Execute the actual search request."""
        try:
            import aiohttp
        except ImportError:
            raise SearchProviderError(
                "aiohttp not installed. Install with: pip install aiohttp",
                provider=self.provider_name,
                error_code="MISSING_DEPENDENCY",
                recoverable=False,
            )
        
        # Get endpoint for search type
        endpoint = self.ENDPOINTS.get(options.search_type, self.ENDPOINTS[SearchType.WEB])
        url = f"{self.BASE_URL}{endpoint}"
        
        params = self._build_params(query, options)
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
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
    
    def _build_params(self, query: str, options: SearchOptions) -> Dict[str, Any]:
        """Build request parameters for Bing API."""
        params: Dict[str, Any] = {
            "q": query,
            "count": min(options.max_results, 50),  # Max 50 per request
        }
        
        # Pagination
        if options.page > 1:
            params["offset"] = (options.page - 1) * options.max_results
        
        # Safe search
        safe_search_map = {
            SafeSearchLevel.OFF: "Off",
            SafeSearchLevel.MODERATE: "Moderate",
            SafeSearchLevel.STRICT: "Strict",
        }
        params["safeSearch"] = safe_search_map.get(options.safe_search, "Moderate")
        
        # Market (country + language)
        if options.country and options.language:
            params["mkt"] = f"{options.language}-{options.country.upper()}"
        elif options.country:
            params["cc"] = options.country.upper()
        
        # Time range (freshness)
        freshness_map = {
            TimeRange.DAY: "Day",
            TimeRange.WEEK: "Week",
            TimeRange.MONTH: "Month",
        }
        if options.time_range in freshness_map:
            params["freshness"] = freshness_map[options.time_range]
        
        return params
    
    def _parse_response(
        self,
        query: str,
        data: Dict[str, Any],
        options: SearchOptions,
    ) -> SearchResponse:
        """Parse Bing API response into SearchResponse."""
        results: List[SearchResult] = []
        
        # Parse based on search type
        if options.search_type == SearchType.WEB:
            results = self._parse_web_results(data)
        elif options.search_type == SearchType.IMAGES:
            results = self._parse_image_results(data)
        elif options.search_type == SearchType.NEWS:
            results = self._parse_news_results(data)
        elif options.search_type == SearchType.VIDEOS:
            results = self._parse_video_results(data)
        
        # Get total results
        web_pages = data.get("webPages", {})
        total_results = web_pages.get("totalEstimatedMatches", len(results) * 10)
        
        # Extract related searches
        metadata: Dict[str, Any] = {
            "search_type": options.search_type.value,
        }
        
        if "relatedSearches" in data:
            metadata["related_searches"] = [
                item.get("text", "") 
                for item in data.get("relatedSearches", {}).get("value", [])
            ]
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=total_results,
            search_time_ms=0,  # Bing doesn't return timing
            provider=SearchProvider.BING_API.value,
            metadata=metadata,
        )
    
    def _parse_web_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse web search results."""
        results = []
        
        web_pages = data.get("webPages", {}).get("value", [])
        
        for i, item in enumerate(web_pages, 1):
            title = item.get("name", "")
            url = item.get("url", "")
            snippet = clean_snippet(item.get("snippet", ""))
            
            if not title or not url:
                continue
            
            metadata = {}
            
            # Extract additional data
            if "dateLastCrawled" in item:
                metadata["date_crawled"] = item["dateLastCrawled"]
            if "displayUrl" in item:
                metadata["display_url"] = item["displayUrl"]
            if "deepLinks" in item:
                metadata["has_sitelinks"] = True
            
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
        
        images = data.get("value", [])
        
        for i, item in enumerate(images, 1):
            title = item.get("name", "")
            url = item.get("hostPageUrl", "")
            
            if not url:
                continue
            
            results.append(SearchResult(
                title=title or "Image",
                url=url,
                snippet=item.get("hostPageDisplayUrl", ""),
                position=i,
                source=self.provider_name,
                metadata={
                    "image_url": item.get("contentUrl", ""),
                    "thumbnail_url": item.get("thumbnailUrl", ""),
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "encoding_format": item.get("encodingFormat", ""),
                },
            ))
        
        return results
    
    def _parse_news_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse news search results."""
        results = []
        
        news = data.get("value", [])
        
        for i, item in enumerate(news, 1):
            title = item.get("name", "")
            url = item.get("url", "")
            snippet = clean_snippet(item.get("description", ""))
            
            if not title or not url:
                continue
            
            # Get provider info
            provider_info = item.get("provider", [{}])[0]
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                position=i,
                source=self.provider_name,
                metadata={
                    "date": item.get("datePublished", ""),
                    "news_source": provider_info.get("name", ""),
                    "category": item.get("category", ""),
                    "image_url": item.get("image", {}).get("thumbnail", {}).get("contentUrl", ""),
                },
            ))
        
        return results
    
    def _parse_video_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse video search results."""
        results = []
        
        videos = data.get("value", [])
        
        for i, item in enumerate(videos, 1):
            title = item.get("name", "")
            url = item.get("hostPageUrl", "") or item.get("contentUrl", "")
            
            if not title or not url:
                continue
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=item.get("description", ""),
                position=i,
                source=self.provider_name,
                metadata={
                    "duration": item.get("duration", ""),
                    "date": item.get("datePublished", ""),
                    "view_count": item.get("viewCount"),
                    "thumbnail_url": item.get("thumbnailUrl", ""),
                    "publisher": item.get("publisher", [{}])[0].get("name", ""),
                    "video_id": item.get("videoId", ""),
                },
            ))
        
        return results
