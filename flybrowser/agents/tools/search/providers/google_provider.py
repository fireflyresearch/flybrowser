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
Google Custom Search API Provider.

Google Custom Search API provides:
- Web search with organic results
- Image search
- Safe search filtering
- Country and language localization

Pricing: $5 per 1000 queries (first 100 queries/day free)
Documentation: https://developers.google.com/custom-search/v1/overview
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
)
from flybrowser.agents.tools.search_utils import (
    SearchResponse,
    SearchResult,
    SearchProvider,
    clean_snippet,
)

logger = logging.getLogger(__name__)


class GoogleProvider(BaseSearchProvider):
    """
    Google Custom Search API provider.
    
    Provides access to Google search through the Custom Search JSON API.
    Requires both an API key and a Custom Search Engine ID (CX).
    
    Example:
        >>> provider = GoogleProvider(
        ...     api_key="your-api-key",
        ...     cx="your-custom-search-engine-id"
        ... )
        >>> response = await provider.search("python tutorials")
    """
    
    provider_name = "google"
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        rate_limit_rpm: int = 100,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Google Custom Search provider.
        
        Args:
            api_key: Google API key
            cx: Custom Search Engine ID
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
        self.cx = cx
    
    def is_configured(self) -> bool:
        """Check if provider is configured with API key and CX."""
        return (
            self.api_key is not None and len(self.api_key) > 0 and
            self.cx is not None and len(self.cx) > 0
        )
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get Google Custom Search capabilities."""
        return ProviderCapabilities(
            supports_pagination=True,
            supports_images=True,
            supports_news=False,  # Not directly supported
            supports_videos=False,
            supports_places=False,
            supports_shopping=False,
            supports_safe_search=True,
            supports_time_range=False,  # Limited support
            supports_country=True,
            supports_language=True,
            supports_knowledge_graph=False,
            supports_answer_box=False,
            supports_related_searches=False,
            max_results_per_page=10,  # Google max is 10 per request
            rate_limit_rpm=self.rate_limit_rpm,
            cost_per_request=0.005,  # $5 per 1000 queries
        )
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
    ) -> SearchResponse:
        """
        Perform a search query using Google Custom Search API.
        
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
                "Google Custom Search not configured. Set GOOGLE_CUSTOM_SEARCH_API_KEY and GOOGLE_CUSTOM_SEARCH_CX.",
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
        
        params = self._build_params(query, options)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.BASE_URL,
                params=params,
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
                        "Rate limit exceeded or quota exhausted",
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
        """Build request parameters for Google API."""
        params: Dict[str, Any] = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(options.max_results, 10),  # Max 10 per request
        }
        
        # Pagination
        if options.page > 1:
            params["start"] = (options.page - 1) * 10 + 1
        
        # Safe search
        if options.safe_search == SafeSearchLevel.OFF:
            params["safe"] = "off"
        elif options.safe_search == SafeSearchLevel.STRICT:
            params["safe"] = "active"
        else:
            params["safe"] = "medium"
        
        # Country
        if options.country:
            params["gl"] = options.country.lower()
        
        # Language
        if options.language:
            params["hl"] = options.language.lower()
        
        # Image search
        if options.search_type == SearchType.IMAGES:
            params["searchType"] = "image"
        
        # File type
        if options.file_type:
            params["fileType"] = options.file_type
        
        # Site filter
        if options.site_filter:
            params["siteSearch"] = options.site_filter
            params["siteSearchFilter"] = "i"  # include
        
        return params
    
    def _parse_response(
        self,
        query: str,
        data: Dict[str, Any],
        options: SearchOptions,
    ) -> SearchResponse:
        """Parse Google API response into SearchResponse."""
        results: List[SearchResult] = []
        
        for i, item in enumerate(data.get("items", []), 1):
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = clean_snippet(item.get("snippet", ""))
            
            if not title or not url:
                continue
            
            metadata = {}
            
            # Extract additional data
            if "pagemap" in item:
                pagemap = item["pagemap"]
                if "metatags" in pagemap:
                    metadata["has_metatags"] = True
                if "cse_image" in pagemap:
                    images = pagemap["cse_image"]
                    if images:
                        metadata["thumbnail"] = images[0].get("src", "")
            
            # Image-specific metadata
            if options.search_type == SearchType.IMAGES:
                if "image" in item:
                    img = item["image"]
                    metadata["width"] = img.get("width")
                    metadata["height"] = img.get("height")
                    metadata["thumbnail_url"] = img.get("thumbnailLink", "")
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                position=i,
                source=self.provider_name,
                metadata=metadata,
            ))
        
        # Get total results
        search_info = data.get("searchInformation", {})
        total_results = int(search_info.get("totalResults", "0"))
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=total_results,
            search_time_ms=float(search_info.get("searchTime", 0)) * 1000,
            provider=SearchProvider.GOOGLE_API.value,
            metadata={
                "search_type": options.search_type.value,
                "formatted_total_results": search_info.get("formattedTotalResults", ""),
            },
        )
