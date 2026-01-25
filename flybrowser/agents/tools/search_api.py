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
API-Based Search Tool for FlyBrowser.

This tool performs web searches using official APIs to avoid bot detection:
- Google Custom Search API (requires API key + CX)
- Bing Web Search API (requires API key)

Note: DuckDuckGo API has been deprecated due to reliability issues.
Use human-like search (search_human) when no API keys are configured.

Features:
- Automatic provider fallback
- Rate limiting and caching
- Structured result extraction
- No browser overhead

Usage:
    tool = SearchAPITool(page_controller)
    result = await tool.execute(query="python tutorials", provider="auto")
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlencode

from .base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from .search_utils import (
    SearchResponse,
    SearchResult,
    SearchProvider,
    normalize_query,
    clean_snippet,
)

if TYPE_CHECKING:
    from flybrowser.core.page import PageController

logger = logging.getLogger(__name__)


class SearchAPITool(BaseTool):
    """
    API-based search tool.
    
    Performs web searches using official APIs to avoid bot detection.
    Supports multiple providers with automatic fallback.
    """
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        """Initialize the search API tool."""
        super().__init__(page_controller)
        
        # Load API keys from environment
        self.google_api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
        self.bing_api_key = os.getenv("BING_SEARCH_API_KEY")
        
        # Simple cache for rate limiting
        self._cache: Dict[str, SearchResponse] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_request_time: Dict[str, float] = {}
    
    def has_api_keys_configured(self) -> bool:
        """
        Check if any search API keys are configured.
        
        Returns:
            True if Google or Bing API keys are available
            
        Note:
            DuckDuckGo API has proven unreliable in practice.
            Without proper API keys, use human-like search instead.
        """
        has_google = bool(self.google_api_key and self.google_cx)
        has_bing = bool(self.bing_api_key)
        return has_google or has_bing
    
    @property
    def metadata(self) -> ToolMetadata:
        """Tool metadata."""
        return ToolMetadata(
            name="search_api",
            description=(
                "Perform web search using APIs (Google, Bing). "
                "Requires API keys. Fast, reliable, avoids bot detection. "
                "Use search_human if no API keys available."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="provider",
                    type="string",
                    description="Search provider: 'auto', 'google', 'bing'",
                    required=False,
                    default="auto",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return (1-20)",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="safe_search",
                    type="boolean",
                    description="Enable safe search filtering",
                    required=False,
                    default=True,
                ),
            ],
            examples=['search_api(query="python tutorials", provider="auto", max_results=10)'],
        )
    
    async def execute(
        self,
        query: str,
        provider: str = "auto",
        max_results: int = 10,
        safe_search: bool = True,
        **kwargs
    ) -> ToolResult:
        """
        Execute API-based search.
        
        Args:
            query: Search query
            provider: Provider to use ('auto', 'google', 'duckduckgo', 'bing')
            max_results: Maximum results to return
            safe_search: Enable safe search
            
        Returns:
            ToolResult with SearchResponse data
        """
        start_time = time.time()
        
        try:
            # Normalize query
            query = normalize_query(query)
            
            # Check cache
            cache_key = f"{provider}:{query}:{max_results}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if time.time() - cached.timestamp < self._cache_ttl:
                    logger.info(f"Returning cached results for: {query}")
                    return ToolResult.success_result(
                        data=cached.to_dict(),
                        message=f"Found {cached.result_count} results (cached)",
                    )
            
            # Determine providers to try
            providers = self._get_providers_to_try(provider)
            
            # Try each provider until one succeeds
            last_error = None
            for prov in providers:
                try:
                    logger.info(f"Trying provider: {prov}")
                    response = await self._search_with_provider(
                        query, prov, max_results, safe_search
                    )
                    
                    if response and response.results:
                        # Cache the response
                        self._cache[cache_key] = response
                        
                        elapsed_ms = (time.time() - start_time) * 1000
                        response.search_time_ms = elapsed_ms
                        
                        return ToolResult.success_result(
                            data=response.to_dict(),
                            message=f"Found {response.result_count} results via {prov}",
                            metadata={"provider": prov},
                        )
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Provider {prov} failed: {e}")
                    continue
            
            # All providers failed
            return ToolResult.error_result(
                error=f"All providers failed. Last error: {last_error}",
                error_code="SEARCH_FAILED",
            )
        
        except Exception as e:
            logger.exception(f"Search API execution failed: {e}")
            return ToolResult.error_result(
                error=f"Search execution error: {str(e)}",
                error_code="EXECUTION_ERROR",
            )
    
    def _get_providers_to_try(self, provider: str) -> List[str]:
        """
        Determine which providers to try.
        
        Args:
            provider: Requested provider or 'auto'
            
        Returns:
            List of provider names to try in order
        """
        if provider == "auto":
            # Try available providers with API keys
            # DuckDuckGo API is disabled due to reliability issues
            providers = []
            if self.google_api_key and self.google_cx:
                providers.append("google")
            if self.bing_api_key:
                providers.append("bing")
            
            # If no keys available, return empty - caller should fall back to human-like
            if not providers:
                logger.warning(
                    "No API keys configured. Use human-like search (search_human) instead. "
                    "Set GOOGLE_CUSTOM_SEARCH_API_KEY/CX or BING_SEARCH_API_KEY to enable API search."
                )
            return providers
        else:
            # Warn if trying to use DuckDuckGo
            if provider == "duckduckgo":
                logger.warning("DuckDuckGo API is disabled due to reliability issues. Use google or bing instead.")
                return []
            return [provider]
    
    async def _search_with_provider(
        self,
        query: str,
        provider: str,
        max_results: int,
        safe_search: bool,
    ) -> Optional[SearchResponse]:
        """
        Search with a specific provider.
        
        Args:
            query: Search query
            provider: Provider name
            max_results: Max results
            safe_search: Safe search enabled
            
        Returns:
            SearchResponse or None
        """
        if provider == "google":
            return await self._search_google(query, max_results, safe_search)
        elif provider == "bing":
            return await self._search_bing(query, max_results, safe_search)
        elif provider == "duckduckgo":
            # DuckDuckGo API is deprecated due to reliability issues
            logger.error("DuckDuckGo API is disabled. Use google or bing, or use human-like search instead.")
            return None
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> Optional[SearchResponse]:
        """
        Search using DuckDuckGo Instant Answer API.
        
        Free, no API key required.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            SearchResponse
        """
        try:
            import aiohttp
            
            # DuckDuckGo HTML search (more results than Instant Answer)
            url = f"https://html.duckduckgo.com/html/?{urlencode({'q': query})}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"User-Agent": "FlyBrowser/1.0"}) as resp:
                    if resp.status != 200:
                        raise Exception(f"DDG returned status {resp.status}")
                    
                    html = await resp.text()
                    
                    # Parse HTML to extract results
                    results = self._parse_duckduckgo_html(html, max_results)
                    
                    return SearchResponse(
                        query=query,
                        results=results,
                        total_results=len(results) * 10,  # Estimate
                        provider=SearchProvider.DUCKDUCKGO_API.value,
                    )
        
        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return None
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return None
    
    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML to extract results."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Find result divs
            result_divs = soup.find_all('div', class_='result')[:max_results]
            
            for i, div in enumerate(result_divs, 1):
                try:
                    # Extract title and URL
                    title_elem = div.find('a', class_='result__a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    snippet = clean_snippet(snippet)
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        position=i,
                        source="duckduckgo",
                    ))
                
                except Exception as e:
                    logger.debug(f"Failed to parse DDG result: {e}")
                    continue
            
            return results
        
        except ImportError:
            logger.error("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            return []
        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo HTML: {e}")
            return []
    
    async def _search_google(
        self,
        query: str,
        max_results: int,
        safe_search: bool,
    ) -> Optional[SearchResponse]:
        """
        Search using Google Custom Search API.
        
        Requires GOOGLE_CUSTOM_SEARCH_API_KEY and GOOGLE_CUSTOM_SEARCH_CX env vars.
        
        Args:
            query: Search query
            max_results: Maximum results
            safe_search: Safe search enabled
            
        Returns:
            SearchResponse
        """
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google API key or CX not configured")
            return None
        
        try:
            import aiohttp
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cx,
                "q": query,
                "num": min(max_results, 10),  # Google max is 10 per request
                "safe": "active" if safe_search else "off",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise Exception(f"Google API returned status {resp.status}")
                    
                    data = await resp.json()
                    
                    results = []
                    for i, item in enumerate(data.get('items', []), 1):
                        results.append(SearchResult(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=clean_snippet(item.get('snippet', '')),
                            position=i,
                            source="google",
                            metadata={
                                "featured": item.get('pagemap', {}).get('metatags') is not None,
                            },
                        ))
                    
                    total_results = int(data.get('searchInformation', {}).get('totalResults', '0'))
                    
                    return SearchResponse(
                        query=query,
                        results=results,
                        total_results=total_results,
                        provider=SearchProvider.GOOGLE_API.value,
                    )
        
        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return None
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return None
    
    async def _search_bing(
        self,
        query: str,
        max_results: int,
        safe_search: bool,
    ) -> Optional[SearchResponse]:
        """
        Search using Bing Web Search API.
        
        Requires BING_SEARCH_API_KEY env var.
        
        Args:
            query: Search query
            max_results: Maximum results
            safe_search: Safe search enabled
            
        Returns:
            SearchResponse
        """
        if not self.bing_api_key:
            logger.warning("Bing API key not configured")
            return None
        
        try:
            import aiohttp
            
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
            params = {
                "q": query,
                "count": min(max_results, 50),  # Bing max is 50
                "safeSearch": "Strict" if safe_search else "Off",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        raise Exception(f"Bing API returned status {resp.status}")
                    
                    data = await resp.json()
                    
                    results = []
                    for i, item in enumerate(data.get('webPages', {}).get('value', []), 1):
                        results.append(SearchResult(
                            title=item.get('name', ''),
                            url=item.get('url', ''),
                            snippet=clean_snippet(item.get('snippet', '')),
                            position=i,
                            source="bing",
                        ))
                    
                    total_results = data.get('webPages', {}).get('totalEstimatedMatches', 0)
                    
                    return SearchResponse(
                        query=query,
                        results=results,
                        total_results=total_results,
                        provider=SearchProvider.BING_API.value,
                    )
        
        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return None
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return None
