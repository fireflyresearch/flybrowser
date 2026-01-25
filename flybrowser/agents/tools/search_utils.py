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
Search Utilities for FlyBrowser.

This module provides shared data structures and utilities for search tools,
supporting both API-based and human-like search implementations.

Data Structures:
    - SearchResult: Individual search result
    - SearchResponse: Complete search response with metadata
    
Utilities:
    - Query normalization
    - URL parsing and validation
    - Result ranking and filtering
    - Response formatting
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SearchEngine(str, Enum):
    """Supported search engines."""
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"
    CUSTOM = "custom"


class SearchProvider(str, Enum):
    """Search API providers."""
    GOOGLE_API = "google_api"
    DUCKDUCKGO_API = "duckduckgo_api"
    BING_API = "bing_api"
    SERP_API = "serp_api"
    BROWSER = "browser"


@dataclass
class SearchResult:
    """
    Individual search result.
    
    Represents a single result from a search query with
    title, URL, snippet, and additional metadata.
    
    Attributes:
        title: Result title/heading
        url: Target URL
        snippet: Text snippet/description
        position: Position in results (1-indexed)
        source: Search engine/provider that returned this result
        metadata: Additional data (featured snippet, rich results, etc.)
    """
    title: str
    url: str
    snippet: str
    position: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        try:
            parsed = urllib.parse.urlparse(self.url)
            return parsed.netloc
        except Exception:
            return ""
    
    @property
    def is_featured(self) -> bool:
        """Check if this is a featured snippet."""
        return self.metadata.get("featured", False)
    
    @property
    def has_rich_result(self) -> bool:
        """Check if this has rich result data."""
        return "rich_data" in self.metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "source": self.source,
            "domain": self.domain,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.position}. {self.title}\n   {self.url}\n   {self.snippet[:100]}..."


@dataclass
class SearchResponse:
    """
    Complete search response.
    
    Contains all results from a search query plus metadata
    about the search itself.
    
    Attributes:
        query: Original search query
        results: List of search results
        total_results: Estimated total results available
        search_time_ms: Time taken to execute search
        provider: Provider/engine used
        metadata: Additional response metadata
        timestamp: When search was executed
    """
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0.0
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.results)
    
    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the top result."""
        return self.results[0] if self.results else None
    
    @property
    def domains(self) -> Set[str]:
        """Get unique domains in results."""
        return {r.domain for r in self.results if r.domain}
    
    def get_results_by_domain(self, domain: str) -> List[SearchResult]:
        """Get all results from a specific domain."""
        return [r for r in self.results if r.domain == domain]
    
    def filter_results(
        self,
        max_results: Optional[int] = None,
        exclude_domains: Optional[List[str]] = None,
        min_position: Optional[int] = None,
        max_position: Optional[int] = None,
    ) -> SearchResponse:
        """
        Filter results based on criteria.
        
        Args:
            max_results: Maximum number of results to keep
            exclude_domains: Domains to exclude
            min_position: Minimum position (inclusive)
            max_position: Maximum position (inclusive)
            
        Returns:
            New SearchResponse with filtered results
        """
        filtered = self.results.copy()
        
        if exclude_domains:
            filtered = [r for r in filtered if r.domain not in exclude_domains]
        
        if min_position is not None:
            filtered = [r for r in filtered if r.position >= min_position]
        
        if max_position is not None:
            filtered = [r for r in filtered if r.position <= max_position]
        
        if max_results is not None:
            filtered = filtered[:max_results]
        
        return SearchResponse(
            query=self.query,
            results=filtered,
            total_results=self.total_results,
            search_time_ms=self.search_time_ms,
            provider=self.provider,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "result_count": self.result_count,
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
            "provider": self.provider,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    def format_brief(self, max_results: int = 5) -> str:
        """Format as brief text summary."""
        lines = [
            f"Search: {self.query}",
            f"Provider: {self.provider}",
            f"Results: {self.result_count} (of ~{self.total_results})",
            f"Time: {self.search_time_ms:.0f}ms",
            "",
            "Top Results:",
        ]
        
        for result in self.results[:max_results]:
            lines.append(f"  {result.position}. {result.title}")
            lines.append(f"     {result.url}")
        
        return "\n".join(lines)


def normalize_query(query: str) -> str:
    """
    Normalize a search query.
    
    Removes extra whitespace, normalizes quotes, and cleans up the query.
    
    Args:
        query: Raw search query
        
    Returns:
        Normalized query string
    """
    # Strip leading/trailing whitespace
    query = query.strip()
    
    # Normalize whitespace (multiple spaces -> single space)
    query = re.sub(r'\s+', ' ', query)
    
    # Normalize quotes (smart quotes -> straight quotes)
    query = query.replace('"', '"').replace('"', '"')
    query = query.replace(''', "'").replace(''', "'")
    
    return query


def parse_search_url(url: str) -> Dict[str, str]:
    """
    Parse a search engine URL to extract query and parameters.
    
    Args:
        url: Search engine URL
        
    Returns:
        Dictionary with 'query' and other parameters
    """
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    
    result = {}
    
    # Extract query based on search engine
    if 'google.com' in parsed.netloc:
        result['query'] = params.get('q', [''])[0]
        result['engine'] = 'google'
    elif 'duckduckgo.com' in parsed.netloc:
        result['query'] = params.get('q', [''])[0]
        result['engine'] = 'duckduckgo'
    elif 'bing.com' in parsed.netloc:
        result['query'] = params.get('q', [''])[0]
        result['engine'] = 'bing'
    else:
        result['query'] = params.get('q', params.get('query', ['']))[0]
        result['engine'] = 'unknown'
    
    return result


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def clean_snippet(snippet: str, max_length: int = 300) -> str:
    """
    Clean and truncate a snippet.
    
    Args:
        snippet: Raw snippet text
        max_length: Maximum length
        
    Returns:
        Cleaned snippet
    """
    # Remove extra whitespace
    snippet = re.sub(r'\s+', ' ', snippet.strip())
    
    # Remove common artifacts
    snippet = snippet.replace('...', ' ')
    snippet = snippet.replace('…', ' ')
    
    # Truncate if needed
    if len(snippet) > max_length:
        snippet = snippet[:max_length].rsplit(' ', 1)[0] + '...'
    
    return snippet


def rank_results(
    results: List[SearchResult],
    boost_domains: Optional[List[str]] = None,
    penalize_domains: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    Re-rank search results based on criteria.
    
    Args:
        results: Original results
        boost_domains: Domains to boost in ranking
        penalize_domains: Domains to penalize in ranking
        
    Returns:
        Re-ranked results
    """
    def score(result: SearchResult) -> float:
        # Start with inverse position (lower position = higher score)
        s = 100.0 / result.position
        
        # Boost featured snippets
        if result.is_featured:
            s *= 1.5
        
        # Boost/penalize domains
        if boost_domains and result.domain in boost_domains:
            s *= 1.3
        if penalize_domains and result.domain in penalize_domains:
            s *= 0.7
        
        return s
    
    # Sort by score (descending)
    ranked = sorted(results, key=score, reverse=True)
    
    # Update positions
    for i, result in enumerate(ranked, 1):
        result.position = i
    
    return ranked


def merge_results(
    responses: List[SearchResponse],
    max_results: int = 10,
    deduplicate: bool = True,
) -> SearchResponse:
    """
    Merge multiple search responses into one.
    
    Useful for combining results from multiple providers.
    
    Args:
        responses: List of search responses to merge
        max_results: Maximum results to keep
        deduplicate: Remove duplicate URLs
        
    Returns:
        Merged SearchResponse
    """
    if not responses:
        return SearchResponse(query="", results=[])
    
    # Use query from first response
    query = responses[0].query
    
    # Collect all results
    all_results = []
    for response in responses:
        all_results.extend(response.results)
    
    # Deduplicate by URL if requested
    if deduplicate:
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        all_results = unique_results
    
    # Sort by original position and take top N
    all_results.sort(key=lambda r: r.position)
    all_results = all_results[:max_results]
    
    # Update positions
    for i, result in enumerate(all_results, 1):
        result.position = i
    
    # Calculate total time
    total_time = sum(r.search_time_ms for r in responses)
    
    return SearchResponse(
        query=query,
        results=all_results,
        total_results=max(r.total_results for r in responses),
        search_time_ms=total_time,
        provider="merged",
        metadata={
            "source_providers": [r.provider for r in responses],
            "merged_count": len(responses),
        },
    )
