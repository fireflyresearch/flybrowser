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
Type definitions for the Search Abstraction Layer.

This module defines all types, enums, and dataclasses used throughout
the search system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SearchType(str, Enum):
    """Types of search queries supported."""
    WEB = "web"
    IMAGES = "images"
    NEWS = "news"
    VIDEOS = "videos"
    PLACES = "places"
    SHOPPING = "shopping"


class ProviderStatus(str, Enum):
    """Health status of a search provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SafeSearchLevel(str, Enum):
    """Safe search filtering levels."""
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class TimeRange(str, Enum):
    """Time range filters for search results."""
    ANY = "any"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class SearchOptions:
    """
    Configuration options for search queries.
    
    Attributes:
        search_type: Type of search (web, images, news, etc.)
        max_results: Maximum number of results to return
        page: Page number for pagination (1-indexed)
        safe_search: Safe search filtering level
        time_range: Time range filter
        country: Country code for localized results (e.g., "us", "gb")
        language: Language code for results (e.g., "en", "es")
        include_related: Include related searches
        include_knowledge_graph: Include knowledge graph data
        include_answer_box: Include answer box/featured snippet
        site_filter: Limit results to specific site (e.g., "site:github.com")
        exclude_sites: Sites to exclude from results
        file_type: Filter by file type (e.g., "pdf", "doc")
    """
    search_type: SearchType = SearchType.WEB
    max_results: int = 10
    page: int = 1
    safe_search: SafeSearchLevel = SafeSearchLevel.MODERATE
    time_range: TimeRange = TimeRange.ANY
    country: Optional[str] = None
    language: Optional[str] = None
    include_related: bool = False
    include_knowledge_graph: bool = False
    include_answer_box: bool = True
    site_filter: Optional[str] = None
    exclude_sites: List[str] = field(default_factory=list)
    file_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary."""
        return {
            "search_type": self.search_type.value,
            "max_results": self.max_results,
            "page": self.page,
            "safe_search": self.safe_search.value,
            "time_range": self.time_range.value,
            "country": self.country,
            "language": self.language,
            "include_related": self.include_related,
            "include_knowledge_graph": self.include_knowledge_graph,
            "include_answer_box": self.include_answer_box,
            "site_filter": self.site_filter,
            "exclude_sites": self.exclude_sites,
            "file_type": self.file_type,
        }


@dataclass
class ProviderHealth:
    """
    Health status of a search provider.
    
    Attributes:
        status: Current health status
        latency_ms: Average response latency in milliseconds
        success_rate: Success rate (0.0 to 1.0)
        last_check: Timestamp of last health check
        error_message: Error message if unhealthy
        requests_today: Number of requests made today
        quota_remaining: Remaining API quota (if applicable)
    """
    status: ProviderStatus
    latency_ms: float = 0.0
    success_rate: float = 1.0
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    requests_today: int = 0
    quota_remaining: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "success_rate": self.success_rate,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "error_message": self.error_message,
            "requests_today": self.requests_today,
            "quota_remaining": self.quota_remaining,
        }


@dataclass
class RankedSearchResult:
    """
    A search result with ranking scores.
    
    Extends the base SearchResult with ranking information.
    
    Attributes:
        title: Result title
        url: Result URL
        snippet: Text snippet/description
        position: Original position from provider
        source: Provider name
        relevance_score: Combined relevance score (0.0 to 1.0)
        ranking_signals: Individual ranking signal scores
        metadata: Additional metadata (featured, rich data, etc.)
    """
    title: str
    url: str
    snippet: str
    position: int
    source: str
    relevance_score: float = 0.0
    ranking_signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            return parsed.netloc
        except Exception:
            return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "source": self.source,
            "domain": self.domain,
            "relevance_score": self.relevance_score,
            "ranking_signals": self.ranking_signals,
            "metadata": self.metadata,
        }


@dataclass
class SearchAgentResponse:
    """
    Response from SearchAgent including results and metadata.
    
    Attributes:
        query: Original search query
        results: Ranked search results
        total_results: Estimated total results available
        search_time_ms: Total time to complete search
        provider_used: Provider that returned results
        fallback_used: Whether fallback provider was used
        cached: Whether response was served from cache
        ranking_applied: Whether ranking was applied
        answer_box: Answer box content if available
        knowledge_graph: Knowledge graph data if available
        related_searches: Related search suggestions
        metadata: Additional response metadata
    """
    query: str
    results: List[RankedSearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0.0
    provider_used: str = ""
    fallback_used: bool = False
    cached: bool = False
    ranking_applied: bool = False
    answer_box: Optional[Dict[str, Any]] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    related_searches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.results)
    
    @property
    def top_result(self) -> Optional[RankedSearchResult]:
        """Get top-ranked result."""
        return self.results[0] if self.results else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "result_count": self.result_count,
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
            "provider_used": self.provider_used,
            "fallback_used": self.fallback_used,
            "cached": self.cached,
            "ranking_applied": self.ranking_applied,
            "answer_box": self.answer_box,
            "knowledge_graph": self.knowledge_graph,
            "related_searches": self.related_searches,
            "metadata": self.metadata,
        }


@dataclass 
class ProviderCapabilities:
    """
    Capabilities supported by a search provider.
    
    Attributes:
        supports_pagination: Whether pagination is supported
        supports_images: Whether image search is supported
        supports_news: Whether news search is supported
        supports_videos: Whether video search is supported
        supports_places: Whether places/local search is supported
        supports_shopping: Whether shopping search is supported
        supports_safe_search: Whether safe search filtering is supported
        supports_time_range: Whether time range filtering is supported
        supports_country: Whether country localization is supported
        supports_language: Whether language filtering is supported
        supports_knowledge_graph: Whether knowledge graph is returned
        supports_answer_box: Whether answer box is returned
        supports_related_searches: Whether related searches are returned
        max_results_per_page: Maximum results per page
        rate_limit_rpm: Requests per minute limit
        cost_per_request: Cost per request in USD
    """
    supports_pagination: bool = True
    supports_images: bool = False
    supports_news: bool = False
    supports_videos: bool = False
    supports_places: bool = False
    supports_shopping: bool = False
    supports_safe_search: bool = True
    supports_time_range: bool = False
    supports_country: bool = True
    supports_language: bool = True
    supports_knowledge_graph: bool = False
    supports_answer_box: bool = False
    supports_related_searches: bool = False
    max_results_per_page: int = 10
    rate_limit_rpm: int = 100
    cost_per_request: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supports_pagination": self.supports_pagination,
            "supports_images": self.supports_images,
            "supports_news": self.supports_news,
            "supports_videos": self.supports_videos,
            "supports_places": self.supports_places,
            "supports_shopping": self.supports_shopping,
            "supports_safe_search": self.supports_safe_search,
            "supports_time_range": self.supports_time_range,
            "supports_country": self.supports_country,
            "supports_language": self.supports_language,
            "supports_knowledge_graph": self.supports_knowledge_graph,
            "supports_answer_box": self.supports_answer_box,
            "supports_related_searches": self.supports_related_searches,
            "max_results_per_page": self.max_results_per_page,
            "rate_limit_rpm": self.rate_limit_rpm,
            "cost_per_request": self.cost_per_request,
        }
