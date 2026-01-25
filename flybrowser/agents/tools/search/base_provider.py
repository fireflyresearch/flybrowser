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
Abstract Base Class for Search Providers.

This module defines the interface that all search providers must implement.
It provides a consistent API for performing searches across different
search engines and APIs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Deque

from flybrowser.agents.tools.search.types import (
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    SearchOptions,
    SearchType,
)
from flybrowser.agents.tools.search_utils import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for tracking request performance."""
    timestamp: datetime
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BaseSearchProvider(ABC):
    """
    Abstract base class for search providers.
    
    All search provider implementations must inherit from this class
    and implement the required abstract methods.
    
    Features:
        - Unified search interface
        - Built-in rate limiting
        - Health monitoring
        - Request metrics tracking
        - Cost estimation
    
    Example:
        >>> class MySearchProvider(BaseSearchProvider):
        ...     provider_name = "my_provider"
        ...     
        ...     async def search(self, query, options):
        ...         # Implementation
        ...         pass
    """
    
    # Must be overridden by subclasses
    provider_name: str = "base"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_rpm: int = 100,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the search provider.
        
        Args:
            api_key: API key for the search provider
            rate_limit_rpm: Maximum requests per minute
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.rate_limit_rpm = rate_limit_rpm
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Rate limiting
        self._request_times: Deque[float] = deque(maxlen=rate_limit_rpm)
        self._rate_limit_lock = asyncio.Lock()
        
        # Metrics tracking
        self._metrics: Deque[RequestMetrics] = deque(maxlen=1000)
        self._requests_today = 0
        self._last_reset_date = datetime.now().date()
        
        # Health status
        self._last_health_check: Optional[datetime] = None
        self._cached_health: Optional[ProviderHealth] = None
    
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities.
        
        Returns:
            ProviderCapabilities describing what this provider supports
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
    ) -> SearchResponse:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            options: Search options and filters
            
        Returns:
            SearchResponse with results
            
        Raises:
            SearchProviderError: If search fails
        """
        pass
    
    @abstractmethod
    async def _execute_search(
        self,
        query: str,
        options: SearchOptions,
    ) -> SearchResponse:
        """
        Internal method to execute the actual search.
        
        This is called by search() after rate limiting and validation.
        Subclasses must implement this method.
        
        Args:
            query: Search query string
            options: Search options
            
        Returns:
            SearchResponse with results
        """
        pass
    
    def is_configured(self) -> bool:
        """
        Check if the provider is properly configured.
        
        Returns:
            True if provider has required configuration (e.g., API key)
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    def supports_search_type(self, search_type: SearchType) -> bool:
        """
        Check if provider supports a specific search type.
        
        Args:
            search_type: Type of search to check
            
        Returns:
            True if search type is supported
        """
        caps = self.capabilities
        type_support = {
            SearchType.WEB: True,  # All providers support web
            SearchType.IMAGES: caps.supports_images,
            SearchType.NEWS: caps.supports_news,
            SearchType.VIDEOS: caps.supports_videos,
            SearchType.PLACES: caps.supports_places,
            SearchType.SHOPPING: caps.supports_shopping,
        }
        return type_support.get(search_type, False)
    
    async def _acquire_rate_limit(self) -> None:
        """
        Acquire rate limit slot. Blocks if rate limit is exceeded.
        """
        async with self._rate_limit_lock:
            now = time.time()
            
            # Remove old entries (older than 60 seconds)
            while self._request_times and now - self._request_times[0] > 60:
                self._request_times.popleft()
            
            # Check if we're at the limit
            if len(self._request_times) >= self.rate_limit_rpm:
                # Wait until oldest request expires
                wait_time = 60 - (now - self._request_times[0])
                if wait_time > 0:
                    logger.debug(f"{self.provider_name}: Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self._request_times.append(time.time())
    
    def _record_request(
        self,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Record request metrics.
        
        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request was successful
            error: Error message if failed
        """
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._requests_today = 0
            self._last_reset_date = today
        
        self._requests_today += 1
        
        metric = RequestMetrics(
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        self._metrics.append(metric)
    
    def get_health_status(self) -> ProviderHealth:
        """
        Get the current health status of the provider.
        
        Returns:
            ProviderHealth with current status and metrics
        """
        if not self._metrics:
            return ProviderHealth(
                status=ProviderStatus.UNKNOWN,
                last_check=self._last_health_check,
                requests_today=self._requests_today,
            )
        
        # Calculate metrics from recent requests
        recent_metrics = list(self._metrics)[-100:]  # Last 100 requests
        
        success_count = sum(1 for m in recent_metrics if m.success)
        success_rate = success_count / len(recent_metrics) if recent_metrics else 1.0
        
        avg_latency = (
            sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
            if recent_metrics else 0.0
        )
        
        # Determine status
        if success_rate >= 0.95:
            status = ProviderStatus.HEALTHY
        elif success_rate >= 0.8:
            status = ProviderStatus.DEGRADED
        else:
            status = ProviderStatus.UNHEALTHY
        
        # Get last error
        last_error = None
        for metric in reversed(recent_metrics):
            if metric.error:
                last_error = metric.error
                break
        
        return ProviderHealth(
            status=status,
            latency_ms=avg_latency,
            success_rate=success_rate,
            last_check=datetime.now(),
            error_message=last_error,
            requests_today=self._requests_today,
        )
    
    def estimate_cost(self, query_count: int = 1) -> float:
        """
        Estimate cost for a number of queries.
        
        Args:
            query_count: Number of queries to estimate
            
        Returns:
            Estimated cost in USD
        """
        return self.capabilities.cost_per_request * query_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get provider statistics.
        
        Returns:
            Dictionary with provider stats
        """
        health = self.get_health_status()
        
        return {
            "provider_name": self.provider_name,
            "configured": self.is_configured(),
            "health": health.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "requests_today": self._requests_today,
            "rate_limit_rpm": self.rate_limit_rpm,
            "estimated_daily_cost": self.estimate_cost(self._requests_today),
        }
    
    async def health_check(self) -> ProviderHealth:
        """
        Perform an active health check.
        
        This performs a simple search to verify the provider is working.
        
        Returns:
            ProviderHealth with current status
        """
        if not self.is_configured():
            return ProviderHealth(
                status=ProviderStatus.UNHEALTHY,
                error_message="Provider not configured (missing API key)",
                last_check=datetime.now(),
            )
        
        try:
            start = time.time()
            # Simple test search
            response = await self.search("test", SearchOptions(max_results=1))
            latency_ms = (time.time() - start) * 1000
            
            self._last_health_check = datetime.now()
            
            if response and response.results:
                return ProviderHealth(
                    status=ProviderStatus.HEALTHY,
                    latency_ms=latency_ms,
                    success_rate=1.0,
                    last_check=self._last_health_check,
                    requests_today=self._requests_today,
                )
            else:
                return ProviderHealth(
                    status=ProviderStatus.DEGRADED,
                    latency_ms=latency_ms,
                    error_message="Search returned no results",
                    last_check=self._last_health_check,
                    requests_today=self._requests_today,
                )
                
        except Exception as e:
            self._last_health_check = datetime.now()
            return ProviderHealth(
                status=ProviderStatus.UNHEALTHY,
                error_message=str(e),
                last_check=self._last_health_check,
                requests_today=self._requests_today,
            )
    
    def __repr__(self) -> str:
        """String representation."""
        configured = "configured" if self.is_configured() else "not configured"
        return f"{self.__class__.__name__}({self.provider_name}, {configured})"


class SearchProviderError(Exception):
    """Exception raised by search providers."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        error_code: Optional[str] = None,
        recoverable: bool = True,
    ) -> None:
        """
        Initialize search provider error.
        
        Args:
            message: Error message
            provider: Provider name that raised the error
            error_code: Optional error code
            recoverable: Whether the error is recoverable (can retry)
        """
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        return f"[{self.provider}] {super().__str__()}"
