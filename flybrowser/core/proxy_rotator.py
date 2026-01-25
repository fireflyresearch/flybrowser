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
Proxy Rotation System for FlyBrowser.

This module provides intelligent proxy rotation to avoid IP-based blocking
and rate limiting. It supports:

- Round-robin rotation
- Health checking and failover
- Geographic distribution
- Performance tracking
- Automatic proxy validation

Example:
    >>> from flybrowser.core.proxy_rotator import ProxyRotator, ProxyConfig
    >>> 
    >>> proxies = [
    ...     ProxyConfig(
    ...         server="http://proxy1.example.com:8080",
    ...         username="user1",
    ...         password="pass1"
    ...     ),
    ...     ProxyConfig(
    ...         server="http://proxy2.example.com:8080",
    ...         username="user2",
    ...         password="pass2"
    ...     ),
    ... ]
    >>> 
    >>> rotator = ProxyRotator(proxies)
    >>> proxy = rotator.get_next_proxy()
    >>> print(f"Using: {proxy.server}")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlparse

from flybrowser.utils.logger import logger


class ProxyProtocol(str, Enum):
    """Proxy protocol type."""
    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"


class ProxyStatus(str, Enum):
    """Proxy health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNTESTED = "untested"


@dataclass
class ProxyConfig:
    """
    Configuration for a single proxy.
    
    Attributes:
        server: Proxy server URL (e.g., "http://proxy.example.com:8080")
        username: Optional authentication username
        password: Optional authentication password
        protocol: Proxy protocol (http, https, socks5)
        country: Optional country code for geographic routing
        max_failures: Maximum consecutive failures before marking as failed
        timeout_seconds: Connection timeout in seconds
    """
    server: str
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: ProxyProtocol = ProxyProtocol.HTTP
    country: Optional[str] = None
    max_failures: int = 3
    timeout_seconds: float = 10.0
    
    # Internal state (not for user configuration)
    status: ProxyStatus = field(default=ProxyStatus.UNTESTED, init=False)
    consecutive_failures: int = field(default=0, init=False)
    total_requests: int = field(default=0, init=False)
    successful_requests: int = field(default=0, init=False)
    avg_response_time_ms: float = field(default=0.0, init=False)
    last_used: float = field(default=0.0, init=False)
    last_failed: float = field(default=0.0, init=False)
    
    def to_playwright_format(self) -> Dict[str, str]:
        """Convert to Playwright proxy format."""
        proxy_dict = {"server": self.server}
        
        if self.username:
            proxy_dict["username"] = self.username
        if self.password:
            proxy_dict["password"] = self.password
        
        return proxy_dict
    
    def get_success_rate(self) -> float:
        """Get success rate (0.0 - 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def mark_success(self, response_time_ms: float) -> None:
        """Mark a successful request."""
        self.consecutive_failures = 0
        self.total_requests += 1
        self.successful_requests += 1
        self.last_used = time.time()
        
        # Update rolling average response time
        if self.avg_response_time_ms == 0:
            self.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            self.avg_response_time_ms = (0.7 * self.avg_response_time_ms) + (0.3 * response_time_ms)
        
        # Update status
        if self.status == ProxyStatus.FAILED:
            self.status = ProxyStatus.DEGRADED
        elif self.status == ProxyStatus.DEGRADED and self.consecutive_failures == 0:
            self.status = ProxyStatus.HEALTHY
    
    def mark_failure(self, error: str = "") -> None:
        """Mark a failed request."""
        self.consecutive_failures += 1
        self.total_requests += 1
        self.last_failed = time.time()
        
        # Update status based on failure count
        if self.consecutive_failures >= self.max_failures:
            self.status = ProxyStatus.FAILED
            logger.warning(f"[PROXY] Proxy {self.server} marked as FAILED after {self.consecutive_failures} failures")
        elif self.consecutive_failures > 1:
            self.status = ProxyStatus.DEGRADED
    
    def is_available(self) -> bool:
        """Check if proxy is available for use."""
        return self.status != ProxyStatus.FAILED
    
    def reset(self) -> None:
        """Reset proxy status and counters."""
        self.status = ProxyStatus.UNTESTED
        self.consecutive_failures = 0


class RotationStrategy(str, Enum):
    """Proxy rotation strategy."""
    ROUND_ROBIN = "round_robin"  # Rotate through proxies in order
    RANDOM = "random"  # Pick random proxy
    LEAST_USED = "least_used"  # Use proxy with least requests
    BEST_PERFORMANCE = "best_performance"  # Use fastest proxy
    GEOGRAPHIC = "geographic"  # Rotate based on geography


class ProxyRotator:
    """
    Manages proxy rotation for avoiding IP-based blocking.
    
    Features:
    - Multiple rotation strategies
    - Health checking and automatic failover
    - Performance tracking
    - Geographic distribution
    - Automatic recovery of failed proxies
    
    Example:
        >>> rotator = ProxyRotator(proxies, strategy=RotationStrategy.ROUND_ROBIN)
        >>> 
        >>> # Get next proxy
        >>> proxy = rotator.get_next_proxy()
        >>> 
        >>> # Mark success/failure
        >>> rotator.mark_success(proxy, response_time_ms=250)
        >>> rotator.mark_failure(proxy, "Connection timeout")
    """
    
    def __init__(
        self,
        proxies: List[ProxyConfig],
        strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
        enable_health_check: bool = True,
        health_check_interval: float = 300.0,  # 5 minutes
        enable_auto_recovery: bool = True,
        recovery_interval: float = 600.0,  # 10 minutes
    ):
        """
        Initialize the ProxyRotator.
        
        Args:
            proxies: List of proxy configurations
            strategy: Rotation strategy to use
            enable_health_check: Whether to perform periodic health checks
            health_check_interval: Interval between health checks (seconds)
            enable_auto_recovery: Whether to auto-recover failed proxies
            recovery_interval: Interval to retry failed proxies (seconds)
        """
        if not proxies:
            raise ValueError("At least one proxy must be provided")
        
        self.proxies = proxies
        self.strategy = strategy
        self.enable_health_check = enable_health_check
        self.health_check_interval = health_check_interval
        self.enable_auto_recovery = enable_auto_recovery
        self.recovery_interval = recovery_interval
        
        # State
        self._current_index = 0
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info(f"[PROXY] Initialized with {len(proxies)} proxies, strategy: {strategy.value}")
    
    async def start(self) -> None:
        """Start background tasks (health checking, etc.)."""
        if self.enable_health_check and self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("[PROXY] Started health check background task")
    
    async def stop(self) -> None:
        """Stop background tasks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("[PROXY] Stopped health check background task")
    
    def get_next_proxy(self, country: Optional[str] = None) -> Optional[ProxyConfig]:
        """
        Get the next proxy to use based on the rotation strategy.
        
        Args:
            country: Optional country code to filter proxies
            
        Returns:
            ProxyConfig or None if no proxies available
        """
        available_proxies = [p for p in self.proxies if p.is_available()]
        
        # Filter by country if specified
        if country:
            country_proxies = [p for p in available_proxies if p.country == country]
            if country_proxies:
                available_proxies = country_proxies
        
        if not available_proxies:
            logger.error("[PROXY] No available proxies!")
            return None
        
        # Select based on strategy
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            proxy = self._round_robin_select(available_proxies)
        elif self.strategy == RotationStrategy.RANDOM:
            proxy = self._random_select(available_proxies)
        elif self.strategy == RotationStrategy.LEAST_USED:
            proxy = self._least_used_select(available_proxies)
        elif self.strategy == RotationStrategy.BEST_PERFORMANCE:
            proxy = self._best_performance_select(available_proxies)
        elif self.strategy == RotationStrategy.GEOGRAPHIC:
            proxy = self._geographic_select(available_proxies)
        else:
            proxy = self._round_robin_select(available_proxies)
        
        if proxy:
            logger.debug(f"[PROXY] Selected: {proxy.server} (status: {proxy.status.value})")
        
        return proxy
    
    def _round_robin_select(self, proxies: List[ProxyConfig]) -> ProxyConfig:
        """Round-robin selection."""
        self._current_index = (self._current_index + 1) % len(proxies)
        return proxies[self._current_index]
    
    def _random_select(self, proxies: List[ProxyConfig]) -> ProxyConfig:
        """Random selection."""
        import random
        return random.choice(proxies)
    
    def _least_used_select(self, proxies: List[ProxyConfig]) -> ProxyConfig:
        """Select proxy with least total requests."""
        return min(proxies, key=lambda p: p.total_requests)
    
    def _best_performance_select(self, proxies: List[ProxyConfig]) -> ProxyConfig:
        """Select proxy with best performance (lowest avg response time)."""
        # Filter out untested proxies
        tested = [p for p in proxies if p.total_requests > 0]
        if not tested:
            return proxies[0]
        return min(tested, key=lambda p: p.avg_response_time_ms)
    
    def _geographic_select(self, proxies: List[ProxyConfig]) -> ProxyConfig:
        """Select based on geographic distribution."""
        # Group by country and rotate within groups
        countries = list(set(p.country for p in proxies if p.country))
        if not countries:
            return self._round_robin_select(proxies)
        
        # Simple rotation through countries
        country = countries[self._current_index % len(countries)]
        country_proxies = [p for p in proxies if p.country == country]
        
        if not country_proxies:
            return self._round_robin_select(proxies)
        
        self._current_index += 1
        return country_proxies[0]
    
    def mark_success(self, proxy: ProxyConfig, response_time_ms: float) -> None:
        """Mark a successful request for a proxy."""
        proxy.mark_success(response_time_ms)
        logger.debug(
            f"[PROXY] Success: {proxy.server} "
            f"({response_time_ms:.0f}ms, success_rate: {proxy.get_success_rate():.1%})"
        )
    
    def mark_failure(self, proxy: ProxyConfig, error: str = "") -> None:
        """Mark a failed request for a proxy."""
        proxy.mark_failure(error)
        logger.warning(
            f"[PROXY] Failure: {proxy.server} "
            f"(failures: {proxy.consecutive_failures}/{proxy.max_failures}, error: {error[:50]})"
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get overall proxy pool statistics."""
        total_proxies = len(self.proxies)
        healthy = sum(1 for p in self.proxies if p.status == ProxyStatus.HEALTHY)
        degraded = sum(1 for p in self.proxies if p.status == ProxyStatus.DEGRADED)
        failed = sum(1 for p in self.proxies if p.status == ProxyStatus.FAILED)
        untested = sum(1 for p in self.proxies if p.status == ProxyStatus.UNTESTED)
        
        total_requests = sum(p.total_requests for p in self.proxies)
        total_successful = sum(p.successful_requests for p in self.proxies)
        
        overall_success_rate = total_successful / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_proxies": total_proxies,
            "healthy": healthy,
            "degraded": degraded,
            "failed": failed,
            "untested": untested,
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "overall_success_rate": overall_success_rate,
            "strategy": self.strategy.value,
        }
    
    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks and recovery."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Attempt to recover failed proxies
                if self.enable_auto_recovery:
                    current_time = time.time()
                    for proxy in self.proxies:
                        if proxy.status == ProxyStatus.FAILED:
                            time_since_failure = current_time - proxy.last_failed
                            if time_since_failure >= self.recovery_interval:
                                logger.info(f"[PROXY] Attempting to recover: {proxy.server}")
                                proxy.reset()
                
                # Log statistics
                stats = self.get_statistics()
                logger.info(
                    f"[PROXY] Statistics: {stats['healthy']}/{stats['total_proxies']} healthy, "
                    f"success_rate: {stats['overall_success_rate']:.1%}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PROXY] Health check error: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass
