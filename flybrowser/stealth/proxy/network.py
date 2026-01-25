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
Intelligent Proxy Super Network for FlyBrowser.

This module provides an intelligent proxy selection system that:
- Selects optimal proxies based on target domain requirements
- Ensures fingerprint-proxy-geolocation consistency
- Integrates with multiple residential proxy providers
- Monitors proxy health and rotates on detection signals
- Maintains session stickiness for login flows

Supported Providers:
- Bright Data (formerly Luminati)
- Oxylabs
- Smartproxy
- IPRoyal
- PacketStream
- Custom HTTP/SOCKS5 proxies

Example:
    >>> from flybrowser.stealth.proxy import ProxyNetwork, ProxyConfig
    >>> 
    >>> network = ProxyNetwork(ProxyConfig(
    ...     providers=[
    ...         {"type": "bright_data", "username": "...", "password": "..."},
    ...     ],
    ...     strategy="smart",
    ...     geolocation="us-west",
    ... ))
    >>> 
    >>> # Get optimal proxy for target
    >>> proxy = await network.get_proxy("https://example.com")
    >>> print(f"Using {proxy.ip} in {proxy.country}")
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import aiohttp

from flybrowser.utils.logger import logger


class ProxyType(str, Enum):
    """Type of proxy."""
    RESIDENTIAL = "residential"
    DATACENTER = "datacenter"
    MOBILE = "mobile"
    ISP = "isp"


class ProxyProvider(str, Enum):
    """Supported proxy providers."""
    BRIGHT_DATA = "bright_data"
    OXYLABS = "oxylabs"
    SMARTPROXY = "smartproxy"
    IPROYAL = "iproyal"
    PACKETSTREAM = "packetstream"
    CUSTOM = "custom"


class SelectionStrategy(str, Enum):
    """Proxy selection strategy."""
    SMART = "smart"          # AI-based selection considering target
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LOWEST_LATENCY = "lowest_latency"
    HIGHEST_SUCCESS = "highest_success"
    GEOGRAPHIC = "geographic"
    STICKY = "sticky"        # Maintain same IP for session


class HealthStatus(str, Enum):
    """Proxy health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class GeoLocation:
    """Geographic location for proxy targeting."""
    country: str = "US"
    country_code: str = "us"
    region: Optional[str] = None      # State/Province
    city: Optional[str] = None
    timezone: str = "America/Los_Angeles"
    language: str = "en-US"
    
    @classmethod
    def from_string(cls, geo_str: str) -> "GeoLocation":
        """Parse geolocation string like 'us-west', 'uk', 'germany'."""
        geo_map = {
            "us-west": cls(country="United States", country_code="us", region="California", 
                          city="Los Angeles", timezone="America/Los_Angeles", language="en-US"),
            "us-east": cls(country="United States", country_code="us", region="New York",
                          city="New York", timezone="America/New_York", language="en-US"),
            "us-central": cls(country="United States", country_code="us", region="Illinois",
                             city="Chicago", timezone="America/Chicago", language="en-US"),
            "uk": cls(country="United Kingdom", country_code="gb", city="London",
                     timezone="Europe/London", language="en-GB"),
            "germany": cls(country="Germany", country_code="de", city="Frankfurt",
                          timezone="Europe/Berlin", language="de-DE"),
            "france": cls(country="France", country_code="fr", city="Paris",
                         timezone="Europe/Paris", language="fr-FR"),
            "japan": cls(country="Japan", country_code="jp", city="Tokyo",
                        timezone="Asia/Tokyo", language="ja-JP"),
            "australia": cls(country="Australia", country_code="au", city="Sydney",
                            timezone="Australia/Sydney", language="en-AU"),
            "brazil": cls(country="Brazil", country_code="br", city="Sao Paulo",
                         timezone="America/Sao_Paulo", language="pt-BR"),
            "india": cls(country="India", country_code="in", city="Mumbai",
                        timezone="Asia/Kolkata", language="en-IN"),
            "singapore": cls(country="Singapore", country_code="sg", city="Singapore",
                            timezone="Asia/Singapore", language="en-SG"),
        }
        return geo_map.get(geo_str.lower(), geo_map["us-west"])


@dataclass
class ProxyEndpoint:
    """A single proxy endpoint with metadata."""
    
    # Connection info
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = "http"  # http, https, socks5
    
    # Provider info
    provider: ProxyProvider = ProxyProvider.CUSTOM
    proxy_type: ProxyType = ProxyType.DATACENTER
    
    # Location info
    country: str = "US"
    country_code: str = "us"
    region: Optional[str] = None
    city: Optional[str] = None
    asn: Optional[str] = None
    isp: Optional[str] = None
    
    # Resolved IP (if known)
    ip: Optional[str] = None
    
    # Health metrics
    health: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    last_used: float = 0.0
    last_failure: Optional[float] = None
    last_success: Optional[float] = None
    
    # Session stickiness
    session_id: Optional[str] = None
    
    def to_url(self) -> str:
        """Get proxy URL for Playwright."""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}"
    
    def to_playwright_format(self) -> Dict[str, str]:
        """Convert to Playwright proxy format."""
        proxy_dict = {"server": f"{self.protocol}://{self.host}:{self.port}"}
        if self.username:
            proxy_dict["username"] = self.username
        if self.password:
            proxy_dict["password"] = self.password
        return proxy_dict
    
    def mark_success(self, latency_ms: float) -> None:
        """Mark a successful request."""
        self.total_requests += 1
        self.last_used = time.time()
        self.last_success = time.time()
        
        # Update latency with exponential moving average
        if self.latency_ms == 0:
            self.latency_ms = latency_ms
        else:
            self.latency_ms = 0.7 * self.latency_ms + 0.3 * latency_ms
        
        # Update success rate
        self.success_rate = 1 - (self.failed_requests / max(1, self.total_requests))
        
        # Update health
        if self.health in (HealthStatus.DEGRADED, HealthStatus.BLOCKED):
            self.health = HealthStatus.HEALTHY
    
    def mark_failure(self, is_blocked: bool = False) -> None:
        """Mark a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_used = time.time()
        self.last_failure = time.time()
        
        # Update success rate
        self.success_rate = 1 - (self.failed_requests / max(1, self.total_requests))
        
        # Update health
        if is_blocked:
            self.health = HealthStatus.BLOCKED
        elif self.success_rate < 0.5:
            self.health = HealthStatus.FAILED
        elif self.success_rate < 0.8:
            self.health = HealthStatus.DEGRADED


@dataclass
class ProviderConfig:
    """Configuration for a proxy provider."""
    
    provider: ProxyProvider
    username: str = ""
    password: str = ""
    api_key: Optional[str] = None
    
    # Zone/product configuration
    zone: Optional[str] = None
    product: Optional[str] = None
    
    # Proxy type preference
    proxy_type: ProxyType = ProxyType.RESIDENTIAL
    
    # Geographic targeting
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    
    # Session settings
    session_duration: int = 600  # seconds
    
    # Rate limits
    max_concurrent: int = 100
    requests_per_minute: int = 600


@dataclass
class ProxyNetworkConfig:
    """Configuration for the proxy network."""
    
    # Provider configurations
    providers: List[ProviderConfig] = field(default_factory=list)
    
    # Selection strategy
    strategy: SelectionStrategy = SelectionStrategy.SMART
    
    # Default geolocation
    geolocation: str = "us-west"
    
    # Health check settings
    health_check_interval: float = 300.0  # 5 minutes
    health_check_url: str = "https://httpbin.org/ip"
    
    # Rotation settings
    rotate_on_captcha: bool = True
    rotate_on_block: bool = True
    rotate_on_403: bool = True
    max_failures_before_rotate: int = 3
    
    # Session stickiness
    sticky_sessions: bool = True
    session_duration: int = 600  # 10 minutes
    
    # ASN diversity
    enforce_asn_diversity: bool = True
    min_asn_diversity: int = 3  # Use at least 3 different ASNs
    
    # Fingerprint consistency
    ensure_fingerprint_consistency: bool = True


# ============================================================================
# Provider Implementations
# ============================================================================

class BaseProxyProvider:
    """Base class for proxy provider integrations."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def get_endpoint(
        self,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ProxyEndpoint:
        """Get a proxy endpoint with the specified targeting."""
        raise NotImplementedError
    
    async def check_balance(self) -> Dict[str, Any]:
        """Check account balance/quota."""
        return {"available": True}


class BrightDataProvider(BaseProxyProvider):
    """Bright Data (Luminati) proxy provider."""
    
    # Bright Data gateway endpoints
    ENDPOINTS = {
        "residential": "brd.superproxy.io",
        "datacenter": "brd.superproxy.io", 
        "mobile": "brd.superproxy.io",
        "isp": "brd.superproxy.io",
    }
    
    def get_endpoint(
        self,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ProxyEndpoint:
        """Get a Bright Data proxy endpoint."""
        # Build username with targeting
        parts = [self.config.username]
        
        if self.config.zone:
            parts.append(f"zone-{self.config.zone}")
        
        if country:
            parts.append(f"country-{country.lower()}")
        if city:
            parts.append(f"city-{city.lower().replace(' ', '_')}")
        
        if session_id:
            parts.append(f"session-{session_id}")
        
        username = "-".join(parts)
        
        host = self.ENDPOINTS.get(self.config.proxy_type.value, self.ENDPOINTS["residential"])
        
        return ProxyEndpoint(
            host=host,
            port=22225,
            username=username,
            password=self.config.password,
            protocol="http",
            provider=ProxyProvider.BRIGHT_DATA,
            proxy_type=self.config.proxy_type,
            country=country or self.config.country or "US",
            country_code=(country or self.config.country or "US").lower()[:2],
            city=city,
            session_id=session_id,
        )


class OxylabsProvider(BaseProxyProvider):
    """Oxylabs proxy provider."""
    
    ENDPOINTS = {
        "residential": "pr.oxylabs.io",
        "datacenter": "dc.pr.oxylabs.io",
        "mobile": "mobile.oxylabs.io",
    }
    
    def get_endpoint(
        self,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ProxyEndpoint:
        """Get an Oxylabs proxy endpoint."""
        # Build username with targeting
        username = self.config.username
        
        if country:
            username = f"customer-{self.config.username}-cc-{country.upper()}"
        
        if session_id:
            username = f"{username}-sessid-{session_id}"
        
        host = self.ENDPOINTS.get(self.config.proxy_type.value, self.ENDPOINTS["residential"])
        
        return ProxyEndpoint(
            host=host,
            port=7777,
            username=username,
            password=self.config.password,
            protocol="http",
            provider=ProxyProvider.OXYLABS,
            proxy_type=self.config.proxy_type,
            country=country or self.config.country or "US",
            country_code=(country or self.config.country or "US").lower()[:2],
            city=city,
            session_id=session_id,
        )


class SmartproxyProvider(BaseProxyProvider):
    """Smartproxy provider."""
    
    ENDPOINTS = {
        "residential": "gate.smartproxy.com",
        "datacenter": "dc.smartproxy.com",
    }
    
    def get_endpoint(
        self,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ProxyEndpoint:
        """Get a Smartproxy endpoint."""
        # Smartproxy uses different ports for different countries
        port_map = {
            "us": 10001,
            "uk": 10002,
            "de": 10003,
            "fr": 10004,
        }
        
        country_code = (country or "us").lower()[:2]
        port = port_map.get(country_code, 10001)
        
        username = self.config.username
        if session_id:
            username = f"{username}-session-{session_id}"
        
        host = self.ENDPOINTS.get(self.config.proxy_type.value, self.ENDPOINTS["residential"])
        
        return ProxyEndpoint(
            host=host,
            port=port,
            username=username,
            password=self.config.password,
            protocol="http",
            provider=ProxyProvider.SMARTPROXY,
            proxy_type=self.config.proxy_type,
            country=country or self.config.country or "US",
            country_code=country_code,
            city=city,
            session_id=session_id,
        )


class CustomProxyProvider(BaseProxyProvider):
    """Custom HTTP/SOCKS5 proxy provider."""
    
    def __init__(self, config: ProviderConfig, endpoints: List[Dict[str, Any]]):
        super().__init__(config)
        self._endpoints = endpoints
        self._current_index = 0
    
    def get_endpoint(
        self,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ProxyEndpoint:
        """Get a custom proxy endpoint."""
        if not self._endpoints:
            raise ValueError("No custom proxy endpoints configured")
        
        # Filter by country if specified
        available = self._endpoints
        if country:
            available = [e for e in self._endpoints 
                        if e.get("country", "").lower() == country.lower()]
            if not available:
                available = self._endpoints
        
        # Round-robin selection
        endpoint = available[self._current_index % len(available)]
        self._current_index += 1
        
        return ProxyEndpoint(
            host=endpoint["host"],
            port=endpoint["port"],
            username=endpoint.get("username"),
            password=endpoint.get("password"),
            protocol=endpoint.get("protocol", "http"),
            provider=ProxyProvider.CUSTOM,
            proxy_type=ProxyType(endpoint.get("type", "datacenter")),
            country=endpoint.get("country", "US"),
            country_code=endpoint.get("country", "US").lower()[:2],
            city=endpoint.get("city"),
        )


# ============================================================================
# Proxy Intelligence
# ============================================================================

class ProxyIntelligence:
    """
    Intelligent proxy selection based on target analysis.
    
    Analyzes target domains to determine optimal proxy characteristics:
    - Geographic requirements
    - Proxy type (residential vs datacenter)
    - Risk level and rotation frequency
    """
    
    # Domain patterns that typically require residential proxies
    RESIDENTIAL_REQUIRED = [
        "google.com", "google.",
        "facebook.com", "fb.com",
        "instagram.com",
        "twitter.com", "x.com",
        "linkedin.com",
        "amazon.", "amzn.",
        "ebay.",
        "walmart.",
        "target.com",
        "bestbuy.com",
        "nike.com",
        "adidas.com",
    ]
    
    # Domain patterns that are more tolerant
    DATACENTER_OK = [
        "httpbin.org",
        "example.com",
        "api.",
        "cdn.",
        "static.",
    ]
    
    # Country-specific domains
    COUNTRY_REQUIREMENTS = {
        ".co.uk": "gb",
        ".de": "de",
        ".fr": "fr",
        ".jp": "jp",
        ".au": "au",
        ".br": "br",
        ".in": "in",
        ".cn": "cn",
        ".ru": "ru",
    }
    
    def analyze_target(self, url: str) -> Dict[str, Any]:
        """
        Analyze target URL to determine proxy requirements.
        
        Returns:
            Dict with recommended proxy configuration
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Determine if residential is required
        needs_residential = any(p in domain for p in self.RESIDENTIAL_REQUIRED)
        datacenter_ok = any(p in domain for p in self.DATACENTER_OK)
        
        # Determine country from domain
        country = None
        for suffix, country_code in self.COUNTRY_REQUIREMENTS.items():
            if domain.endswith(suffix):
                country = country_code
                break
        
        # Risk assessment
        if needs_residential:
            risk_level = "high"
            rotation_frequency = "per_request"
        elif datacenter_ok:
            risk_level = "low"
            rotation_frequency = "per_session"
        else:
            risk_level = "medium"
            rotation_frequency = "on_block"
        
        return {
            "domain": domain,
            "needs_residential": needs_residential,
            "recommended_type": ProxyType.RESIDENTIAL if needs_residential else ProxyType.DATACENTER,
            "country": country,
            "risk_level": risk_level,
            "rotation_frequency": rotation_frequency,
        }


# ============================================================================
# Main Proxy Network
# ============================================================================

class ProxyNetwork:
    """
    Intelligent Proxy Super Network.
    
    Coordinates proxy selection across multiple providers with:
    - Target-aware selection
    - Fingerprint consistency
    - Health monitoring
    - Automatic rotation
    
    Example:
        >>> network = ProxyNetwork(config)
        >>> 
        >>> # Get proxy for target
        >>> proxy = await network.get_proxy("https://example.com")
        >>> 
        >>> # Report success/failure
        >>> network.report_success(proxy, latency_ms=150)
        >>> network.report_failure(proxy, is_blocked=True)
    """
    
    PROVIDER_CLASSES = {
        ProxyProvider.BRIGHT_DATA: BrightDataProvider,
        ProxyProvider.OXYLABS: OxylabsProvider,
        ProxyProvider.SMARTPROXY: SmartproxyProvider,
    }
    
    def __init__(self, config: ProxyNetworkConfig):
        """Initialize the proxy network."""
        self.config = config
        self._intelligence = ProxyIntelligence()
        self._providers: List[BaseProxyProvider] = []
        self._active_proxies: Dict[str, ProxyEndpoint] = {}  # session_id -> proxy
        self._used_asns: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Initialize providers
        for provider_config in config.providers:
            self._init_provider(provider_config)
        
        # Set default geolocation
        self._default_geo = GeoLocation.from_string(config.geolocation)
        
        logger.info(f"[PROXY] Initialized with {len(self._providers)} providers")
    
    def _init_provider(self, config: ProviderConfig) -> None:
        """Initialize a provider."""
        provider_class = self.PROVIDER_CLASSES.get(config.provider)
        if provider_class:
            self._providers.append(provider_class(config))
        else:
            logger.warning(f"[PROXY] Unknown provider: {config.provider}")
    
    async def close(self) -> None:
        """Close all provider sessions."""
        for provider in self._providers:
            await provider.close()
    
    async def get_proxy(
        self,
        target_url: Optional[str] = None,
        session_id: Optional[str] = None,
        fingerprint: Optional[Dict[str, Any]] = None,
    ) -> ProxyEndpoint:
        """
        Get optimal proxy for the target.
        
        Args:
            target_url: Target URL to analyze
            session_id: Session ID for sticky sessions
            fingerprint: Fingerprint profile for consistency
            
        Returns:
            ProxyEndpoint configured for the target
        """
        async with self._lock:
            # Check for sticky session
            if session_id and session_id in self._active_proxies:
                proxy = self._active_proxies[session_id]
                if proxy.health not in (HealthStatus.BLOCKED, HealthStatus.FAILED):
                    return proxy
            
            # Analyze target
            target_analysis = None
            if target_url:
                target_analysis = self._intelligence.analyze_target(target_url)
            
            # Determine targeting
            country = None
            city = None
            
            # From fingerprint (highest priority)
            if fingerprint and self.config.ensure_fingerprint_consistency:
                locale = fingerprint.get("locale", {})
                tz = locale.get("timezone", "")
                if "America" in tz:
                    country = "us"
                elif "Europe/London" in tz:
                    country = "gb"
                elif "Europe/Berlin" in tz:
                    country = "de"
                elif "Europe/Paris" in tz:
                    country = "fr"
                elif "Asia/Tokyo" in tz:
                    country = "jp"
            
            # From target analysis
            if not country and target_analysis and target_analysis.get("country"):
                country = target_analysis["country"]
            
            # From default
            if not country:
                country = self._default_geo.country_code
            
            # Select provider based on requirements
            proxy_type = ProxyType.RESIDENTIAL
            if target_analysis:
                proxy_type = target_analysis.get("recommended_type", ProxyType.RESIDENTIAL)
            
            # Find suitable provider
            provider = self._select_provider(proxy_type)
            if not provider:
                # Fallback to any provider
                provider = self._providers[0] if self._providers else None
            
            if not provider:
                raise ValueError("No proxy providers configured")
            
            # Generate session ID for stickiness
            if not session_id and self.config.sticky_sessions:
                session_id = hashlib.md5(
                    f"{time.time()}{random.random()}".encode()
                ).hexdigest()[:16]
            
            # Get endpoint
            proxy = provider.get_endpoint(
                country=country,
                city=city,
                session_id=session_id,
            )
            
            # Store for session
            if session_id:
                self._active_proxies[session_id] = proxy
            
            logger.info(
                f"[PROXY] Selected {proxy.provider.value} proxy: "
                f"{proxy.country} ({proxy.proxy_type.value})"
            )
            
            return proxy
    
    def _select_provider(self, proxy_type: ProxyType) -> Optional[BaseProxyProvider]:
        """Select a provider that supports the proxy type."""
        for provider in self._providers:
            if provider.config.proxy_type == proxy_type:
                return provider
        return None
    
    def report_success(self, proxy: ProxyEndpoint, latency_ms: float) -> None:
        """Report successful request through proxy."""
        proxy.mark_success(latency_ms)
        logger.debug(f"[PROXY] Success: {proxy.host} ({latency_ms:.0f}ms)")
    
    def report_failure(
        self, 
        proxy: ProxyEndpoint, 
        is_blocked: bool = False,
        is_captcha: bool = False,
    ) -> None:
        """Report failed request through proxy."""
        proxy.mark_failure(is_blocked)
        
        if is_blocked or is_captcha:
            logger.warning(
                f"[PROXY] {'Blocked' if is_blocked else 'CAPTCHA'}: {proxy.host}"
            )
            
            # Remove from active sessions if blocked
            if proxy.session_id and proxy.session_id in self._active_proxies:
                del self._active_proxies[proxy.session_id]
        else:
            logger.debug(f"[PROXY] Failure: {proxy.host}")
    
    async def rotate_proxy(
        self, 
        current: ProxyEndpoint,
        target_url: Optional[str] = None,
    ) -> ProxyEndpoint:
        """
        Rotate to a new proxy.
        
        Args:
            current: Current proxy to rotate from
            target_url: Target URL for new proxy selection
            
        Returns:
            New ProxyEndpoint
        """
        # Mark current as needing rotation
        if current.session_id and current.session_id in self._active_proxies:
            del self._active_proxies[current.session_id]
        
        # Get new proxy with different session
        new_session = hashlib.md5(
            f"{time.time()}{random.random()}".encode()
        ).hexdigest()[:16]
        
        return await self.get_proxy(
            target_url=target_url,
            session_id=new_session,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        total_requests = sum(p.total_requests for p in self._active_proxies.values())
        total_failures = sum(p.failed_requests for p in self._active_proxies.values())
        
        return {
            "providers_count": len(self._providers),
            "active_sessions": len(self._active_proxies),
            "total_requests": total_requests,
            "total_failures": total_failures,
            "success_rate": 1 - (total_failures / max(1, total_requests)),
        }
