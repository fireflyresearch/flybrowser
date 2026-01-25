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
Provider Registry for Search Abstraction Layer.

This module provides the ProviderRegistry class for registering, discovering,
and managing search providers. It supports automatic failover, health-based
selection, and capability filtering.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Type

from flybrowser.agents.tools.search.base_provider import (
    BaseSearchProvider,
    SearchProviderError,
)
from flybrowser.agents.tools.search.types import (
    ProviderHealth,
    ProviderStatus,
    SearchType,
)

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry for managing search providers.
    
    Thread-safe registry that maintains a catalog of all registered providers
    and provides methods for provider discovery, selection, and failover.
    
    Features:
        - Dynamic provider registration
        - Health-based provider selection
        - Automatic failover on failure
        - Capability-based filtering
        - Cost-aware selection
    
    Example:
        >>> registry = ProviderRegistry()
        >>> registry.register(SerperProvider(api_key="..."))
        >>> registry.register(GoogleProvider(api_key="...", cx="..."))
        >>> 
        >>> # Get best available provider
        >>> provider = registry.get_best_provider()
        >>> 
        >>> # Get provider for specific search type
        >>> provider = registry.get_provider_for_type(SearchType.IMAGES)
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: Dict[str, BaseSearchProvider] = {}
        self._priority_order: List[str] = []  # Provider names in priority order
        self._lock = threading.RLock()
    
    def register(
        self,
        provider: BaseSearchProvider,
        priority: Optional[int] = None,
    ) -> None:
        """
        Register a search provider.
        
        Args:
            provider: Provider instance to register
            priority: Priority order (lower = higher priority). If None, appends to end.
            
        Raises:
            ValueError: If provider with same name already registered
        """
        with self._lock:
            name = provider.provider_name
            
            if name in self._providers:
                raise ValueError(
                    f"Provider '{name}' is already registered. "
                    "Use unregister() first or use a unique provider name."
                )
            
            self._providers[name] = provider
            
            if priority is not None and 0 <= priority <= len(self._priority_order):
                self._priority_order.insert(priority, name)
            else:
                self._priority_order.append(name)
            
            logger.info(
                f"Registered search provider: {name} "
                f"(configured: {provider.is_configured()})"
            )
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a provider by name.
        
        Args:
            name: Provider name to unregister
            
        Returns:
            True if provider was unregistered, False if not found
        """
        with self._lock:
            if name not in self._providers:
                return False
            
            del self._providers[name]
            self._priority_order = [n for n in self._priority_order if n != name]
            
            logger.info(f"Unregistered search provider: {name}")
            return True
    
    def get_provider(self, name: str) -> Optional[BaseSearchProvider]:
        """
        Get a provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance or None if not found
        """
        with self._lock:
            return self._providers.get(name)
    
    def has_provider(self, name: str) -> bool:
        """Check if a provider is registered."""
        with self._lock:
            return name in self._providers
    
    def list_providers(self) -> List[str]:
        """Get list of registered provider names in priority order."""
        with self._lock:
            return self._priority_order.copy()
    
    def list_configured_providers(self) -> List[str]:
        """Get list of configured (ready-to-use) provider names."""
        with self._lock:
            return [
                name for name in self._priority_order
                if self._providers[name].is_configured()
            ]
    
    def get_best_provider(
        self,
        search_type: SearchType = SearchType.WEB,
        prefer_cost_effective: bool = False,
    ) -> Optional[BaseSearchProvider]:
        """
        Get the best available provider based on health and capabilities.
        
        Selection criteria (in order):
        1. Must be configured (has API key)
        2. Must support the requested search type
        3. Must be healthy or degraded (not unhealthy)
        4. If prefer_cost_effective, sort by cost
        5. Otherwise, use priority order
        
        Args:
            search_type: Type of search to perform
            prefer_cost_effective: Prefer cheaper providers
            
        Returns:
            Best available provider or None if none available
        """
        with self._lock:
            candidates = []
            
            for name in self._priority_order:
                provider = self._providers[name]
                
                # Check configuration
                if not provider.is_configured():
                    continue
                
                # Check search type support
                if not provider.supports_search_type(search_type):
                    continue
                
                # Check health
                health = provider.get_health_status()
                if health.status == ProviderStatus.UNHEALTHY:
                    continue
                
                candidates.append(provider)
            
            if not candidates:
                return None
            
            if prefer_cost_effective:
                # Sort by cost (ascending)
                candidates.sort(key=lambda p: p.capabilities.cost_per_request)
            
            return candidates[0]
    
    def get_provider_for_type(self, search_type: SearchType) -> Optional[BaseSearchProvider]:
        """
        Get a provider that supports a specific search type.
        
        Args:
            search_type: Type of search (web, images, news, etc.)
            
        Returns:
            Provider that supports the search type, or None
        """
        return self.get_best_provider(search_type=search_type)
    
    def get_fallback_providers(
        self,
        exclude: Optional[str] = None,
        search_type: SearchType = SearchType.WEB,
    ) -> List[BaseSearchProvider]:
        """
        Get list of fallback providers (excluding a specific one).
        
        Args:
            exclude: Provider name to exclude (e.g., the one that just failed)
            search_type: Type of search to perform
            
        Returns:
            List of available fallback providers in priority order
        """
        with self._lock:
            fallbacks = []
            
            for name in self._priority_order:
                if name == exclude:
                    continue
                
                provider = self._providers[name]
                
                if not provider.is_configured():
                    continue
                
                if not provider.supports_search_type(search_type):
                    continue
                
                health = provider.get_health_status()
                if health.status != ProviderStatus.UNHEALTHY:
                    fallbacks.append(provider)
            
            return fallbacks
    
    def get_all_health_status(self) -> Dict[str, ProviderHealth]:
        """
        Get health status of all registered providers.
        
        Returns:
            Dictionary mapping provider names to their health status
        """
        with self._lock:
            return {
                name: provider.get_health_status()
                for name, provider in self._providers.items()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry stats
        """
        with self._lock:
            health_statuses = self.get_all_health_status()
            
            healthy_count = sum(
                1 for h in health_statuses.values()
                if h.status == ProviderStatus.HEALTHY
            )
            
            return {
                "total_providers": len(self._providers),
                "configured_providers": len(self.list_configured_providers()),
                "healthy_providers": healthy_count,
                "priority_order": self._priority_order.copy(),
                "provider_stats": {
                    name: provider.get_stats()
                    for name, provider in self._providers.items()
                },
            }
    
    def set_priority(self, provider_names: List[str]) -> None:
        """
        Set the priority order of providers.
        
        Args:
            provider_names: List of provider names in desired priority order
            
        Raises:
            ValueError: If any provider name is not registered
        """
        with self._lock:
            # Validate all names exist
            for name in provider_names:
                if name not in self._providers:
                    raise ValueError(f"Provider '{name}' is not registered")
            
            # Update priority order
            self._priority_order = provider_names.copy()
    
    @classmethod
    def create_default(cls) -> "ProviderRegistry":
        """
        Create a registry with default providers from environment variables.
        
        Looks for these environment variables:
        - SERPER_API_KEY: Serper.dev API key
        - GOOGLE_CUSTOM_SEARCH_API_KEY + GOOGLE_CUSTOM_SEARCH_CX: Google
        - BING_SEARCH_API_KEY: Bing
        
        Returns:
            Configured ProviderRegistry
        """
        # Import providers here to avoid circular imports
        from flybrowser.agents.tools.search.providers import (
            SerperProvider,
            GoogleProvider,
            BingProvider,
        )
        
        registry = cls()
        
        # Register Serper (highest priority - best value)
        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            registry.register(SerperProvider(api_key=serper_key), priority=0)
        
        # Register Google Custom Search
        google_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        google_cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
        if google_key and google_cx:
            registry.register(
                GoogleProvider(api_key=google_key, cx=google_cx),
                priority=1
            )
        
        # Register Bing
        bing_key = os.getenv("BING_SEARCH_API_KEY")
        if bing_key:
            registry.register(BingProvider(api_key=bing_key), priority=2)
        
        logger.info(
            f"Created default registry with {len(registry.list_configured_providers())} "
            f"configured providers: {registry.list_configured_providers()}"
        )
        
        return registry
    
    def __len__(self) -> int:
        """Return number of registered providers."""
        with self._lock:
            return len(self._providers)
    
    def __contains__(self, name: str) -> bool:
        """Check if provider is registered."""
        return self.has_provider(name)
    
    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            configured = len(self.list_configured_providers())
            total = len(self._providers)
            return f"ProviderRegistry({configured}/{total} providers configured)"
