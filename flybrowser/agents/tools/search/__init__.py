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
Search Tool Abstraction Layer for FlyBrowser.

This module provides a best-in-class search abstraction supporting multiple
search providers with intelligent ranking, automatic failover, and caching.

Architecture:
    - BaseSearchProvider: Abstract interface for search providers
    - ProviderRegistry: Registration and discovery of providers
    - SearchAgent: High-level orchestrator with ranking and failover
    - Ranking System: BM25, semantic, freshness, and composite ranking

Supported Providers:
    - Serper.dev: Fast, affordable Google search API (recommended)
    - Google Custom Search API: Official Google search
    - Bing Web Search API: Microsoft search
    - Browser-based: Human-like search automation

Example Usage:
    ```python
    from flybrowser.agents.tools.search import (
        SearchAgent,
        ProviderRegistry,
        SerperProvider,
        SearchOptions,
    )
    
    # Create provider registry
    registry = ProviderRegistry()
    registry.register(SerperProvider(api_key=os.getenv("SERPER_API_KEY")))
    
    # Create search agent
    agent = SearchAgent(registry)
    
    # Perform search with intelligent ranking
    results = await agent.search(
        query="python web scraping tutorial",
        options=SearchOptions(max_results=10)
    )
    ```
"""

# Core types and enums
from flybrowser.agents.tools.search.types import (
    SearchOptions,
    SearchType,
    ProviderHealth,
    ProviderStatus,
    RankedSearchResult,
    SearchAgentResponse,
    ProviderCapabilities,
    SafeSearchLevel,
    TimeRange,
)

# Base classes
from flybrowser.agents.tools.search.base_provider import BaseSearchProvider

# Provider registry
from flybrowser.agents.tools.search.provider_registry import ProviderRegistry

# Concrete providers
from flybrowser.agents.tools.search.providers import (
    SerperProvider,
    GoogleProvider,
    GoogleProvider as GoogleSearchProvider,  # Alias for backward compatibility
    BingProvider,
    BingProvider as BingSearchProvider,  # Alias for backward compatibility
)

# Ranking system
from flybrowser.agents.tools.search.ranking import (
    BaseRanker,
    BM25Ranker,
    FreshnessRanker,
    DomainAuthorityRanker,
    CompositeRanker,
)

# High-level agent
from flybrowser.agents.tools.search.search_agent import SearchAgent

# Tool integration for ReAct framework
from flybrowser.agents.tools.search.search_agent_tool import SearchAgentTool

# Intent detection
from flybrowser.agents.tools.search.intent_detector import (
    SearchIntent,
    SearchIntentDetector,
    detect_search_intent,
)

# Coordinator for unified API/browser search
from flybrowser.agents.tools.search.coordinator import (
    SearchCoordinator,
    SearchCoordinatorConfig,
    SearchMethod,
    CoordinatedSearchResult,
)

__all__ = [
    # Types
    "SearchOptions",
    "SearchType",
    "ProviderHealth",
    "ProviderStatus",
    "RankedSearchResult",
    "SearchAgentResponse",
    "ProviderCapabilities",
    "SafeSearchLevel",
    "TimeRange",
    # Base
    "BaseSearchProvider",
    # Registry
    "ProviderRegistry",
    # Providers
    "SerperProvider",
    "GoogleProvider",
    "GoogleSearchProvider",  # Alias
    "BingProvider",
    "BingSearchProvider",  # Alias
    # Ranking
    "BaseRanker",
    "BM25Ranker",
    "FreshnessRanker",
    "DomainAuthorityRanker",
    "CompositeRanker",
    # Agent
    "SearchAgent",
    # Tool integration
    "SearchAgentTool",
    # Intent detection
    "SearchIntent",
    "SearchIntentDetector",
    "detect_search_intent",
    # Coordinator
    "SearchCoordinator",
    "SearchCoordinatorConfig",
    "SearchMethod",
    "CoordinatedSearchResult",
]
