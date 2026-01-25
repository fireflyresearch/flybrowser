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
LLM-Based Search Intent Detector.

This module provides intelligent detection of search intent from user queries,
determining:
1. Whether a query requires a search operation
2. What type of search (web, images, news, videos, etc.)
3. Optimized search query extraction
4. Ranking preferences based on intent

Uses LLM for semantic understanding rather than hardcoded rules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flybrowser.agents.tools.search.types import SearchType

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


# LLM prompt for search intent detection
INTENT_DETECTION_PROMPT = """Analyze the following user instruction and determine if it requires a web search.

User instruction: "{instruction}"

Respond with a JSON object containing:
{{
    "requires_search": boolean,  // true if the instruction needs to search the web
    "search_type": string,  // one of: "web", "images", "news", "videos", "places", "shopping", or null if no search needed
    "optimized_query": string,  // the optimal search query to use, or null if no search needed
    "ranking_preference": string,  // one of: "relevance", "freshness", "authority", "balanced"
    "confidence": float,  // 0.0 to 1.0 confidence in this analysis
    "reasoning": string  // brief explanation of the decision
}}

Guidelines:
- "requires_search": true if the user wants to find information on the web, search for something, look up data, etc.
- Search types:
  - "web": General web search (default for most queries)
  - "images": Looking for pictures, photos, graphics
  - "news": Current events, recent news, breaking stories
  - "videos": Looking for video content
  - "places": Local businesses, restaurants, locations
  - "shopping": Products, prices, buying items
- "optimized_query": Extract the core search terms, remove filler words, make it concise
- "ranking_preference":
  - "freshness": For time-sensitive queries (news, recent events, latest updates)
  - "authority": For research, academic, or authoritative sources needed
  - "relevance": For general queries where keyword match matters most
  - "balanced": Default for most queries

Examples:
- "Search for Python tutorials" -> requires_search: true, type: "web", query: "Python tutorials"
- "Find images of cats" -> requires_search: true, type: "images", query: "cats"
- "What's the latest news about AI?" -> requires_search: true, type: "news", query: "AI news", ranking: "freshness"
- "Click the login button" -> requires_search: false
- "Navigate to google.com" -> requires_search: false

Respond ONLY with the JSON object, no additional text."""


@dataclass
class SearchIntent:
    """
    Detected search intent from user instruction.
    
    Attributes:
        requires_search: Whether the instruction requires a search
        search_type: Type of search (web, images, news, etc.)
        optimized_query: Optimized search query
        ranking_preference: Preferred ranking strategy
        confidence: Confidence in the detection (0.0-1.0)
        reasoning: Explanation of the decision
        raw_instruction: Original user instruction
    """
    requires_search: bool = False
    search_type: Optional[SearchType] = None
    optimized_query: Optional[str] = None
    ranking_preference: str = "balanced"
    confidence: float = 0.0
    reasoning: str = ""
    raw_instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requires_search": self.requires_search,
            "search_type": self.search_type.value if self.search_type else None,
            "optimized_query": self.optimized_query,
            "ranking_preference": self.ranking_preference,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "raw_instruction": self.raw_instruction,
            "metadata": self.metadata,
        }


class SearchIntentDetector:
    """
    LLM-powered search intent detector.
    
    Uses semantic understanding to determine if a user instruction
    requires a search operation and what type of search to perform.
    
    Example:
        ```python
        detector = SearchIntentDetector(llm_provider)
        intent = await detector.detect("Find the latest news about Python 4.0")
        if intent.requires_search:
            print(f"Search type: {intent.search_type}")
            print(f"Query: {intent.optimized_query}")
        ```
    """
    
    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        cache_results: bool = True,
        max_cache_size: int = 100,
    ) -> None:
        """
        Initialize the intent detector.
        
        Args:
            llm_provider: LLM provider for semantic analysis
            cache_results: Whether to cache detection results
            max_cache_size: Maximum cache entries
        """
        self.llm = llm_provider
        self.cache_results = cache_results
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, SearchIntent] = {}
    
    async def detect(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SearchIntent:
        """
        Detect search intent from user instruction.
        
        Args:
            instruction: User instruction to analyze
            context: Optional context (current URL, page title, etc.)
            
        Returns:
            SearchIntent with detected parameters
        """
        # Check cache first
        cache_key = instruction.lower().strip()
        if self.cache_results and cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"Cache hit for intent detection: {instruction[:50]}...")
            return cached
        
        try:
            # Call LLM for intent detection
            prompt = INTENT_DETECTION_PROMPT.format(instruction=instruction)
            
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500,
            )
            
            # Parse JSON response
            intent = self._parse_response(response, instruction)
            
            # Cache the result
            if self.cache_results:
                self._add_to_cache(cache_key, intent)
            
            return intent
            
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}, using fallback")
            return self._fallback_detection(instruction)
    
    def _parse_response(self, response: Any, instruction: str) -> SearchIntent:
        """Parse LLM response into SearchIntent."""
        try:
            # Handle LLMResponse object - extract content string
            if hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                response = str(response)
            
            # Clean up response - extract JSON
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(
                    line for line in lines 
                    if not line.startswith("```")
                )
            
            data = json.loads(response)
            
            # Map search type string to enum
            search_type = None
            if data.get("search_type"):
                type_map = {
                    "web": SearchType.WEB,
                    "images": SearchType.IMAGES,
                    "news": SearchType.NEWS,
                    "videos": SearchType.VIDEOS,
                    "places": SearchType.PLACES,
                    "shopping": SearchType.SHOPPING,
                }
                search_type = type_map.get(data["search_type"].lower())
            
            return SearchIntent(
                requires_search=data.get("requires_search", False),
                search_type=search_type,
                optimized_query=data.get("optimized_query"),
                ranking_preference=data.get("ranking_preference", "balanced"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                raw_instruction=instruction,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_detection(instruction)
    
    def _fallback_detection(self, instruction: str) -> SearchIntent:
        """
        Fallback detection using keyword heuristics.
        
        Used when LLM fails or is unavailable.
        """
        instruction_lower = instruction.lower()
        
        # Keywords indicating search intent
        search_keywords = [
            "search", "find", "look up", "lookup", "google", "bing",
            "what is", "what are", "who is", "where is", "when is",
            "how to", "how do", "tell me about", "information about",
            "latest", "recent", "news about", "updates on",
        ]
        
        # Keywords indicating image search
        image_keywords = ["image", "picture", "photo", "graphic", "icon", "logo"]
        
        # Keywords indicating news search
        news_keywords = ["news", "headline", "breaking", "current events", "latest"]
        
        # Keywords indicating video search
        video_keywords = ["video", "watch", "youtube", "tutorial video"]
        
        # Keywords indicating shopping
        shopping_keywords = ["buy", "purchase", "price", "cheap", "deal", "shop"]
        
        # Keywords indicating places
        places_keywords = ["near me", "restaurant", "hotel", "store", "location"]
        
        # Check for search intent
        requires_search = any(kw in instruction_lower for kw in search_keywords)
        
        if not requires_search:
            return SearchIntent(
                requires_search=False,
                reasoning="No search keywords detected in instruction",
                raw_instruction=instruction,
                confidence=0.6,
            )
        
        # Determine search type
        search_type = SearchType.WEB  # Default
        
        if any(kw in instruction_lower for kw in image_keywords):
            search_type = SearchType.IMAGES
        elif any(kw in instruction_lower for kw in video_keywords):
            search_type = SearchType.VIDEOS
        elif any(kw in instruction_lower for kw in news_keywords):
            search_type = SearchType.NEWS
        elif any(kw in instruction_lower for kw in shopping_keywords):
            search_type = SearchType.SHOPPING
        elif any(kw in instruction_lower for kw in places_keywords):
            search_type = SearchType.PLACES
        
        # Extract query (simple heuristic)
        query = instruction
        for kw in ["search for", "find", "look up", "lookup", "search"]:
            if kw in instruction_lower:
                idx = instruction_lower.find(kw) + len(kw)
                query = instruction[idx:].strip()
                break
        
        # Determine ranking preference
        ranking = "balanced"
        if search_type == SearchType.NEWS:
            ranking = "freshness"
        elif any(kw in instruction_lower for kw in ["research", "study", "academic"]):
            ranking = "authority"
        
        return SearchIntent(
            requires_search=True,
            search_type=search_type,
            optimized_query=query,
            ranking_preference=ranking,
            confidence=0.5,  # Lower confidence for fallback
            reasoning="Fallback keyword-based detection",
            raw_instruction=instruction,
        )
    
    def _add_to_cache(self, key: str, intent: SearchIntent) -> None:
        """Add intent to cache with size limit."""
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = intent
    
    def clear_cache(self) -> None:
        """Clear the intent cache."""
        self._cache.clear()


async def detect_search_intent(
    instruction: str,
    llm_provider: "BaseLLMProvider",
    context: Optional[Dict[str, Any]] = None,
) -> SearchIntent:
    """
    Convenience function to detect search intent.
    
    Args:
        instruction: User instruction
        llm_provider: LLM provider
        context: Optional context
        
    Returns:
        SearchIntent
    """
    detector = SearchIntentDetector(llm_provider)
    return await detector.detect(instruction, context)
