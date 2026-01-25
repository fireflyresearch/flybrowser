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
Composite Ranker for Combining Multiple Ranking Signals.

This ranker combines scores from multiple rankers using configurable
weights to produce a final relevance score for each result.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from flybrowser.agents.tools.search.ranking.base_ranker import BaseRanker
from flybrowser.agents.tools.search.ranking.bm25_ranker import BM25Ranker
from flybrowser.agents.tools.search.ranking.freshness_ranker import FreshnessRanker
from flybrowser.agents.tools.search.ranking.domain_ranker import DomainAuthorityRanker
from flybrowser.agents.tools.search.types import RankedSearchResult

logger = logging.getLogger(__name__)


class CompositeRanker(BaseRanker):
    """
    Composite ranker that combines multiple ranking signals.
    
    Applies multiple rankers and combines their scores using
    configurable weights to produce a final relevance score.
    
    Default weights:
        - bm25: 0.35 (keyword relevance)
        - freshness: 0.20 (recency)
        - domain_authority: 0.15 (source quality)
        - position: 0.30 (original search engine ranking)
    
    Example:
        >>> ranker = CompositeRanker(
        ...     weights={"bm25": 0.4, "freshness": 0.3, "domain_authority": 0.3}
        ... )
        >>> ranked_results = ranker.rank("python tutorial", results)
    """
    
    ranker_name = "composite"
    
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "bm25": 0.35,
        "freshness": 0.20,
        "domain_authority": 0.15,
        "position": 0.30,  # Original search engine position
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        rankers: Optional[List[BaseRanker]] = None,
        normalize_weights: bool = True,
    ) -> None:
        """
        Initialize composite ranker.
        
        Args:
            weights: Custom weights for ranking signals
            rankers: List of rankers to use (defaults to standard rankers)
            normalize_weights: Whether to normalize weights to sum to 1.0
        """
        # Set up weights
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        
        if normalize_weights:
            self._normalize_weights()
        
        # Set up rankers
        if rankers is not None:
            self.rankers = rankers
        else:
            # Default rankers
            self.rankers = [
                BM25Ranker(),
                FreshnessRanker(),
                DomainAuthorityRanker(),
            ]
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def rank(
        self,
        query: str,
        results: List[RankedSearchResult],
    ) -> List[RankedSearchResult]:
        """
        Rank results using multiple signals.
        
        Process:
        1. Apply each ranker to add individual scores
        2. Add position-based score from original ranking
        3. Compute weighted composite score
        4. Sort results by composite score
        
        Args:
            query: Search query
            results: List of results to rank
            
        Returns:
            Results sorted by composite relevance score
        """
        if not results:
            return results
        
        # Apply each ranker
        for ranker in self.rankers:
            try:
                results = ranker.rank(query, results)
            except Exception as e:
                logger.warning(f"Ranker {ranker.ranker_name} failed: {e}")
                # Assign neutral scores if ranker fails
                for result in results:
                    if ranker.ranker_name not in result.ranking_signals:
                        result.ranking_signals[ranker.ranker_name] = 0.5
        
        # Add position-based scores (inverse of position, normalized)
        max_position = len(results)
        for result in results:
            # Higher score for lower position (position 1 = highest score)
            position_score = 1.0 - (result.position - 1) / max_position if max_position > 1 else 1.0
            result.ranking_signals["position"] = position_score
        
        # Compute composite scores
        for result in results:
            composite_score = self._compute_composite_score(result)
            result.relevance_score = composite_score
        
        # Sort by composite score (descending)
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        # Update positions
        for i, result in enumerate(results, 1):
            result.position = i
        
        return results
    
    def _compute_composite_score(self, result: RankedSearchResult) -> float:
        """
        Compute weighted composite score for a result.
        
        Args:
            result: Search result with ranking signals
            
        Returns:
            Weighted composite score (0.0 to 1.0)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.weights.items():
            if signal_name in result.ranking_signals:
                signal_score = result.ranking_signals[signal_name]
                total_score += weight * signal_score
                total_weight += weight
        
        # Return weighted average
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5  # Neutral score if no signals
    
    def add_ranker(self, ranker: BaseRanker, weight: float = 0.1) -> None:
        """
        Add a custom ranker.
        
        Args:
            ranker: Ranker to add
            weight: Weight for this ranker's signal
        """
        self.rankers.append(ranker)
        self.weights[ranker.ranker_name] = weight
        self._normalize_weights()
    
    def set_weight(self, signal_name: str, weight: float) -> None:
        """
        Set weight for a specific signal.
        
        Args:
            signal_name: Name of the ranking signal
            weight: New weight value
        """
        self.weights[signal_name] = weight
        self._normalize_weights()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return dict(self.weights)
    
    @classmethod
    def create_for_news(cls) -> "CompositeRanker":
        """
        Create a ranker optimized for news searches.
        
        Prioritizes freshness over other signals.
        """
        return cls(
            weights={
                "bm25": 0.25,
                "freshness": 0.45,  # High weight on recency
                "domain_authority": 0.15,
                "position": 0.15,
            }
        )
    
    @classmethod
    def create_for_research(cls) -> "CompositeRanker":
        """
        Create a ranker optimized for research/academic searches.
        
        Prioritizes domain authority and relevance.
        """
        return cls(
            weights={
                "bm25": 0.40,
                "freshness": 0.10,
                "domain_authority": 0.35,  # High weight on authority
                "position": 0.15,
            }
        )
    
    @classmethod
    def create_for_tutorials(cls) -> "CompositeRanker":
        """
        Create a ranker optimized for tutorial/how-to searches.
        
        Balances relevance with source quality.
        """
        return cls(
            weights={
                "bm25": 0.40,
                "freshness": 0.15,
                "domain_authority": 0.25,
                "position": 0.20,
            }
        )
    
    def __repr__(self) -> str:
        return f"CompositeRanker(rankers={len(self.rankers)}, weights={self.weights})"
