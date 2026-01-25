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
Base Ranker Abstract Class.

This module defines the interface that all rankers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from flybrowser.agents.tools.search.types import RankedSearchResult


class BaseRanker(ABC):
    """
    Abstract base class for search result rankers.
    
    All ranker implementations must inherit from this class and
    implement the rank() method.
    
    Example:
        >>> class MyRanker(BaseRanker):
        ...     ranker_name = "my_ranker"
        ...     
        ...     def rank(self, query, results):
        ...         # Score each result
        ...         for result in results:
        ...             result.ranking_signals[self.ranker_name] = compute_score(result)
        ...         return results
    """
    
    ranker_name: str = "base"
    
    @abstractmethod
    def rank(
        self,
        query: str,
        results: List[RankedSearchResult],
    ) -> List[RankedSearchResult]:
        """
        Rank search results and add ranking signals.
        
        This method should:
        1. Compute a score for each result based on the ranker's criteria
        2. Store the score in result.ranking_signals[self.ranker_name]
        3. Return the results (order unchanged, scores added)
        
        The actual re-ordering is done by CompositeRanker which combines
        all signals with weights.
        
        Args:
            query: Original search query
            results: List of search results to rank
            
        Returns:
            Results with ranking signals added
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
