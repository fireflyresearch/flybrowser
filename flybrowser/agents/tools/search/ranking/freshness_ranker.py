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
Freshness Ranker for Time-Decay Scoring.

This ranker scores results based on recency, giving higher scores
to more recent content using exponential time decay.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import List, Optional

from flybrowser.agents.tools.search.ranking.base_ranker import BaseRanker
from flybrowser.agents.tools.search.types import RankedSearchResult


class FreshnessRanker(BaseRanker):
    """
    Freshness ranker using time-decay scoring.
    
    Scores results based on publication/crawl date with exponential decay.
    Results without dates get a neutral score.
    
    Parameters:
        half_life_days: Days until score drops to 50% (default: 30)
        max_age_days: Maximum age to consider (default: 365)
    
    Example:
        >>> ranker = FreshnessRanker(half_life_days=30)
        >>> ranked = ranker.rank("latest news", results)
    """
    
    ranker_name = "freshness"
    
    def __init__(
        self,
        half_life_days: int = 30,
        max_age_days: int = 365,
    ) -> None:
        """
        Initialize freshness ranker.
        
        Args:
            half_life_days: Days until score drops to 50%
            max_age_days: Maximum age to consider (older = minimum score)
        """
        self.half_life_days = half_life_days
        self.max_age_days = max_age_days
        
        # Pre-calculate decay constant
        self.decay_constant = math.log(2) / half_life_days
    
    def rank(
        self,
        query: str,
        results: List[RankedSearchResult],
    ) -> List[RankedSearchResult]:
        """
        Compute freshness scores for results.
        
        Args:
            query: Search query (unused, but required by interface)
            results: List of results to rank
            
        Returns:
            Results with freshness scores added
        """
        if not results:
            return results
        
        now = datetime.now()
        
        for result in results:
            date = self._extract_date(result)
            
            if date:
                # Calculate age in days
                age_days = (now - date).days
                age_days = max(0, min(age_days, self.max_age_days))
                
                # Exponential decay: score = e^(-λt)
                score = math.exp(-self.decay_constant * age_days)
            else:
                # No date available, assign neutral score
                score = 0.5
            
            result.ranking_signals[self.ranker_name] = score
        
        return results
    
    def _extract_date(self, result: RankedSearchResult) -> Optional[datetime]:
        """
        Extract date from result metadata or snippet.
        
        Args:
            result: Search result
            
        Returns:
            Extracted datetime or None
        """
        # Try metadata fields first
        for field in ["date", "date_published", "datePublished", "date_crawled"]:
            date_str = result.metadata.get(field)
            if date_str:
                parsed = self._parse_date(str(date_str))
                if parsed:
                    return parsed
        
        # Try to extract from snippet
        snippet_date = self._extract_date_from_text(result.snippet)
        if snippet_date:
            return snippet_date
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string in various formats.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed datetime or None
        """
        # Common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """
        Extract date from text using pattern matching.
        
        Args:
            text: Text to search for dates
            
        Returns:
            Extracted datetime or None
        """
        # Pattern for relative dates like "2 days ago", "1 week ago"
        relative_patterns = [
            (r"(\d+)\s*(?:hour|hr)s?\s*ago", lambda m: timedelta(hours=int(m.group(1)))),
            (r"(\d+)\s*days?\s*ago", lambda m: timedelta(days=int(m.group(1)))),
            (r"(\d+)\s*weeks?\s*ago", lambda m: timedelta(weeks=int(m.group(1)))),
            (r"(\d+)\s*months?\s*ago", lambda m: timedelta(days=int(m.group(1)) * 30)),
            (r"yesterday", lambda m: timedelta(days=1)),
            (r"today", lambda m: timedelta(days=0)),
        ]
        
        text_lower = text.lower()
        now = datetime.now()
        
        for pattern, delta_func in relative_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    delta = delta_func(match)
                    return now - delta
                except Exception:
                    continue
        
        return None
    
    def __repr__(self) -> str:
        return f"FreshnessRanker(half_life_days={self.half_life_days})"
