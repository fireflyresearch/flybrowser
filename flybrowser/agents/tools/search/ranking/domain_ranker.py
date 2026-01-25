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
Domain Authority Ranker.

This ranker scores results based on domain reputation/authority.
It uses a pre-defined list of trusted domains and penalizes
known low-quality domains.
"""

from __future__ import annotations

from typing import Dict, List, Set
from urllib.parse import urlparse

from flybrowser.agents.tools.search.ranking.base_ranker import BaseRanker
from flybrowser.agents.tools.search.types import RankedSearchResult


class DomainAuthorityRanker(BaseRanker):
    """
    Domain authority ranker.
    
    Scores results based on domain reputation. Uses a curated list
    of high-authority domains (boosted) and low-quality domains (penalized).
    
    Example:
        >>> ranker = DomainAuthorityRanker()
        >>> ranked = ranker.rank("programming tutorial", results)
    """
    
    ranker_name = "domain_authority"
    
    # High authority domains (score boost)
    HIGH_AUTHORITY_DOMAINS: Dict[str, float] = {
        # Documentation & Official Sources
        "docs.python.org": 0.95,
        "developer.mozilla.org": 0.95,
        "docs.microsoft.com": 0.90,
        "cloud.google.com": 0.90,
        "aws.amazon.com": 0.90,
        "developer.apple.com": 0.90,
        
        # Reference & Education
        "wikipedia.org": 0.85,
        "stackoverflow.com": 0.85,
        "github.com": 0.85,
        "arxiv.org": 0.90,
        "scholar.google.com": 0.90,
        
        # Quality Tech Sites
        "realpython.com": 0.85,
        "medium.com": 0.70,
        "dev.to": 0.75,
        "hackernews.com": 0.75,
        "news.ycombinator.com": 0.75,
        "techcrunch.com": 0.80,
        "arstechnica.com": 0.80,
        "wired.com": 0.80,
        "theverge.com": 0.75,
        
        # Major News Sources
        "nytimes.com": 0.85,
        "washingtonpost.com": 0.85,
        "bbc.com": 0.85,
        "bbc.co.uk": 0.85,
        "reuters.com": 0.90,
        "apnews.com": 0.90,
        "theguardian.com": 0.85,
        "economist.com": 0.85,
        
        # Government & Organizations
        "gov": 0.90,  # .gov TLD
        "edu": 0.85,  # .edu TLD
        "org": 0.70,  # .org TLD (general boost)
        
        # Shopping (for product searches)
        "amazon.com": 0.80,
        "ebay.com": 0.75,
        
        # Social Media (context-dependent)
        "reddit.com": 0.70,
        "twitter.com": 0.65,
        "linkedin.com": 0.70,
    }
    
    # Low quality domains (score penalty)
    LOW_QUALITY_DOMAINS: Set[str] = {
        # Content farms / Low quality aggregators
        "ehow.com",
        "wikihow.com",
        "answers.com",
        "about.com",
        
        # Known spam/SEO farms
        "articlesbase.com",
        "ezinearticles.com",
        "hubpages.com",
    }
    
    def __init__(
        self,
        custom_authority: Dict[str, float] = None,
        custom_low_quality: Set[str] = None,
        default_score: float = 0.5,
    ) -> None:
        """
        Initialize domain authority ranker.
        
        Args:
            custom_authority: Additional high-authority domains with scores
            custom_low_quality: Additional low-quality domains
            default_score: Default score for unknown domains
        """
        self.authority_scores = dict(self.HIGH_AUTHORITY_DOMAINS)
        if custom_authority:
            self.authority_scores.update(custom_authority)
        
        self.low_quality = set(self.LOW_QUALITY_DOMAINS)
        if custom_low_quality:
            self.low_quality.update(custom_low_quality)
        
        self.default_score = default_score
    
    def rank(
        self,
        query: str,
        results: List[RankedSearchResult],
    ) -> List[RankedSearchResult]:
        """
        Compute domain authority scores for results.
        
        Args:
            query: Search query (unused, but required by interface)
            results: List of results to rank
            
        Returns:
            Results with domain authority scores added
        """
        for result in results:
            domain = self._extract_domain(result.url)
            score = self._get_domain_score(domain)
            result.ranking_signals[self.ranker_name] = score
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: Full URL
            
        Returns:
            Domain name (e.g., "example.com")
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            return domain
        except Exception:
            return ""
    
    def _get_domain_score(self, domain: str) -> float:
        """
        Get authority score for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Authority score (0.0 to 1.0)
        """
        if not domain:
            return self.default_score
        
        # Check low quality first
        for lq_domain in self.low_quality:
            if domain == lq_domain or domain.endswith(f".{lq_domain}"):
                return 0.2  # Low quality penalty
        
        # Check exact match
        if domain in self.authority_scores:
            return self.authority_scores[domain]
        
        # Check parent domain (e.g., docs.python.org -> python.org)
        parts = domain.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            if parent in self.authority_scores:
                return self.authority_scores[parent]
        
        # Check TLD bonus
        tld = parts[-1] if parts else ""
        if tld in self.authority_scores:
            return self.authority_scores[tld]
        
        return self.default_score
    
    def __repr__(self) -> str:
        return f"DomainAuthorityRanker(domains={len(self.authority_scores)})"
