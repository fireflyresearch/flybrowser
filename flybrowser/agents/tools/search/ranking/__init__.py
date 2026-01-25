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
Search Result Ranking System.

This module provides intelligent ranking of search results using multiple
signals including keyword relevance (BM25), freshness, and domain authority.

Components:
    - BaseRanker: Abstract interface for rankers
    - BM25Ranker: BM25 keyword relevance scoring
    - FreshnessRanker: Time-decay scoring for recency
    - DomainAuthorityRanker: Domain reputation scoring
    - CompositeRanker: Combine multiple ranking signals
"""

from flybrowser.agents.tools.search.ranking.base_ranker import BaseRanker
from flybrowser.agents.tools.search.ranking.bm25_ranker import BM25Ranker
from flybrowser.agents.tools.search.ranking.freshness_ranker import FreshnessRanker
from flybrowser.agents.tools.search.ranking.domain_ranker import DomainAuthorityRanker
from flybrowser.agents.tools.search.ranking.composite_ranker import CompositeRanker

__all__ = [
    "BaseRanker",
    "BM25Ranker",
    "FreshnessRanker",
    "DomainAuthorityRanker",
    "CompositeRanker",
]
