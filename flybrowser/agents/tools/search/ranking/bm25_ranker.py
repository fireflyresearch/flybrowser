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
BM25 Ranker for Keyword Relevance Scoring.

BM25 (Best Matching 25) is a ranking function used by search engines
to estimate the relevance of documents to a given search query.

It is based on:
- Term Frequency (TF): How often a term appears in a document
- Inverse Document Frequency (IDF): How rare/common a term is
- Document length normalization
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Set

from flybrowser.agents.tools.search.ranking.base_ranker import BaseRanker
from flybrowser.agents.tools.search.types import RankedSearchResult


class BM25Ranker(BaseRanker):
    """
    BM25 keyword relevance ranker.
    
    Computes BM25 scores for search results based on query term
    frequency and document length normalization.
    
    Parameters:
        k1: Term frequency saturation parameter (default: 1.5)
        b: Length normalization parameter (default: 0.75)
    
    Example:
        >>> ranker = BM25Ranker(k1=1.5, b=0.75)
        >>> ranked = ranker.rank("python tutorial", results)
    """
    
    ranker_name = "bm25"
    
    # Common stop words to filter out
    STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "what", "which", "who", "whom", "whose", "where",
        "when", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "also",
    }
    
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Initialize BM25 ranker.
        
        Args:
            k1: Term frequency saturation (higher = less saturation)
            b: Length normalization (0 = none, 1 = full)
        """
        self.k1 = k1
        self.b = b
    
    def rank(
        self,
        query: str,
        results: List[RankedSearchResult],
    ) -> List[RankedSearchResult]:
        """
        Compute BM25 scores for results.
        
        Args:
            query: Search query
            results: List of results to rank
            
        Returns:
            Results with BM25 scores added
        """
        if not results:
            return results
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            # No meaningful query terms, assign equal scores
            for result in results:
                result.ranking_signals[self.ranker_name] = 0.5
            return results
        
        # Build corpus from results
        documents = []
        for result in results:
            # Combine title and snippet for scoring
            doc_text = f"{result.title} {result.snippet}"
            documents.append(self._tokenize(doc_text))
        
        # Calculate average document length
        avg_doc_len = sum(len(doc) for doc in documents) / len(documents) if documents else 1
        
        # Calculate IDF for query terms
        idf_scores = self._calculate_idf(query_terms, documents)
        
        # Score each document
        scores = []
        for doc in documents:
            score = self._score_document(query_terms, doc, idf_scores, avg_doc_len)
            scores.append(score)
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        score_range = max_score - min_score if max_score > min_score else 1
        
        for result, score in zip(results, scores):
            normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5
            result.ranking_signals[self.ranker_name] = normalized_score
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase terms (stop words removed)
        """
        # Convert to lowercase and extract words
        text = text.lower()
        terms = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Remove stop words
        terms = [t for t in terms if t not in self.STOP_WORDS and len(t) > 1]
        
        return terms
    
    def _calculate_idf(
        self,
        query_terms: List[str],
        documents: List[List[str]],
    ) -> dict:
        """
        Calculate IDF scores for query terms.
        
        Args:
            query_terms: List of query terms
            documents: List of tokenized documents
            
        Returns:
            Dictionary mapping term to IDF score
        """
        n = len(documents)
        idf = {}
        
        for term in set(query_terms):
            # Count documents containing term
            doc_freq = sum(1 for doc in documents if term in doc)
            
            # BM25 IDF formula
            idf[term] = math.log(
                (n - doc_freq + 0.5) / (doc_freq + 0.5) + 1
            )
        
        return idf
    
    def _score_document(
        self,
        query_terms: List[str],
        doc: List[str],
        idf_scores: dict,
        avg_doc_len: float,
    ) -> float:
        """
        Calculate BM25 score for a document.
        
        Args:
            query_terms: List of query terms
            doc: Tokenized document
            idf_scores: IDF scores for query terms
            avg_doc_len: Average document length
            
        Returns:
            BM25 score
        """
        doc_len = len(doc)
        doc_term_counts = Counter(doc)
        
        score = 0.0
        for term in query_terms:
            if term not in idf_scores:
                continue
            
            tf = doc_term_counts.get(term, 0)
            idf = idf_scores[term]
            
            # BM25 TF component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
            
            score += idf * numerator / denominator
        
        return score
    
    def __repr__(self) -> str:
        return f"BM25Ranker(k1={self.k1}, b={self.b})"
