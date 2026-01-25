# Copyright 2026 Firefly Software Solutions Inc.
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
SearchRank Tool for FlyBrowser.

This tool analyzes search results using LLM to:
- Rank results by relevance to the user's query/goal
- Extract key information from snippets
- Recommend which links to visit
- Provide reasoning for recommendations

Usage:
    tool = SearchRankTool(page_controller)
    result = await tool.execute(
        query="user's search intent",
        search_results=[...],
        goal="what user wants to achieve"
    )
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from .search_utils import SearchResult, SearchResponse
from flybrowser.prompts import PromptManager
from flybrowser.agents.structured_llm import StructuredLLMWrapper

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class SearchRankTool(BaseTool):
    """
    Intelligent search result ranking and analysis tool.
    
    Uses LLM to analyze search results and recommend which links
    to visit based on relevance, credibility, and goal alignment.
    """
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        """Initialize the search rank tool."""
        super().__init__(page_controller)
        self.prompt_manager = PromptManager()
    
    @property
    def metadata(self) -> ToolMetadata:
        """Tool metadata."""
        return ToolMetadata(
            name="search_rank",
            description=(
                "Analyze and rank search results by relevance. "
                "Provides intelligent recommendations on which links to visit. "
                "Extracts key information from snippets."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Original search query or user intent",
                    required=True,
                ),
                ToolParameter(
                    name="search_results",
                    type="array",
                    description="Array of search results to analyze (from search_human or search_api)",
                    required=True,
                ),
                ToolParameter(
                    name="goal",
                    type="string",
                    description="User's goal or what they want to accomplish",
                    required=True,
                ),
                ToolParameter(
                    name="top_n",
                    type="integer",
                    description="Number of top results to return (default: 5)",
                    required=False,
                    default=5,
                ),
            ],
            examples=['search_rank(query="Taidy startup", search_results=results, goal="Get business information about Taidy", top_n=5)'],
        )
    
    async def execute(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        goal: str,
        top_n: int = 5,
        **kwargs
    ) -> ToolResult:
        """
        Analyze and rank search results.
        
        Args:
            query: Original search query
            search_results: List of search results (dicts with title, url, snippet)
            goal: User's goal
            top_n: Number of top results to return
            
        Returns:
            ToolResult with ranked results and recommendations
        """
        try:
            # Get LLM from tool context
            llm = getattr(self, 'llm_provider', None)
            if not llm:
                return ToolResult.error_result(
                    error="LLM not available for result analysis",
                    error_code="LLM_NOT_AVAILABLE"
                )
            
            if not search_results:
                return ToolResult.error_result(
                    error="No search results to analyze",
                    error_code="NO_RESULTS"
                )
            
            logger.info(f"Analyzing {len(search_results)} search results for: {query}")
            
            # Prepare results for LLM analysis
            results_text = self._format_results_for_analysis(search_results[:15])  # Limit to 15 for context
            
            # Get prompts from template manager
            prompts = self.prompt_manager.get_prompt(
                "search_result_ranking",
                query=query,
                goal=goal,
                results_text=results_text,
            )
            
            # Define schema for search ranking response
            ranking_schema = {
                "type": "object",
                "properties": {
                    "analysis_summary": {
                        "type": "string",
                        "description": "Brief summary of the search results analysis"
                    },
                    "key_insights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key insights from analyzing the results"
                    },
                    "ranked_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "relevance_score": {"type": "number"},
                                "reason": {"type": "string"}
                            },
                            "required": ["title", "url", "relevance_score"]
                        },
                        "description": "Results ranked by relevance"
                    },
                    "recommended_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommended next actions"
                    }
                },
                "required": ["analysis_summary", "ranked_results"]
            }
            
            # Use configured temperature or default to 0.3
            temperature = getattr(self, 'agent_config', None)
            if temperature and hasattr(temperature, 'search_ranking_temperature'):
                temp_value = temperature.search_ranking_temperature
            else:
                temp_value = 0.3  # Fallback default
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            wrapper = StructuredLLMWrapper(llm, max_repair_attempts=2)
            
            try:
                analysis = await wrapper.generate_structured(
                    prompt=prompts["user"],
                    schema=ranking_schema,
                    system_prompt=prompts["system"],
                    temperature=temp_value,
                )
            except ValueError as e:
                logger.error(f"Structured output failed after repairs: {e}")
                return ToolResult.error_result(
                    error=f"Could not parse LLM analysis response: {e}",
                    error_code="PARSE_ERROR"
                )
            
            # Extract top N results
            ranked = analysis.get('ranked_results', [])[:top_n]
            
            # Build response
            result_data = {
                'analysis_summary': analysis.get('analysis_summary', ''),
                'key_insights': analysis.get('key_insights', []),
                'ranked_results': ranked,
                'recommended_actions': analysis.get('recommended_actions', []),
                'total_analyzed': len(search_results),
                'top_n_returned': len(ranked),
            }
            
            # Create human-readable message
            top_result = ranked[0] if ranked else None
            message = f"Analyzed {len(search_results)} results. "
            if top_result:
                message += f"Top recommendation: {top_result.get('title', 'Unknown')} (relevance: {top_result.get('relevance_score', 0):.2f})"
            
            logger.info(f"Search analysis complete. Top {len(ranked)} results ranked.")
            
            return ToolResult.success_result(
                data=result_data,
                message=message,
                metadata={'query': query, 'goal': goal}
            )
        
        except Exception as e:
            logger.exception(f"Search ranking failed: {e}")
            return ToolResult.error_result(
                error=f"Analysis error: {str(e)}",
                error_code="EXECUTION_ERROR"
            )
    
    def _format_results_for_analysis(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for LLM analysis.
        
        Args:
            results: List of search result dicts
            
        Returns:
            Formatted string
        """
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', 'No description')
            
            formatted.append(f"""
[{i}] {title}
URL: {url}
Description: {snippet}
""")
        
        return "\n".join(formatted)
