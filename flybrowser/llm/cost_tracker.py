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
Cost tracking for LLM usage.

This module provides cost tracking functionality for LLM API usage. It tracks
token usage and calculates costs based on current pricing for different providers
and models.

Features:
- Automatic cost calculation based on token usage
- Support for multiple providers (OpenAI, Anthropic, etc.)
- Per-request and aggregate cost tracking
- Usage statistics and reporting
- Budget alerts and warnings
- Export to CSV/JSON for analysis

The pricing table is updated as of January 2026 and should be periodically
reviewed for accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from flybrowser.llm.config import CostTrackingConfig, LLMProviderType
from flybrowser.utils.logger import logger


# Provider name mapping for registry lookups
_PROVIDER_NAME_MAP = {
    LLMProviderType.OPENAI: "openai",
    LLMProviderType.ANTHROPIC: "anthropic",
    LLMProviderType.GEMINI: "gemini",
    LLMProviderType.OLLAMA: "ollama",
    LLMProviderType.LM_STUDIO: "ollama",  # Uses same registry as ollama
    LLMProviderType.LOCAL_AI: "ollama",
    LLMProviderType.VLLM: "ollama",
}


@dataclass
class UsageRecord:
    """
    Record of a single LLM API usage.

    This dataclass stores detailed information about each LLM request
    for cost tracking and analysis purposes.

    Attributes:
        timestamp: When the request was made
        provider: LLM provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        cost: Calculated cost in USD for this request
        cached: Whether this response was served from cache (zero cost)
        metadata: Additional metadata about the request

    Example:
        >>> record = UsageRecord(
        ...     timestamp=datetime.now(),
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     prompt_tokens=100,
        ...     completion_tokens=50,
        ...     total_tokens=150,
        ...     cost=0.000625,
        ...     cached=False
        ... )
    """

    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    cached: bool = False
    metadata: Dict = field(default_factory=dict)


class CostTracker:
    """
    Tracks LLM usage and calculates costs.

    This class monitors all LLM API calls and calculates costs based on
    token usage and current pricing. It provides statistics, alerts, and
    export capabilities for cost analysis.

    Attributes:
        config: Cost tracking configuration
        records: List of all usage records
        _total_cost: Cumulative cost across all requests
        _total_tokens: Cumulative tokens across all requests
        _total_requests: Total number of requests tracked

    Example:
        >>> from flybrowser.llm.config import CostTrackingConfig, LLMProviderType
        >>> config = CostTrackingConfig(
        ...     enabled=True,
        ...     budget_limit_usd=10.0,
        ...     alert_threshold_usd=8.0
        ... )
        >>> tracker = CostTracker(config)
        >>>
        >>> # Track a request
        >>> cost = tracker.calculate_cost(
        ...     LLMProviderType.OPENAI,
        ...     "gpt-4o",
        ...     prompt_tokens=100,
        ...     completion_tokens=50
        ... )
        >>> tracker.track_usage(
        ...     LLMProviderType.OPENAI,
        ...     "gpt-4o",
        ...     prompt_tokens=100,
        ...     completion_tokens=50
        ... )
        >>>
        >>> # Get summary
        >>> summary = tracker.get_summary()
        >>> print(f"Total cost: ${summary['total_cost']:.4f}")
    """

    def __init__(self, config: CostTrackingConfig) -> None:
        """
        Initialize the cost tracker with configuration.

        Args:
            config: Cost tracking configuration containing:
                - enabled: Whether cost tracking is enabled
                - budget_limit_usd: Maximum budget in USD (optional)
                - alert_threshold_usd: Alert when cost exceeds this (optional)
                - track_by_session: Whether to track costs per session

        Example:
            >>> config = CostTrackingConfig(
            ...     enabled=True,
            ...     budget_limit_usd=100.0,
            ...     alert_threshold_usd=80.0
            ... )
            >>> tracker = CostTracker(config)
        """
        self.config = config
        self.records: List[UsageRecord] = []
        self._total_cost = 0.0
        self._total_tokens = 0
        self._total_requests = 0

    def calculate_cost(
        self,
        provider: LLMProviderType,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate cost for a request using simple hardcoded pricing.

        Args:
            provider: Provider type
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        # Simple hardcoded pricing (approximate values as of Jan 2026)
        model_lower = model.lower()
        
        if provider == LLMProviderType.OPENAI:
            if "gpt-5" in model_lower:
                input_cost = 0.010
                output_cost = 0.030
            elif "gpt-4o" in model_lower:
                input_cost = 0.0025
                output_cost = 0.010
            elif "gpt-4" in model_lower:
                input_cost = 0.030
                output_cost = 0.060
            elif "o1" in model_lower or "o3" in model_lower:
                input_cost = 0.015
                output_cost = 0.060
            elif "gpt-3.5" in model_lower:
                input_cost = 0.0005
                output_cost = 0.0015
            else:
                input_cost = 0.0
                output_cost = 0.0
        elif provider == LLMProviderType.ANTHROPIC:
            if "opus" in model_lower:
                input_cost = 0.015
                output_cost = 0.075
            elif "sonnet" in model_lower:
                input_cost = 0.003
                output_cost = 0.015
            elif "haiku" in model_lower:
                input_cost = 0.00025
                output_cost = 0.00125
            else:
                input_cost = 0.0
                output_cost = 0.0
        elif provider == LLMProviderType.GEMINI or provider == LLMProviderType.GOOGLE:
            if "pro" in model_lower:
                input_cost = 0.00125
                output_cost = 0.005
            else:
                input_cost = 0.0  # Free tier
                output_cost = 0.0
        else:
            # Local models (Ollama, etc.) are free
            input_cost = 0.0
            output_cost = 0.0
        
        # Calculate cost based on actual tokens
        cost = (prompt_tokens / 1000) * input_cost + (completion_tokens / 1000) * output_cost
        return cost

    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool = False,
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        """
        Record LLM usage.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cached: Whether response was cached
            metadata: Additional metadata

        Returns:
            Usage record
        """
        if not self.config.enabled:
            return None

        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        try:
            provider_type = LLMProviderType(provider.lower())
            cost = self.calculate_cost(provider_type, model, prompt_tokens, completion_tokens)
        except ValueError:
            cost = 0.0

        # Create record
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            cached=cached,
            metadata=metadata or {},
        )

        # Store record
        self.records.append(record)

        # Update totals
        if not cached:  # Don't count cached responses in totals
            self._total_cost += cost
            self._total_tokens += total_tokens
            self._total_requests += 1

        # Log if enabled
        if self.config.log_costs and cost > 0:
            logger.info(
                f"LLM usage: {model} - {total_tokens} tokens - ${cost:.6f} "
                f"(cached: {cached})"
            )

        return record

    def get_summary(self) -> Dict:
        """
        Get usage summary.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 6),
            "cached_requests": sum(1 for r in self.records if r.cached),
            "records_count": len(self.records),
        }

    def get_breakdown_by_model(self) -> Dict[str, Dict]:
        """
        Get cost breakdown by model.

        Returns:
            Dictionary with per-model statistics
        """
        breakdown = {}
        
        for record in self.records:
            key = f"{record.provider}/{record.model}"
            if key not in breakdown:
                breakdown[key] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                }
            
            if not record.cached:
                breakdown[key]["requests"] += 1
                breakdown[key]["tokens"] += record.total_tokens
                breakdown[key]["cost"] += record.cost

        return breakdown

    def reset(self) -> None:
        """Reset all tracking data."""
        self.records.clear()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._total_requests = 0
        logger.info("Cost tracker reset")

