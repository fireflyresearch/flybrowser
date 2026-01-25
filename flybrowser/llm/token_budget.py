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
Token Budget Manager for conversation context management.

This module provides utilities for estimating token counts and managing
token budgets across multi-turn conversations to prevent context overflow
and rate limit errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from flybrowser.utils.logger import logger


class ContentType(str, Enum):
    """Types of content for token estimation."""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    CODE = "code"
    IMAGE = "image"


@dataclass
class TokenEstimate:
    """Token estimation result with confidence."""
    tokens: int
    content_type: ContentType
    confidence: float = 0.9  # Estimation confidence (0-1)
    raw_size: int = 0  # Original size in chars/bytes
    
    @property
    def with_buffer(self) -> int:
        """Get token count with safety buffer based on confidence."""
        # Lower confidence = higher buffer
        buffer_multiplier = 1.0 + (1.0 - self.confidence) * 0.5
        return int(self.tokens * buffer_multiplier)


@dataclass 
class BudgetAllocation:
    """Budget allocation for different components."""
    system_prompt: int = 0
    conversation_history: int = 0
    current_message: int = 0
    response_reserve: int = 0
    safety_buffer: int = 0
    
    @property
    def total_input(self) -> int:
        """Total input tokens (everything except response reserve)."""
        return (self.system_prompt + self.conversation_history + 
                self.current_message + self.safety_buffer)
    
    @property
    def total(self) -> int:
        """Total allocated tokens."""
        return self.total_input + self.response_reserve


class TokenEstimator:
    """
    Estimates token counts for various content types.
    
    Uses heuristics based on content type:
    - English text: ~4 chars per token (GPT tokenizer average)
    - Code: ~3.5 chars per token (more symbols)
    - JSON: ~3 chars per token (structural overhead)
    - HTML: ~2.5 chars per token (lots of tags)
    
    For accurate counts, use tiktoken or provider-specific tokenizers.
    """
    
    # Chars per token ratios by content type
    CHARS_PER_TOKEN = {
        ContentType.TEXT: 4.0,
        ContentType.CODE: 3.5,
        ContentType.JSON: 3.0,
        ContentType.HTML: 2.5,
        ContentType.IMAGE: 1.0,  # Images use special token counts
    }
    
    # Image token estimates (based on OpenAI GPT-4V)
    IMAGE_TOKENS = {
        "low": 85,      # Low detail
        "high_small": 765,   # High detail, small image (<512x512)
        "high_medium": 1105,  # High detail, medium image
        "high_large": 1445,   # High detail, large image (>2048x2048)
    }
    
    @classmethod
    def estimate(
        cls,
        content: Union[str, bytes, Dict, List],
        content_type: Optional[ContentType] = None,
    ) -> TokenEstimate:
        """
        Estimate token count for content.
        
        Args:
            content: Content to estimate (text, bytes, dict, or list)
            content_type: Optional explicit content type (auto-detected if None)
            
        Returns:
            TokenEstimate with count and metadata
        """
        # Handle bytes (likely image)
        if isinstance(content, bytes):
            return cls._estimate_image(content)
        
        # Convert to string if needed
        if isinstance(content, (dict, list)):
            import json
            content_str = json.dumps(content, ensure_ascii=False)
            detected_type = ContentType.JSON
        else:
            content_str = str(content)
            detected_type = cls._detect_content_type(content_str)
        
        # Use provided type or detected
        final_type = content_type or detected_type
        
        # Calculate tokens
        chars_per_token = cls.CHARS_PER_TOKEN.get(final_type, 4.0)
        raw_size = len(content_str)
        tokens = max(1, int(raw_size / chars_per_token))
        
        # Confidence based on content type (some are harder to estimate)
        confidence = 0.85 if final_type in (ContentType.HTML, ContentType.CODE) else 0.9
        
        return TokenEstimate(
            tokens=tokens,
            content_type=final_type,
            confidence=confidence,
            raw_size=raw_size,
        )
    
    @classmethod
    def _detect_content_type(cls, content: str) -> ContentType:
        """Auto-detect content type from string."""
        content_sample = content[:2000]  # Sample for detection
        
        # Check for HTML
        if re.search(r'<[a-zA-Z][^>]*>', content_sample):
            html_tag_ratio = len(re.findall(r'<[^>]+>', content_sample)) / max(1, len(content_sample.split()))
            if html_tag_ratio > 0.1:
                return ContentType.HTML
        
        # Check for JSON
        stripped = content.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            return ContentType.JSON
        
        # Check for code patterns
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python
            r'function\s+\w+\s*\(',  # JavaScript
            r'class\s+\w+',  # Class definitions
            r'import\s+\w+',  # Imports
            r'\breturn\b',  # Return statements
            r'{\s*\n',  # Code blocks
        ]
        code_matches = sum(1 for pattern in code_indicators if re.search(pattern, content_sample))
        if code_matches >= 2:
            return ContentType.CODE
        
        return ContentType.TEXT
    
    @classmethod
    def _estimate_image(cls, image_bytes: bytes) -> TokenEstimate:
        """Estimate tokens for image data."""
        size_kb = len(image_bytes) / 1024
        
        # Rough estimation based on image size
        # Larger images = more detail = more tokens
        if size_kb < 100:
            tokens = cls.IMAGE_TOKENS["low"]
        elif size_kb < 500:
            tokens = cls.IMAGE_TOKENS["high_small"]
        elif size_kb < 1500:
            tokens = cls.IMAGE_TOKENS["high_medium"]
        else:
            tokens = cls.IMAGE_TOKENS["high_large"]
        
        return TokenEstimate(
            tokens=tokens,
            content_type=ContentType.IMAGE,
            confidence=0.75,  # Image estimation is less precise
            raw_size=len(image_bytes),
        )
    
    @classmethod
    def estimate_messages(cls, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate total tokens for a list of messages.
        
        Accounts for message overhead (role, formatting).
        """
        total = 0
        for msg in messages:
            # Message overhead (role token, formatting)
            overhead = 4
            
            content = msg.get("content", "")
            if isinstance(content, str):
                estimate = cls.estimate(content)
                total += estimate.tokens + overhead
            elif isinstance(content, list):
                # Multi-part content (text + images)
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            estimate = cls.estimate(part.get("text", ""))
                            total += estimate.tokens
                        elif part.get("type") == "image_url":
                            total += cls.IMAGE_TOKENS["high_medium"]
                total += overhead
        
        return total


@dataclass
class TokenBudgetManager:
    """
    Manages token budget for conversations.
    
    Tracks usage and provides budget allocation for different
    components of a conversation (system prompt, history, current message).
    
    Attributes:
        context_window: Total context window size for the model
        max_output_tokens: Maximum tokens reserved for response
        rate_limit_tpm: Tokens per minute rate limit (if any)
        safety_margin: Percentage of budget to keep as safety buffer
    """
    
    context_window: int = 128000
    max_output_tokens: int = 8192
    rate_limit_tpm: Optional[int] = None
    safety_margin: float = 0.1  # 10% safety buffer
    
    # Internal tracking
    _used_tokens: int = field(default=0, init=False)
    _message_tokens: List[int] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Initialize internal state."""
        self._used_tokens = 0
        self._message_tokens = []
    
    @property
    def available_for_input(self) -> int:
        """Tokens available for input (excluding response reserve)."""
        safety_buffer = int(self.context_window * self.safety_margin)
        return self.context_window - self.max_output_tokens - safety_buffer - self._used_tokens
    
    @property
    def available_for_response(self) -> int:
        """Tokens available for model response."""
        return self.max_output_tokens
    
    @property
    def total_used(self) -> int:
        """Total tokens used so far."""
        return self._used_tokens
    
    def can_fit(self, content: Union[str, bytes, Dict, List], content_type: Optional[ContentType] = None) -> bool:
        """
        Check if content can fit in remaining budget.
        
        Args:
            content: Content to check
            content_type: Optional content type hint
            
        Returns:
            True if content fits within budget
        """
        estimate = TokenEstimator.estimate(content, content_type)
        return estimate.with_buffer <= self.available_for_input
    
    def allocate(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, Any]],
        current_content: Union[str, bytes, Dict, List],
    ) -> BudgetAllocation:
        """
        Calculate budget allocation for a conversation turn.
        
        Args:
            system_prompt: System prompt content
            conversation_history: Previous messages
            current_content: Content for current message
            
        Returns:
            BudgetAllocation with token counts for each component
        """
        system_tokens = TokenEstimator.estimate(system_prompt).with_buffer if system_prompt else 0
        history_tokens = TokenEstimator.estimate_messages(conversation_history)
        current_tokens = TokenEstimator.estimate(current_content).with_buffer
        
        safety_buffer = int(self.context_window * self.safety_margin)
        
        return BudgetAllocation(
            system_prompt=system_tokens,
            conversation_history=history_tokens,
            current_message=current_tokens,
            response_reserve=self.max_output_tokens,
            safety_buffer=safety_buffer,
        )
    
    def would_exceed_budget(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, Any]],
        current_content: Union[str, bytes, Dict, List],
    ) -> tuple[bool, int]:
        """
        Check if adding content would exceed budget.
        
        Args:
            system_prompt: System prompt
            conversation_history: Previous messages
            current_content: New content to add
            
        Returns:
            Tuple of (would_exceed, overflow_tokens)
        """
        allocation = self.allocate(system_prompt, conversation_history, current_content)
        overflow = allocation.total - self.context_window
        return overflow > 0, max(0, overflow)
    
    def record_usage(self, tokens: int) -> None:
        """Record token usage for a message."""
        self._used_tokens += tokens
        self._message_tokens.append(tokens)
    
    def calculate_chunk_size(
        self,
        total_content_tokens: int,
        reserved_for_other: int = 0,
    ) -> int:
        """
        Calculate optimal chunk size for splitting large content.
        
        Args:
            total_content_tokens: Total tokens in content to split
            reserved_for_other: Tokens reserved for other purposes
            
        Returns:
            Recommended tokens per chunk
        """
        available = self.available_for_input - reserved_for_other
        
        # Aim for chunks that fit comfortably (60% of available)
        target_chunk = int(available * 0.6)
        
        # Minimum meaningful chunk size
        min_chunk = 1000
        
        return max(min_chunk, target_chunk)
    
    def reset(self) -> None:
        """Reset budget tracking for new conversation."""
        self._used_tokens = 0
        self._message_tokens.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get budget usage statistics."""
        return {
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "used_tokens": self._used_tokens,
            "available_for_input": self.available_for_input,
            "message_count": len(self._message_tokens),
            "average_message_tokens": (
                sum(self._message_tokens) / len(self._message_tokens) 
                if self._message_tokens else 0
            ),
        }
