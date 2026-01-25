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
Conversation Manager for Multi-Turn LLM Interactions.

This module provides the ConversationManager class which handles:
- Multi-turn conversation history tracking
- Large content splitting across multiple turns
- Token budget management to prevent overflow
- Structured output preservation across turns
- Context window optimization

The ConversationManager acts as a layer between the ReAct agent and
LLM providers, ensuring conversations stay within token limits while
preserving all necessary context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from flybrowser.llm.token_budget import (
    TokenEstimator, TokenBudgetManager, ContentType, BudgetAllocation
)
from flybrowser.llm.chunking import (
    Chunk, ChunkingStrategy, SmartChunker, get_chunker
)
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider, ModelInfo


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    """
    A message in a conversation.
    
    Attributes:
        role: Message role (system, user, assistant)
        content: Message content (text or structured)
        tokens: Estimated token count
        timestamp: When message was created
        metadata: Additional message metadata
    """
    role: MessageRole
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API message format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }
    
    @classmethod
    def system(cls, content: str) -> "ConversationMessage":
        """Create a system message."""
        tokens = TokenEstimator.estimate(content).tokens
        return cls(role=MessageRole.SYSTEM, content=content, tokens=tokens)
    
    @classmethod
    def user(cls, content: Union[str, List[Dict[str, Any]]]) -> "ConversationMessage":
        """Create a user message."""
        if isinstance(content, str):
            tokens = TokenEstimator.estimate(content).tokens
        else:
            tokens = TokenEstimator.estimate_messages([{"content": content}])
        return cls(role=MessageRole.USER, content=content, tokens=tokens)
    
    @classmethod
    def assistant(cls, content: str) -> "ConversationMessage":
        """Create an assistant message."""
        tokens = TokenEstimator.estimate(content).tokens
        return cls(role=MessageRole.ASSISTANT, content=content, tokens=tokens)


@dataclass
class ConversationHistory:
    """
    Tracks conversation history with token management.
    
    Maintains a list of messages and provides methods for
    pruning history to stay within token budgets.
    """
    messages: List[ConversationMessage] = field(default_factory=list)
    system_message: Optional[ConversationMessage] = None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens in conversation history."""
        total = sum(m.tokens for m in self.messages)
        if self.system_message:
            total += self.system_message.tokens
        return total
    
    @property
    def message_count(self) -> int:
        """Number of messages (excluding system)."""
        return len(self.messages)
    
    def add(self, message: ConversationMessage) -> None:
        """Add a message to history."""
        if message.role == MessageRole.SYSTEM:
            self.system_message = message
        else:
            self.messages.append(message)
    
    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages in API format."""
        result = []
        if self.system_message:
            result.append(self.system_message.to_api_format())
        result.extend(m.to_api_format() for m in self.messages)
        return result
    
    def prune_to_fit(self, max_tokens: int, keep_recent: int = 2) -> int:
        """
        Prune history to fit within token budget.
        
        Removes oldest messages first, but always keeps the most recent
        `keep_recent` messages.
        
        Args:
            max_tokens: Maximum total tokens allowed
            keep_recent: Minimum recent messages to keep
            
        Returns:
            Number of messages removed
        """
        removed = 0
        
        while self.total_tokens > max_tokens and len(self.messages) > keep_recent:
            # Remove oldest non-system message
            self.messages.pop(0)
            removed += 1
        
        return removed
    
    def clear(self) -> None:
        """Clear conversation history (keeps system message)."""
        self.messages.clear()
    
    def reset(self) -> None:
        """Reset everything including system message."""
        self.messages.clear()
        self.system_message = None


class AccumulationPhase(str, Enum):
    """Phases for multi-turn accumulation protocol."""
    SINGLE = "single"          # No accumulation needed
    ACCUMULATING = "accumulating"  # Processing chunks
    SYNTHESIZING = "synthesizing"  # Final synthesis


@dataclass
class AccumulationContext:
    """Context for multi-turn content accumulation."""
    phase: AccumulationPhase = AccumulationPhase.SINGLE
    total_chunks: int = 0
    processed_chunks: int = 0
    chunk_summaries: List[str] = field(default_factory=list)
    original_instruction: str = ""
    
    @property
    def is_complete(self) -> bool:
        """Check if accumulation is complete."""
        return self.phase == AccumulationPhase.SINGLE or \
               (self.phase == AccumulationPhase.SYNTHESIZING and 
                self.processed_chunks >= self.total_chunks)
    
    @property
    def progress(self) -> float:
        """Get accumulation progress (0-1)."""
        if self.total_chunks == 0:
            return 1.0
        return self.processed_chunks / self.total_chunks


class ConversationManager:
    """
    Manages multi-turn conversations with LLM providers.
    
    Handles:
    - Conversation history tracking
    - Token budget management
    - Large content chunking and multi-turn processing
    - Structured output across turns
    
    The ConversationManager ensures that conversations stay within
    token limits while preserving all necessary context for the agent
    to make informed decisions.
    
    Example:
        >>> manager = ConversationManager(llm_provider)
        >>> manager.set_system_prompt("You are a helpful assistant.")
        >>> 
        >>> # Simple single-turn
        >>> response = await manager.send_structured(
        ...     "What is the capital of France?",
        ...     schema={"type": "object", "properties": {"answer": {"type": "string"}}}
        ... )
        >>> 
        >>> # Large content with automatic chunking
        >>> response = await manager.send_with_large_content(
        ...     large_html_content,
        ...     instruction="Extract all product names",
        ...     schema=product_schema
        ... )
    """
    
    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        model_info: Optional["ModelInfo"] = None,
        max_history_tokens: Optional[int] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None,
    ) -> None:
        """
        Initialize the ConversationManager.
        
        Args:
            llm_provider: LLM provider instance
            model_info: Model information (fetched from provider if not given)
            max_history_tokens: Maximum tokens for history (default: 50% of context)
            chunking_strategy: Strategy for splitting large content
        """
        self.llm = llm_provider
        
        # Get model info
        if model_info is None:
            self.model_info = llm_provider.get_model_info()
        else:
            self.model_info = model_info
        
        # Initialize budget manager
        self.budget = TokenBudgetManager(
            context_window=self.model_info.context_window,
            max_output_tokens=self.model_info.max_output_tokens,
            safety_margin=0.1,
        )
        
        # History token budget (default 50% of available)
        if max_history_tokens is None:
            max_history_tokens = int(self.budget.available_for_input * 0.5)
        self.max_history_tokens = max_history_tokens
        
        # Initialize history
        self.history = ConversationHistory()
        
        # Chunking strategy
        self.chunker = chunking_strategy or SmartChunker()
        
        # Accumulation context for multi-turn processing
        self._accumulation: Optional[AccumulationContext] = None
        
        # Statistics
        self._total_requests = 0
        self._total_tokens_used = 0
        self._multi_turn_requests = 0
        
        logger.info(
            f"ConversationManager initialized: context_window={self.model_info.context_window}, "
            f"max_output={self.model_info.max_output_tokens}, max_history={max_history_tokens}"
        )
    
    def set_system_prompt(self, content: str) -> None:
        """Set the system prompt for the conversation."""
        self.history.add(ConversationMessage.system(content))
        logger.debug(f"System prompt set ({TokenEstimator.estimate(content).tokens} tokens)")
    
    def add_user_message(self, content: Union[str, List[Dict[str, Any]]]) -> None:
        """Add a user message to history."""
        self.history.add(ConversationMessage.user(content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.history.add(ConversationMessage.assistant(content))
    
    async def send_structured(
        self,
        content: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        add_to_history: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a message and get structured response.
        
        Automatically handles:
        - Token budget checking
        - History pruning if needed
        - Response validation
        
        Args:
            content: User message content
            schema: JSON schema for response
            temperature: Sampling temperature
            add_to_history: Whether to add to conversation history
            **kwargs: Additional LLM parameters
            
        Returns:
            Structured response matching schema
        """
        # Check if content fits
        content_tokens = TokenEstimator.estimate(content).tokens
        available = self.budget.available_for_input - self.history.total_tokens
        
        if content_tokens > available:
            logger.warning(
                f"Content too large for single turn ({content_tokens} > {available}). "
                f"Consider using send_with_large_content()."
            )
            # Prune history to make room
            self.history.prune_to_fit(available - content_tokens)
        
        # Build messages
        messages = self.history.get_messages_for_api()
        messages.append({"role": "user", "content": content})
        
        # Get system prompt from history if present
        system_prompt = None
        if self.history.system_message:
            system_prompt = self.history.system_message.content
            # Remove system from messages (will be passed separately)
            messages = [m for m in messages if m["role"] != "system"]
        
        # Call LLM
        self._total_requests += 1
        
        try:
            # Format messages as prompt for generate_structured
            formatted_prompt = self._format_messages_as_prompt(messages)
            
            response = await self.llm.generate_structured(
                prompt=formatted_prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
            
            # Estimate token usage since generate_structured returns Dict, not LLMResponse
            # This is an approximation based on input/output sizes
            input_tokens = TokenEstimator.estimate(formatted_prompt).tokens
            if system_prompt:
                input_tokens += TokenEstimator.estimate(system_prompt).tokens
            output_tokens = TokenEstimator.estimate(json.dumps(response)).tokens
            self._total_tokens_used += input_tokens + output_tokens
            
            # Add to history
            if add_to_history:
                self.add_user_message(content)
                self.add_assistant_message(json.dumps(response))
            
            return response
            
        except Exception as e:
            logger.error(f"ConversationManager.send_structured failed: {e}")
            raise
    
    async def send_with_large_content(
        self,
        content: str,
        instruction: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        content_type: Optional[ContentType] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send large content that may need to be split across multiple turns.
        
        Implements the accumulation protocol:
        1. If content fits, send as single turn
        2. If too large, chunk and process incrementally
        3. Final synthesis turn produces structured output
        
        Args:
            content: Large content to process
            instruction: What to do with the content
            schema: JSON schema for final response
            temperature: Sampling temperature
            content_type: Optional content type hint
            **kwargs: Additional LLM parameters
            
        Returns:
            Structured response matching schema
        """
        # Estimate content size
        content_estimate = TokenEstimator.estimate(content, content_type)
        
        # Calculate available space (leaving room for instruction and response)
        instruction_tokens = TokenEstimator.estimate(instruction).tokens
        available = self.budget.available_for_input - self.history.total_tokens - instruction_tokens - 500
        
        logger.info(
            f"Processing large content: {content_estimate.tokens} tokens, "
            f"available: {available}, type: {content_estimate.content_type.value}"
        )
        
        # If content fits, send as single turn
        if content_estimate.tokens <= available:
            combined = f"{instruction}\n\n---\nContent to process:\n{content}"
            return await self.send_structured(combined, schema, temperature, **kwargs)
        
        # Need multi-turn processing
        self._multi_turn_requests += 1
        
        # Get appropriate chunker
        chunker = get_chunker(content_type or content_estimate.content_type)
        
        # Calculate chunk size (leave room for accumulation overhead)
        chunk_budget = int(available * 0.7)  # 70% for content, 30% for overhead
        chunks = chunker.chunk(content, chunk_budget)
        
        logger.info(f"Split content into {len(chunks)} chunks for multi-turn processing")
        
        # Initialize accumulation context
        self._accumulation = AccumulationContext(
            phase=AccumulationPhase.ACCUMULATING,
            total_chunks=len(chunks),
            processed_chunks=0,
            original_instruction=instruction,
        )
        
        # Process chunks
        for chunk in chunks:
            await self._process_accumulation_chunk(chunk)
        
        # Synthesis phase
        return await self._synthesize_accumulated(schema, temperature, **kwargs)
    
    async def _process_accumulation_chunk(self, chunk: Chunk) -> None:
        """Process a single chunk during accumulation phase."""
        if not self._accumulation:
            raise ValueError("No accumulation context")
        
        # Build accumulation prompt
        prompt = self._build_accumulation_prompt(chunk)
        
        # Simple acknowledgment schema
        ack_schema = {
            "type": "object",
            "properties": {
                "acknowledged": {"type": "boolean"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points extracted from this chunk"
                }
            },
            "required": ["acknowledged", "key_points"]
        }
        
        try:
            response = await self.send_structured(
                prompt,
                schema=ack_schema,
                temperature=0.3,  # Low temp for consistent accumulation
                add_to_history=False,  # Don't bloat history with chunks
            )
            
            # Store key points for synthesis
            if response.get("key_points"):
                self._accumulation.chunk_summaries.extend(response["key_points"])
            
            self._accumulation.processed_chunks += 1
            
            logger.debug(
                f"Processed chunk {chunk.index + 1}/{chunk.total_chunks}, "
                f"extracted {len(response.get('key_points', []))} key points"
            )
            
        except Exception as e:
            logger.warning(f"Chunk processing failed: {e}, continuing...")
            self._accumulation.processed_chunks += 1
    
    async def _synthesize_accumulated(
        self,
        schema: Dict[str, Any],
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synthesize final response from accumulated chunks."""
        if not self._accumulation:
            raise ValueError("No accumulation context")
        
        self._accumulation.phase = AccumulationPhase.SYNTHESIZING
        
        # Build synthesis prompt with accumulated key points
        synthesis_prompt = self._build_synthesis_prompt()
        
        logger.info(
            f"Synthesizing from {len(self._accumulation.chunk_summaries)} key points"
        )
        
        try:
            response = await self.send_structured(
                synthesis_prompt,
                schema=schema,
                temperature=temperature,
                **kwargs,
            )
            
            return response
            
        finally:
            # Clear accumulation context
            self._accumulation = None
    
    def _build_accumulation_prompt(self, chunk: Chunk) -> str:
        """Build prompt for accumulation phase."""
        header = chunk.format_header()
        instruction = self._accumulation.original_instruction if self._accumulation else ""
        
        return f"""I'm processing a large document in multiple parts. This is {header}.

INSTRUCTION: {instruction}

For now, just acknowledge receiving this chunk and extract the key points relevant to the instruction.
Do NOT produce the final answer yet - that will come after all chunks are processed.

---
CHUNK CONTENT:
{chunk.content}
---

Extract key points from this chunk that are relevant to the instruction."""
    
    def _build_synthesis_prompt(self) -> str:
        """Build prompt for synthesis phase."""
        if not self._accumulation:
            return ""
        
        key_points = self._accumulation.chunk_summaries
        instruction = self._accumulation.original_instruction
        
        key_points_text = "\n".join(f"- {point}" for point in key_points)
        
        return f"""You have processed a large document in {self._accumulation.total_chunks} chunks.
Here are the key points extracted from all chunks:

{key_points_text}

Now, based on these key points, fulfill the original instruction:
INSTRUCTION: {instruction}

Provide your final, complete response based on all the information gathered."""
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages array as a single prompt string."""
        # For providers that don't support messages array natively
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, list):
                # Multi-part content (text + images)
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = "\n".join(text_parts)
            
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        return "\n\n".join(parts)
    
    def get_available_tokens(self) -> int:
        """Get tokens available for next message."""
        return self.budget.available_for_input - self.history.total_tokens
    
    def would_exceed_budget(self, content: str) -> tuple[bool, int]:
        """Check if content would exceed budget."""
        content_tokens = TokenEstimator.estimate(content).tokens
        available = self.get_available_tokens()
        overflow = content_tokens - available
        return overflow > 0, max(0, overflow)
    
    def reset(self) -> None:
        """Reset conversation (clear history, keep system prompt)."""
        self.history.clear()
        self._accumulation = None
        self.budget.reset()
        logger.debug("Conversation reset")
    
    def full_reset(self) -> None:
        """Full reset including system prompt."""
        self.history.reset()
        self._accumulation = None
        self.budget.reset()
        logger.debug("Full conversation reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "total_requests": self._total_requests,
            "multi_turn_requests": self._multi_turn_requests,
            "total_tokens_used": self._total_tokens_used,
            "history_messages": self.history.message_count,
            "history_tokens": self.history.total_tokens,
            "available_tokens": self.get_available_tokens(),
            "budget_stats": self.budget.get_stats(),
        }
