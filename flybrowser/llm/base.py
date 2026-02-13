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
Base LLM provider interface with production features.

This module defines the abstract base class for all LLM providers in FlyBrowser.
It provides a unified interface for interacting with different LLM services
(OpenAI, Anthropic, Ollama, Google Gemini, etc.) and includes production-ready
features like caching, retry logic, rate limiting, and cost tracking.

The module also defines the LLMResponse dataclass which standardizes responses
across all providers.

Key Features:
- Unified API across all providers
- Vision/multimodal support with single and multiple images
- Streaming response support
- Tool/function calling support
- Structured output with JSON schemas
- Embeddings generation
- Model capability introspection
- Production features (caching, retry, rate limiting, cost tracking)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from flybrowser.llm.provider_status import ProviderStatus
from flybrowser.utils.logger import logger
from flybrowser.utils.execution_logger import get_execution_logger, LogVerbosity


# Stubs for removed modules (functionality now handled by fireflyframework-genai)
class LLMCache:
    """Stub -- caching is now handled by fireflyframework-genai."""


class LLMProviderConfig:
    """Stub -- provider config is now handled by fireflyframework-genai model strings."""


class CostTracker:
    """Stub -- cost tracking is now handled by fireflyframework-genai."""


class RateLimiter:
    """Stub -- rate limiting is now handled by fireflyframework-genai."""


class RetryHandler:
    """Stub -- retry logic is now handled by fireflyframework-genai."""


class ModelCapability(str, Enum):
    """Capabilities that a model may support."""

    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    MULTI_IMAGE_VISION = "multi_image_vision"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    EMBEDDINGS = "embeddings"
    CODE_EXECUTION = "code_execution"
    EXTENDED_THINKING = "extended_thinking"


@dataclass
class ImageInput:
    """
    Represents an image input for vision-capable models.

    Attributes:
        data: Image data as bytes or base64 string
        media_type: MIME type of the image (e.g., "image/png", "image/jpeg")
        detail: Level of detail for image analysis ("low", "high", "auto")
        source_type: Whether data is "bytes" or "base64"
    """

    data: Union[bytes, str]
    media_type: str = "image/png"
    detail: str = "auto"
    source_type: str = "bytes"

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str = "image/png", detail: str = "auto") -> "ImageInput":
        """Create ImageInput from raw bytes."""
        return cls(data=data, media_type=media_type, detail=detail, source_type="bytes")

    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png", detail: str = "auto") -> "ImageInput":
        """Create ImageInput from base64 string."""
        return cls(data=data, media_type=media_type, detail=detail, source_type="base64")

    @classmethod
    def from_url(cls, url: str, detail: str = "auto") -> "ImageInput":
        """Create ImageInput from URL (for providers that support it)."""
        return cls(data=url, media_type="url", detail=detail, source_type="url")


@dataclass
class ToolDefinition:
    """
    Definition of a tool/function that can be called by the LLM.

    Attributes:
        name: Name of the tool
        description: Description of what the tool does
        parameters: JSON schema for the tool's parameters
        required: List of required parameter names
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """
    Represents a tool call made by the LLM.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool being called
        arguments: Arguments passed to the tool (as dict)
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelInfo:
    """
    Information about a model's capabilities and limits.

    Attributes:
        name: Model name/identifier
        provider: Provider name
        capabilities: Set of supported capabilities
        context_window: Maximum context window size in tokens
        max_output_tokens: Maximum output tokens
        supports_system_prompt: Whether model supports system prompts
        cost_per_1k_input_tokens: Cost per 1000 input tokens (USD)
        cost_per_1k_output_tokens: Cost per 1000 output tokens (USD)
    """

    name: str
    provider: str
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_window: int = 128000
    max_output_tokens: int = 8192  # Generous default to avoid truncation
    supports_system_prompt: bool = True
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None


@dataclass
class LLMResponse:
    """
    Standardized response from an LLM provider.

    This dataclass encapsulates the response from any LLM provider, providing
    a consistent interface regardless of the underlying service.

    Attributes:
        content: The generated text content from the LLM
        model: Name of the model that generated the response
        usage: Token usage statistics with keys:
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - total_tokens: Total tokens used
        metadata: Additional provider-specific metadata
        cached: Whether this response was served from cache
        tool_calls: List of tool calls made by the model
        finish_reason: Reason for completion (e.g., "stop", "tool_calls", "length")

    Example:
        >>> response = LLMResponse(
        ...     content="The capital of France is Paris.",
        ...     model="gpt-4o",
        ...     usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        ...     cached=False
        ... )
    """

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    cached: bool = False
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers with production features.

    This class defines the interface that all LLM providers must implement.
    It also provides production-ready features like caching, retry logic,
    rate limiting, and cost tracking when configured.

    Subclasses must implement:
    - generate(): Basic text generation
    - generate_with_vision(): Generation with image input
    - generate_structured(): Generation with structured output

    Attributes:
        model: Model name/identifier
        api_key: API key for the LLM provider (e.g., OpenAI API key)
        provider_config: Full provider configuration
        extra_config: Additional provider-specific configuration
        cache: LLM response cache (if configured)
        cost_tracker: Token usage and cost tracker (if configured)
        rate_limiter: Rate limiter for API requests (if configured)
        retry_handler: Retry handler with exponential backoff (if configured)

    Example:
        Subclass implementation:

        >>> class MyLLMProvider(BaseLLMProvider):
        ...     async def generate(self, prompt, **kwargs):
        ...         # Implementation here
        ...         pass
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LLM provider with configuration.

        Args:
            model: Model name/identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key for the LLM provider (e.g., OpenAI API key, Anthropic API key).
                Not required for local providers like Ollama.
            config: Full provider configuration including retry, cache, rate limit settings.
                If not provided, production features will be disabled.
            **kwargs: Additional provider-specific configuration options:
                - llm_logging: Enable LLM request/response logging (default: False)
                - vision_enabled: Override auto-detected vision capability (Optional[bool])
                    - None: Use auto-detected capabilities (default)
                    - True: Force vision enabled
                    - False: Force vision disabled

        Note:
            When config is provided, the following production features are enabled:
            - Response caching (LRU cache with TTL)
            - Automatic retry with exponential backoff
            - Rate limiting (requests/min and tokens/min)
            - Cost tracking and reporting
        """
        self.model = model
        self.api_key = api_key
        self.provider_config = config
        self.extra_config = kwargs
        
        # Vision capability override
        # None = use auto-detected, True = force on, False = force off
        self._vision_enabled_override: Optional[bool] = kwargs.get("vision_enabled")
        
        # Standardized capability caching (all providers should use these)
        self._capabilities_detected: bool = False
        self._model_capabilities_cache: Optional[List[ModelCapability]] = None
        self._model_info_cache: Optional[ModelInfo] = None

        # Initialize production features if config provided
        if config:
            self.cache = LLMCache(config.cache_config)
            self.cost_tracker = CostTracker(config.cost_tracking_config)
            self.rate_limiter = RateLimiter(config.rate_limit_config)
            self.retry_handler = RetryHandler(config.retry_config)
        else:
            # No config provided - production features disabled
            self.cache = None
            self.cost_tracker = None
            self.rate_limiter = None
            self.retry_handler = None
        
        # Session-level usage tracking (always enabled)
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_total_tokens = 0
        self._session_cost = 0.0
        self._session_calls = 0
        self._session_cached_calls = 0
        
        # LLM request/response logging (configurable)
        # Levels: False/0 = disabled, True/1 = basic, 2 = detailed (shows prompts/responses)
        llm_logging_value = kwargs.get("llm_logging", False)
        if isinstance(llm_logging_value, bool):
            self._llm_logging_level = 1 if llm_logging_value else 0
        else:
            self._llm_logging_level = int(llm_logging_value)
        self._llm_logging_enabled = self._llm_logging_level > 0
        
        # Execution logger for hierarchical logging
        self._elog = get_execution_logger()

    async def generate_with_features(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate with production features (caching, rate limiting, retry, cost tracking).

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            use_cache: Whether to use caching
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        # Check cache first
        if use_cache and self.cache:
            cached_response = self.cache.get(
                prompt, system_prompt, self.model, temperature, **kwargs
            )
            if cached_response:
                logger.debug("Returning cached response")
                cached_response.cached = True
                return cached_response

        # Acquire rate limit
        if self.rate_limiter:
            estimated_tokens = len(prompt.split()) * 2  # Rough estimate
            await self.rate_limiter.acquire(estimated_tokens)

        try:
            # Execute with retry
            if self.retry_handler:
                response = await self.retry_handler.execute_with_retry(
                    self._generate_impl,
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    **kwargs,
                )
            else:
                response = await self._generate_impl(
                    prompt, system_prompt, temperature, max_tokens, **kwargs
                )

            # Track cost
            if response.usage:
                prompt_tokens = response.usage.get("prompt_tokens", 0)
                completion_tokens = response.usage.get("completion_tokens", 0)
                
                # Update session-level tracking
                self._session_prompt_tokens += prompt_tokens
                self._session_completion_tokens += completion_tokens
                self._session_total_tokens += prompt_tokens + completion_tokens
                self._session_calls += 1
                
                if self.cost_tracker:
                    record = self.cost_tracker.record_usage(
                        provider=self.__class__.__name__.replace("Provider", "").lower(),
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached=False,
                    )
                    if record:
                        self._session_cost += record.cost

            # Cache response
            if use_cache and self.cache:
                self.cache.set(
                    prompt, system_prompt, self.model, temperature, response, **kwargs
                )

            return response

        finally:
            # Release rate limit
            if self.rate_limiter:
                actual_tokens = (
                    response.usage.get("total_tokens", 0) if response.usage else 0
                )
                self.rate_limiter.release(actual_tokens)

    def enable_llm_logging(self, enabled: bool = True, level: int = 1) -> None:
        """
        Enable or disable LLM request/response logging.
        
        Args:
            enabled: Whether to enable logging (default: True)
            level: Logging level (default: 1)
                - 0: Disabled
                - 1: Basic (shows request/response timing)
                - 2: Detailed (also shows prompts and responses)
        """
        if not enabled:
            self._llm_logging_level = 0
        else:
            self._llm_logging_level = level
        self._llm_logging_enabled = self._llm_logging_level > 0
        
        level_names = {0: "disabled", 1: "basic", 2: "detailed"}
        logger.info(f"LLM logging: {level_names.get(self._llm_logging_level, 'unknown')} (level {self._llm_logging_level})")

    def _log_llm_request(self, method: str, model: str, extra_info: str = "", prompt: str = "", system_prompt: str = "") -> float:
        """
        Log LLM request if logging is enabled. Returns start time.
        
        Args:
            method: Method name (GENERATE, VISION, STRUCTURED)
            model: Model name
            extra_info: Additional info to show
            prompt: The user prompt (logged at level 2)
            system_prompt: The system prompt (logged at level 2)
        """
        import time
        
        # Use execution logger for hierarchical tracking
        # llm_request only takes model and operation, extra_info is ignored at this level
        self._elog.llm_request(model, method.lower())
        
        # Legacy logging for llm_logging mode (deprecated but kept for compatibility)
        if self._llm_logging_enabled:
            # Level 2: Show detailed prompts (DEBUG level in execution logger)
            if self._llm_logging_level >= 2:
                if system_prompt:
                    sys_display = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt
                    logger.debug(f"  LLM system: {sys_display}")
                if prompt:
                    prompt_display = prompt[:1000] + "..." if len(prompt) > 1000 else prompt
                    logger.debug(f"  LLM prompt: {prompt_display}")
        return time.time()

    def _log_llm_response(self, method: str, start_time: float, tokens: int = 0, response_content: str = "") -> None:
        """
        Log LLM response if logging is enabled.
        
        Args:
            method: Method name
            start_time: Start time from _log_llm_request
            tokens: Token count
            response_content: The response content (logged at level 2)
        """
        import time
        elapsed_s = time.time() - start_time
        elapsed_ms = elapsed_s * 1000  # Convert to milliseconds
        
        # Use execution logger for hierarchical tracking
        self._elog.llm_response(self.model, elapsed_ms, tokens if tokens > 0 else None)
        
        # Legacy logging for llm_logging mode (deprecated but kept for compatibility)
        if self._llm_logging_enabled and self._llm_logging_level >= 2 and response_content:
            resp_display = response_content[:2000] + "..." if len(response_content) > 2000 else response_content
            logger.debug(f"  LLM response: {resp_display}")

    def supports_vision(self) -> bool:
        """
        Check if this provider/model supports vision capabilities.
        
        Returns:
            True if vision is supported, False otherwise
        """
        model_info = self.get_model_info()
        return ModelCapability.VISION in model_info.capabilities
    
    @property
    def vision_enabled(self) -> bool:
        """
        Property to check if vision is enabled for this provider/model.
        
        This is the canonical way to check vision capability. Use this instead
        of supports_vision() for cleaner code.
        
        Checks in order:
        1. Explicit override (vision_enabled parameter)
        2. Auto-detected from model capabilities
        
        Returns:
            True if vision is supported, False otherwise
        """
        # Check explicit override first
        if self._vision_enabled_override is not None:
            return self._vision_enabled_override
        
        # Fall back to auto-detected capabilities
        return self.supports_vision()

    async def _execute_with_rate_limit_retry(
        self,
        api_call: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an API call with automatic rate limit retry using exponential backoff.
        
        This is a standardized retry mechanism that all LLM providers should use
        to handle rate limit errors (HTTP 429). It implements:
        - Exponential backoff with jitter
        - Delay parsing from API error messages
        - Configurable max retries and delays
        
        Args:
            api_call: Async callable to execute (e.g., lambda: client.messages.create(...))
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for first retry (default: 1.0)
            max_delay: Maximum delay between retries (default: 60.0)
            **kwargs: Additional arguments (reserved for future use)
            
        Returns:
            The result from the API call
            
        Raises:
            The original exception if all retries are exhausted or it's a non-retryable error
            
        Example:
            >>> result = await self._execute_with_rate_limit_retry(
            ...     lambda: self.client.messages.create(model="...", messages=[...])
            ... )
        """
        import asyncio
        import random
        import re
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await api_call()
                
            except Exception as e:
                # Check if this is a rate limit error (429)
                # Different providers may use different exception types
                is_rate_limit = (
                    "rate" in str(e).lower() and "limit" in str(e).lower()
                ) or (
                    hasattr(e, "status_code") and e.status_code == 429
                ) or (
                    "429" in str(e)
                )
                
                if not is_rate_limit:
                    # Not a rate limit error - don't retry
                    raise
                
                last_exception = e
                
                if attempt >= max_retries:
                    logger.error(
                        f"[{self.__class__.__name__}] Rate limit: max retries ({max_retries}) "
                        f"exhausted. Last error: {e}"
                    )
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                
                # Try to parse delay from error message (e.g., "retry after 5 seconds")
                error_str = str(e).lower()
                retry_match = re.search(r"retry.*?(\d+\.?\d*)\s*s", error_str)
                if retry_match:
                    suggested_delay = float(retry_match.group(1))
                    delay = min(suggested_delay, max_delay)
                
                # Add jitter (±25%)
                jitter = delay * 0.25 * (2 * random.random() - 1)
                delay = max(0.1, delay + jitter)
                
                logger.warning(
                    f"[{self.__class__.__name__}] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Waiting {delay:.2f}s before retry..."
                )
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state in rate limit retry")

    async def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> LLMResponse:
        """Internal implementation that calls the abstract generate method."""
        return await self.generate(prompt, system_prompt, temperature, max_tokens, **kwargs)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def generate_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, ImageInput, List[ImageInput]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM with vision capabilities.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).

        Args:
            prompt: User prompt
            image_data: Image data as bytes, ImageInput, or list of ImageInput for multi-image
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a structured response matching the provided schema.

        Args:
            prompt: User prompt
            schema: JSON schema for the expected response
            system_prompt: System prompt for context
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Structured data matching the schema
        """
        pass

    async def generate_structured_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, ImageInput, List[ImageInput]],
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a structured response with vision input, matching the provided schema.
        
        This combines vision capabilities with structured output for deterministic
        JSON responses. Useful for ReAct agents where consistent output format is critical.

        Args:
            prompt: User prompt
            image_data: Image data as bytes, ImageInput, or list of ImageInput
            schema: JSON schema for the expected response
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Structured data matching the schema

        Note:
            Default implementation falls back to generate_with_vision and parses JSON.
            Providers with native structured output support should override this method.
        """
        # Default: fall back to vision + JSON parsing
        response = await self.generate_with_vision(
            prompt=prompt,
            image_data=image_data,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Try to parse JSON from response
        import json
        content = response.content.strip()
        
        # Extract JSON from markdown code blocks if present
        if content.startswith("```"):
            import re
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        
        # Find JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse structured response: {e}")
            return {"error": "Failed to parse response", "raw_content": response.content}

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response with tool/function calling capabilities.

        Args:
            prompt: User prompt
            tools: List of tool definitions available to the model
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tool_choice: How to select tools ("auto", "none", or specific tool name)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object with potential tool_calls

        Note:
            Default implementation raises NotImplementedError.
            Providers that support tool calling should override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tool calling. "
            "Override generate_with_tools() to add support."
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Response chunks as strings

        Note:
            Default implementation falls back to non-streaming.
            Providers that support streaming should override this method.
        """
        # Default implementation: non-streaming fallback
        response = await self.generate(prompt, system_prompt, temperature, max_tokens, **kwargs)
        yield response.content

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            model: Optional embedding model (uses provider default if not specified)
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors (one per input text)

        Note:
            Default implementation raises NotImplementedError.
            Providers that support embeddings should override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings. "
            "Override generate_embeddings() to add support."
        )

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model's capabilities.

        Returns:
            ModelInfo object with model capabilities and limits

        Note:
            Default implementation returns basic info.
            Providers should override to provide accurate model information.
        """
        return ModelInfo(
            name=self.model,
            provider=self.__class__.__name__.replace("Provider", "").lower(),
            capabilities=[ModelCapability.TEXT_GENERATION],
            context_window=128000,
            max_output_tokens=8192,  # Generous default to avoid truncation
            supports_system_prompt=True,
        )

    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if the current model supports a specific capability.

        Args:
            capability: The capability to check

        Returns:
            True if the capability is supported
        """
        model_info = self.get_model_info()
        return capability in model_info.capabilities

    def get_stats(self) -> Dict[str, Any]:
        """
        Get provider statistics.

        Returns:
            Dictionary with stats including cache, cost, and rate limit info
        """
        stats = {
            "model": self.model,
            "provider": self.__class__.__name__,
            "capabilities": [c.value for c in self.get_model_info().capabilities],
        }

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        if self.cost_tracker:
            stats["cost"] = self.cost_tracker.get_summary()

        if self.rate_limiter:
            stats["rate_limit"] = self.rate_limiter.get_stats()

        return stats
    
    def get_session_usage(self) -> Dict[str, Any]:
        """
        Get accumulated usage statistics for the current session.
        
        This tracks all LLM calls made through this provider instance,
        regardless of whether cost_tracker is configured.
        
        Returns:
            Dictionary with session usage statistics:
            - prompt_tokens: Total input tokens
            - completion_tokens: Total output tokens
            - total_tokens: Total tokens
            - cost_usd: Total cost (0.0 if cost_tracker not configured)
            - calls_count: Number of API calls
            - cached_calls: Number of cached responses
            - model: Model name
        """
        return {
            "prompt_tokens": self._session_prompt_tokens,
            "completion_tokens": self._session_completion_tokens,
            "total_tokens": self._session_total_tokens,
            "cost_usd": self._session_cost,
            "calls_count": self._session_calls,
            "cached_calls": self._session_cached_calls,
            "model": self.model,
            "models_used": [self.model] if self.model else [],
        }
    
    def reset_session_usage(self) -> None:
        """
        Reset session-level usage tracking.
        
        Call this to start fresh tracking for a new operation.
        """
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_total_tokens = 0
        self._session_cost = 0.0
        self._session_calls = 0
        self._session_cached_calls = 0
    
    def record_cached_call(self) -> None:
        """
        Record that a cached response was used.
        
        Call this when returning a cached response to track cache hits.
        """
        self._session_cached_calls += 1
    
    def _track_usage(self, response: LLMResponse) -> None:
        """
        Track usage from an LLMResponse.
        
        Call this after getting a response from generate() or generate_with_vision()
        to update session-level tracking.
        
        Args:
            response: LLMResponse from a generate call
        """
        if response.usage:
            self._session_prompt_tokens += response.usage.get("prompt_tokens", 0)
            self._session_completion_tokens += response.usage.get("completion_tokens", 0)
            self._session_total_tokens += response.usage.get("total_tokens", 0)
            self._session_calls += 1
            
            # Track cost if cost_tracker is available
            if self.cost_tracker:
                record = self.cost_tracker.record_usage(
                    provider=self.__class__.__name__.replace("Provider", "").lower(),
                    model=self.model,
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    cached=response.cached,
                )
                if record:
                    self._session_cost += record.cost
        
        if response.cached:
            self._session_cached_calls += 1

    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if this provider is available and properly configured.

        This method should be overridden by each provider to check:
        - API key configuration (if required)
        - Connectivity to the service (for local providers)
        - Any other provider-specific requirements

        Returns:
            ProviderStatus object with availability information

        Note:
            Default implementation returns INFO status indicating
            the provider hasn't implemented availability checking.
        """
        provider_name = cls.__name__.replace("Provider", "")
        return ProviderStatus.info(
            name=provider_name,
            message="Availability check not implemented",
        )

    def _normalize_images(
        self, image_data: Union[bytes, ImageInput, List[Union[bytes, ImageInput]]]
    ) -> List[ImageInput]:
        """
        Normalize image input to a list of ImageInput objects.

        Args:
            image_data: Image data in various formats (bytes, ImageInput, or list of either)

        Returns:
            List of ImageInput objects
        """
        if isinstance(image_data, bytes):
            return [ImageInput.from_bytes(image_data)]
        elif isinstance(image_data, ImageInput):
            return [image_data]
        elif isinstance(image_data, list):
            # Normalize each item in the list
            result = []
            for item in image_data:
                if isinstance(item, bytes):
                    result.append(ImageInput.from_bytes(item))
                elif isinstance(item, ImageInput):
                    result.append(item)
                else:
                    raise ValueError(f"Unsupported image item type in list: {type(item)}")
            return result
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")

