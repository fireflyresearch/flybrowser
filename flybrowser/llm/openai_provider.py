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
OpenAI LLM provider implementation.

This module provides the OpenAIProvider class which implements the BaseLLMProvider
interface for OpenAI's GPT models. It supports both text and vision capabilities
using OpenAI's Chat Completions API.

Supported models include (as of January 2026):
- GPT-5.2 (default): Latest flagship model for coding and agentic tasks
- GPT-5 mini: Faster, cost-efficient version of GPT-5
- GPT-5 nano: Fastest, most cost-efficient version of GPT-5
- GPT-4.1: Smartest non-reasoning model
- GPT-4o: Fast, intelligent, flexible GPT model (legacy)

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images
- Structured output with JSON schemas
- Tool/function calling
- Streaming responses
- Embeddings generation
- Token usage tracking
- Error handling and retries
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI, RateLimitError

import os

from flybrowser.exceptions import LLMProviderError
from flybrowser.llm.base import (
    BaseLLMProvider,
    ImageInput,
    LLMResponse,
    ModelCapability,
    ModelInfo,
    ToolCall,
    ToolDefinition,
)
from flybrowser.llm.provider_status import ProviderStatus
from flybrowser.utils.logger import logger


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation using Chat Completions API.

    This provider supports all OpenAI chat models including GPT-5.2, GPT-5 mini,
    GPT-4.1, and GPT-4o. It provides text generation, vision capabilities, and
    structured output support.

    Attributes:
        client: AsyncOpenAI client instance for API calls
        model: OpenAI model name (e.g., "gpt-5.2", "gpt-5-mini", "gpt-4.1")
        api_key: OpenAI API key for authentication

    Example:
        >>> provider = OpenAIProvider(model="gpt-5.2", api_key="sk-...")
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize OpenAI provider with API credentials.

        Args:
            model: OpenAI model name. Supported models (as of January 2026):
                - "gpt-5.2": Latest flagship model (default)
                - "gpt-5-mini": Faster, cost-efficient version
                - "gpt-5-nano": Fastest, most cost-efficient
                - "gpt-4.1": Smartest non-reasoning model
                - "gpt-4o": Fast, intelligent model (legacy)
            api_key: OpenAI API key (starts with "sk-"). Required for API access.
                Can also be set via OPENAI_API_KEY environment variable.
            **kwargs: Additional configuration passed to BaseLLMProvider

        Example:
            >>> provider = OpenAIProvider(
            ...     model="gpt-4o",
            ...     api_key="sk-proj-YOUR_KEY_HERE"
            ... )
        """
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Cache for model capabilities (populated on first use)
        self._model_capabilities_cache: Optional[List[ModelCapability]] = None
        self._model_info_cache: Optional[ModelInfo] = None
        self._capabilities_detected: bool = False
        
        # Track if this is a reasoning model (o1, o3)
        self._is_reasoning_model = self._detect_reasoning_model()
    
    def _detect_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o1, o3)."""
        model_lower = self.model.lower()
        return model_lower.startswith("o1") or model_lower.startswith("o3")
    
    async def initialize(self) -> None:
        """
        Initialize the provider.

        Vision support is determined by model name patterns and can be overridden
        with the vision_enabled parameter.

        This is called automatically on first use, but can be called explicitly
        for early initialization.
        """
        if not self._capabilities_detected:
            self._model_capabilities_cache = self._get_basic_capabilities()
            self._capabilities_detected = True

            has_vision = ModelCapability.VISION in self._model_capabilities_cache
            if has_vision:
                logger.info(f"Vision: [INFO] ENABLED for model {self.model}")
            else:
                logger.debug(f"Vision: [INFO] DISABLED for model {self.model}")

    def _add_max_tokens_param(self, api_kwargs: Dict[str, Any], max_tokens: int) -> None:
        """
        Add the appropriate max tokens parameter to API kwargs based on model type.
        
        OpenAI's reasoning models (o1, o3) require 'max_completion_tokens' parameter
        instead of 'max_tokens'.
        
        Args:
            api_kwargs: Dictionary of API kwargs to modify in place
            max_tokens: The max tokens value to set
        """
        if self._is_reasoning_model:
            api_kwargs["max_completion_tokens"] = max_tokens
        else:
            api_kwargs["max_tokens"] = max_tokens
    
    def _build_api_kwargs(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build API kwargs with proper parameter handling for different model types.
        
        This method handles model-specific parameter restrictions:
        - Reasoning models (o1, o3, GPT-5) don't support custom temperature (only 1)
        - Reasoning models use 'max_completion_tokens' instead of 'max_tokens'
        - Filters out any invalid OpenAI API parameters
        
        Args:
            messages: List of message dicts for the API call
            temperature: Desired temperature (will be ignored for reasoning models)
            max_tokens: Max tokens to generate (optional)
            **kwargs: Additional API parameters
            
        Returns:
            Dictionary of API kwargs ready for the OpenAI client
        """
        # Filter kwargs to only include valid OpenAI API parameters
        # This prevents invalid params (like 'images') from leaking through
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in self._VALID_OPENAI_API_PARAMS
        }
        
        # Log if we filtered out any invalid params (debug level)
        invalid_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if invalid_params:
            logger.debug(f"[OpenAI] Filtered out invalid API params: {invalid_params}")
        
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            **filtered_kwargs,
        }
        
        # Only add temperature if the model supports it (reasoning models don't)
        if not self._is_reasoning_model:
            api_kwargs["temperature"] = temperature
        
        # Add max tokens with correct parameter name if provided
        if max_tokens is not None:
            self._add_max_tokens_param(api_kwargs, max_tokens)
        
        return api_kwargs
    
    def _is_reasoning_model_error(self, error: Exception) -> bool:
        """
        Check if an error indicates the model is a reasoning model.
        
        Reasoning models reject temperature != 1 and require max_completion_tokens.
        
        Args:
            error: The exception from the API call
            
        Returns:
            True if error indicates reasoning model restrictions
        """
        error_str = str(error).lower()
        
        # Check for temperature rejection
        if "temperature" in error_str and ("unsupported" in error_str or "not supported" in error_str):
            return True
        
        # Check for max_tokens rejection requiring max_completion_tokens
        if "max_tokens" in error_str and "max_completion_tokens" in error_str:
            return True
        
        return False
    
    def _mark_as_reasoning_model(self) -> None:
        """
        Mark the current model as a reasoning model.
        
        This updates the local flag so subsequent calls use the correct parameters.
        """
        self._is_reasoning_model = True
        logger.info(
            f"[OpenAI] Detected {self.model} as reasoning model via API response. "
            f"Will use max_completion_tokens and skip temperature parameter."
        )
    
    # Valid OpenAI Chat Completions API parameters (for filtering during retry)
    _VALID_OPENAI_API_PARAMS = {
        "model", "messages", "temperature", "max_tokens", "max_completion_tokens",
        "top_p", "n", "stream", "stream_options", "stop", "presence_penalty",
        "frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "user",
        "response_format", "seed", "tools", "tool_choice", "parallel_tool_calls",
        "functions", "function_call",  # Legacy function calling
        "reasoning_effort",  # For reasoning models
    }
    
    async def _execute_with_reasoning_detection(
        self,
        api_call_func,
        api_kwargs: Dict[str, Any],
        original_temperature: float,
        original_max_tokens: Optional[int],
    ):
        """
        Execute an API call with automatic reasoning model detection and rate limit retry.
        
        This method handles:
        1. Rate limit errors (429) with exponential backoff retry
        2. Reasoning model detection and parameter correction
        
        Rate limit retry configuration:
        - Max retries: 3 attempts
        - Initial delay: 1 second (or from API response)
        - Exponential backoff with jitter
        - Max delay: 60 seconds
        
        Args:
            api_call_func: Async function to call (e.g., self.client.chat.completions.create)
            api_kwargs: The API kwargs to use
            original_temperature: Original temperature value (for rebuilding)
            original_max_tokens: Original max_tokens value (for rebuilding)
            
        Returns:
            The API response
            
        Raises:
            The original exception if all retries are exhausted or it's a non-retryable error
        """
        max_retries = 3
        base_delay = 1.0
        max_delay = 60.0
        
        last_exception = None
        current_kwargs = api_kwargs
        
        for attempt in range(max_retries + 1):
            try:
                return await api_call_func(**current_kwargs)
                
            except RateLimitError as e:
                last_exception = e
                
                if attempt >= max_retries:
                    logger.error(
                        f"[OpenAI] Rate limit: max retries ({max_retries}) exhausted. "
                        f"Last error: {e}"
                    )
                    raise
                
                # Parse retry delay from error message if available
                # Error format: "Please try again in 734ms"
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                error_msg = str(e)
                
                if "try again in" in error_msg.lower():
                    try:
                        # Extract delay from message
                        import re
                        match = re.search(r'try again in (\d+(?:\.\d+)?)(ms|s)', error_msg.lower())
                        if match:
                            value = float(match.group(1))
                            unit = match.group(2)
                            if unit == 'ms':
                                delay = value / 1000.0
                            else:
                                delay = value
                            # Add some buffer
                            delay = delay + 0.5
                    except Exception:
                        pass  # Use calculated delay
                
                # Add jitter (±25%)
                delay = delay * (0.75 + random.random() * 0.5)
                delay = min(delay, max_delay)
                
                logger.warning(
                    f"[OpenAI] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Waiting {delay:.2f}s before retry..."
                )
                await asyncio.sleep(delay)
                continue
                
            except Exception as e:
                if self._is_reasoning_model_error(e):
                    # Detected as reasoning model - update registry and retry
                    self._mark_as_reasoning_model()
                    
                    # Rebuild kwargs with corrected parameters
                    messages = current_kwargs.get("messages", [])
                    
                    # Filter to only valid OpenAI API parameters
                    excluded_params = {"temperature", "max_tokens", "max_completion_tokens", "messages", "model"}
                    retry_kwargs = {
                        k: v for k, v in current_kwargs.items() 
                        if k in self._VALID_OPENAI_API_PARAMS and k not in excluded_params
                    }
                    
                    # Rebuild with correct parameters
                    current_kwargs = self._build_api_kwargs(
                        messages=messages,
                        temperature=original_temperature,
                        max_tokens=original_max_tokens,
                        **retry_kwargs,
                    )
                    
                    logger.debug(f"[OpenAI] Retrying request with corrected parameters for reasoning model")
                    # Continue to retry with corrected kwargs
                    continue
                else:
                    # Not a retryable error - re-raise immediately
                    raise
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if OpenAI provider is available.

        Checks for OPENAI_API_KEY environment variable.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return ProviderStatus.ok(
                name="OpenAI",
                message="API key configured",
                api_key_env_var="OPENAI_API_KEY",
                requires_api_key=True,
            )
        return ProviderStatus.info(
            name="OpenAI",
            message="API key not set (optional)",
            api_key_env_var="OPENAI_API_KEY",
            requires_api_key=True,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a text response using OpenAI's Chat Completions API.

        Args:
            prompt: User prompt/message to send to the model
            system_prompt: Optional system prompt to set model behavior.
                Example: "You are a helpful assistant that extracts data from web pages."
            temperature: Sampling temperature (0.0-2.0). Higher values make output
                more random, lower values more deterministic. Default: 0.7
            max_tokens: Maximum tokens to generate. If None, uses model default.
            **kwargs: Additional parameters for OpenAI API:
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Penalize frequent tokens
                - presence_penalty: Penalize tokens based on presence
                - stop: Stop sequences

        Returns:
            LLMResponse containing:
            - content: Generated text
            - model: Model used for generation
            - usage: Token usage statistics

        Raises:
            LLMProviderError: If API call fails

        Example:
            >>> response = await provider.generate(
            ...     prompt="What is the capital of France?",
            ...     system_prompt="You are a geography expert.",
            ...     temperature=0.3
            ... )
            >>> print(response.content)
            'The capital of France is Paris.'
        """
        # Ensure capabilities are detected
        await self.initialize()
        
        try:
            # Build messages array with optional system prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Log LLM request (with prompts at level 2)
            start_time = self._log_llm_request(
                "GENERATE", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # Build API kwargs with proper parameter handling for model type
            api_kwargs = self._build_api_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Call OpenAI Chat Completions API with automatic reasoning model detection
            response = await self._execute_with_reasoning_detection(
                self.client.chat.completions.create,
                api_kwargs,
                original_temperature=temperature,
                original_max_tokens=max_tokens,
            )

            # Log response (with content at level 2)
            total_tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content or ""
            self._log_llm_response("GENERATE", start_time, total_tokens, response_content)

            # Extract response and usage information
            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
            
            # Track usage at session level
            self._track_usage(llm_response)
            
            return llm_response
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise LLMProviderError(f"OpenAI generation failed: {e}") from e

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
        Generate a response with vision using OpenAI.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).
        
        If the model does not support vision, falls back to text-only generation.
        """
        # Ensure capabilities are detected
        await self.initialize()
        
        # Check if model supports vision
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Vision-capable models: gpt-4o, gpt-4-turbo, gpt-4-vision. "
                f"Falling back to text-only generation (image will be ignored)."
            )
            # Fall back to text-only generation
            return await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        
        try:
            # Normalize images to list of ImageInput
            images = self._normalize_images(image_data)

            # Log LLM vision request (with prompts at level 2)
            start_time = self._log_llm_request(
                "VISION", self.model, f"{len(images) if isinstance(images, list) else 1} images",
                prompt=prompt, system_prompt=system_prompt or ""
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Build content with text and images
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

            for img in images:
                if img.source_type == "url":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img.data, "detail": img.detail},
                    })
                else:
                    # Convert bytes to base64 if needed
                    if img.source_type == "bytes":
                        img_base64 = base64.b64encode(img.data).decode("utf-8")
                    else:
                        img_base64 = img.data

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.media_type};base64,{img_base64}",
                            "detail": img.detail,
                        },
                    })

            messages.append({"role": "user", "content": content})

            # Build API kwargs with proper parameter handling for model type
            api_kwargs = self._build_api_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 8192,  # Generous default to avoid truncation
                **kwargs,
            )

            # Call with automatic reasoning model detection
            response = await self._execute_with_reasoning_detection(
                self.client.chat.completions.create,
                api_kwargs,
                original_temperature=temperature,
                original_max_tokens=max_tokens or 8192,
            )

            # Log response (with content at level 2)
            total_tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content or ""
            self._log_llm_response("VISION", start_time, total_tokens, response_content)

            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
            )
            
            # Track usage at session level
            self._track_usage(llm_response)
            
            return llm_response
        except Exception as e:
            logger.error(f"OpenAI vision generation error: {e}")
            raise LLMProviderError(f"OpenAI vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output using OpenAI."""
        # Ensure capabilities are detected (including reasoning model probing)
        await self.initialize()
        
        try:
            # Log structured request (with prompts at level 2)
            start_time = self._log_llm_request(
                "STRUCTURED", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # OpenAI requires 'json' in messages when using response_format={"type": "json_object"}
            # Add safety check and inject if missing from both prompt and system_prompt
            combined_text = (prompt + (system_prompt or "")).lower()
            if "json" not in combined_text:
                # Inject JSON instruction into prompt to satisfy OpenAI API requirement
                prompt = prompt + "\n\nRespond with valid JSON."
                logger.debug("[OpenAI] Injected 'json' into prompt to satisfy API requirement")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            # Build API kwargs with proper parameter handling for model type
            # Use generous default max_tokens for structured output
            effective_max_tokens = kwargs.pop("max_tokens", None) or 8192
            api_kwargs = self._build_api_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                response_format={"type": "json_object"},
                **kwargs,
            )

            # Call with automatic reasoning model detection
            response = await self._execute_with_reasoning_detection(
                self.client.chat.completions.create,
                api_kwargs,
                original_temperature=temperature,
                original_max_tokens=effective_max_tokens,
            )

            content = response.choices[0].message.content or "{}"
            total_tokens = response.usage.total_tokens if response.usage else 0
            self._log_llm_response("STRUCTURED", start_time, total_tokens, response_content=content)
            
            # Track usage at session level
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"OpenAI structured generation error: {e}")
            raise LLMProviderError(f"OpenAI structured generation failed: {e}") from e

    async def generate_structured_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, "ImageInput", List["ImageInput"]],
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output with vision using OpenAI.
        
        Combines vision capabilities with JSON mode for deterministic structured output.
        Uses response_format={"type": "json_object"} to enforce JSON output.
        
        Args:
            prompt: User prompt (should mention JSON format for best results)
            image_data: Image data as bytes, ImageInput, or list of ImageInput
            schema: JSON schema (used for documentation, not enforcement by API)
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response as dictionary
        """
        # Ensure capabilities are detected
        await self.initialize()
        
        # Check if model supports vision
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Falling back to text-only structured generation."
            )
            return await self.generate_structured(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
        
        try:
            # Normalize images to list of ImageInput
            images = self._normalize_images(image_data)
            
            # Calculate total image size for logging
            total_image_bytes = 0
            for img in images:
                if img.source_type == "bytes":
                    total_image_bytes += len(img.data)
                elif img.source_type == "base64":
                    total_image_bytes += len(img.data) * 3 // 4  # Approx decoded size
            
            logger.info(
                f"[VISION] Preparing {len(images)} image(s) for upload "
                f"(~{total_image_bytes // 1024}KB total)"
            )
            
            # OpenAI requires 'json' in messages when using response_format={"type": "json_object"}
            # Add safety check and inject if missing from both prompt and system_prompt
            combined_text = (prompt + (system_prompt or "")).lower()
            if "json" not in combined_text:
                # Inject JSON instruction into prompt to satisfy OpenAI API requirement
                prompt = prompt + "\n\nRespond with valid JSON."
                logger.debug("[OpenAI] Injected 'json' into vision prompt to satisfy API requirement")
            
            # Log request
            start_time = self._log_llm_request(
                "STRUCTURED_VISION", self.model,
                f"{len(images)} images (~{total_image_bytes // 1024}KB)",
                prompt=prompt, system_prompt=system_prompt or ""
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Build content with text and images
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            
            for idx, img in enumerate(images):
                if img.source_type == "url":
                    logger.debug(f"[VISION] Image {idx+1}: URL source, detail={img.detail}")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img.data, "detail": img.detail},
                    })
                else:
                    # Convert bytes to base64 if needed
                    if img.source_type == "bytes":
                        img_base64 = base64.b64encode(img.data).decode("utf-8")
                        logger.debug(
                            f"[VISION] Image {idx+1}: {len(img.data)//1024}KB bytes -> "
                            f"{len(img_base64)//1024}KB base64, media={img.media_type}, detail={img.detail}"
                        )
                    else:
                        img_base64 = img.data
                        logger.debug(
                            f"[VISION] Image {idx+1}: base64 string, "
                            f"media={img.media_type}, detail={img.detail}"
                        )
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.media_type};base64,{img_base64}",
                            "detail": img.detail,
                        },
                    })
            
            messages.append({"role": "user", "content": content})
            logger.info(f"[VISION] Sending request with {len(content)-1} image(s) to {self.model}")
            
            # Build API kwargs with JSON mode enabled
            api_kwargs = self._build_api_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 8192,  # Generous default to avoid truncation
                response_format={"type": "json_object"},
                **kwargs,
            )
            
            # Call with automatic reasoning model detection
            response = await self._execute_with_reasoning_detection(
                self.client.chat.completions.create,
                api_kwargs,
                original_temperature=temperature,
                original_max_tokens=max_tokens or 8192,
            )
            
            # Log response
            total_tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content or "{}"
            self._log_llm_response("STRUCTURED_VISION", start_time, total_tokens, response_content)
            
            # Track usage at session level
            llm_response = LLMResponse(
                content=response_content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            # Parse and return JSON
            return json.loads(response_content)
            
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI structured vision response not valid JSON: {e}")
            raise LLMProviderError(f"OpenAI structured vision response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"OpenAI structured vision generation error: {e}")
            raise LLMProviderError(f"OpenAI structured vision generation failed: {e}") from e

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
        """Generate a response with tool/function calling capabilities."""
        # Ensure capabilities are detected (including reasoning model probing)
        await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Convert ToolDefinition to OpenAI format
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": tool.required,
                        },
                    },
                }
                for tool in tools
            ]

            # Build API kwargs with proper parameter handling for model type
            api_kwargs = self._build_api_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=openai_tools,
                **kwargs,
            )

            if tool_choice:
                api_kwargs["tool_choice"] = tool_choice

            # Call with automatic reasoning model detection
            response = await self._execute_with_reasoning_detection(
                self.client.chat.completions.create,
                api_kwargs,
                original_temperature=temperature,
                original_max_tokens=max_tokens,
            )

            # Extract tool calls if present
            tool_calls = None
            if response.choices[0].message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in response.choices[0].message.tool_calls
                ]

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI tool calling error: {e}")
            raise LLMProviderError(f"OpenAI tool calling failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI."""
        # Ensure capabilities are detected (including reasoning model probing)
        await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Build API kwargs with proper parameter handling for model type
            api_kwargs = self._build_api_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            # Call with automatic reasoning model detection
            stream = await self._execute_with_reasoning_detection(
                self.client.chat.completions.create,
                api_kwargs,
                original_temperature=temperature,
                original_max_tokens=max_tokens,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMProviderError(f"OpenAI streaming failed: {e}") from e

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            response = await self.client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=texts,
                **kwargs,
            )

            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            raise LLMProviderError(f"OpenAI embeddings failed: {e}") from e

    def _get_basic_capabilities(self) -> List[ModelCapability]:
        """
        Get model capabilities based on model name patterns.

        Vision support is determined by model name (gpt-4o, gpt-5, etc.) and can be
        overridden with vision_enabled parameter.

        Returns:
            List of ModelCapability enum values
        """
        model_lower = self.model.lower()
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.STREAMING]

        # Most GPT models support tool calling and structured output
        if "gpt-4" in model_lower or "gpt-5" in model_lower or "gpt-3.5" in model_lower:
            capabilities.append(ModelCapability.TOOL_CALLING)
            capabilities.append(ModelCapability.STRUCTURED_OUTPUT)
        
        # Vision models: gpt-4o, gpt-4-turbo, gpt-5, gpt-4-vision
        # Check override first
        if self._vision_enabled_override is not None:
            if self._vision_enabled_override:
                capabilities.append(ModelCapability.VISION)
        elif any(x in model_lower for x in ["gpt-4o", "gpt-5", "gpt-4-turbo", "gpt-4-vision"]):
            capabilities.append(ModelCapability.VISION)
        
        # Reasoning models
        if self._is_reasoning_model:
            capabilities.append(ModelCapability.EXTENDED_THINKING)

        return capabilities
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current OpenAI model.
        
        Uses simple hardcoded values based on model name patterns.
        """
        # Use cached capabilities if available
        if self._model_capabilities_cache is not None:
            capabilities = self._model_capabilities_cache
        else:
            capabilities = self._get_basic_capabilities()

        # Simple hardcoded context windows and pricing
        model_lower = self.model.lower()
        
        if "gpt-5" in model_lower:
            context_window = 256000
            max_output = 16384
            input_cost = 0.010
            output_cost = 0.030
        elif "gpt-4o" in model_lower:
            context_window = 128000
            max_output = 16384
            input_cost = 0.0025
            output_cost = 0.010
        elif "gpt-4" in model_lower:
            context_window = 128000
            max_output = 8192
            input_cost = 0.030
            output_cost = 0.060
        elif "o1" in model_lower or "o3" in model_lower:
            context_window = 200000
            max_output = 32768
            input_cost = 0.015
            output_cost = 0.060
        elif "gpt-3.5" in model_lower:
            context_window = 16385
            max_output = 4096
            input_cost = 0.0005
            output_cost = 0.0015
        else:
            context_window = 128000
            max_output = 4096
            input_cost = 0.0
            output_cost = 0.0
        
        return ModelInfo(
            name=self.model,
            provider="openai",
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=input_cost,
            cost_per_1k_output_tokens=output_cost,
        )

