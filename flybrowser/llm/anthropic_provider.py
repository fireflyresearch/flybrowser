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
Anthropic LLM provider implementation.

This module provides the AnthropicProvider class which implements the BaseLLMProvider
interface for Anthropic's Claude models. It supports both text and vision capabilities
using Anthropic's Messages API.

Supported models include (as of January 2026):
- Claude Sonnet 4.5 (default): Smart model for complex agents and coding
- Claude Haiku 4.5: Fastest model with near-frontier intelligence
- Claude Opus 4.5: Premium model with maximum intelligence
- Claude 3.5 Sonnet: Previous generation high-performance model (legacy)

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images
- Structured output with JSON schemas
- Tool/function calling
- Streaming responses
- Token usage tracking
- Error handling and retries
- Extended thinking support (Claude 4.5 models)
"""

from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from anthropic import AsyncAnthropic

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


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM provider implementation using Messages API.

    This provider supports all Claude models including Claude Sonnet 4.5,
    Claude Haiku 4.5, and Claude Opus 4.5. It provides text generation, vision
    capabilities, extended thinking, and structured output support.

    Attributes:
        client: AsyncAnthropic client instance for API calls
        model: Anthropic model name (e.g., "claude-sonnet-4-5-20250929")
        api_key: Anthropic API key for authentication

    Example:
        >>> provider = AnthropicProvider(
        ...     model="claude-sonnet-4-5-20250929",
        ...     api_key="sk-ant-..."
        ... )
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(
        self, model: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None, **kwargs: Any
    ) -> None:
        """
        Initialize Anthropic provider with API credentials.

        Args:
            model: Anthropic model name. Supported models (as of January 2026):
                - "claude-sonnet-4-5-20250929": Smart model for agents (default)
                - "claude-haiku-4-5-20251001": Fastest model
                - "claude-opus-4-5-20251101": Premium model
                - "claude-3-5-sonnet-20241022": Previous generation (legacy)
            api_key: Anthropic API key (starts with "sk-ant-"). Required for API access.
                Can also be set via ANTHROPIC_API_KEY environment variable.
            **kwargs: Additional configuration passed to BaseLLMProvider

        Example:
            >>> provider = AnthropicProvider(
            ...     model="claude-sonnet-4-5-20250929",
            ...     api_key="sk-ant-YOUR_KEY_HERE"
            ... )
        """
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
        
        # Track initialization state
        self._capabilities_detected: bool = False
        self._model_capabilities_cache: Optional[List[ModelCapability]] = None

    async def initialize(self) -> None:
        """
        Initialize the provider.
        
        Vision support is determined by model name patterns and can be overridden
        with the vision_enabled parameter.
        """
        if not self._capabilities_detected:
            self._model_capabilities_cache = self._get_basic_capabilities()
            self._capabilities_detected = True
            
            has_vision = ModelCapability.VISION in self._model_capabilities_cache
            if has_vision:
                logger.info(f"Vision: [INFO] ENABLED for model {self.model}")
            else:
                logger.debug(f"Vision: [INFO] DISABLED for model {self.model}")

    def _get_basic_capabilities(self) -> List[ModelCapability]:
        """
        Get model capabilities based on model name patterns.
        
        Vision support is determined by model name (claude-3, claude-4) and can be
        overridden with vision_enabled parameter.
        
        Returns:
            List of ModelCapability enum values
        """
        model_lower = self.model.lower()
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.STREAMING,
            ModelCapability.TOOL_CALLING,
        ]
        
        # All Claude 3+ models have vision
        # Check override first
        if self._vision_enabled_override is not None:
            if self._vision_enabled_override:
                capabilities.append(ModelCapability.VISION)
        elif "claude-3" in model_lower or "claude-4" in model_lower:
            capabilities.append(ModelCapability.VISION)
        
        # Claude 4.5 models have extended thinking
        if "claude-4" in model_lower:
            capabilities.append(ModelCapability.EXTENDED_THINKING)
        
        return capabilities


    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if Anthropic provider is available.

        Checks for ANTHROPIC_API_KEY environment variable.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return ProviderStatus.ok(
                name="Anthropic",
                message="API key configured",
                api_key_env_var="ANTHROPIC_API_KEY",
                requires_api_key=True,
            )
        return ProviderStatus.info(
            name="Anthropic",
            message="API key not set (optional)",
            api_key_env_var="ANTHROPIC_API_KEY",
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
        Generate a text response using Anthropic's Messages API.

        Args:
            prompt: User prompt/message to send to the model
            system_prompt: Optional system prompt to set model behavior.
                Example: "You are a helpful assistant that extracts data from web pages."
            temperature: Sampling temperature (0.0-1.0). Higher values make output
                more random, lower values more deterministic. Default: 0.7
            max_tokens: Maximum tokens to generate. If None, uses 4096 as default.
                Claude models support up to 4096 output tokens.
            **kwargs: Additional parameters for Anthropic API:
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - stop_sequences: List of sequences that stop generation

        Returns:
            LLMResponse containing:
            - content: Generated text
            - model: Model used for generation
            - usage: Token usage statistics (input_tokens, output_tokens, total_tokens)

        Raises:
            LLMProviderError: If API call fails

        Example:
            >>> response = await provider.generate(
            ...     prompt="What is the capital of France?",
            ...     system_prompt="You are a geography expert.",
            ...     temperature=0.3,
            ...     max_tokens=100
            ... )
            >>> print(response.content)
            'The capital of France is Paris.'
        """
        # Ensure capabilities are detected via API
        await self.initialize()

        try:
            # Log LLM request (with prompts at level 2)
            start_time = self._log_llm_request(
                "GENERATE", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # Call Anthropic Messages API
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 8192,  # Generous default to avoid truncation
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Log response (with content at level 2)
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            response_content = response.content[0].text if response.content else ""
            self._log_llm_response("GENERATE", start_time, total_tokens, response_content)

            # Extract response and usage information
            llm_response = LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": total_tokens,
                },
            )

            # Track usage at session level
            self._track_usage(llm_response)

            return llm_response
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise LLMProviderError(f"Anthropic generation failed: {e}") from e

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
        Generate a response with vision using Anthropic.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).

        If the model does not support vision, falls back to text-only generation.
        """
        # Check if model supports vision
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Vision-capable Claude models: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet. "
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
                "VISION", self.model, f"{len(images)} images",
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # Build content with images first, then text (Anthropic format)
            content: List[Dict[str, Any]] = []

            for img in images:
                if img.source_type == "url":
                    # Anthropic supports URL sources
                    content.append({
                        "type": "image",
                        "source": {"type": "url", "url": img.data},
                    })
                else:
                    # Convert bytes to base64 if needed
                    if img.source_type == "bytes":
                        img_base64 = base64.b64encode(img.data).decode("utf-8")
                    else:
                        img_base64 = img.data

                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": img_base64,
                        },
                    })

            content.append({"type": "text", "text": prompt})

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 8192,  # Generous default to avoid truncation
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": content}],
                **kwargs,
            )

            # Log response (with content at level 2)
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            response_content = response.content[0].text if response.content else ""
            self._log_llm_response("VISION", start_time, total_tokens, response_content)

            llm_response = LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": total_tokens,
                },
                finish_reason=response.stop_reason,
            )

            # Track usage at session level
            self._track_usage(llm_response)

            return llm_response
        except Exception as e:
            logger.error(f"Anthropic vision generation error: {e}")
            raise LLMProviderError(f"Anthropic vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output using Anthropic."""
        try:
            # Add JSON schema instruction to system prompt
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, just pure JSON matching this schema: {json.dumps(schema)}"
            )
            full_system_prompt = (system_prompt or "") + schema_instruction

            # Log structured request (with prompts at level 2)
            start_time = self._log_llm_request(
                "STRUCTURED", self.model,
                prompt=prompt, system_prompt=full_system_prompt
            )

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=8192,  # Generous default to avoid truncation
                temperature=temperature,
                system=full_system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            content = response.content[0].text if response.content else "{}"
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            self._log_llm_response("STRUCTURED", start_time, total_tokens, response_content=content)
            
            # Track usage at session level
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            # Extract JSON from response (handle markdown code blocks)
            content = content.strip()
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
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Anthropic structured response not valid JSON: {e}")
            raise LLMProviderError(f"Anthropic structured response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Anthropic structured generation error: {e}")
            raise LLMProviderError(f"Anthropic structured generation failed: {e}") from e

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
        Generate structured output with vision using Anthropic.
        
        Combines vision capabilities with JSON output instructions for deterministic structured output.
        
        Args:
            prompt: User prompt (should mention JSON format for best results)
            image_data: Image data as bytes, ImageInput, or list of ImageInput
            schema: JSON schema for the expected response
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response as dictionary
        """
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
            
            # Add JSON schema instruction to system prompt
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, just pure JSON matching this schema: {json.dumps(schema)}"
            )
            full_system_prompt = (system_prompt or "") + schema_instruction
            
            # Log request
            start_time = self._log_llm_request(
                "STRUCTURED_VISION", self.model,
                f"{len(images)} images",
                prompt=prompt, system_prompt=full_system_prompt
            )
            
            # Build content with images first, then text (Anthropic format)
            content: List[Dict[str, Any]] = []
            
            for img in images:
                if img.source_type == "url":
                    content.append({
                        "type": "image",
                        "source": {"type": "url", "url": img.data},
                    })
                else:
                    # Convert bytes to base64 if needed
                    if img.source_type == "bytes":
                        img_base64 = base64.b64encode(img.data).decode("utf-8")
                    else:
                        img_base64 = img.data
                    
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": img_base64,
                        },
                    })
            
            content.append({"type": "text", "text": prompt})
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 8192,  # Generous default to avoid truncation
                temperature=temperature,
                system=full_system_prompt,
                messages=[{"role": "user", "content": content}],
                **kwargs,
            )
            
            # Log response
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            response_content = response.content[0].text if response.content else "{}"
            self._log_llm_response("STRUCTURED_VISION", start_time, total_tokens, response_content)
            
            # Track usage at session level
            llm_response = LLMResponse(
                content=response_content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            # Extract JSON from response
            result_content = response_content.strip()
            if result_content.startswith("```"):
                import re
                json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", result_content, re.DOTALL)
                if json_match:
                    result_content = json_match.group(1).strip()
            
            # Find JSON object
            start = result_content.find("{")
            end = result_content.rfind("}")
            if start != -1 and end != -1 and end > start:
                result_content = result_content[start:end + 1]
            
            return json.loads(result_content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Anthropic structured vision response not valid JSON: {e}")
            raise LLMProviderError(f"Anthropic structured vision response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Anthropic structured vision generation error: {e}")
            raise LLMProviderError(f"Anthropic structured vision generation failed: {e}") from e

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
        try:
            # Convert ToolDefinition to Anthropic format
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required,
                    },
                }
                for tool in tools
            ]

            api_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens or 8192,  # Generous default to avoid truncation
                "temperature": temperature,
                "system": system_prompt or "",
                "messages": [{"role": "user", "content": prompt}],
                "tools": anthropic_tools,
                **kwargs,
            }

            if tool_choice:
                api_kwargs["tool_choice"] = {"type": tool_choice}

            response = await self.client.messages.create(**api_kwargs)

            # Extract tool calls if present
            tool_calls = None
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                tool_calls=tool_calls,
                finish_reason=response.stop_reason,
            )
        except Exception as e:
            logger.error(f"Anthropic tool calling error: {e}")
            raise LLMProviderError(f"Anthropic tool calling failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Anthropic."""
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or 8192,  # Generous default to avoid truncation
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMProviderError(f"Anthropic streaming failed: {e}") from e

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Anthropic model.
        
        Uses simple hardcoded values based on model name patterns.
        """
        # Use cached capabilities if available
        if self._model_capabilities_cache is not None:
            capabilities = self._model_capabilities_cache
        else:
            capabilities = self._get_basic_capabilities()
        
        # Simple hardcoded pricing and context windows
        model_lower = self.model.lower()
        
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
        
        # All Claude 3+ models have 200K context
        context_window = 200000
        max_output = 8192
        
        return ModelInfo(
            name=self.model,
            provider="anthropic",
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=input_cost,
            cost_per_1k_output_tokens=output_cost,
        )

