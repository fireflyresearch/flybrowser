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
Google Gemini LLM provider implementation.

This module provides the GeminiProvider class which implements the BaseLLMProvider
interface for Google's Gemini models. It supports both text and vision capabilities
using Google's Generative AI API.

Supported models include (as of January 2026):
- Gemini 2.0 Flash (default): Fast, versatile model with multimodal capabilities
- Gemini 2.0 Pro: Most capable model for complex tasks
- Gemini 1.5 Pro: Previous generation high-performance model
- Gemini 1.5 Flash: Fast, efficient model

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images
- Structured output with JSON schemas
- Tool/function calling
- Streaming responses
- Token usage tracking
- Error handling and retries
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import os

import aiohttp

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


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider implementation using Generative AI API.

    This provider supports all Gemini models including Gemini 2.0 Flash,
    Gemini 2.0 Pro, and Gemini 1.5 series. It provides text generation, vision
    capabilities, tool calling, and structured output support.

    Attributes:
        model: Gemini model name (e.g., "gemini-2.0-flash", "gemini-2.0-pro")
        api_key: Google AI API key for authentication
        base_url: API base URL

    Example:
        >>> provider = GeminiProvider(
        ...     model="gemini-2.0-flash",
        ...     api_key="AIza..."
        ... )
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Gemini provider with API credentials.

        Args:
            model: Gemini model name. Supported models (as of January 2026):
                - "gemini-2.0-flash": Fast, versatile model (default)
                - "gemini-2.0-pro": Most capable model
                - "gemini-1.5-pro": Previous generation pro model
                - "gemini-1.5-flash": Fast, efficient model
            api_key: Google AI API key. Required for API access.
                Can also be set via GOOGLE_API_KEY environment variable.
            base_url: API base URL. Default: Google's Generative AI API
            **kwargs: Additional configuration passed to BaseLLMProvider
        """
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url.rstrip("/")
        
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
        
        Vision support is determined by model name (gemini-1.5+, gemini-2.0+) and can be
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
        
        # All Gemini 1.5+ and 2.0+ models have vision
        # Check override first
        if self._vision_enabled_override is not None:
            if self._vision_enabled_override:
                capabilities.append(ModelCapability.VISION)
        elif "gemini-1.5" in model_lower or "gemini-2.0" in model_lower or "gemini-2" in model_lower:
            capabilities.append(ModelCapability.VISION)
        
        return capabilities


    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if Gemini provider is available.

        Checks for GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
        """
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            return ProviderStatus.ok(
                name="Gemini",
                message="API key configured",
                api_key_env_var="GOOGLE_API_KEY",
                requires_api_key=True,
            )
        return ProviderStatus.info(
            name="Gemini",
            message="API key not set (optional)",
            api_key_env_var="GOOGLE_API_KEY",
            requires_api_key=True,
        )

    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Gemini API with rate limit retry.
        
        Handles rate limit errors (429) with exponential backoff and jitter.
        """
        url = f"{self.base_url}/models/{self.model}:{endpoint}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        max_retries = 3
        base_delay = 1.0
        max_delay = 60.0
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 429:
                            # Rate limit - will retry
                            error_text = await response.text()
                            raise LLMProviderError(
                                f"Rate limit: {response.status} - {error_text}"
                            )
                        elif response.status != 200:
                            error_text = await response.text()
                            raise LLMProviderError(
                                f"Gemini API request failed: {response.status} - {error_text}"
                            )
                        return await response.json()
                        
            except LLMProviderError as e:
                error_msg = str(e)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(
                            f"[Gemini] Rate limit: max retries ({max_retries}) exhausted. "
                            f"Last error: {e}"
                        )
                        raise
                    
                    # Parse retry delay from error message if available
                    delay = base_delay * (2 ** attempt)
                    
                    if "try again in" in error_msg.lower():
                        try:
                            match = re.search(r'try again in (\d+(?:\.\d+)?)(ms|s)', error_msg.lower())
                            if match:
                                value = float(match.group(1))
                                unit = match.group(2)
                                if unit == 'ms':
                                    delay = value / 1000.0
                                else:
                                    delay = value
                                delay = delay + 0.5
                        except Exception:
                            pass
                    
                    # Add jitter (±25%)
                    delay = delay * (0.75 + random.random() * 0.5)
                    delay = min(delay, max_delay)
                    
                    logger.warning(
                        f"[Gemini] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.2f}s before retry..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-rate-limit error, re-raise immediately
                    raise
        
        if last_exception:
            raise last_exception

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a text response using Google's Generative AI API."""
        # Ensure capabilities are detected via API
        await self.initialize()

        try:
            # Log LLM request (with prompts at level 2)
            start_time = self._log_llm_request(
                "GENERATE", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # Build request payload
            contents = []

            # Add system instruction if provided
            system_instruction = None
            if system_prompt:
                system_instruction = {"parts": [{"text": system_prompt}]}

            contents.append({"role": "user", "parts": [{"text": prompt}]})

            payload: Dict[str, Any] = {"contents": contents}

            if system_instruction:
                payload["systemInstruction"] = system_instruction

            # Add generation config
            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            result = await self._make_request("generateContent", payload)

            # Extract response
            content = ""
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    content = candidate["content"]["parts"][0].get("text", "")

            # Extract usage
            usage_metadata = result.get("usageMetadata", {})

            # Log response (with content at level 2)
            total_tokens = usage_metadata.get("totalTokenCount", 0)
            self._log_llm_response("GENERATE", start_time, total_tokens, content)

            llm_response = LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": total_tokens,
                },
                finish_reason=result.get("candidates", [{}])[0].get("finishReason"),
            )

            # Track usage at session level
            self._track_usage(llm_response)

            return llm_response
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMProviderError(f"Gemini generation failed: {e}") from e

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
        Generate a response with vision using Gemini.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).

        If the model does not support vision, falls back to text-only generation.
        """
        # Check if model supports vision
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Vision-capable Gemini models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp. "
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

            # Build parts with images and text
            parts: List[Dict[str, Any]] = []

            for img in images:
                if img.source_type == "bytes":
                    img_base64 = base64.b64encode(img.data).decode("utf-8")
                else:
                    img_base64 = img.data

                parts.append({
                    "inlineData": {
                        "mimeType": img.media_type,
                        "data": img_base64,
                    }
                })

            parts.append({"text": prompt})

            contents = [{"role": "user", "parts": parts}]

            payload: Dict[str, Any] = {"contents": contents}

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            result = await self._make_request("generateContent", payload)

            # Extract response
            content = ""
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    content = candidate["content"]["parts"][0].get("text", "")

            usage_metadata = result.get("usageMetadata", {})

            # Log response (with content at level 2)
            total_tokens = usage_metadata.get("totalTokenCount", 0)
            self._log_llm_response("VISION", start_time, total_tokens, content)

            llm_response = LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": total_tokens,
                },
                finish_reason=result.get("candidates", [{}])[0].get("finishReason"),
            )

            # Track usage at session level
            self._track_usage(llm_response)

            return llm_response
        except Exception as e:
            logger.error(f"Gemini vision generation error: {e}")
            raise LLMProviderError(f"Gemini vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output using Gemini."""
        try:
            # Add JSON schema instruction to prompt
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, just pure JSON matching this schema: "
                f"{json.dumps(schema)}"
            )
            full_prompt = prompt + schema_instruction

            response = await self.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )

            # Parse JSON from response
            content = response.content.strip()

            # Try to extract JSON if wrapped in markdown code blocks
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
            logger.error(f"Gemini structured response not valid JSON: {e}")
            raise LLMProviderError(f"Gemini structured response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Gemini structured generation error: {e}")
            raise LLMProviderError(f"Gemini structured generation failed: {e}") from e

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
        Generate structured output with vision using Gemini.
        
        Combines vision capabilities with JSON output instructions for deterministic structured output.
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
            
            # Add JSON schema instruction to prompt
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, just pure JSON matching this schema: "
                f"{json.dumps(schema)}"
            )
            full_prompt = prompt + schema_instruction
            
            # Log request
            start_time = self._log_llm_request(
                "STRUCTURED_VISION", self.model,
                f"{len(images)} images",
                prompt=full_prompt, system_prompt=system_prompt or ""
            )
            
            # Build parts with images and text
            parts: List[Dict[str, Any]] = []
            
            for img in images:
                if img.source_type == "bytes":
                    img_base64 = base64.b64encode(img.data).decode("utf-8")
                else:
                    img_base64 = img.data
                
                parts.append({
                    "inlineData": {
                        "mimeType": img.media_type,
                        "data": img_base64,
                    }
                })
            
            parts.append({"text": full_prompt})
            
            contents = [{"role": "user", "parts": parts}]
            
            payload: Dict[str, Any] = {"contents": contents}
            
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            
            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config
            
            result = await self._make_request("generateContent", payload)
            
            # Extract response
            content = ""
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    content = candidate["content"]["parts"][0].get("text", "")
            
            usage_metadata = result.get("usageMetadata", {})
            total_tokens = usage_metadata.get("totalTokenCount", 0)
            self._log_llm_response("STRUCTURED_VISION", start_time, total_tokens, content)
            
            # Track usage at session level
            llm_response = LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            # Extract JSON from response
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
            logger.error(f"Gemini structured vision response not valid JSON: {e}")
            raise LLMProviderError(f"Gemini structured vision response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Gemini structured vision generation error: {e}")
            raise LLMProviderError(f"Gemini structured vision generation failed: {e}") from e

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
            # Convert ToolDefinition to Gemini format
            gemini_tools = [{
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": tool.required,
                        },
                    }
                    for tool in tools
                ]
            }]

            contents = [{"role": "user", "parts": [{"text": prompt}]}]

            payload: Dict[str, Any] = {
                "contents": contents,
                "tools": gemini_tools,
            }

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            result = await self._make_request("generateContent", payload)

            # Extract tool calls if present
            tool_calls = None
            content = ""

            if result.get("candidates"):
                candidate = result["candidates"][0]
                parts = candidate.get("content", {}).get("parts", [])

                for part in parts:
                    if "text" in part:
                        content = part["text"]
                    elif "functionCall" in part:
                        if tool_calls is None:
                            tool_calls = []
                        fc = part["functionCall"]
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{len(tool_calls)}",
                                name=fc.get("name", ""),
                                arguments=fc.get("args", {}),
                            )
                        )

            usage_metadata = result.get("usageMetadata", {})

            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": usage_metadata.get("totalTokenCount", 0),
                },
                tool_calls=tool_calls,
                finish_reason=result.get("candidates", [{}])[0].get("finishReason"),
            )
        except Exception as e:
            logger.error(f"Gemini tool calling error: {e}")
            raise LLMProviderError(f"Gemini tool calling failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Gemini."""
        try:
            url = f"{self.base_url}/models/{self.model}:streamGenerateContent?key={self.api_key}"

            contents = [{"role": "user", "parts": [{"text": prompt}]}]

            payload: Dict[str, Any] = {"contents": contents}

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            headers = {"Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Gemini streaming request failed: {response.status} - {error_text}"
                        )

                    async for line in response.content:
                        if line:
                            try:
                                # Gemini streams JSON objects
                                text = line.decode("utf-8").strip()
                                if text.startswith("["):
                                    text = text[1:]
                                if text.endswith("]"):
                                    text = text[:-1]
                                if text.startswith(","):
                                    text = text[1:]
                                if text:
                                    data = json.loads(text)
                                    if data.get("candidates"):
                                        parts = data["candidates"][0].get("content", {}).get("parts", [])
                                        for part in parts:
                                            if "text" in part:
                                                yield part["text"]
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise LLMProviderError(f"Gemini streaming failed: {e}") from e

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings using Gemini."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            embed_model = model or "text-embedding-004"
            url = f"{self.base_url}/models/{embed_model}:batchEmbedContents?key={self.api_key}"

            requests = [
                {"model": f"models/{embed_model}", "content": {"parts": [{"text": text}]}}
                for text in texts
            ]

            payload = {"requests": requests}
            headers = {"Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Gemini embeddings request failed: {response.status} - {error_text}"
                        )

                    result = await response.json()

                    embeddings = []
                    for embedding in result.get("embeddings", []):
                        embeddings.append(embedding.get("values", []))

                    return embeddings
        except Exception as e:
            logger.error(f"Gemini embeddings error: {e}")
            raise LLMProviderError(f"Gemini embeddings failed: {e}") from e

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Gemini model.
        
        Uses simple hardcoded values based on model name patterns.
        """
        # Use cached capabilities if available
        if self._model_capabilities_cache is not None:
            capabilities = self._model_capabilities_cache
        else:
            capabilities = self._get_basic_capabilities()
        
        # Gemini always supports embeddings
        if ModelCapability.EMBEDDINGS not in capabilities:
            capabilities = capabilities + [ModelCapability.EMBEDDINGS]
        
        # Simple hardcoded pricing and context windows
        model_lower = self.model.lower()
        
        if "2.0" in model_lower or "2-0" in model_lower:
            context_window = 1000000
            input_cost = 0.0  # Free tier
            output_cost = 0.0
        elif "1.5-pro" in model_lower:
            context_window = 2000000
            input_cost = 0.00125
            output_cost = 0.005
        elif "1.5" in model_lower:
            context_window = 1000000
            input_cost = 0.0  # Free tier
            output_cost = 0.0
        else:
            context_window = 128000
            input_cost = 0.0
            output_cost = 0.0
        
        max_output = 8192
        
        return ModelInfo(
            name=self.model,
            provider="gemini",
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=input_cost,
            cost_per_1k_output_tokens=output_cost,
        )


