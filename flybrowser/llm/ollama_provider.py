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
Ollama LLM provider implementation.

This module provides the OllamaProvider class which implements the BaseLLMProvider
interface for locally-hosted Ollama models. Ollama allows running open-source LLMs
locally without API costs or privacy concerns.

Supported models include (as of January 2026):
- Qwen3 (default): Alibaba's latest generation with thinking support
- Llama 3.2: Meta's latest small models (1B, 3B)
- Gemma 3: Google's most capable model for single GPU
- DeepSeek-R1: Open reasoning model with performance near O3
- GPT-OSS: OpenAI's open-weight models (20B, 120B)
- Mistral: High-performance open model
- Phi-4: Microsoft's state-of-the-art 14B model
- And many more available through Ollama

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images
- Thinking/reasoning mode (for models like qwen3, deepseek-r1)
- Streaming responses
- Local inference without API keys
- Custom Ollama server URLs
- Error handling and retries

Note: Ollama must be running locally or on a specified server.
Install from: https://ollama.ai
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiohttp
import urllib.request
import urllib.error

from flybrowser.exceptions import LLMProviderError
from flybrowser.llm.base import (
    BaseLLMProvider,
    ImageInput,
    LLMResponse,
    ModelCapability,
    ModelInfo,
)
from flybrowser.llm.provider_status import ProviderStatus
from flybrowser.utils.logger import logger


class OllamaProvider(BaseLLMProvider):
    """
    Ollama local LLM provider implementation.

    This provider enables using locally-hosted Ollama models for browser automation.
    Ollama provides a simple way to run open-source LLMs locally without API costs
    or sending data to external services.

    Attributes:
        client: Not used (Ollama uses HTTP API)
        model: Ollama model name (e.g., "qwen3:8b", "gemma3:12b")
        base_url: Ollama server URL (default: http://localhost:11434)
        api_key: Not used for Ollama (kept for interface compatibility)

    Example:
        >>> provider = OllamaProvider(
        ...     model="qwen3:8b",
        ...     base_url="http://localhost:11434"
        ... )
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ollama provider with server configuration.

        Args:
            model: Ollama model name. Popular models (as of January 2026):
                - "qwen3:8b": Alibaba's Qwen3 8B model (default)
                - "qwen3:32b": Larger Qwen3 with thinking support
                - "gemma3:12b": Google's Gemma 3 12B model
                - "llama3.2:3b": Meta's Llama 3.2 3B model
                - "deepseek-r1:8b": DeepSeek reasoning model
                - "phi4": Microsoft's Phi-4 14B model
                - "gpt-oss:20b": OpenAI's open-weight model
                Run `ollama list` to see installed models
            api_key: Not used for Ollama (kept for interface compatibility).
                Ollama runs locally and doesn't require API keys.
            base_url: Ollama server URL. Default: "http://localhost:11434"
                Change this if Ollama is running on a different host/port.
            **kwargs: Additional configuration passed to BaseLLMProvider

        Example:
            Local Ollama:
            >>> provider = OllamaProvider(model="qwen3:8b")

            Remote Ollama server:
            >>> provider = OllamaProvider(
            ...     model="gemma3:12b",
            ...     base_url="http://192.168.1.100:11434"
            ... )
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
        
        Vision support is determined by model name (llava, llama3.2-vision, etc.) and can be
        overridden with vision_enabled parameter.
        
        Returns:
            List of ModelCapability enum values
        """
        model_lower = self.model.lower()
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.STREAMING]
        
        # Vision models
        # Check override first
        if self._vision_enabled_override is not None:
            if self._vision_enabled_override:
                capabilities.append(ModelCapability.VISION)
        elif any(x in model_lower for x in ["vision", "llava", "bakllava", "moondream"]):
            capabilities.append(ModelCapability.VISION)
        
        # Tool calling (most recent models)
        if any(x in model_lower for x in ["llama3", "qwen", "phi", "gemma3"]):
            capabilities.append(ModelCapability.TOOL_CALLING)
        
        return capabilities

    async def _make_request_with_retry(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with rate limit retry.
        
        Handles rate limit errors (429) with exponential backoff and jitter.
        
        Args:
            url: The API endpoint URL
            payload: The request payload
            
        Returns:
            The JSON response
            
        Raises:
            LLMProviderError if all retries are exhausted or non-retryable error
        """
        max_retries = 3
        base_delay = 1.0
        max_delay = 60.0
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 429:
                            error_text = await response.text()
                            raise LLMProviderError(
                                f"Rate limit: {response.status} - {error_text}"
                            )
                        elif response.status != 200:
                            error_text = await response.text()
                            raise LLMProviderError(
                                f"Ollama request failed: {response.status} - {error_text}"
                            )
                        return await response.json()
                        
            except LLMProviderError as e:
                error_msg = str(e)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(
                            f"[Ollama] Rate limit: max retries ({max_retries}) exhausted. "
                            f"Last error: {e}"
                        )
                        raise
                    
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
                        f"[Ollama] Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.2f}s before retry..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
        
        if last_exception:
            raise last_exception


    @classmethod
    def check_availability(cls, base_url: str = "http://localhost:11434") -> ProviderStatus:
        """
        Check if Ollama provider is available.

        Checks connectivity to the Ollama server.
        """
        try:
            url = f"{base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return ProviderStatus.ok(
                        name="Ollama",
                        message=f"Running at {base_url}",
                        requires_api_key=False,
                        base_url=base_url,
                        connectivity_checked=True,
                        connectivity_ok=True,
                    )
        except urllib.error.URLError:
            pass
        except Exception:
            pass

        return ProviderStatus.info(
            name="Ollama",
            message="Not running (optional)",
            requires_api_key=False,
            base_url=base_url,
            connectivity_checked=True,
            connectivity_ok=False,
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
        Generate a text response using Ollama's API.

        Args:
            prompt: User prompt/message to send to the model
            system_prompt: Optional system prompt to set model behavior.
                Example: "You are a helpful assistant that extracts data from web pages."
            temperature: Sampling temperature (0.0-2.0). Higher values make output
                more random, lower values more deterministic. Default: 0.7
            max_tokens: Maximum tokens to generate. If None, uses model default.
                Maps to Ollama's "num_predict" parameter.
            **kwargs: Additional parameters for Ollama API:
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - repeat_penalty: Penalty for repeating tokens

        Returns:
            LLMResponse containing:
            - content: Generated text
            - model: Model used for generation
            - usage: Token usage statistics (estimated for Ollama)

        Raises:
            LLMProviderError: If API call fails or Ollama server is unreachable

        Example:
            >>> response = await provider.generate(
            ...     prompt="What is the capital of France?",
            ...     system_prompt="You are a geography expert.",
            ...     temperature=0.3
            ... )
            >>> print(response.content)
            'The capital of France is Paris.'
        """
        # Ensure capabilities are detected via API
        await self.initialize()

        try:
            url = f"{self.base_url}/api/generate"

            # Build request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,  # Use non-streaming mode for simplicity
                "options": {
                    "temperature": temperature,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            # Log LLM request (with prompts at level 2)
            start_time = self._log_llm_request(
                "GENERATE", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # Make HTTP request to Ollama server with retry
            result = await self._make_request_with_retry(url, payload)

            # Log response (with content at level 2)
            total_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            response_content = result.get("response", "")
            self._log_llm_response("GENERATE", start_time, total_tokens, response_content)

            response = LLMResponse(
                content=result.get("response", ""),
                model=self.model,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0)
                    + result.get("eval_count", 0),
                },
                metadata={
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "eval_duration": result.get("eval_duration"),
                },
            )

            # Track usage at session level
            self._track_usage(response)

            return response

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise LLMProviderError(f"Ollama generation failed: {e}") from e

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
        Generate a response with vision using Ollama.

        Requires a vision-capable model like llava, qwen3-vl, or gemma3.
        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).

        If the model does not support vision, falls back to text-only generation.
        """
        # Check if model supports vision
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Vision-capable models: llava, qwen3-vl, gemma3, bakllava, moondream. "
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
            url = f"{self.base_url}/api/generate"

            # Normalize images to list of ImageInput
            images = self._normalize_images(image_data)

            # Convert all images to base64
            image_base64_list = []
            for img in images:
                if img.source_type == "bytes":
                    img_base64 = base64.b64encode(img.data).decode("utf-8")
                else:
                    img_base64 = img.data
                image_base64_list.append(img_base64)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": image_base64_list,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            # Log LLM vision request (with prompts at level 2)
            start_time = self._log_llm_request(
                "VISION", self.model, f"{len(image_base64_list)} images",
                prompt=prompt, system_prompt=system_prompt or ""
            )

            result = await self._make_request_with_retry(url, payload)

            # Log response (with content at level 2)
            total_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            response_content = result.get("response", "")
            self._log_llm_response("VISION", start_time, total_tokens, response_content)

            response = LLMResponse(
                content=result.get("response", ""),
                model=self.model,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": total_tokens,
                },
            )

            # Track usage at session level
            self._track_usage(response)

            return response

        except Exception as e:
            logger.error(f"Ollama vision generation error: {e}")
            raise LLMProviderError(f"Ollama vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output using Ollama."""
        try:
            # Add JSON schema instruction to prompt
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, just pure JSON matching this schema: "
                f"{json.dumps(schema)}"
            )
            full_prompt = prompt + schema_instruction

            # Log structured request (with prompts at level 2)
            start_time = self._log_llm_request(
                "STRUCTURED", self.model,
                prompt=full_prompt, system_prompt=system_prompt or ""
            )

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

            # Log with parsed content at level 2
            self._log_llm_response("STRUCTURED", start_time, response_content=content)
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Ollama structured response not valid JSON: {e}")
            raise LLMProviderError(f"Ollama structured response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Ollama structured generation error: {e}")
            raise LLMProviderError(f"Ollama structured generation failed: {e}") from e

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
        Generate structured output with vision using Ollama.
        
        Combines vision capabilities with JSON output instructions for deterministic structured output.
        Requires a vision-capable model like llava, qwen3-vl, or gemma3.
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
            url = f"{self.base_url}/api/generate"
            
            # Normalize images to list of ImageInput
            images = self._normalize_images(image_data)
            
            # Convert all images to base64
            image_base64_list = []
            for img in images:
                if img.source_type == "bytes":
                    img_base64 = base64.b64encode(img.data).decode("utf-8")
                else:
                    img_base64 = img.data
                image_base64_list.append(img_base64)
            
            # Add JSON schema instruction to prompt
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON only. No markdown, no code blocks, just pure JSON matching this schema: "
                f"{json.dumps(schema)}"
            )
            full_prompt = prompt + schema_instruction
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "images": image_base64_list,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            # Log request
            start_time = self._log_llm_request(
                "STRUCTURED_VISION", self.model, f"{len(image_base64_list)} images",
                prompt=full_prompt, system_prompt=system_prompt or ""
            )
            
            result = await self._make_request_with_retry(url, payload)
            
            total_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            content = result.get("response", "")
            self._log_llm_response("STRUCTURED_VISION", start_time, total_tokens, content)
            
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
            logger.error(f"Ollama structured vision response not valid JSON: {e}")
            raise LLMProviderError(f"Ollama structured vision response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Ollama structured vision generation error: {e}")
            raise LLMProviderError(f"Ollama structured vision generation failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Ollama."""
        try:
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Ollama streaming request failed: {response.status} - {error_text}"
                        )

                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode("utf-8"))
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise LLMProviderError(f"Ollama streaming failed: {e}") from e

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            url = f"{self.base_url}/api/embeddings"
            embeddings = []

            for text in texts:
                payload = {
                    "model": model or self.model,
                    "prompt": text,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise LLMProviderError(
                                f"Ollama embeddings request failed: {response.status} - {error_text}"
                            )

                        result = await response.json()
                        embeddings.append(result.get("embedding", []))

            return embeddings

        except Exception as e:
            logger.error(f"Ollama embeddings error: {e}")
            raise LLMProviderError(f"Ollama embeddings failed: {e}") from e

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Ollama model.
        
        Uses simple hardcoded values. Ollama models are local and have no cost.
        """
        # Use cached capabilities if available
        if self._model_capabilities_cache is not None:
            capabilities = self._model_capabilities_cache
        else:
            capabilities = self._get_basic_capabilities()
        
        # Ollama always supports embeddings
        if ModelCapability.EMBEDDINGS not in capabilities:
            capabilities = capabilities + [ModelCapability.EMBEDDINGS]
        
        # Simple context window estimates based on model size
        model_lower = self.model.lower()
        
        if "70b" in model_lower or "90b" in model_lower:
            context_window = 128000
        elif "qwen" in model_lower:
            context_window = 131072
        else:
            context_window = 8192
        
        max_output = 4096
        
        return ModelInfo(
            name=self.model,
            provider="ollama",
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=0.0,  # Local, no cost
            cost_per_1k_output_tokens=0.0,
        )

