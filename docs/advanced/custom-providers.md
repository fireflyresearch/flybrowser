# Custom LLM Providers

FlyBrowser supports custom LLM providers for integrating with proprietary or specialized language models. This guide explains how to implement and register custom providers.

## Overview

Custom providers enable:
- Integration with proprietary LLM services
- Support for self-hosted models
- Custom API wrappers
- Specialized model handling

## Creating a Custom Provider

### Basic Structure

```python
from flybrowser.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    ModelInfo,
    ModelCapability,
    ImageInput,
)
from typing import Any, Dict, List, Optional, Union

class MyCustomProvider(BaseLLMProvider):
    """Custom LLM provider implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional["LLMProviderConfig"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, config, **kwargs)
        # Custom initialization
        self.client = MyAPIClient(api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text response."""
        # Implementation
        ...
    
    async def generate_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, ImageInput, List[ImageInput]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with image input."""
        # Implementation
        ...
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        # Implementation
        ...
    
    def get_model_info(self) -> ModelInfo:
        """Return model capabilities."""
        return ModelInfo(
            name=self.model,
            provider="my_custom",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
            ],
            context_window=128000,
            max_output_tokens=8192,
        )
```

## Implementing Required Methods

### generate()

Basic text generation:

```python
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> LLMResponse:
    """Generate a response from the LLM."""
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Call your API
    response = await self.client.chat(
        model=self.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or 4096,
    )
    
    # Return standardized response
    return LLMResponse(
        content=response["choices"][0]["message"]["content"],
        model=self.model,
        usage={
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
        },
        finish_reason=response["choices"][0]["finish_reason"],
    )
```

### generate_with_vision()

Vision/multimodal generation:

```python
async def generate_with_vision(
    self,
    prompt: str,
    image_data: Union[bytes, ImageInput, List[ImageInput]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> LLMResponse:
    """Generate response with image input."""
    
    # Normalize images to list of ImageInput
    images = self._normalize_images(image_data)
    
    # Build content with images
    content = []
    for image in images:
        if image.source_type == "bytes":
            import base64
            b64 = base64.b64encode(image.data).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": b64,
                }
            })
        elif image.source_type == "base64":
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.data,
                }
            })
    
    content.append({"type": "text", "text": prompt})
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    
    # Call API
    response = await self.client.chat(
        model=self.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or 4096,
    )
    
    return LLMResponse(
        content=response["choices"][0]["message"]["content"],
        model=self.model,
        usage=response.get("usage"),
    )
```

### generate_structured()

Structured JSON output:

```python
async def generate_structured(
    self,
    prompt: str,
    schema: Dict[str, Any],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate structured JSON response."""
    import json
    
    # Add schema to system prompt
    schema_str = json.dumps(schema, indent=2)
    enhanced_system = f"""{system_prompt or "You are a helpful assistant."}

You must respond with valid JSON matching this schema:
{schema_str}

Respond ONLY with the JSON object, no additional text."""

    # Generate response
    response = await self.generate(
        prompt=prompt,
        system_prompt=enhanced_system,
        temperature=temperature,
        **kwargs,
    )
    
    # Parse JSON from response
    content = response.content.strip()
    
    # Handle markdown code blocks
    if content.startswith("```"):
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    
    # Find JSON object
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]
    
    return json.loads(content)
```

## Optional Methods

### Streaming Support

```python
async def generate_stream(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Generate streaming response."""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    async for chunk in self.client.chat_stream(
        model=self.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if content := delta.get("content"):
                yield content
```

### Tool Calling

```python
async def generate_with_tools(
    self,
    prompt: str,
    tools: List["ToolDefinition"],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> LLMResponse:
    """Generate with tool calling support."""
    
    # Convert tools to API format
    api_tools = []
    for tool in tools:
        api_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        })
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = await self.client.chat(
        model=self.model,
        messages=messages,
        tools=api_tools,
        tool_choice=tool_choice or "auto",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Parse tool calls if any
    tool_calls = None
    if response["choices"][0].get("message", {}).get("tool_calls"):
        from flybrowser.llm.base import ToolCall
        import json
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"]),
            )
            for tc in response["choices"][0]["message"]["tool_calls"]
        ]
    
    return LLMResponse(
        content=response["choices"][0]["message"].get("content", ""),
        model=self.model,
        usage=response.get("usage"),
        tool_calls=tool_calls,
    )
```

### Embeddings

```python
async def generate_embeddings(
    self,
    texts: Union[str, List[str]],
    model: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    """Generate embeddings for text(s)."""
    
    if isinstance(texts, str):
        texts = [texts]
    
    response = await self.client.embeddings(
        model=model or "text-embedding-model",
        input=texts,
    )
    
    return [item["embedding"] for item in response["data"]]
```

## Model Information

Provide accurate model capabilities:

```python
def get_model_info(self) -> ModelInfo:
    """Return model capabilities and limits."""
    
    # Define capabilities based on model
    capabilities = [ModelCapability.TEXT_GENERATION]
    
    if "vision" in self.model or self.model in VISION_MODELS:
        capabilities.append(ModelCapability.VISION)
        capabilities.append(ModelCapability.MULTI_IMAGE_VISION)
    
    if self.model in STREAMING_MODELS:
        capabilities.append(ModelCapability.STREAMING)
    
    if self.model in TOOL_CALLING_MODELS:
        capabilities.append(ModelCapability.TOOL_CALLING)
    
    # Get context window for model
    context_windows = {
        "model-small": 32000,
        "model-large": 128000,
    }
    context_window = context_windows.get(self.model, 32000)
    
    return ModelInfo(
        name=self.model,
        provider="my_custom",
        capabilities=capabilities,
        context_window=context_window,
        max_output_tokens=8192,
        supports_system_prompt=True,
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.002,
    )
```

## Availability Checking

```python
@classmethod
def check_availability(cls) -> "ProviderStatus":
    """Check if provider is available."""
    import os
    from flybrowser.llm.provider_status import ProviderStatus
    
    api_key = os.environ.get("MY_CUSTOM_API_KEY")
    
    if not api_key:
        return ProviderStatus.error(
            name="MyCustom",
            message="MY_CUSTOM_API_KEY environment variable not set",
        )
    
    # Optionally test connectivity
    try:
        # Quick API test
        ...
        return ProviderStatus.ok(name="MyCustom")
    except Exception as e:
        return ProviderStatus.warn(
            name="MyCustom",
            message=f"API key set but connectivity test failed: {e}",
        )
```

## Registering the Provider

### Manual Registration

```python
from flybrowser.llm.factory import LLMProviderFactory
from my_providers import MyCustomProvider

# Register the provider
LLMProviderFactory.register_provider("my_custom", MyCustomProvider)

# Now use it
provider = LLMProviderFactory.create(
    provider="my_custom",
    model="my-model",
    api_key="...",
)
```

### Using with FlyBrowser

```python
from flybrowser import FlyBrowser
from flybrowser.llm.factory import LLMProviderFactory
from my_providers import MyCustomProvider

# Register before creating browser
LLMProviderFactory.register_provider("my_custom", MyCustomProvider)

async with FlyBrowser(
    llm_provider="my_custom",
    llm_model="my-model",
    llm_api_key="...",
) as browser:
    await browser.goto("https://example.com")
    result = await browser.extract("Get the title")
```

## Complete Example

```python
"""Custom LLM provider for MyService API."""

from typing import Any, AsyncIterator, Dict, List, Optional, Union
import httpx

from flybrowser.llm.base import (
    BaseLLMProvider,
    ImageInput,
    LLMResponse,
    ModelCapability,
    ModelInfo,
    ToolCall,
    ToolDefinition,
)
from flybrowser.llm.config import LLMProviderConfig
from flybrowser.llm.provider_status import ProviderStatus


class MyServiceProvider(BaseLLMProvider):
    """Provider for MyService LLM API."""
    
    BASE_URL = "https://api.myservice.com/v1"
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, config, **kwargs)
        self.base_url = kwargs.get("base_url", self.BASE_URL)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            usage=data.get("usage"),
            finish_reason=data["choices"][0].get("finish_reason"),
        )
    
    async def generate_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, ImageInput, List[ImageInput]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import base64
        
        images = self._normalize_images(image_data)
        
        content = []
        for img in images:
            if img.source_type == "bytes":
                b64 = base64.b64encode(img.data).decode()
            else:
                b64 = img.data
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.media_type};base64,{b64}"}
            })
        content.append({"type": "text", "text": prompt})
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            usage=data.get("usage"),
        )
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import json
        
        schema_str = json.dumps(schema, indent=2)
        enhanced_system = f"""{system_prompt or ""}
Respond with valid JSON matching this schema:
{schema_str}"""
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=enhanced_system.strip(),
            temperature=temperature,
            **kwargs,
        )
        
        content = response.content.strip()
        if content.startswith("```"):
            import re
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
            if match:
                content = match.group(1)
        
        return json.loads(content)
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="myservice",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
                ModelCapability.STRUCTURED_OUTPUT,
            ],
            context_window=128000,
            max_output_tokens=8192,
            supports_system_prompt=True,
        )
    
    @classmethod
    def check_availability(cls) -> ProviderStatus:
        import os
        api_key = os.environ.get("MYSERVICE_API_KEY")
        if not api_key:
            return ProviderStatus.error(
                name="MyService",
                message="MYSERVICE_API_KEY not set",
            )
        return ProviderStatus.ok(name="MyService")
```

## See Also

- [LLM Integration Architecture](../architecture/llm-integration.md) - LLM system overview
- [Configuration Reference](../reference/configuration.md) - Configuration options
