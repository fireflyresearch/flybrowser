# LLM Integration

FlyBrowser provides a unified interface for multiple LLM providers. This document explains the LLM integration architecture.

## Overview

The LLM integration system provides:

- Unified API across all providers (OpenAI, Anthropic, Ollama, Gemini, Qwen)
- Vision/multimodal support with single and multiple images
- Streaming response support
- Tool/function calling support
- Structured output with JSON schemas
- Model capability introspection
- Production features (caching, retry, rate limiting, cost tracking)

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            LLMProviderFactory                             │
│       ┌───────────────────────────────────────────────────────────┐       │
│       │  create() | create_from_config() | register_provider()    │       │
│       └──────────────────────────┬────────────────────────────────┘       │
│                                  │                                        │
│   ┌──────────────────────────────┼────────────────────────────┐           │
│   │           │           │           │           │           │           │
│   ▼           ▼           ▼           ▼           ▼           ▼           │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐    │
│ │ OpenAI  │ │Anthropic│ │ Ollama  │ │ Gemini  │ │  Qwen   │ │Custom  │    │
│ │Provider │ │Provider │ │Provider │ │Provider │ │Provider │ │Provider│    │
│ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬───┘    │
│      │           │           │           │           │           │        │
│      └───────────┴───────────┴───┼───────┴───────────┴───────────┘        │
│                                  │                                        │
│                                  ▼                                        │
│                          ┌─────────────────┐                              │
│                          │ BaseLLMProvider │                              │
│                          │   (Abstract)    │                              │
│                          └─────────────────┘                              │
└───────────────────────────────────────────────────────────────────────────┘
```

## BaseLLMProvider

All providers inherit from the abstract `BaseLLMProvider` class:

```python
class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers with production features."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM provider with configuration."""
        self.model = model
        self.api_key = api_key
        self.provider_config = config
        self.extra_config = kwargs
        
        # Production features (when config provided)
        self.cache = LLMCache(config.cache_config) if config else None
        self.cost_tracker = CostTracker(config.cost_tracking_config) if config else None
        self.rate_limiter = RateLimiter(config.rate_limit_config) if config else None
        self.retry_handler = RetryHandler(config.retry_config) if config else None
```

### Required Methods

Providers must implement these abstract methods:

```python
@abstractmethod
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> LLMResponse:
    """Generate a response from the LLM."""
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
    """Generate a response with vision capabilities."""
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
    """Generate a structured response matching the provided schema."""
    pass
```

### Optional Methods

Providers may optionally implement:

```python
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
    """Generate with tool/function calling capabilities."""

async def generate_stream(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Generate a streaming response."""

async def generate_embeddings(
    self,
    texts: Union[str, List[str]],
    model: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    """Generate embeddings for text(s)."""
```

## LLMResponse

All providers return standardized `LLMResponse` objects:

```python
@dataclass
class LLMResponse:
    """Standardized response from an LLM provider."""
    
    content: str                           # Generated text
    model: str                             # Model that generated response
    usage: Optional[Dict[str, int]]        # Token usage statistics
    metadata: Optional[Dict[str, Any]]     # Provider-specific metadata
    cached: bool                           # Whether served from cache
    tool_calls: Optional[List[ToolCall]]   # Tool calls made
    finish_reason: Optional[str]           # Completion reason
```

Usage statistics include:
- `prompt_tokens`: Number of input tokens
- `completion_tokens`: Number of output tokens
- `total_tokens`: Total tokens used

## Model Capabilities

Models declare their capabilities through `ModelCapability` enum:

```python
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
```

Check capabilities:

```python
# Check if model supports vision
if provider.supports_capability(ModelCapability.VISION):
    response = await provider.generate_with_vision(prompt, image_data)

# Or use the property
if provider.vision_enabled:
    # Use vision features
    ...
```

## Model Information

Get detailed model information:

```python
@dataclass
class ModelInfo:
    """Information about a model's capabilities and limits."""
    
    name: str                           # Model name
    provider: str                       # Provider name
    capabilities: List[ModelCapability] # Supported capabilities
    context_window: int                 # Max context size (tokens)
    max_output_tokens: int              # Max output tokens
    supports_system_prompt: bool        # System prompt support
    cost_per_1k_input_tokens: Optional[float]
    cost_per_1k_output_tokens: Optional[float]

# Get model info
info = provider.get_model_info()
print(f"Model: {info.name}, Context: {info.context_window}")
```

## Supported Providers

### OpenAI

```python
from flybrowser.llm.factory import LLMProviderFactory

provider = LLMProviderFactory.create(
    provider="openai",
    model="gpt-5.2",  # or gpt-4o, gpt-4o-mini
    api_key="sk-...",
)
```

### Anthropic

```python
provider = LLMProviderFactory.create(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",  # or claude-3-opus
    api_key="sk-ant-...",
)
```

### Ollama (Local)

```python
provider = LLMProviderFactory.create(
    provider="ollama",
    model="qwen3:8b",  # or llama3, mistral, etc.
    # No API key needed for local
)
```

### Google Gemini

```python
provider = LLMProviderFactory.create(
    provider="gemini",  # or "google"
    model="gemini-2.0-flash",
    api_key="...",
)
```

### Qwen (Alibaba Cloud)

Qwen models are accessed via Alibaba Cloud's DashScope service, which provides an OpenAI-compatible API:

```python
provider = LLMProviderFactory.create(
    provider="qwen",  # or "dashscope"
    model="qwen-plus",  # or qwen-turbo, qwen-max, qwen3-235b-a22b
    api_key="sk-...",  # DashScope API key
)
```

Qwen supports multiple regions:

```python
# Default (China mainland)
provider = LLMProviderFactory.create(
    provider="qwen",
    model="qwen-plus",
    region="default",
)

# International
provider = LLMProviderFactory.create(
    provider="qwen",
    model="qwen-plus",
    region="international",
)

# US region
provider = LLMProviderFactory.create(
    provider="qwen",
    model="qwen-plus",
    region="us",
)
```

Qwen vision models (qwen-vl-max, qwen-vl-plus) are automatically detected and support image inputs.

## Provider Factory

The `LLMProviderFactory` manages provider creation:

```python
class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        vision_enabled: Optional[bool] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        ...
    
    @classmethod
    def create_from_config(cls, config: LLMProviderConfig) -> BaseLLMProvider:
        """Create a provider from a configuration object."""
        ...
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a custom LLM provider."""
        ...
    
    @classmethod
    def list_providers(cls, include_aliases: bool = True) -> list:
        """List all registered providers."""
        ...
    
    @classmethod
    def get_all_provider_statuses(cls) -> Dict[str, ProviderStatus]:
        """Get availability status for all providers."""
        ...
```

## Provider Configuration

Configure providers with `LLMProviderConfig`:

```python
class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    provider_type: LLMProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    
    # Production features
    retry_config: RetryConfig
    rate_limit_config: RateLimitConfig
    cache_config: CacheConfig
    cost_tracking_config: CostTrackingConfig
    
    # Provider-specific options
    extra_options: Dict[str, Any]
```

### Retry Configuration

```python
class RetryConfig(BaseModel):
    """Retry configuration for LLM requests."""
    
    max_retries: int = 3                  # 0-10
    initial_delay: float = 1.0            # seconds
    max_delay: float = 60.0               # seconds
    exponential_base: float = 2.0
    jitter: bool = True
```

### Rate Limiting

```python
class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    concurrent_requests: int = 10
```

### Caching

```python
class CacheConfig(BaseModel):
    """Cache configuration for LLM responses."""
    
    enabled: bool = True
    ttl_seconds: int = 3600               # 1 hour
    max_size: int = 1000                  # entries
    cache_key_prefix: str = "flybrowser:llm"
```

### Cost Tracking

```python
class CostTrackingConfig(BaseModel):
    """Cost tracking configuration."""
    
    enabled: bool = True
    track_tokens: bool = True
    track_requests: bool = True
    log_costs: bool = True
```

## Image Input

For vision capabilities, use `ImageInput`:

```python
@dataclass
class ImageInput:
    """Represents an image input for vision-capable models."""
    
    data: Union[bytes, str]
    media_type: str = "image/png"
    detail: str = "auto"                  # "low", "high", "auto"
    source_type: str = "bytes"
    
    @classmethod
    def from_bytes(cls, data: bytes, media_type: str = "image/png", detail: str = "auto"):
        """Create ImageInput from raw bytes."""
        ...
    
    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png", detail: str = "auto"):
        """Create ImageInput from base64 string."""
        ...
    
    @classmethod
    def from_url(cls, url: str, detail: str = "auto"):
        """Create ImageInput from URL."""
        ...
```

Example usage:

```python
# Single image from bytes
response = await provider.generate_with_vision(
    "What's in this image?",
    ImageInput.from_bytes(screenshot_bytes, "image/png"),
)

# Multiple images
images = [
    ImageInput.from_bytes(image1_bytes),
    ImageInput.from_bytes(image2_bytes),
]
response = await provider.generate_with_vision(
    "Compare these images",
    images,
)
```

## Tool Calling

Define tools for function calling:

```python
@dataclass
class ToolDefinition:
    """Definition of a tool/function that can be called by the LLM."""
    
    name: str
    description: str
    parameters: Dict[str, Any]           # JSON schema
    required: List[str]

@dataclass
class ToolCall:
    """Represents a tool call made by the LLM."""
    
    id: str
    name: str
    arguments: Dict[str, Any]
```

Example:

```python
tools = [
    ToolDefinition(
        name="search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
        },
        required=["query"],
    ),
]

response = await provider.generate_with_tools(
    "Find information about Python programming",
    tools=tools,
)

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call.name}, Args: {call.arguments}")
```

## Production Features

### With Features

Use production features via `generate_with_features`:

```python
response = await provider.generate_with_features(
    prompt="Hello, world!",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=100,
    use_cache=True,
)

if response.cached:
    print("Response served from cache")
```

### Usage Tracking

Track session usage:

```python
# Get accumulated usage
usage = provider.get_session_usage()
print(f"Tokens: {usage['total_tokens']}, Cost: ${usage['cost_usd']:.4f}")

# Reset tracking
provider.reset_session_usage()
```

### Provider Statistics

Get comprehensive stats:

```python
stats = provider.get_stats()
# {
#     "model": "gpt-4o",
#     "provider": "OpenAIProvider",
#     "capabilities": ["text_generation", "vision", "streaming"],
#     "cache": {"hits": 10, "misses": 5},
#     "cost": {"total_usd": 0.15},
#     "rate_limit": {"current_rpm": 45}
# }
```

### Availability Checking

Check provider availability:

```python
# Check specific provider
status = LLMProviderFactory.get_provider_status("openai")
print(f"OpenAI: {status.available}, Message: {status.message}")

# Check all providers
all_statuses = LLMProviderFactory.get_all_provider_statuses()
for name, status in all_statuses.items():
    print(f"{name}: {'OK' if status.available else 'Unavailable'}")
```

## Default Models

Default models per provider:

| Provider | Default Model |
|----------|---------------|
| OpenAI | gpt-5.2 |
| Anthropic | claude-sonnet-4-5-20250929 |
| Ollama | qwen3:8b |
| Gemini | gemini-2.0-flash |
| Qwen | qwen-plus |

## Custom Providers

Register custom providers:

```python
from flybrowser.llm.base import BaseLLMProvider, LLMResponse

class MyCustomProvider(BaseLLMProvider):
    """Custom LLM provider implementation."""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        ...
    
    async def generate_with_vision(self, prompt: str, image_data, **kwargs) -> LLMResponse:
        # Implementation
        ...
    
    async def generate_structured(self, prompt: str, schema: Dict, **kwargs) -> Dict:
        # Implementation
        ...

# Register the provider
LLMProviderFactory.register_provider("custom", MyCustomProvider)

# Use it
provider = LLMProviderFactory.create("custom", model="my-model")
```

## LLM Logging

Enable LLM request/response logging:

```python
# Enable basic logging
provider.enable_llm_logging(enabled=True, level=1)

# Enable detailed logging (shows prompts/responses)
provider.enable_llm_logging(enabled=True, level=2)

# Disable logging
provider.enable_llm_logging(enabled=False)
```

Logging levels:
- `0`: Disabled
- `1`: Basic (shows request/response timing)
- `2`: Detailed (also shows prompts and responses)

## Conversation Management

FlyBrowser uses a **mandatory** `ConversationManager` for ALL LLM interactions. This ensures consistent token tracking, budget management, and conversation history across the entire framework.

### ConversationManager

The `ConversationManager` is the **central interface** for ALL LLM calls. It handles:
- Multi-turn conversation history tracking
- Token budget management to prevent context overflow
- Large content chunking and multi-turn processing
- Structured output preservation across turns
- Vision/VLM support with image token estimation
- Unified logging and statistics

**All LLM calls in FlyBrowser are routed through ConversationManager**, including:
- `StructuredLLMWrapper.generate_structured()` → `ConversationManager.send_structured()`
- `StructuredLLMWrapper.generate_structured_with_vision()` → `ConversationManager.send_structured_with_vision()`
- `StructuredLLMWrapper._repair_response()` → `ConversationManager.send_structured()`
- `ReActAgent._generate_with_optional_vision()` → `ConversationManager`

```python
from flybrowser.llm.conversation import ConversationManager

# Initialize with an LLM provider
manager = ConversationManager(llm_provider)

# Set system prompt
manager.set_system_prompt("You are a helpful assistant.")

# Simple structured request
response = await manager.send_structured(
    "What is the capital of France?",
    schema={"type": "object", "properties": {"answer": {"type": "string"}}}
)

# Vision request (for VLM models)
response = await manager.send_structured_with_vision(
    "Analyze this screenshot",
    image_data=screenshot_bytes,
    schema={"type": "object", "properties": {"elements": {"type": "array"}}}
)

# Handle large content automatically
response = await manager.send_with_large_content(
    large_html_content,
    instruction="Extract all product names",
    schema=product_schema
)
```

### Token Budget Management

The `TokenBudgetManager` tracks and allocates tokens across conversation turns:

```python
from flybrowser.llm.token_budget import TokenBudgetManager, TokenEstimator

# Estimate tokens for content
estimate = TokenEstimator.estimate(content)
print(f"Tokens: {estimate.tokens}, Type: {estimate.content_type.value}")

# Check if content fits
manager = TokenBudgetManager(context_window=128000, max_output_tokens=8192)
if manager.can_fit(content):
    # Process content
    ...
else:
    # Need to chunk or summarize
    ...
```

### Content Chunking

For content that exceeds token limits, FlyBrowser provides intelligent chunking:

```python
from flybrowser.llm.chunking import SmartChunker, get_chunker, ContentType

# Auto-detect content type and chunk appropriately
chunker = SmartChunker()
chunks = chunker.chunk(large_content, max_tokens_per_chunk=4000)

# Or use specific chunker for known content types
html_chunker = get_chunker(ContentType.HTML)
json_chunker = get_chunker(ContentType.JSON)
```

Chunking strategies:
- **TextChunker**: Splits at paragraph/sentence boundaries
- **HTMLChunker**: Preserves DOM structure, splits at block elements
- **JSONChunker**: Maintains valid JSON structure
- **SmartChunker**: Auto-detects content type

### Multi-Turn Accumulation Protocol

For very large content, the `ConversationManager` implements a multi-turn accumulation protocol:

1. **Single Turn**: If content fits, send as single request
2. **Accumulation Phase**: Split content into chunks, extract key points from each
3. **Synthesis Phase**: Combine key points into final structured output

```python
# This happens automatically when content is too large
response = await manager.send_with_large_content(
    very_large_html,  # 200K+ chars
    instruction="Summarize all the articles",
    schema={"type": "object", "properties": {"summaries": {"type": "array"}}}
)
# Manager automatically chunks, processes, and synthesizes
```

### Vision Support

ConversationManager supports vision/VLM models:

```python
# Check if model supports vision
if manager.has_vision:
    response = await manager.send_structured_with_vision(
        content="What elements are visible on this page?",
        image_data=screenshot_bytes,
        schema=element_schema,
    )

# Image tokens are estimated automatically
# Base: 85 tokens + 170 tokens per 512x512 tile
```

### Integration with Components

**All components that use StructuredLLMWrapper automatically route through ConversationManager:**

- **ReActAgent**: Main reasoning loop and vision requests
- **TaskPlanner**: Planning structured outputs
- **ObstacleDetector**: Page analysis and obstacle detection
- **SitemapGraph**: Link analysis
- **PageAnalyzer**: Page structure analysis

```python
# StructuredLLMWrapper creates ConversationManager automatically
wrapper = StructuredLLMWrapper(llm_provider)
# wrapper.conversation is the ConversationManager

# All calls go through ConversationManager
result = await wrapper.generate_structured(prompt, schema)  # → ConversationManager.send_structured()
result = await wrapper.generate_structured_with_vision(prompt, image, schema)  # → ConversationManager.send_structured_with_vision()
```

The ReActAgent also uses ConversationManager directly for its reasoning loop:

```python
# ReActAgent has its own ConversationManager instance
agent = ReActAgent(page, llm_provider, tool_registry, memory)

# During execution, large extraction data is handled automatically
# The agent's _check_and_handle_large_context() method uses the
# ConversationManager to truncate context when needed
```

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Memory System](memory.md) - Memory management and context
- [Configuration Reference](../reference/configuration.md) - Configuration options
- [Agent Feature](../features/agent.md) - How agents use LLMs
