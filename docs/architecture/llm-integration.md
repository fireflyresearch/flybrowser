# LLM Integration

FlyBrowser delegates LLM orchestration to **fireflyframework-genai**. The `FireflyAgent` wraps Pydantic AI Agent and handles model selection, tool calling, structured output, and multi-provider support.

## Architecture

FlyBrowser does not manage LLM providers directly. Instead:

1. `BrowserAgentConfig.model` specifies the model in `provider:model` format (e.g., `"openai:gpt-4o"`)
2. `FireflyAgent` (from the framework) handles provider creation, API calls, retries, and streaming
3. FlyBrowser focuses on browser-specific tools, memory, and middleware

```
FlyBrowser SDK
    |
    +-- BrowserAgent
            |
            +-- FireflyAgent(model="openai:gpt-4o")
                    |
                    +-- Pydantic AI Agent (handles LLM calls)
                            |
                            +-- OpenAI / Anthropic / Gemini / Ollama
```

## Supported Providers

The framework supports multiple providers via the `model` parameter:

| Provider | Model Format | Example |
|----------|-------------|---------|
| OpenAI | `openai:model-name` | `openai:gpt-4o` |
| Anthropic | `anthropic:model-name` | `anthropic:claude-sonnet-4-5-20250929` |
| Google Gemini | `gemini:model-name` | `gemini:gemini-2.0-flash` |
| Ollama | `ollama:model-name` | `ollama:qwen3:8b` |
| Qwen | `qwen:model-name` | `qwen:qwen-plus` |

## Usage via FlyBrowser SDK

```python
from flybrowser import FlyBrowser

# OpenAI
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://example.com")

# Anthropic
async with FlyBrowser(llm_provider="anthropic", api_key="sk-ant-...") as browser:
    await browser.goto("https://example.com")

# Local Ollama (no API key)
async with FlyBrowser(llm_provider="ollama", llm_model="qwen3:8b") as browser:
    await browser.goto("https://example.com")
```

## How It Works Internally

In `BrowserAgent.__init__`:

```python
self._agent = FireflyAgent(
    name="flybrowser",
    model=config.model,  # e.g., "openai:gpt-4o"
    instructions=_SYSTEM_INSTRUCTIONS,
    tools=self._toolkits,
    middleware=self._middleware,
)
```

The `FireflyAgent` parses the model string to select the appropriate provider and handles all LLM communication, including:

- API authentication
- Structured output / JSON mode
- Tool calling / function calling
- Vision / multimodal inputs
- Streaming responses
- Retry logic and error handling

## See Also

- [Architecture Overview](overview.md) - System architecture
- [ReAct Framework](react.md) - How the agent reasons with LLMs
