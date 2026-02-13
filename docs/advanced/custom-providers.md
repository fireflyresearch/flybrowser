# Custom LLM Providers

FlyBrowser uses **fireflyframework-genai** for LLM orchestration. The framework supports OpenAI, Anthropic, Gemini, Ollama, and Qwen out of the box via the `FireflyAgent` model parameter.

## Supported Providers

Configure the provider via the `llm_provider` and `llm_model` parameters on `FlyBrowser`:

```python
from flybrowser import FlyBrowser

# OpenAI
browser = FlyBrowser(llm_provider="openai", llm_model="gpt-4o", api_key="sk-...")

# Anthropic
browser = FlyBrowser(llm_provider="anthropic", llm_model="claude-sonnet-4-5-20250929", api_key="sk-ant-...")

# Ollama (local, no API key)
browser = FlyBrowser(llm_provider="ollama", llm_model="qwen3:8b")

# Gemini
browser = FlyBrowser(llm_provider="gemini", llm_model="gemini-2.0-flash", api_key="AIza...")

# Qwen
browser = FlyBrowser(llm_provider="qwen", llm_model="qwen-plus", api_key="sk-...")
```

## Custom Provider Support

For custom or self-hosted LLM endpoints, you can use any OpenAI-compatible API by configuring the model string and base URL through the framework. Consult the fireflyframework-genai documentation for details on registering custom model providers.

## See Also

- [LLM Integration Architecture](../architecture/llm-integration.md) - How LLM providers are used
- [Configuration Reference](../reference/configuration.md) - Configuration options
