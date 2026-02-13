# FlyBrowser Documentation

FlyBrowser is a browser automation and web scraping framework built on top of [fireflyframework-genai](https://github.com/fireflyframework/fireflyframework-genai) that uses Large Language Models to enable natural language control of web browsers. Instead of writing complex selectors and handling the intricacies of web page interaction manually, you describe what you want in plain English and FlyBrowser figures out how to accomplish it.

## What Makes FlyBrowser Different

Traditional browser automation requires you to know the exact structure of web pages: CSS selectors, XPath expressions, waiting for specific elements, handling dynamic content, and dealing with the countless edge cases that make web automation brittle. FlyBrowser takes a fundamentally different approach.

When you tell FlyBrowser to "click the login button", it uses an LLM to understand the page structure, identify the relevant element, and perform the action. When you ask it to "extract all product prices from this page", it reasons about the page layout and returns structured data. This natural language interface means your automation scripts are more readable, more maintainable, and more resilient to website changes.

## Core Capabilities

FlyBrowser provides several primary methods for browser automation:

**Navigation and Actions**

- `goto(url)` - Navigate directly to a URL
- `navigate(instruction)` - Navigate using natural language ("go to the login page")
- `act(instruction)` - Perform actions ("click the submit button", "type hello in the search box")

**Data Extraction**

- `extract(query, schema)` - Extract data from pages using natural language queries with optional JSON Schema validation
- `observe(query)` - Find and analyze elements on the page

**Autonomous Operations**

- `agent(task, context)` - Execute complex multi-step tasks autonomously
- `execute_task(task)` - Run tasks with full ReAct reasoning

**Recording and Streaming**

- `screenshot()` - Capture page screenshots
- `start_recording()` / `stop_recording()` - Record browser sessions
- `start_stream()` / `stop_stream()` - Live stream browser sessions via HLS, DASH, or RTMP

## Deployment Modes

FlyBrowser runs in three modes, all using the same API:

**Embedded Mode** - The browser runs directly in your Python process. Ideal for scripts, notebooks, and development.

```python
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://example.com")
    data = await browser.extract("Get the main heading")
```

**Standalone Server Mode** - FlyBrowser runs as an HTTP server, allowing multiple clients to create and manage browser sessions.

```python
async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    await browser.goto("https://example.com")
    data = await browser.extract("Get the main heading")
```

**Cluster Mode** - Multiple FlyBrowser nodes coordinate via Raft consensus for high availability, automatic failover, and load balancing.

The same code works in all three modes. The only difference is how you initialize the FlyBrowser instance.

## Supported LLM Providers

LLM orchestration is handled by [fireflyframework-genai](https://github.com/fireflyframework/fireflyframework-genai). FlyBrowser works with multiple providers:

- **OpenAI** — GPT-5.2, GPT-5-mini, GPT-4o, GPT-4o-mini
- **Anthropic** — Claude 4.5 Sonnet, Claude 3.5 Sonnet
- **Google** — Gemini 2.0 Flash, Gemini 1.5 Pro
- **Qwen** — Qwen3, Qwen-Plus, Qwen-VL
- **Ollama** — Local models like Qwen3, Llama 3.2, Gemma 3

Vision-capable models enable FlyBrowser to analyze screenshots for better understanding of complex page layouts.

## Documentation Sections

### Getting Started

- [Installation](getting-started/installation.md) - System requirements and installation methods
- [Quickstart](getting-started/quickstart.md) - Your first automation in five minutes
- [Core Concepts](getting-started/concepts.md) - Understanding sessions, actions, and the ReAct framework

### User Guides

- [Basic Automation](guides/basic-automation.md) - Navigation, clicking, typing, and scrolling
- [Data Extraction](guides/data-extraction.md) - Extracting structured data from web pages
- [Form Automation](guides/form-automation.md) - Filling and submitting forms
- [Multi-Page Workflows](guides/multi-page-workflows.md) - Complex tasks spanning multiple pages
- [Authentication](guides/authentication.md) - Handling login flows securely
- [Error Handling](guides/error-handling.md) - Dealing with failures and retries

### Features

- [Act](features/act.md) - The act() method in depth
- [Extract](features/extract.md) - The extract() method in depth
- [Agent](features/agent.md) - The agent() method for complex tasks
- [Observe](features/observe.md) - The observe() method
- [Navigation](features/navigation.md) - URL and natural language navigation
- [Screenshots](features/screenshots.md) - Capturing browser sessions
- [Streaming](features/streaming.md) - HLS, DASH, and RTMP streaming
- [PII Handling](features/pii.md) - Secure credential handling
- [Search](features/search.md) - Multi-provider web search with intelligent ranking
- [Stealth Mode](features/stealth.md) - Fingerprint generation, CAPTCHA solving, and proxy network
- [Obstacle Detection](features/obstacle-detection.md) - Automatic popup/modal dismissal
- [Observability](features/observability.md) - Command logging, source capture, live view, and completion page

### Architecture

- [System Overview](architecture/overview.md) - High-level architecture
- [ReAct Framework](architecture/react.md) - The reasoning and acting loop
- [Tools System](architecture/tools.md) - ToolKit architecture
- [Memory System](architecture/memory.md) - BrowserMemoryManager details
- [LLM Integration](architecture/llm-integration.md) - fireflyframework-genai provider delegation
- [Response Validation](architecture/validation.md) - Ensuring quality outputs
- [Context System](architecture/context.md) - Context building and management

### Deployment

- [Embedded Mode](deployment/embedded.md) - Running in scripts and notebooks
- [Standalone Server](deployment/standalone.md) - Single server deployment
- [Cluster Mode](deployment/cluster.md) - Distributed high-availability deployment
- [Docker](deployment/docker.md) - Container deployment
- [Kubernetes](deployment/kubernetes.md) - Orchestrated deployment

### Reference

- [SDK Reference](reference/sdk.md) - Complete Python API documentation
- [REST API Reference](reference/rest-api.md) - HTTP endpoint documentation
- [CLI Reference](reference/cli.md) - Command-line tools
- [Configuration](reference/configuration.md) - All configuration options

### Advanced Topics

- [Custom Tools](advanced/custom-tools.md) - Creating your own browser tools
- [Custom Providers](advanced/custom-providers.md) - Adding LLM provider support
- [Performance](advanced/performance.md) - Optimization strategies
- [Troubleshooting](advanced/troubleshooting.md) - Common issues and solutions

### Examples

- [Examples Overview](examples/index.md) - Guide to example code
- [Web Scraping](examples/web-scraping.md) - Scraping examples
- [UI Testing](examples/ui-testing.md) - Testing examples
- [Workflow Automation](examples/workflow-automation.md) - End-to-end automation examples

## Quick Example

Here is a complete example that navigates to a website, performs some actions, and extracts data:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=True,
    ) as browser:
        # Navigate to the page
        await browser.goto("https://news.ycombinator.com")
        
        # Extract structured data
        result = await browser.extract(
            "Get the titles and scores of the top 5 stories",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "score": {"type": "integer"}
                    }
                }
            }
        )
        
        if result.success:
            for story in result.data:
                print(f"{story['title']} - {story['score']} points")
        
        # Show execution statistics
        result.pprint()

asyncio.run(main())
```

## Version

This documentation covers FlyBrowser version 26.02.01 (CalVer). Requires Python 3.13+.

## License

FlyBrowser is licensed under the Apache License 2.0. See the LICENSE file for details.
