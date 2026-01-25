# FlyBrowser Documentation

FlyBrowser is a browser automation and web scraping framework that uses Large Language Models to enable natural language control of web browsers. Instead of writing complex selectors and handling the intricacies of web page interaction manually, you describe what you want in plain English and FlyBrowser figures out how to accomplish it.

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

FlyBrowser works with multiple LLM providers:

- **OpenAI** - GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google** - Gemini 2.0 Flash, Gemini 1.5 Pro
- **Ollama** - Local models like Qwen3, Llama 3.2, Gemma 3

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

- [Natural Language Actions](features/natural-language-actions.md) - The act() method in depth
- [Intelligent Extraction](features/intelligent-extraction.md) - The extract() method in depth
- [Autonomous Agent](features/autonomous-agent.md) - The agent() method for complex tasks
- [Element Observation](features/element-observation.md) - The observe() method
- [Navigation](features/navigation.md) - URL and natural language navigation
- [Screenshots and Recording](features/screenshots-recording.md) - Capturing browser sessions
- [Live Streaming](features/live-streaming.md) - HLS, DASH, and RTMP streaming
- [PII Protection](features/pii-protection.md) - Secure credential handling

### Architecture

- [System Overview](architecture/overview.md) - High-level architecture
- [ReAct Framework](architecture/react-framework.md) - The reasoning and acting loop
- [Tool System](architecture/tool-system.md) - How browser tools work
- [Memory System](architecture/memory-system.md) - Context and memory management
- [LLM Integration](architecture/llm-integration.md) - Provider abstraction and capabilities
- [Response Validation](architecture/response-validation.md) - Ensuring quality outputs

### Deployment

- [Embedded Mode](deployment/embedded-mode.md) - Running in scripts and notebooks
- [Standalone Server](deployment/standalone-server.md) - Single server deployment
- [Cluster Mode](deployment/cluster-mode.md) - Distributed high-availability deployment
- [Docker](deployment/docker.md) - Container deployment
- [Kubernetes](deployment/kubernetes.md) - Orchestrated deployment

### Reference

- [SDK Reference](reference/sdk.md) - Complete Python API documentation
- [REST API Reference](reference/rest-api.md) - HTTP endpoint documentation
- [CLI Reference](reference/cli.md) - Command-line tools
- [Configuration](reference/configuration.md) - All configuration options
- [Environment Variables](reference/environment-variables.md) - Environment variable reference

### Advanced Topics

- [Custom Tools](advanced/custom-tools.md) - Creating your own browser tools
- [Custom LLM Providers](advanced/custom-llm-providers.md) - Adding new LLM backends
- [Performance Tuning](advanced/performance-tuning.md) - Optimization strategies
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

This documentation covers FlyBrowser version 1.26.1.

## License

FlyBrowser is licensed under the Apache License 2.0. See the LICENSE file for details.
