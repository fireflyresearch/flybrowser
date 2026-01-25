# Examples

This section provides practical, working examples demonstrating FlyBrowser's capabilities across common automation scenarios. Each example includes complete code with detailed explanations.

## Overview

FlyBrowser excels at three primary use cases:

1. **Web Scraping** - Extracting structured data from websites
2. **UI Testing** - Automated testing of web applications
3. **Workflow Automation** - Multi-step business process automation

## Quick Reference

### SDK Methods

| Method | Purpose | Operation Mode |
|--------|---------|----------------|
| `goto(url)` | Direct navigation to URL | - |
| `navigate(instruction)` | Natural language navigation | NAVIGATE |
| `act(instruction)` | Single action execution | EXECUTE |
| `extract(query)` | Data extraction | SCRAPE |
| `observe(query)` | Find page elements | RESEARCH |
| `agent(task)` | Complex multi-step tasks | AUTO |

### Basic Pattern

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com")
        result = await browser.extract("Get the main heading")
        print(result.data)

asyncio.run(main())
```

## Examples by Category

### Web Scraping Examples

- [Single Page Extraction](#) - Extract data from a single page
- [Multi-Page Scraping](#) - Navigate and scrape multiple pages
- [Structured Data Extraction](#) - Extract data with JSON schemas
- [Dynamic Content Scraping](#) - Handle JavaScript-rendered content

See [Web Scraping Examples](web-scraping.md) for complete code.

### UI Testing Examples

- [Form Validation Testing](#) - Test form inputs and validation
- [Navigation Testing](#) - Verify navigation flows
- [Visual Regression](#) - Capture screenshots for comparison
- [Authentication Testing](#) - Test login flows

See [UI Testing Examples](ui-testing.md) for complete code.

### Workflow Automation Examples

- [E-commerce Checkout](#) - Automate shopping workflows
- [Form Filling](#) - Automate multi-step forms
- [Report Generation](#) - Navigate, extract, and compile data
- [Monitoring Tasks](#) - Periodic checks and alerts

See [Workflow Automation Examples](workflow-automation.md) for complete code.

## Running the Examples

### Prerequisites

1. Install FlyBrowser:

```bash
pip install flybrowser
```

2. Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Running Examples

Each example can be run directly:

```bash
python example_name.py
```

Or from the examples directory:

```bash
cd examples/
python scraping/hackernews.py
```

## Example Structure

All examples follow a consistent structure:

```python path=null start=null
"""
Example: [Title]

Description of what this example demonstrates.

Prerequisites:
- Any specific requirements
"""

import asyncio
from flybrowser import FlyBrowser

# Configuration
CONFIG = {
    "llm_provider": "openai",
    # Additional settings
}

async def main():
    """Main example function."""
    async with FlyBrowser(**CONFIG) as browser:
        # Example implementation
        pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Error Handling

Always handle potential failures:

```python path=null start=null
async with FlyBrowser(...) as browser:
    try:
        result = await browser.extract("Get data")
        if result.success:
            print(result.data)
        else:
            print(f"Extraction failed: {result.error}")
    except Exception as e:
        print(f"Error: {e}")
```

### Resource Management

Use context managers for automatic cleanup:

```python path=null start=null
# Recommended: automatic cleanup
async with FlyBrowser(...) as browser:
    await browser.goto("https://example.com")
# Browser automatically closed

# Manual management (if needed)
browser = FlyBrowser(...)
await browser.start()
try:
    await browser.goto("https://example.com")
finally:
    await browser.stop()  # Always cleanup
```

### Logging

Enable appropriate logging for debugging:

```python path=null start=null
# Verbose logging for development
browser = FlyBrowser(
    log_verbosity="verbose",
    pretty_logs=True,
)

# Minimal logging for production
browser = FlyBrowser(
    log_verbosity="minimal",
    pretty_logs=False,
)
```

### Performance

Optimize for your use case:

```python path=null start=null
# Fast execution (single actions)
browser = FlyBrowser(
    speed_preset="fast",
)

# Thorough execution (complex pages)
browser = FlyBrowser(
    speed_preset="thorough",
)
```

## Troubleshooting Examples

### Common Issues

**Example not working?**

1. Check API key is set correctly
2. Verify network connectivity
3. Try increasing `max_iterations` for complex tasks
4. Enable `log_verbosity="debug"` for detailed output

**Timeout errors?**

```python path=null start=null
# Increase timeouts for slow sites
browser = FlyBrowser(
    timeout=60.0,
    speed_preset="thorough",
)
```

**Element not found?**

```python path=null start=null
# Use observe to debug element detection
elements = await browser.observe("find the submit button")
print(f"Found {len(elements.data)} elements")
for elem in elements.data:
    print(f"  - {elem.get('selector')}: {elem.get('text')}")
```

## Contributing Examples

We welcome example contributions. Please ensure:

1. Code runs without modification (except API keys)
2. Includes comprehensive comments
3. Follows the standard structure
4. Targets publicly accessible websites
5. Includes expected output examples

Submit examples via pull request to the `examples/` directory.

## Next Steps

- [Web Scraping Examples](web-scraping.md) - Data extraction patterns
- [UI Testing Examples](ui-testing.md) - Testing automation
- [Workflow Automation](workflow-automation.md) - Complex task automation
