# Embedded Deployment

Embedded deployment runs FlyBrowser directly within your Python application. This is the simplest deployment mode and ideal for development, testing, and single-process applications.

## Overview

In embedded mode:
- Browser instances run in the same process as your application
- No separate server required
- Direct API access via Python SDK
- Automatic resource management with context managers

## When to Use

Embedded deployment is best for:

- Development and testing
- Simple automation scripts
- Single-user applications
- Applications with light browser usage
- Integration into existing Python applications

## Basic Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        headless=True,
        llm_provider="openai",
        llm_model="gpt-4o",
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("Get the page title")
        print(data)

asyncio.run(main())
```

## Configuration Options

### Browser Settings

```python
browser = FlyBrowser(
    # Browser options
    headless=True,                    # Run without visible window
    browser_type="chromium",          # chromium, firefox, webkit
    slow_mo=100,                      # Slow down operations (ms)
    timeout=30000,                    # Default timeout (ms)
    
    # Proxy settings
    proxy={
        "server": "http://proxy.example.com:8080",
        "username": "user",
        "password": "pass",
    },
    
    # Viewport
    viewport={"width": 1920, "height": 1080},
)
```

### LLM Settings

```python
browser = FlyBrowser(
    # LLM provider
    llm_provider="openai",            # openai, anthropic, ollama, gemini
    llm_model="gpt-4o",               # Model name
    llm_api_key="sk-...",             # API key (or use env var)
    
    # Agent settings
    max_steps=30,                     # Max agent steps
    max_retries=3,                    # Retries on failure
    temperature=0.2,                  # LLM temperature
    
    # Vision
    vision_enabled=True,              # Enable screenshot analysis
)
```

### Recording Settings

```python
browser = FlyBrowser(
    # Recording
    recording_enabled=True,
    recording_output_dir="./recordings",
    recording_format="mp4",
    recording_fps=30,
)
```

## Resource Management

Always use context managers to ensure proper cleanup:

```python
# Recommended: async context manager
async with FlyBrowser(...) as browser:
    # Use browser
    ...
# Browser automatically closed

# Alternative: manual management
browser = FlyBrowser(...)
try:
    await browser.start()
    # Use browser
    ...
finally:
    await browser.close()
```

## Concurrent Sessions

Run multiple browser sessions in parallel:

```python
import asyncio
from flybrowser import FlyBrowser

async def scrape_page(url):
    async with FlyBrowser(headless=True) as browser:
        await browser.goto(url)
        return await browser.extract("Get main content")

async def main():
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]
    
    # Run concurrently
    results = await asyncio.gather(*[
        scrape_page(url) for url in urls
    ])
    
    for url, result in zip(urls, results):
        print(f"{url}: {result}")

asyncio.run(main())
```

## Memory Management

For long-running applications, manage memory:

```python
async def process_many_pages(urls):
    # Process in batches to limit memory usage
    batch_size = 10
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        
        async with FlyBrowser(headless=True) as browser:
            for url in batch:
                await browser.goto(url)
                # Process page
                ...
        
        # Browser closed, memory freed
        # Optionally: await asyncio.sleep(1)  # Brief pause
```

## Error Handling

```python
from flybrowser import FlyBrowser
from flybrowser.exceptions import (
    BrowserError,
    NavigationError,
    TimeoutError,
    LLMError,
)

async def safe_automation():
    try:
        async with FlyBrowser(headless=True) as browser:
            await browser.goto("https://example.com")
            return await browser.extract("Get data")
            
    except NavigationError as e:
        print(f"Navigation failed: {e}")
        return None
        
    except TimeoutError as e:
        print(f"Operation timed out: {e}")
        return None
        
    except LLMError as e:
        print(f"LLM error: {e}")
        return None
        
    except BrowserError as e:
        print(f"Browser error: {e}")
        return None
```

## Environment Variables

Configure via environment variables:

```bash
# LLM
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Browser
export FLYBROWSER_HEADLESS=true
export FLYBROWSER_BROWSER_TYPE=chromium
export FLYBROWSER_TIMEOUT=30000

# Logging
export FLYBROWSER_LOG_LEVEL=INFO
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from flybrowser import FlyBrowser

app = FastAPI()

@app.post("/extract")
async def extract_data(url: str, instruction: str):
    async with FlyBrowser(headless=True) as browser:
        await browser.goto(url)
        data = await browser.extract(instruction)
        return {"data": data}
```

### Flask Integration (Async)

```python
from flask import Flask
import asyncio
from flybrowser import FlyBrowser

app = Flask(__name__)

def run_async(coro):
    return asyncio.run(coro)

@app.route("/extract")
def extract_data():
    async def do_extract():
        async with FlyBrowser(headless=True) as browser:
            await browser.goto("https://example.com")
            return await browser.extract("Get title")
    
    return {"data": run_async(do_extract())}
```

### Django Integration

```python
# views.py
import asyncio
from django.http import JsonResponse
from flybrowser import FlyBrowser

def extract_view(request):
    url = request.GET.get("url")
    
    async def do_extract():
        async with FlyBrowser(headless=True) as browser:
            await browser.goto(url)
            return await browser.extract("Get main content")
    
    data = asyncio.run(do_extract())
    return JsonResponse({"data": data})
```

## Limitations

Embedded deployment has some limitations:

1. **Single process** - All browsers share process resources
2. **Scaling** - Limited to available CPU/memory on single machine
3. **Availability** - No automatic failover
4. **Persistence** - State lost on process restart

For production workloads requiring scaling or high availability, consider [Standalone](standalone.md) or [Cluster](cluster.md) deployment.

## See Also

- [Standalone Deployment](standalone.md) - Server-based deployment
- [Cluster Deployment](cluster.md) - High-availability deployment
- [Configuration Reference](../reference/configuration.md) - All configuration options
