# Performance Optimization

This guide covers techniques for optimizing FlyBrowser performance in production environments.

## Browser Performance

### Headless Mode

Always use headless mode in production:

```python
async with FlyBrowser(headless=True) as browser:
    ...
```

Headless mode:
- Uses less memory (no GPU rendering)
- Faster startup
- No display server required

### Browser Pool Sizing

Configure pool size based on your workload:

```bash
# Light usage (1-2 concurrent tasks)
FLYBROWSER_POOL__MIN_SIZE=1
FLYBROWSER_POOL__MAX_SIZE=3

# Medium usage (5-10 concurrent tasks)
FLYBROWSER_POOL__MIN_SIZE=2
FLYBROWSER_POOL__MAX_SIZE=10

# Heavy usage (20+ concurrent tasks)
FLYBROWSER_POOL__MIN_SIZE=5
FLYBROWSER_POOL__MAX_SIZE=30
```

**Memory estimate:** ~300-500MB per browser instance

### Session Management

Reuse sessions for related operations:

```python
# Inefficient: New browser for each operation
for url in urls:
    async with FlyBrowser() as browser:
        await browser.goto(url)
        data = await browser.extract(instruction)

# Efficient: Reuse browser session
async with FlyBrowser() as browser:
    for url in urls:
        await browser.goto(url)
        data = await browser.extract(instruction)
```

### Session Timeouts

Configure appropriate timeouts:

```bash
# Close idle browsers quickly
FLYBROWSER_POOL__IDLE_TIMEOUT_SECONDS=120

# Recycle long-running sessions
FLYBROWSER_POOL__MAX_SESSION_AGE_SECONDS=1800
```

## LLM Performance

### Model Selection

Choose models based on task complexity:

| Task | Recommended Model | Speed |
|------|-------------------|-------|
| Simple extraction | gpt-4o-mini | Fast |
| Complex reasoning | gpt-4o | Medium |
| Vision tasks | gpt-4o | Medium |
| Local/private | ollama (llama3) | Varies |

### Temperature Settings

Lower temperature = faster, more deterministic:

```python
# Fast, deterministic operations
browser = FlyBrowser(temperature=0.1)

# Creative tasks (slower)
browser = FlyBrowser(temperature=0.7)
```

### Token Management

Limit token usage:

```python
# SDK level
browser = FlyBrowser(
    max_tokens=2048,  # Limit output tokens
)

# Per-operation
await browser.extract(
    instruction="Get the title",
    max_tokens=500,  # Short responses
)
```

### Token Budget Awareness

FlyBrowser uses `BrowserMemoryManager` to keep context within LLM limits. The memory system automatically formats page history, navigation graph, and obstacle cache into a prompt-friendly representation.

For large content, use more specific extraction instructions:

```python
# Instead of extracting entire page
await browser.extract("Get all text")  # May overflow

# Extract specific sections
await browser.extract("Get only the main article content")
```

### Vision Optimization

Reduce image sizes for vision tasks:

```python
# Take optimized screenshots
screenshot = await browser.screenshot(
    quality=80,        # JPEG quality (0-100)
    full_page=False,   # Viewport only
)

# Or resize before sending
from PIL import Image
img = Image.open(BytesIO(screenshot))
img.thumbnail((1280, 720))  # Max dimensions
```

## Agent Performance

### Step Limits

Set appropriate step limits:

```python
# Simple tasks
result = await browser.agent(
    task="Click the login button",
    max_steps=5,
)

# Complex multi-step tasks
result = await browser.agent(
    task="Navigate site and extract data",
    max_steps=20,
)
```

### Reasoning Strategy

Choose appropriate reasoning strategy:

```python
from flybrowser.agents.types import ReasoningStrategy

# Fast, simple reasoning
browser = FlyBrowser(
    reasoning_strategy=ReasoningStrategy.REACT_STANDARD,
)

# Thorough reasoning (slower)
browser = FlyBrowser(
    reasoning_strategy=ReasoningStrategy.REACT_PLUS,
)
```

### Operation Modes

Use specific operation modes:

```python
from flybrowser.agents.types import OperationMode

# Extraction-focused (optimized for data retrieval)
result = await browser.agent(
    task="Get all product prices",
    mode=OperationMode.SCRAPE,
)

# Action-focused (optimized for interactions)
result = await browser.agent(
    task="Fill out the contact form",
    mode=OperationMode.EXECUTE,
)
```

## Network Performance

### Proxy Configuration

Use proxies for distributed scraping:

```python
browser = FlyBrowser(
    proxy={
        "server": "http://proxy.example.com:8080",
    }
)
```

### Request Interception

Block unnecessary resources:

```python
async with FlyBrowser() as browser:
    # Block images, fonts, media
    await browser.page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,mp4}", 
                             lambda route: route.abort())
    
    await browser.goto("https://example.com")
```

### Timeout Tuning

Adjust timeouts for your network:

```python
browser = FlyBrowser(
    timeout=15000,           # Default timeout (ms)
    navigation_timeout=30000, # Navigation timeout (ms)
)
```

## Memory Management

### Batch Processing

Process in batches to control memory:

```python
async def process_urls(urls: list, batch_size: int = 10):
    results = []
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        
        async with FlyBrowser(headless=True) as browser:
            for url in batch:
                await browser.goto(url)
                data = await browser.extract("Get content")
                results.append(data)
        
        # Memory freed after context manager
        
    return results
```

### Garbage Collection

Force garbage collection between heavy operations:

```python
import gc

async def heavy_operation():
    async with FlyBrowser() as browser:
        # Heavy processing
        ...
    
    # Force cleanup
    gc.collect()
```

### Context Clearing

Clear browser context periodically:

```python
async with FlyBrowser() as browser:
    for i, url in enumerate(urls):
        await browser.goto(url)
        await browser.extract("Get data")
        
        # Clear cookies/storage every 50 pages
        if (i + 1) % 50 == 0:
            await browser.context.clear_cookies()
```

## Concurrency

### Parallel Operations

Run operations in parallel:

```python
import asyncio

async def scrape_url(url: str):
    async with FlyBrowser(headless=True) as browser:
        await browser.goto(url)
        return await browser.extract("Get title")

async def main():
    urls = ["https://example1.com", "https://example2.com", ...]
    
    # Run up to 5 concurrent operations
    semaphore = asyncio.Semaphore(5)
    
    async def limited_scrape(url):
        async with semaphore:
            return await scrape_url(url)
    
    results = await asyncio.gather(*[
        limited_scrape(url) for url in urls
    ])
```

### Connection Pooling

Use connection pools for REST API:

```python
from flybrowser import FlyBrowserClient

# Reuse client with connection pool
client = FlyBrowserClient(
    "http://localhost:8000",
    pool_size=10,  # Connection pool size
)

async def main():
    tasks = [
        client.extract(session_id, instruction)
        for session_id in session_ids
    ]
    results = await asyncio.gather(*tasks)
```

## Monitoring

### Performance Metrics

Track key metrics:

```python
import time

async def timed_operation():
    start = time.time()
    
    async with FlyBrowser() as browser:
        await browser.goto("https://example.com")
        result = await browser.extract("Get content")
    
    duration = time.time() - start
    print(f"Operation took {duration:.2f}s")
    
    # Log to monitoring system
    metrics.histogram("flybrowser.operation.duration", duration)
```

### LLM Cost Tracking

Monitor LLM costs:

```python
async with FlyBrowser() as browser:
    await browser.goto("https://example.com")
    await browser.extract("Get data")
    
    # Get usage statistics
    usage = browser.get_llm_usage()
    print(f"Tokens: {usage['total_tokens']}")
    print(f"Cost: ${usage['cost_usd']:.4f}")
```

### Health Monitoring

Monitor server health:

```bash
# Prometheus endpoint
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

## Best Practices

### 1. Use Appropriate Tools

| Task | Method |
|------|--------|
| Simple data extraction | `extract()` |
| Single action | `act()` |
| Multi-step workflow | `agent()` |
| Page inspection | `observe()` |

### 2. Minimize LLM Calls

```python
# Inefficient: Multiple extract calls
title = await browser.extract("Get title")
content = await browser.extract("Get content")
links = await browser.extract("Get links")

# Efficient: Single structured extraction
data = await browser.extract(
    "Get title, content summary, and navigation links",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "links": {"type": "array", "items": {"type": "string"}}
        }
    }
)
```

### 3. Reuse Sessions

```python
# Create session once
async with FlyBrowser() as browser:
    # Reuse for multiple pages
    for url in urls:
        await browser.goto(url)
        # Process...
```

### 4. Use Specific Instructions

```python
# Vague (slower, less reliable)
await browser.extract("Get the data")

# Specific (faster, more reliable)
await browser.extract(
    "Get the product name and price from the main content area"
)
```

### 5. Handle Errors Gracefully

```python
async def resilient_extract(browser, url, instruction, retries=3):
    for attempt in range(retries):
        try:
            await browser.goto(url)
            return await browser.extract(instruction)
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## See Also

- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Configuration Reference](../reference/configuration.md) - All settings
- [Deployment Guide](../deployment/standalone.md) - Production deployment
