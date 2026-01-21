# Embedded Mode Deployment

Embedded mode integrates FlyBrowser directly into your Python application as a library. This deployment mode provides the lowest latency and tightest integration, making it ideal for scripts, automation tools, and applications that require direct control over browser operations.

## Overview

In embedded mode, FlyBrowser runs within your application's process. The browser instance is created, managed, and destroyed by your code, giving you complete control over the automation lifecycle.

**Advantages:**
- No network overhead between your code and the browser
- Direct access to all SDK features
- Simplest deployment model
- Ideal for single-user applications

**Considerations:**
- Browser resources are tied to your application's lifecycle
- Scaling requires running multiple application instances
- Not suitable for multi-tenant scenarios

## Installation

Install FlyBrowser from source:

```bash path=null start=null
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh
```

## Basic Setup

### Minimal Configuration

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-api-key"
)
```

### Full Configuration

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    # LLM Configuration - supports all providers
    llm_provider="openai",       # "openai", "anthropic", "gemini", "google",
                                 # "ollama", "lm_studio", "localai", "vllm"
    llm_model="gpt-5.2",         # Model name (uses provider default if not set)
    # api_key="your-api-key",    # Or use environment variables
    # base_url=None,             # Custom endpoint for local providers
    
    # Browser Configuration
    headless=True,               # True for no visible window
    browser_type="chromium",     # "chromium", "firefox", or "webkit"
    timeout=30.0,                # Default timeout in seconds
    
    # Feature Flags
    recording_enabled=False,     # Enable session recording
    pii_masking_enabled=True     # Enable automatic PII masking (default)
)
```

### Provider Examples

**Google Gemini:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="gemini",
    llm_model="gemini-2.0-flash"
)
```

**Ollama (local):**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b"  # or gemma3:12b, llama3.2:3b, phi4
)
```

## Lifecycle Management

### Starting and Stopping

Always ensure proper cleanup of browser resources:

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def main():
browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2"
    )
    
    # Start initializes Playwright and launches the browser
    await browser.start()
    
    try:
        # Perform automation tasks
        await browser.goto("https://example.com")
        data = await browser.extract("Extract the page title")
        print(data)
        
    finally:
        # Stop closes the browser and releases resources
        await browser.stop()

asyncio.run(main())
```

### Context Manager Pattern

For cleaner resource management, use the async context manager:

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def main():
browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2"
    )
    
    async with browser:
        await browser.goto("https://example.com")
        data = await browser.extract("Extract the main content")
        print(data)

asyncio.run(main())
```

## Navigation Operations

### Direct URL Navigation

```python path=null start=null
# Navigate to a specific URL
await browser.goto("https://example.com")

# Navigate and wait for network idle
await browser.goto("https://example.com/dashboard")
```

### LLM-Powered Navigation

```python path=null start=null
# Natural language navigation
await browser.navigate("Go to the user settings page")

# Multi-step navigation
await browser.navigate("Click on Products, then select the first item")
```

## Data Extraction

### Simple Extraction

```python path=null start=null
# Extract text content
title = await browser.extract("What is the page title?")

# Extract specific elements
prices = await browser.extract("List all product prices on this page")
```

### Structured Extraction

```python path=null start=null
# Extract as structured data
product_info = await browser.extract(
    "Extract product name, price, and availability for each item"
)

# Extract with format specification
contact = await browser.extract(
    "Extract the contact information including email and phone number"
)
```

## Page Actions

### Form Interactions

```python path=null start=null
# Fill text fields
await browser.act("Type 'john.doe@example.com' in the email field")

# Click buttons
await browser.act("Click the Submit button")

# Select dropdowns
await browser.act("Select 'California' from the state dropdown")

# Check/uncheck boxes
await browser.act("Check the 'Remember me' checkbox")
```

### Complex Interactions

```python path=null start=null
# Hover actions
await browser.act("Hover over the user menu")

# Keyboard actions
await browser.act("Press Enter in the search field")

# Scroll actions
await browser.act("Scroll down to the footer")
```

## Workflows

Workflows define reusable automation sequences:

```python path=null start=null
login_workflow = {
    "name": "user_login",
    "steps": [
        {
            "action": "goto",
            "url": "https://example.com/login"
        },
        {
            "action": "act",
            "command": "Type 'user@example.com' in the email field"
        },
        {
            "action": "act",
            "command": "Type 'password123' in the password field"
        },
        {
            "action": "act",
            "command": "Click the Login button"
        },
        {
            "action": "extract",
            "query": "Confirm successful login by extracting the welcome message"
        }
    ]
}

result = await browser.run_workflow(login_workflow)
print(f"Workflow result: {result}")
```

## Security Features

### PII Masking

Enable automatic PII masking for all LLM interactions:

```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-api-key",
    pii_masking_enabled=True
)
```

Manually mask sensitive data:

```python path=null start=null
# Mask PII in text
original = "Contact John at john@example.com or 555-123-4567"
masked = await browser.mask_pii(original)
# Result: "Contact [NAME] at [EMAIL] or [PHONE]"
```

### Credential Management

Store and use credentials securely:

```python path=null start=null
# Store credentials (encrypted at rest)
await browser.store_credential("work_email", "employee@company.com")
await browser.store_credential("work_password", "secure_password")

# Use credentials for form filling
await browser.secure_fill("email", "work_email")
await browser.secure_fill("password", "work_password")
```

## Screenshots and Recording

### Capturing Screenshots

```python path=null start=null
# Capture current viewport
screenshot = await browser.screenshot()

# Save to file
with open("page_screenshot.png", "wb") as f:
    f.write(screenshot)
```

### Session Recording

Enable recording to capture all browser operations:

```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-api-key",
    recording_enabled=True
)

async with browser:
    await browser.goto("https://example.com")
    await browser.act("Click the Products link")
    # Recording is automatically saved when browser stops
```

## Monitoring

Monitor browser state during automation:

```python path=null start=null
# Get current browser status
status = await browser.monitor()

print(f"Current URL: {status.get('url')}")
print(f"Page title: {status.get('title')}")
print(f"Browser state: {status.get('state')}")
```

## Error Handling

### Timeout Handling

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-api-key",
    timeout=15000  # 15 second timeout
)

async with browser:
    try:
        await browser.goto("https://slow-loading-site.com")
        await browser.extract("Get the content")
    except TimeoutError:
        print("Page took too long to load or respond")
```

### Comprehensive Error Handling

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def robust_automation():
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4",
        api_key="your-api-key"
    )
    
    try:
        await browser.start()
        
        await browser.goto("https://example.com")
        
        result = await browser.extract("Extract data")
        return result
        
    except TimeoutError as e:
        print(f"Operation timed out: {e}")
        return None
        
    except ConnectionError as e:
        print(f"Connection failed: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
        
    finally:
        await browser.stop()

result = asyncio.run(robust_automation())
```

## Integration Patterns

### With Web Frameworks

Integration with FastAPI:

```python path=null start=null
from fastapi import FastAPI
from flybrowser import FlyBrowser

app = FastAPI()
browser = None

@app.on_event("startup")
async def startup():
    global browser
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4",
        api_key="your-api-key"
    )
    await browser.start()

@app.on_event("shutdown")
async def shutdown():
    global browser
    if browser:
        await browser.stop()

@app.post("/extract")
async def extract_data(url: str, query: str):
    await browser.goto(url)
    result = await browser.extract(query)
    return {"result": result}
```

### With Background Task Queues

Integration with Celery:

```python path=null start=null
import asyncio
from celery import Celery
from flybrowser import FlyBrowser

celery_app = Celery("tasks", broker="redis://localhost:6379")

@celery_app.task
def scrape_page(url: str, extraction_query: str):
    async def _scrape():
        browser = FlyBrowser(
            llm_provider="openai",
            llm_model="gpt-4",
            api_key="your-api-key"
        )
        
        async with browser:
            await browser.goto(url)
            return await browser.extract(extraction_query)
    
    return asyncio.run(_scrape())
```

### With Testing Frameworks

Integration with pytest:

```python path=null start=null
import pytest
from flybrowser import FlyBrowser

@pytest.fixture
async def browser():
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4",
        api_key="your-api-key",
        headless=True
    )
    await browser.start()
    yield browser
    await browser.stop()

@pytest.mark.asyncio
async def test_homepage_title(browser):
    await browser.goto("https://example.com")
    title = await browser.extract("What is the page title?")
    assert "Example" in title
```

## Performance Considerations

### Browser Reuse

Reuse browser instances for multiple operations to avoid startup overhead:

```python path=null start=null
async def process_urls(urls: list):
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4",
        api_key="your-api-key"
    )
    
    await browser.start()
    results = []
    
    try:
        for url in urls:
            await browser.goto(url)
            data = await browser.extract("Extract the main content")
            results.append(data)
    finally:
        await browser.stop()
    
    return results
```

### Headless Mode

Always use headless mode in production for better performance:

```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="your-api-key",
    headless=True  # No visible browser window
)
```

## Next Steps

- [Standalone Mode](standalone.md) - Deploy FlyBrowser as a service
- [Cluster Mode](cluster.md) - Deploy a distributed cluster
- [SDK Reference](../reference/sdk.md) - Complete SDK documentation
