# Navigation

FlyBrowser provides multiple ways to navigate web pages, from direct URL navigation to intelligent natural language instructions.

## Direct Navigation

### goto()

Navigate directly to a URL:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        # Basic navigation
        await browser.goto("https://example.com")
        
        # Wait for network to be idle
        await browser.goto("https://example.com", wait_until="networkidle")

asyncio.run(main())
```

### Wait Until Options

| Value | Description |
|-------|-------------|
| `"load"` | Wait for the load event |
| `"domcontentloaded"` | Wait for DOMContentLoaded (default) |
| `"networkidle"` | Wait until no network activity for 500ms |

## Natural Language Navigation

### navigate()

Use natural language to describe where to go:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://shop.example.com")
    
    # Navigate using natural language
    result = await browser.navigate("go to the login page")
    
    # Click navigation elements
    result = await browser.navigate("click on Products in the main menu")
    
    # Find and navigate
    result = await browser.navigate("find the contact page")
```

### Method Signature

```python
async def navigate(
    instruction: str,
    use_vision: bool = True
) -> dict
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction` | `str` | Required | Natural language navigation instruction |
| `use_vision` | `bool` | `True` | Use screenshot for visual context |

### Returns

Dictionary with:
- `success` - Whether navigation succeeded
- `url` - Current URL after navigation
- `title` - Page title after navigation
- `error` - Error message if failed

## Navigation with act()

For click-based navigation:

```python
# Click navigation links
await browser.act("click the About Us link")

# Click menu items
await browser.act("click Products in the header menu")

# Click with context
await browser.act("click the first item in the sidebar navigation")
```

## Browser History

### Going Back and Forward

```python
# Go back
await browser.act("go back")
await browser.act("click the back button")

# Go forward
await browser.act("go forward")

# Refresh
await browser.act("refresh the page")
```

## Page Load Strategies

### Waiting for Content

```python
import asyncio

# Wait after navigation
await browser.goto("https://example.com")
await asyncio.sleep(2)  # Wait for dynamic content

# Navigate with networkidle for SPAs
await browser.goto("https://spa-example.com", wait_until="networkidle")
```

### Handling Slow Pages

```python
# For slow-loading pages
await browser.goto("https://slow-site.com", wait_until="networkidle")

# Or wait explicitly
await browser.goto("https://slow-site.com")
await asyncio.sleep(5)
```

## Navigation Patterns

### Multi-Step Navigation

```python
async def navigate_to_product(browser, category, product_name):
    await browser.goto("https://shop.example.com")
    
    # Navigate through categories
    await browser.navigate(f"go to the {category} category")
    
    # Find specific product
    await browser.navigate(f"find {product_name}")
```

### Breadcrumb Navigation

```python
async def navigate_via_breadcrumbs(browser):
    # Extract current breadcrumb path
    breadcrumbs = await browser.extract("get the breadcrumb navigation")
    
    # Navigate to parent
    await browser.act("click the parent category in the breadcrumbs")
```

### Handling Redirects

```python
async def handle_redirects(browser, url):
    await browser.goto(url)
    
    # Check final URL after redirects
    result = await browser.extract("what is the current URL?")
    print(f"Ended up at: {result.data}")
```

## Error Handling

```python
async def safe_navigation(browser, url):
    try:
        await browser.goto(url)
    except Exception as e:
        error = str(e).lower()
        
        if "timeout" in error:
            print("Page took too long to load")
        elif "dns" in error or "resolve" in error:
            print("Site not found")
        elif "ssl" in error:
            print("SSL certificate error")
        else:
            print(f"Navigation error: {e}")
```

## Best Practices

### Use Appropriate Wait Strategies

```python
# For static sites
await browser.goto(url, wait_until="domcontentloaded")

# For dynamic/SPA sites
await browser.goto(url, wait_until="networkidle")

# For sites with heavy JavaScript
await browser.goto(url, wait_until="networkidle")
await asyncio.sleep(1)  # Extra wait for JS execution
```

### Verify Navigation Success

```python
await browser.goto("https://example.com/login")

# Verify we're on the right page
result = await browser.extract("what page am I on?")
if "login" not in str(result.data).lower():
    print("Navigation may have failed")
```

### Handle Dynamic URLs

```python
async def navigate_with_params(browser, base_url, params):
    from urllib.parse import urlencode
    url = f"{base_url}?{urlencode(params)}"
    await browser.goto(url)
```

## Operation Mode

The `navigate()` method uses `NAVIGATE` operation mode, optimized for:
- High-frequency vision (screenshots every 2 iterations)
- Comprehensive page exploration
- Detailed memory of site structure

## Related Methods

- [act()](act.md) - Click-based navigation
- [agent()](agent.md) - Multi-step navigation tasks
- [observe()](observe.md) - Find navigation elements

## See Also

- [Basic Automation Guide](../guides/basic-automation.md) - Navigation examples
- [Multi-Page Workflows](../guides/multi-page-workflows.md) - Complex navigation patterns
