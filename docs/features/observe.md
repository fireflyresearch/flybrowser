# observe() - Element Discovery

The `observe()` method finds and describes elements on a page without interacting with them. It is useful for understanding page structure before taking actions.

## Basic Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com")
        
        # Find elements matching a description
        result = await browser.observe("find all buttons on this page")
        
        print(f"Found {len(result.data)} buttons")
        for element in result.data:
            print(f"  - {element}")

asyncio.run(main())
```

## Method Signature

```python
async def observe(
    self,
    query: str,
    return_selectors: bool = False,
    return_metadata: bool = False,
) -> AgentRequestResponse
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Natural language description of what to find |
| `return_selectors` | `bool` | `False` | Include CSS selectors for found elements |
| `return_metadata` | `bool` | `False` | Include detailed execution metadata |

### Returns

`AgentRequestResponse` with:
- `success` - Whether observation succeeded
- `data` - List of found elements with descriptions
- `error` - Error message if failed
- `operation` - "observe"
- `query` - The original query

## Finding Elements

### By Type

```python
# Buttons
result = await browser.observe("find all buttons")

# Links
result = await browser.observe("find all navigation links")

# Forms
result = await browser.observe("find form fields")

# Images
result = await browser.observe("find all images with alt text")
```

### By Location

```python
# In specific area
result = await browser.observe("find buttons in the header")

# By position
result = await browser.observe("find the first input field")

# Relative to other elements
result = await browser.observe("find links near the product image")
```

### By Purpose

```python
# By function
result = await browser.observe("find the search functionality")

# By content
result = await browser.observe("find elements containing the price")

# By state
result = await browser.observe("find disabled buttons")
```

## Getting Selectors

Request CSS selectors for programmatic use:

```python
result = await browser.observe(
    "find the login button",
    return_selectors=True
)

if result.data:
    element = result.data[0]
    print(f"Element: {element.get('text', 'unknown')}")
    print(f"Selector: {element.get('selector', 'none')}")
    
    # Use selector directly if needed
    selector = element.get("selector")
    if selector:
        await browser.act(f"click element with selector {selector}")
```

## Response Format

### Basic Response

```python
result = await browser.observe("find all buttons")

# result.data is a list of elements
for element in result.data:
    print(element)
    # Each element contains descriptive information
```

### With Selectors

```python
result = await browser.observe(
    "find form inputs",
    return_selectors=True
)

for element in result.data:
    print(f"Type: {element.get('type', 'unknown')}")
    print(f"Name: {element.get('name', '')}")
    print(f"Placeholder: {element.get('placeholder', '')}")
    print(f"Selector: {element.get('selector', '')}")
```

### With Metadata

```python
result = await browser.observe(
    "find interactive elements",
    return_metadata=True
)

print(f"Found: {len(result.data)} elements")
print(f"Iterations: {result.execution.iterations}")
print(f"Duration: {result.execution.duration_seconds}s")
```

## Use Cases

### Understanding Page Structure

```python
async def analyze_page(browser, url):
    await browser.goto(url)
    
    # Find main sections
    sections = await browser.observe("find main content sections")
    
    # Find navigation
    nav = await browser.observe("find navigation menus")
    
    # Find forms
    forms = await browser.observe("find forms on this page")
    
    return {
        "sections": sections.data,
        "navigation": nav.data,
        "forms": forms.data
    }
```

### Before Taking Action

```python
async def smart_click(browser, description):
    # First, observe what's available
    result = await browser.observe(f"find elements matching: {description}")
    
    if not result.data:
        print(f"No elements found for: {description}")
        return None
    
    # Report what was found
    print(f"Found {len(result.data)} matching elements:")
    for i, elem in enumerate(result.data):
        print(f"  {i+1}. {elem}")
    
    # Then click the first one
    return await browser.act(f"click {description}")
```

### Validating Page State

```python
async def verify_elements_present(browser, expected_elements):
    """Check that required elements exist."""
    missing = []
    
    for element_desc in expected_elements:
        result = await browser.observe(f"find {element_desc}")
        if not result.data:
            missing.append(element_desc)
    
    if missing:
        print(f"Missing elements: {missing}")
        return False
    
    return True

# Usage
await verify_elements_present(browser, [
    "login button",
    "email input field",
    "password input field"
])
```

### Dynamic Element Discovery

```python
async def find_actionable_items(browser):
    """Discover what can be interacted with."""
    
    buttons = await browser.observe("find all clickable buttons")
    links = await browser.observe("find all links")
    inputs = await browser.observe("find all input fields")
    
    return {
        "buttons": [b for b in buttons.data if b],
        "links": [l for l in links.data if l],
        "inputs": [i for i in inputs.data if i]
    }
```

## Combining with Other Methods

### observe() + act()

```python
# Find available options, then select one
options = await browser.observe("find filter options")

if options.data:
    # Pick the first option
    first_option = options.data[0]
    await browser.act(f"click the {first_option.get('text', 'first')} filter")
```

### observe() + extract()

```python
# Find data regions, then extract from them
regions = await browser.observe("find product listing areas")

products = []
for region in regions.data:
    # Extract from identified region
    data = await browser.extract(
        f"extract product details from the {region} section"
    )
    products.extend(data.data if data.data else [])
```

## Best Practices

### Be Specific

```python
# Too broad
result = await browser.observe("find elements")

# Specific and useful
result = await browser.observe("find submit buttons in the checkout form")
```

### Use for Discovery

```python
# Good: Use observe() to understand the page
elements = await browser.observe("find all ways to filter products")
print(f"Filter options: {elements.data}")

# Then act based on findings
await browser.act("apply the first filter")
```

### Handle Empty Results

```python
result = await browser.observe("find the special element")

if not result.data:
    print("Element not found on this page")
    # Try alternative approach or report error
else:
    print(f"Found {len(result.data)} elements")
```

## Comparison with extract()

| Feature | observe() | extract() |
|---------|-----------|-----------|
| Purpose | Find elements | Get data |
| Returns | Element descriptions | Actual content |
| Selectors | Optional | No |
| Best for | Discovery, verification | Data collection |

## Related Methods

- [act()](act.md) - Execute actions on found elements
- [extract()](extract.md) - Extract data from page
- [agent()](agent.md) - Autonomous tasks using observation

## See Also

- [Basic Automation Guide](../guides/basic-automation.md) - Using observe in workflows
- [SDK Reference](../reference/sdk.md) - Complete API documentation
