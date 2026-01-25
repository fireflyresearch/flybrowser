# Basic Automation Guide

This guide covers the fundamental browser automation patterns with FlyBrowser. You will learn how to navigate, interact with elements, and build reliable automation scripts.

## Navigation

### Direct URL Navigation

The `goto()` method navigates directly to a URL:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        # Navigate to a URL
        await browser.goto("https://example.com")
        
        # Wait for specific events during navigation
        await browser.goto(
            "https://example.com",
            wait_until="networkidle"  # Wait for network to be idle
        )

asyncio.run(main())
```

The `wait_until` parameter accepts:
- `"load"` - Wait for the load event (default)
- `"domcontentloaded"` - Wait for DOMContentLoaded event
- `"networkidle"` - Wait until no network connections for 500ms

### Natural Language Navigation

The `navigate()` method uses natural language to describe where to go:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://shop.example.com")
    
    # Natural language navigation
    await browser.navigate("go to the electronics section")
    await browser.navigate("find the contact page")
    await browser.navigate("look for laptop deals")
```

The agent interprets your instruction and determines how to get there - clicking links, using navigation menus, or searching.

### Navigation with Vision

Enable vision for better navigation on complex pages:

```python
await browser.navigate(
    "find the login button in the header",
    use_vision=True  # Uses screenshot for context
)
```

Vision helps when:
- Page structure is complex
- Elements are visually distinct but semantically ambiguous
- You need to find elements by appearance

## Clicking Elements

The `act()` method handles clicks and other interactions:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://example.com")
    
    # Click by description
    await browser.act("click the Sign In button")
    
    # Click by location
    await browser.act("click the first item in the navigation menu")
    
    # Click with context
    await browser.act("click the Add to Cart button next to the MacBook")
```

### Click Strategies

FlyBrowser tries multiple strategies to find and click elements:

1. **Semantic matching** - Understands "Sign In", "Login", "Log In" are equivalent
2. **Position hints** - "first", "last", "second from top"
3. **Contextual hints** - "next to", "below", "in the header"
4. **Visual matching** - Uses screenshots when enabled

### Handling Click Failures

When clicks fail, FlyBrowser provides useful diagnostics:

```python
result = await browser.act("click the Submit button")

if not result.success:
    print(f"Click failed: {result.error}")
    
    # Check what elements were found
    observe_result = await browser.observe("find submit buttons")
    print(f"Found elements: {observe_result.data}")
```

## Typing Text

Enter text into form fields:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://example.com/search")
    
    # Type in a field by description
    await browser.act("type 'python tutorials' in the search box")
    
    # Clear and type
    await browser.act("clear the email field and type 'user@example.com'")
    
    # Type in specific fields
    await browser.act("enter 'John Doe' in the Name field")
    await browser.act("type '123 Main St' in the Address input")
```

### Form Field Identification

FlyBrowser identifies form fields by:
- Labels (`<label>` elements)
- Placeholder text
- `name` and `id` attributes
- ARIA labels
- Visual proximity to labels

### Special Keys

Some typing operations need special handling:

```python
# Press Enter to submit
await browser.act("type 'search query' in the search box and press Enter")

# Tab to next field
await browser.act("press Tab to move to the next field")

# Keyboard shortcuts
await browser.act("press Ctrl+A to select all")
```

## Form Interactions

### Checkboxes and Radio Buttons

```python
# Check a checkbox
await browser.act("check the 'I agree to terms' checkbox")

# Uncheck
await browser.act("uncheck the newsletter subscription")

# Radio buttons
await browser.act("select the 'Express Shipping' option")
```

### Dropdowns

```python
# Select by visible text
await browser.act("select 'California' from the State dropdown")

# Select by description
await browser.act("choose 'Medium' size from the size selector")
```

### File Uploads

```python
# Note: File uploads require specifying the file path
await browser.act("upload 'document.pdf' to the file input")
```

## Scrolling

### Basic Scrolling

```python
# Scroll down
await browser.act("scroll down")

# Scroll up
await browser.act("scroll to the top of the page")

# Scroll by amount
await browser.act("scroll down 500 pixels")
```

### Scrolling to Elements

```python
# Scroll element into view
await browser.act("scroll to the product reviews section")

# Scroll within containers
await browser.act("scroll down in the sidebar")
```

### Infinite Scroll Pages

For pages with infinite scroll:

```python
async def load_all_items():
    async with FlyBrowser(...) as browser:
        await browser.goto("https://example.com/feed")
        
        items = []
        last_count = 0
        
        while True:
            # Extract current items
            result = await browser.extract("get all post titles")
            items = result.data
            
            # Check if we got new items
            if len(items) == last_count:
                break  # No more items loading
            
            last_count = len(items)
            
            # Scroll to load more
            await browser.act("scroll to the bottom of the page")
            
            # Small wait for content to load
            await asyncio.sleep(1)
        
        return items
```

## Waiting

### Explicit Waits

Sometimes you need to wait for page changes:

```python
import asyncio

async with FlyBrowser(...) as browser:
    await browser.goto("https://example.com")
    
    await browser.act("click the Load More button")
    
    # Wait for content to appear
    await asyncio.sleep(2)
    
    result = await browser.extract("get all items")
```

### Conditional Waiting

The agent handles most waits automatically, but you can be explicit:

```python
# The agent will wait for the element to appear before interacting
await browser.act("wait for the loading spinner to disappear, then click Submit")

# Or use multi-step instructions
await browser.agent(
    task="Click Submit and wait for the confirmation message",
    max_iterations=10
)
```

## Combining Actions

### Sequential Actions

For related actions, chain them:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://shop.example.com")
    
    # Each action builds on the previous
    await browser.act("search for 'wireless headphones'")
    await browser.act("click on the first product")
    await browser.act("select the Black color option")
    await browser.act("click Add to Cart")
    await browser.act("proceed to checkout")
```

### Complex Workflows with agent()

For multi-step tasks, use the autonomous agent:

```python
result = await browser.agent(
    task="""
    1. Go to the products page
    2. Find the most expensive item
    3. Add it to the cart
    4. Go to checkout
    5. Fill in the shipping address with test data
    """,
    max_iterations=30
)

print(f"Completed: {result.success}")
print(f"Steps taken: {result.execution.iterations}")
```

## Handling Dynamic Content

### Single Page Applications (SPAs)

SPAs require attention to dynamic content:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://spa-example.com")
    
    # Navigate within the SPA
    await browser.act("click the Dashboard link")
    
    # Wait for SPA routing to complete
    await asyncio.sleep(1)
    
    # Now interact with the new view
    result = await browser.extract("get dashboard statistics")
```

### AJAX Content

For content that loads asynchronously:

```python
# Click triggers AJAX load
await browser.act("click 'Show Details'")

# The agent observes the page and waits appropriately
# but you can add explicit waits for reliability
await asyncio.sleep(2)

# Extract the loaded content
result = await browser.extract("get the details that just loaded")
```

## Working with Iframes

FlyBrowser can interact with content inside iframes:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://example.com/embedded-form")
    
    # The agent automatically handles iframe context when needed
    await browser.act("in the payment iframe, enter card number '4111111111111111'")
```

## Screenshots for Debugging

Capture screenshots to understand page state:

```python
import base64

async with FlyBrowser(...) as browser:
    await browser.goto("https://example.com")
    
    # Take a screenshot
    screenshot = await browser.screenshot()
    
    # Save to file
    with open("debug.png", "wb") as f:
        f.write(base64.b64decode(screenshot["data_base64"]))
    
    # Full page screenshot
    full_screenshot = await browser.screenshot(full_page=True)
```

## Best Practices

### Be Specific

More specific instructions yield better results:

```python
# Less reliable
await browser.act("click the button")

# More reliable
await browser.act("click the blue Submit button at the bottom of the form")
```

### Handle Errors Gracefully

```python
async def robust_click(browser, instruction, retries=3):
    for attempt in range(retries):
        result = await browser.act(instruction)
        if result.success:
            return result
        print(f"Attempt {attempt + 1} failed: {result.error}")
        await asyncio.sleep(1)
    return result
```

### Use Vision for Complex Pages

```python
# For pages with complex layouts
result = await browser.act(
    "click the search icon in the top right corner",
    use_vision=True
)
```

### Monitor Execution

```python
result = await browser.agent(task="Complete the checkout process")

# Review what happened
print(f"Iterations: {result.execution.iterations}")
print(f"Duration: {result.execution.duration_seconds}s")
print(f"Actions: {result.execution.actions_taken}")

# Detailed step-by-step
for step in result.execution.history:
    print(f"Step {step['step']}: {step['action']}")
```

## Common Patterns

### Login Flow

```python
async def login(browser, username, password):
    await browser.goto("https://example.com/login")
    await browser.act(f"type '{username}' in the username field")
    await browser.act(f"type '{password}' in the password field")
    await browser.act("click the Login button")
    
    # Verify login succeeded
    result = await browser.extract("check if we are logged in")
    return "welcome" in str(result.data).lower()
```

### Search and Select

```python
async def search_and_select(browser, search_term, result_index=0):
    await browser.act(f"type '{search_term}' in the search box")
    await browser.act("press Enter or click Search")
    await asyncio.sleep(2)  # Wait for results
    
    await browser.act(f"click the {result_index + 1}{'st' if result_index == 0 else 'th'} result")
```

### Pagination

```python
async def process_all_pages(browser):
    all_data = []
    page_num = 1
    
    while True:
        # Extract current page data
        result = await browser.extract("get all items on this page")
        all_data.extend(result.data)
        
        # Try to go to next page
        next_result = await browser.act("click the Next page button if it exists")
        
        if not next_result.success or "no next" in str(next_result.error).lower():
            break
            
        page_num += 1
        await asyncio.sleep(1)
    
    return all_data
```

## Next Steps

- [Data Extraction Guide](data-extraction.md) - Advanced data extraction patterns
- [Form Automation Guide](form-automation.md) - Complex form handling
- [Multi-Page Workflows](multi-page-workflows.md) - Building complex automations
