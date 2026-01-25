# act() - Action Execution

The `act()` method executes browser actions using natural language instructions. It interprets your intent and performs the appropriate browser interaction.

## Basic Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com")
        
        # Click a button
        result = await browser.act("click the Sign In button")
        
        # Type text
        result = await browser.act("type 'hello world' in the search box")
        
        # Complex instruction
        result = await browser.act("scroll down and click the Learn More link")

asyncio.run(main())
```

## Method Signature

```python
async def act(
    self,
    instruction: str,
    use_vision: bool = False,
    return_metadata: bool = False,
) -> AgentRequestResponse
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction` | `str` | Required | Natural language description of the action to perform |
| `use_vision` | `bool` | `False` | Include screenshot for visual context |
| `return_metadata` | `bool` | `False` | Include detailed execution metadata in response |

### Returns

`AgentRequestResponse` with:
- `success` - Whether the action completed successfully
- `data` - Result data from the action
- `error` - Error message if failed
- `operation` - "act"
- `query` - The original instruction

When `return_metadata=True`, also includes:
- `execution` - ExecutionInfo with iterations, duration, history
- `llm_usage` - LLMUsageInfo with token counts and cost

## Supported Actions

### Clicking

```python
# By element text
await browser.act("click the Submit button")
await browser.act("click 'Learn More'")

# By location
await browser.act("click the first link in the navigation")
await browser.act("click the button at the bottom of the form")

# By context
await browser.act("click the Add to Cart button next to the iPhone")
await browser.act("click the delete icon in the third row")

# Multiple options
await browser.act("click Sign In or Log In button")
```

### Typing

```python
# In identified fields
await browser.act("type 'john@example.com' in the email field")
await browser.act("enter 'password123' in the password input")

# With field clearing
await browser.act("clear the search box and type 'new query'")

# Multi-line text
await browser.act("type 'Line 1\nLine 2\nLine 3' in the message area")
```

### Form Controls

```python
# Checkboxes
await browser.act("check the 'I agree' checkbox")
await browser.act("uncheck the newsletter option")

# Radio buttons
await browser.act("select 'Express Shipping' option")

# Dropdowns
await browser.act("select 'California' from the state dropdown")
await browser.act("choose 'Large' from the size selector")
```

### Navigation

```python
# Scrolling
await browser.act("scroll down")
await browser.act("scroll to the bottom of the page")
await browser.act("scroll up 300 pixels")
await browser.act("scroll to the Reviews section")

# Browser navigation
await browser.act("go back")
await browser.act("refresh the page")

# Within the page
await browser.act("click the Products link in the header")
```

### Special Actions

```python
# Hovering
await browser.act("hover over the dropdown menu")

# Keyboard
await browser.act("press Enter")
await browser.act("press Ctrl+A to select all")

# File upload
await browser.act("upload 'document.pdf' to the file input")

# Waiting
await browser.act("wait for the loading spinner to disappear")
```

## Using Vision

Enable vision mode for complex layouts or visual identification:

```python
# When element position matters
result = await browser.act(
    "click the search icon in the top right corner",
    use_vision=True
)

# When elements look similar but are positioned differently
result = await browser.act(
    "click the red Delete button, not the gray one",
    use_vision=True
)

# When text is in images
result = await browser.act(
    "click on the banner that says 'Sale'",
    use_vision=True
)
```

## Return Values

### Basic Response

```python
result = await browser.act("click Submit")

print(result.success)  # True/False
print(result.data)     # Action result details
print(result.error)    # Error message if failed
```

### With Metadata

```python
result = await browser.act("click Submit", return_metadata=True)

# Execution details
print(f"Iterations: {result.execution.iterations}")
print(f"Duration: {result.execution.duration_seconds}s")
print(f"Actions: {result.execution.actions_taken}")

# Step history
for step in result.execution.history:
    print(f"  {step['action']}: {step.get('success', 'unknown')}")

# LLM costs
print(f"Tokens: {result.llm_usage.total_tokens}")
print(f"Cost: ${result.llm_usage.cost_usd:.4f}")
```

## Error Handling

```python
result = await browser.act("click the nonexistent button")

if not result.success:
    print(f"Failed: {result.error}")
    
    # Common error patterns
    error = result.error.lower()
    
    if "not found" in error:
        # Element doesn't exist
        pass
    elif "timeout" in error:
        # Action took too long
        pass
    elif "intercepted" in error:
        # Something blocking the element
        pass
```

## Best Practices

### Be Specific

```python
# Less reliable
await browser.act("click the button")

# More reliable
await browser.act("click the blue Submit button at the bottom of the form")
```

### Use Context

```python
# When there are multiple similar elements
await browser.act("click the Edit button in the first row")
await browser.act("click Delete next to 'John Doe'")
```

### Handle Dynamic Content

```python
# Wait for content to load
await browser.goto("https://example.com")
await asyncio.sleep(1)  # Wait for dynamic content
await browser.act("click the dynamically loaded button")

# Or use compound instructions
await browser.act("wait for the menu to appear and click Options")
```

### Combine with observe()

```python
# First understand the page
elements = await browser.observe("find all action buttons")

# Then act on specific element
if elements.data:
    button_text = elements.data[0].get("text", "")
    await browser.act(f"click the {button_text} button")
```

## Operation Mode

The `act()` method internally sets the operation mode to `EXECUTE`, which optimizes for:
- Fast, targeted interactions
- Minimal vision overhead (only used on failures)
- Focused prompts for specific actions

## Related Methods

- [extract()](extract.md) - Extract data from pages
- [observe()](observe.md) - Find elements without acting
- [navigate()](navigation.md) - Natural language navigation
- [agent()](agent.md) - Autonomous multi-step tasks

## See Also

- [Basic Automation Guide](../guides/basic-automation.md) - Comprehensive action examples
- [Form Automation Guide](../guides/form-automation.md) - Form interaction patterns
- [Error Handling Guide](../guides/error-handling.md) - Handling action failures
