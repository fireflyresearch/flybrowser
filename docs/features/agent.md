# agent() - Autonomous Tasks

The `agent()` method executes complex, multi-step tasks autonomously. It plans, reasons, acts, and adapts to complete tasks that require multiple interactions.

## Basic Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://shop.example.com")
        
        # Autonomous multi-step task
        result = await browser.agent(
            task="Find the cheapest laptop under $500, add it to cart, and report the total"
        )
        
        if result.success:
            print(f"Task completed: {result.data}")
        else:
            print(f"Task failed: {result.error}")

asyncio.run(main())
```

## Method Signature

```python
async def agent(
    self,
    task: str,
    context: dict | None = None,
    max_iterations: int = 50,
    max_time_seconds: float | None = None,
    return_metadata: bool = False,
) -> AgentRequestResponse
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | Required | Natural language description of the task |
| `context` | `dict` | `None` | Additional context to help the agent |
| `max_iterations` | `int` | `50` | Maximum ReAct cycles before stopping |
| `max_time_seconds` | `float` | `None` | Time limit for task execution |
| `return_metadata` | `bool` | `False` | Include detailed execution metadata |

### Returns

`AgentRequestResponse` with:
- `success` - Whether task completed successfully
- `data` - Final result or extracted data
- `error` - Error message if failed
- `operation` - "agent"
- `query` - The original task

When `return_metadata=True`:
- `execution` - Detailed execution info with history
- `llm_usage` - Token usage and cost information

## Task Examples

### Simple Multi-Step

```python
result = await browser.agent(
    task="Go to Google, search for 'Python tutorials', and get the first 3 results"
)
```

### Complex Workflow

```python
result = await browser.agent(
    task="""
    1. Navigate to the products page
    2. Filter by category 'Electronics'
    3. Sort by price low to high
    4. Find a laptop with at least 4.5 star rating
    5. Add it to cart
    6. Report the product name and price
    """
)
```

### With Context

Provide context to help the agent:

```python
result = await browser.agent(
    task="Fill out the registration form and submit",
    context={
        "name": "John Doe",
        "email": "john@example.com",
        "password": "SecurePass123",
        "country": "United States"
    }
)
```

### Research Tasks

```python
result = await browser.agent(
    task="""
    Research the company 'Example Corp':
    1. Find their about page
    2. Extract the company mission statement
    3. Find their contact information
    4. List their main products or services
    """
)
```

## Execution Control

### Setting Limits

```python
# Limit iterations
result = await browser.agent(
    task="Find and purchase the item",
    max_iterations=30  # Stop after 30 ReAct cycles
)

# Time limit
result = await browser.agent(
    task="Complete the checkout process",
    max_time_seconds=180  # Stop after 3 minutes
)

# Both limits
result = await browser.agent(
    task="Scrape all products",
    max_iterations=100,
    max_time_seconds=600
)
```

### Monitoring Progress

```python
result = await browser.agent(
    task="Complete the multi-page form",
    return_metadata=True
)

# Review what happened
print(f"Success: {result.success}")
print(f"Iterations: {result.execution.iterations}")
print(f"Duration: {result.execution.duration_seconds}s")

# Step-by-step history
for step in result.execution.history:
    print(f"Step {step['step']}: {step['action']}")
    if step.get('error'):
        print(f"  Error: {step['error']}")
```

## How It Works

The agent uses the ReAct (Reasoning and Acting) paradigm:

1. **Thought** - Reasons about the current state and what to do next
2. **Action** - Executes a tool (click, type, extract, etc.)
3. **Observation** - Observes the result
4. **Repeat** - Continues until task is complete or limits reached

### Planning Mode

For complex tasks, the agent creates a structured plan:

```python
result = await browser.agent(
    task="""
    Complete the checkout process:
    - Fill shipping address
    - Choose payment method
    - Review order
    - Submit
    """
)

# The plan is included in metadata
if result.execution:
    if plan := result.metadata.get("execution_plan"):
        print(f"Plan phases: {len(plan['phases'])}")
        for phase in plan['phases']:
            print(f"  - {phase['name']}: {phase['status']}")
```

### Adaptive Execution

The agent adapts when things don't go as expected:

- Retries failed actions with different approaches
- Handles unexpected popups and obstacles (including JavaScript-triggered modals)
- Adjusts plan when encountering blockers
- Provides alternative suggestions on failures
- Auto-recovery when clicks are intercepted by overlays

### Dynamic Obstacle Handling

The agent automatically detects and dismisses obstacles that appear dynamically via JavaScript **after** initial page load:

```python
# Obstacles handled automatically during execution:
# - Newsletter signup popups (MailPoet, Mailchimp, HubSpot, Klaviyo)
# - Cookie consent banners (OneTrust, CookieBot, Quantcast, Termly)
# - Modal dialogs (Bootstrap, MUI, React-Modal, Ant Design)
# - Age verification gates
# - Promotional overlays

result = await browser.agent(
    task="Add the first product to cart and checkout"
)
# Agent automatically dismisses popups that appear during the task
```

**Two-Phase Detection:**
1. **Quick DOM Check** (~10ms): Multi-point sampling detects obstacles without LLM calls
2. **VLM Analysis** (if needed): AI-driven dismissal strategies when confidence > 0.3

**Auto-Recovery on Click Failures:**
When a click fails because another element intercepts it (common with popups), the agent automatically:
1. Detects the blocking obstacle
2. Dismisses it using appropriate strategies
3. Retries the original click

```python
# This works even when a newsletter popup appears mid-task
result = await browser.agent(
    task="Click the 'Add to Cart' button"
)
# If popup blocks the click, agent dismisses it and retries
```

## Return Values

### Basic Response

```python
result = await browser.agent(task="Find the contact page")

print(result.success)  # True/False
print(result.data)     # Task result
print(result.error)    # Error if failed
```

### Detailed Response

```python
result = await browser.agent(
    task="Complete registration",
    return_metadata=True
)

# Execution summary
exec_info = result.execution
print(f"Iterations: {exec_info.iterations}/{exec_info.max_iterations}")
print(f"Duration: {exec_info.duration_seconds:.2f}s")
print(f"Success: {exec_info.success}")
print(f"Summary: {exec_info.summary}")

# Actions taken
print(f"Actions: {exec_info.actions_taken}")

# Full history
for step in exec_info.history:
    status = "OK" if step.get('success') else "FAIL"
    print(f"[{status}] {step.get('thought', '')[:50]}...")
    print(f"        Action: {step.get('action')}")

# LLM costs
llm = result.llm_usage
print(f"Total tokens: {llm.total_tokens}")
print(f"Cost: ${llm.cost_usd:.4f}")
```

## Completion Page

When running in non-headless mode, the browser displays an interactive **Completion Page** after task execution. This provides a visual summary of the agent's work.

### Completion Page Features

**Metrics Overview:**
- **Duration**: Total execution time (formatted as ms, seconds, or minutes)
- **Iterations**: ReAct cycles used vs. maximum allowed
- **Tokens**: Total LLM tokens consumed (prompt + completion)
- **Cost**: Estimated LLM cost in USD

**Expandable Sections:**
- **LLM Usage Details**: Model name, provider, token breakdown (prompt/completion), API call count, average latency
- **Tools Executed**: Complete list of all tools invoked with individual durations and success/failure status
- **Reasoning Steps**: Timeline showing each thought → action pair in sequence

**Result Data Explorer:**
- **Tree View**: Interactive, collapsible JSON tree for exploring complex nested results
- **Raw View**: Syntax-highlighted JSON with proper indentation
- One-click toggle between views
- Deep nesting auto-collapsed for readability
- Copy to clipboard functionality

**Metadata Footer:**
- Session ID for debugging and tracing
- Reasoning strategy used (react_standard, planning, etc.)
- Stop reason (completed, max_iterations, timeout, error)

**Error Display (on failure):**
- Clear error message
- Optional stack trace for debugging

### Example Completion Page

```
┌─────────────────────────────────────────────────────────────┐
│  ✓ AGENT COMPLETED SUCCESSFULLY                             │
│  Task: Find the cheapest laptop and add to cart             │
├─────────────────────────────────────────────────────────────┤
│  Duration     Iterations     Tokens      Cost               │
│  12.4s        8/50           2,450       $0.0024            │
├─────────────────────────────────────────────────────────────┤
│  ▼ LLM Usage Details                                        │
│    Model: gpt-4o | Provider: openai                         │
│    Prompt: 1,820 | Completion: 630 | Latency: 1.2s          │
├─────────────────────────────────────────────────────────────┤
│  ▼ Tools Executed (8)                                       │
│    navigate (1.2s) → click (0.3s) → extract (2.1s) → ...    │
├─────────────────────────────────────────────────────────────┤
│  ▼ Reasoning Steps                                          │
│    1. "Navigate to laptops" → navigate                      │
│    2. "Sort by price" → click                               │
│    3. "Find cheapest" → extract                             │
├─────────────────────────────────────────────────────────────┤
│  Result Data  [Tree] [Raw]                                  │
│  ▼ {                                                        │
│      "product": "Laptop Pro 15"                             │
│      "price": 449.99                                        │
│      "added_to_cart": true                                  │
│    }                                                        │
├─────────────────────────────────────────────────────────────┤
│  Session: sess_abc123 | Strategy: completed | Stop: success │
└─────────────────────────────────────────────────────────────┘
```

### Viewing the Completion Page

The completion page appears automatically when:
- Running with `headless=False`
- Using embedded mode (not server mode)
- The `agent()` method completes (success or failure)

```python
# Completion page will be visible in the browser window
browser = FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    headless=False,  # Show browser window
)

async with browser:
    await browser.goto("https://shop.example.com")
    result = await browser.agent("Add the cheapest item to cart")
    # Completion page now visible in browser
    # User can explore the JSON tree, review steps, etc.
```

### Data Extraction & Robustness

The completion page handles various data formats safely:

- **ReActStep objects**: Extracts thought, action, observation, duration
- **Dictionary representations**: Handles serialized step data
- **Missing fields**: Gracefully handles incomplete data with sensible defaults
- **LLM usage**: Normalizes token counts, costs, and latency metrics
- **Error cases**: Displays error messages and optional stack traces

The page will never fail to render due to missing or malformed data—all fields use defensive defaults.

## Error Handling

```python
result = await browser.agent(task="Complete impossible task")

if not result.success:
    print(f"Task failed: {result.error}")
    
    # Analyze what went wrong
    if result.execution:
        # Check if we ran out of iterations
        if result.execution.iterations >= result.execution.max_iterations:
            print("Hit iteration limit")
        
        # Check last steps for errors
        history = result.execution.history
        failed_steps = [s for s in history if not s.get('success', True)]
        print(f"Failed steps: {len(failed_steps)}/{len(history)}")
```

## Best Practices

### Clear Task Descriptions

```python
# Vague - might not work well
result = await browser.agent(task="buy something")

# Clear and specific
result = await browser.agent(
    task="""
    Purchase the first available laptop:
    1. Go to laptops category
    2. Select the first in-stock item
    3. Add to cart
    4. Proceed to checkout (stop before payment)
    """
)
```

### Use Context Effectively

```python
# Without context - agent has to figure everything out
result = await browser.agent(task="Log in")

# With context - agent knows what to use
result = await browser.agent(
    task="Log in to the account",
    context={
        "username": "user@example.com",
        "password": "mypassword"
    }
)
```

### Set Appropriate Limits

```python
# Short task - fewer iterations needed
result = await browser.agent(
    task="Click the login button",
    max_iterations=10
)

# Complex task - may need more
result = await browser.agent(
    task="Complete entire checkout flow",
    max_iterations=50,
    max_time_seconds=300
)
```

### Monitor Long Tasks

```python
result = await browser.agent(
    task="Scrape all products from every category",
    max_iterations=200,
    max_time_seconds=1800,  # 30 minutes
    return_metadata=True
)

# Always check results of long tasks
if result.success:
    print(f"Completed in {result.execution.duration_seconds}s")
    print(f"Used {result.execution.iterations} iterations")
else:
    print(f"Failed after {result.execution.iterations} iterations")
    print(f"Last error: {result.error}")
```

## Operation Mode

The `agent()` method uses `RESEARCH` operation mode, optimized for:
- Adaptive vision based on task needs (skips blank pages automatically)
- Smart page exploration with dynamic obstacle detection
- Comprehensive memory for complex workflows
- Autonomous planning and replanning
- Auto-recovery from intercepted clicks and blocked interactions

## Comparison with Other Methods

| Feature | agent() | act() | extract() |
|---------|---------|-------|-----------|
| Multi-step | Yes | No | No |
| Planning | Yes | No | No |
| Autonomous | Yes | Guided | Guided |
| Best for | Complex tasks | Single actions | Data retrieval |

## Related Methods

- [act()](act.md) - Single action execution
- [extract()](extract.md) - Data extraction
- [observe()](observe.md) - Element discovery
- [navigate()](navigation.md) - Page navigation

## See Also

- [Multi-Page Workflows](../guides/multi-page-workflows.md) - Complex workflow patterns
- [Error Handling Guide](../guides/error-handling.md) - Handling agent failures
- [Architecture: ReAct Agent](../architecture/react.md) - How the agent works
