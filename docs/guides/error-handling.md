# Error Handling Guide

This guide covers handling errors and building robust automation with FlyBrowser. You will learn about different error types, recovery strategies, and best practices for reliable scripts.

## Understanding Response Objects

### Checking for Success

Every FlyBrowser operation returns a response with success status:

```python
import asyncio
from flybrowser import FlyBrowser

async def check_results():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com")
        
        result = await browser.act("click the Sign In button")
        
        if result.success:
            print("Action succeeded!")
            print(f"Data: {result.data}")
        else:
            print(f"Action failed: {result.error}")
```

### Response Fields

```python
result = await browser.extract("get the price")

# Core fields
result.success      # bool - Did operation succeed?
result.data         # Any - The returned data
result.error        # str | None - Error message if failed
result.operation    # str - What operation was attempted
result.query        # str - The original query

# Execution details (when return_metadata=True)
result.execution.iterations      # How many ReAct cycles
result.execution.duration_seconds # Total time
result.execution.actions_taken    # List of tools used
result.execution.history          # Full step history

# LLM usage
result.llm_usage.total_tokens    # Tokens consumed
result.llm_usage.cost_usd        # Estimated cost
```

## Common Error Types

### Element Not Found

```python
result = await browser.act("click the Subscribe button")

if not result.success:
    if "not found" in str(result.error).lower():
        print("Element doesn't exist on this page")
        
        # Try observing what's available
        elements = await browser.observe("find all buttons on the page")
        print(f"Available buttons: {elements.data}")
```

### Timeout Errors

```python
async def handle_timeout():
    try:
        result = await browser.goto("https://slow-site.example.com")
    except asyncio.TimeoutError:
        print("Page took too long to load")
        # Retry with longer timeout or skip

async def with_timeout(coro, timeout_seconds=30):
    """Wrap any operation with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"success": False, "error": "Operation timed out"}
```

### Navigation Errors

```python
result = await browser.goto("https://nonexistent-site.example.com")

if not result.success:
    error = str(result.error).lower()
    
    if "dns" in error or "resolve" in error:
        print("Site doesn't exist or DNS failed")
    elif "refused" in error:
        print("Connection refused by server")
    elif "timeout" in error:
        print("Connection timed out")
    elif "ssl" in error or "certificate" in error:
        print("SSL/TLS certificate error")
```

### LLM Errors

```python
result = await browser.extract("get complex data")

if not result.success:
    error = str(result.error).lower()
    
    if "rate limit" in error:
        print("Hit API rate limit, waiting...")
        await asyncio.sleep(60)
    elif "token" in error or "context" in error:
        print("Request too large for model context")
        # FlyBrowser handles this automatically in most cases
        # Use more specific extraction instructions to reduce content size
    elif "api key" in error:
        print("Invalid or expired API key")
```

### Token Overflow Errors

FlyBrowser automatically prevents most token overflow errors, but you can handle edge cases:

```python
# If content is too large, use more targeted extraction
result = await browser.extract(
    "get only the main article content, excluding navigation and ads"
)
```

The agent's memory system automatically:
- Limits extraction data to 25% of context budget
- Truncates individual extractions to max 32K chars
- Prunes old history entries when context is full

## Retry Strategies

### Simple Retry

```python
async def with_retry(operation, max_retries=3):
    """Retry an operation multiple times."""
    last_error = None
    
    for attempt in range(max_retries):
        result = await operation()
        
        if result.success:
            return result
        
        last_error = result.error
        print(f"Attempt {attempt + 1} failed: {last_error}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(1)
    
    return result  # Return last failed result

# Usage
result = await with_retry(
    lambda: browser.act("click the Submit button")
)
```

### Exponential Backoff

```python
async def with_exponential_backoff(operation, max_retries=5, base_delay=1):
    """Retry with exponential backoff."""
    for attempt in range(max_retries):
        result = await operation()
        
        if result.success:
            return result
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
            print(f"Attempt {attempt + 1} failed, waiting {delay}s...")
            await asyncio.sleep(delay)
    
    return result
```

### Conditional Retry

```python
async def smart_retry(browser, instruction, max_retries=3):
    """Retry with different strategies based on failure type."""
    for attempt in range(max_retries):
        result = await browser.act(instruction)
        
        if result.success:
            return result
        
        error = str(result.error).lower()
        
        # Choose strategy based on error
        if "not found" in error:
            # Element might not be visible - try scrolling
            await browser.act("scroll down")
            await asyncio.sleep(0.5)
            
        elif "timeout" in error:
            # Page might be slow - wait longer
            await asyncio.sleep(3)
            
        elif "intercepted" in error:
            # Something blocking the element - try to close it
            await browser.act("close any popups or modals")
            await asyncio.sleep(0.5)
            
        else:
            # Unknown error - simple wait
            await asyncio.sleep(1)
    
    return result
```

## Recovery Patterns

### Alternative Approaches

```python
async def click_with_alternatives(browser, element_description):
    """Try multiple approaches to click an element."""
    
    # Approach 1: Natural language
    result = await browser.act(f"click {element_description}")
    if result.success:
        return result
    
    # Approach 2: With vision
    result = await browser.act(f"click {element_description}", use_vision=True)
    if result.success:
        return result
    
    # Approach 3: Find first, then click
    observe_result = await browser.observe(f"find {element_description}")
    if observe_result.success and observe_result.data:
        first_element = observe_result.data[0]
        if selector := first_element.get("selector"):
            result = await browser.act(f"click element with selector {selector}")
            if result.success:
                return result
    
    # Approach 4: Scroll into view first
    await browser.act(f"scroll to {element_description}")
    await asyncio.sleep(0.5)
    result = await browser.act(f"click {element_description}")
    
    return result
```

### Page State Recovery

```python
async def ensure_correct_page(browser, expected_url_pattern, go_to_url):
    """Ensure we're on the correct page, navigate if not."""
    
    # Extract current URL
    current = await browser.extract("what is the current page URL?")
    current_url = str(current.data).lower()
    
    if expected_url_pattern.lower() not in current_url:
        print(f"Wrong page ({current_url}), navigating to correct page...")
        await browser.goto(go_to_url)
        await asyncio.sleep(1)
        return False
    
    return True

# Usage
await ensure_correct_page(
    browser, 
    "checkout",
    "https://shop.example.com/checkout"
)
```

### Session Recovery

```python
async def with_session_recovery(browser, operation, login_func):
    """Wrap operation with automatic re-login on session expiry."""
    result = await operation()
    
    if not result.success:
        error = str(result.error).lower()
        
        if any(word in error for word in ["session", "login", "unauthorized", "401"]):
            print("Session expired, re-authenticating...")
            await login_func(browser)
            result = await operation()
    
    return result
```

## Error Boundaries

### Try-Except Wrappers

```python
async def safe_operation(browser, operation_name, operation_func):
    """Execute operation with comprehensive error handling."""
    try:
        result = await operation_func()
        
        if result.success:
            return {"success": True, "data": result.data}
        else:
            return {
                "success": False,
                "error": result.error,
                "operation": operation_name
            }
            
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": "Operation timed out",
            "operation": operation_name
        }
    except asyncio.CancelledError:
        return {
            "success": False,
            "error": "Operation cancelled",
            "operation": operation_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": operation_name,
            "exception_type": type(e).__name__
        }
```

### Workflow Error Collection

```python
async def workflow_with_error_collection(browser, steps):
    """Execute workflow, collecting errors but continuing."""
    results = []
    errors = []
    
    for step_name, step_func in steps:
        try:
            result = await step_func()
            results.append({
                "step": step_name,
                "success": result.success,
                "data": result.data if result.success else None
            })
            
            if not result.success:
                errors.append({
                    "step": step_name,
                    "error": result.error
                })
                
        except Exception as e:
            errors.append({
                "step": step_name,
                "error": str(e),
                "exception": True
            })
    
    return {
        "results": results,
        "errors": errors,
        "total_steps": len(steps),
        "failed_steps": len(errors)
    }

# Usage
steps = [
    ("navigate", lambda: browser.goto("https://example.com")),
    ("search", lambda: browser.act("search for 'test'")),
    ("extract", lambda: browser.extract("get results")),
]

outcome = await workflow_with_error_collection(browser, steps)
print(f"Completed with {outcome['failed_steps']} errors")
```

## Logging and Debugging

### Structured Logging

```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flybrowser_automation")

async def logged_operation(browser, operation_name, operation_func):
    """Execute with detailed logging."""
    logger.info(f"Starting: {operation_name}")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = await operation_func()
        duration = asyncio.get_event_loop().time() - start_time
        
        if result.success:
            logger.info(
                f"Success: {operation_name} "
                f"(duration={duration:.2f}s)"
            )
        else:
            logger.warning(
                f"Failed: {operation_name} "
                f"(duration={duration:.2f}s, error={result.error})"
            )
        
        return result
        
    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Exception: {operation_name} "
            f"(duration={duration:.2f}s, error={str(e)})"
        )
        raise
```

### Debug Screenshots

```python
import base64
from datetime import datetime

async def save_debug_screenshot(browser, name="debug"):
    """Save screenshot for debugging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    
    try:
        screenshot = await browser.screenshot(full_page=True)
        image_data = base64.b64decode(screenshot["data_base64"])
        
        with open(filename, "wb") as f:
            f.write(image_data)
        
        print(f"Debug screenshot saved: {filename}")
        return filename
        
    except Exception as e:
        print(f"Could not save screenshot: {e}")
        return None

async def operation_with_debug(browser, operation_func):
    """Execute operation, save screenshot on failure."""
    result = await operation_func()
    
    if not result.success:
        await save_debug_screenshot(browser, "error")
    
    return result
```

### Execution History Analysis

```python
async def analyze_execution(browser, task):
    """Execute task and analyze what happened."""
    result = await browser.agent(task=task, return_metadata=True)
    
    # Print execution summary
    print(f"Task: {task}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.execution.iterations}")
    print(f"Duration: {result.execution.duration_seconds:.2f}s")
    print(f"Actions taken: {result.execution.actions_taken}")
    
    # Analyze step history
    if result.execution.history:
        print("\nStep-by-step breakdown:")
        for step in result.execution.history:
            status = "OK" if step.get("success", False) else "FAIL"
            print(f"  [{status}] {step.get('action', 'unknown')}")
            if step.get("error"):
                print(f"         Error: {step['error']}")
    
    # Check for patterns
    failed_actions = [
        s for s in result.execution.history 
        if not s.get("success", False)
    ]
    
    if len(failed_actions) > 2:
        print(f"\nWarning: {len(failed_actions)} failed actions detected")
    
    return result
```

## Best Practices

### Defensive Coding

```python
async def defensive_workflow(browser):
    """Workflow with defensive checks at each step."""
    
    # Step 1: Navigate with verification
    await browser.goto("https://shop.example.com")
    
    page_check = await browser.extract("is this the shop homepage?")
    if not page_check.data or "yes" not in str(page_check.data).lower():
        raise Exception("Failed to load shop homepage")
    
    # Step 2: Search with verification
    await browser.act("search for 'laptop'")
    await asyncio.sleep(1)
    
    results_check = await browser.extract("are there search results displayed?")
    if not results_check.data:
        raise Exception("Search returned no results")
    
    # Step 3: Click with verification
    await browser.act("click the first product")
    await asyncio.sleep(1)
    
    product_check = await browser.extract("am I on a product detail page?")
    if not product_check.data:
        raise Exception("Failed to navigate to product page")
    
    # Continue with confidence...
```

### Graceful Degradation

```python
async def extract_with_fallbacks(browser, queries):
    """Try multiple extraction queries, return first success."""
    for query in queries:
        result = await browser.extract(query)
        if result.success and result.data:
            return result
    
    # Return last result even if failed
    return result

# Usage
result = await extract_with_fallbacks(browser, [
    "get the product price from the main display",
    "find any price on this page",
    "extract all numbers that look like prices"
])
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.is_open = False
    
    async def execute(self, operation):
        # Check if circuit should reset
        if self.is_open:
            if (asyncio.get_event_loop().time() - self.last_failure_time 
                > self.reset_timeout):
                self.is_open = False
                self.failures = 0
            else:
                raise Exception("Circuit breaker is open")
        
        result = await operation()
        
        if result.success:
            self.failures = 0
            return result
        else:
            self.failures += 1
            self.last_failure_time = asyncio.get_event_loop().time()
            
            if self.failures >= self.threshold:
                self.is_open = True
                raise Exception("Circuit breaker triggered")
            
            return result

# Usage
breaker = CircuitBreaker()

try:
    result = await breaker.execute(
        lambda: browser.act("click Submit")
    )
except Exception as e:
    print(f"Circuit breaker: {e}")
```

### Error Reporting

```python
async def report_error(error_info):
    """Send error report (implement your reporting logic)."""
    print(f"ERROR REPORT: {json.dumps(error_info, indent=2)}")
    # In production: send to monitoring service, log aggregator, etc.

async def monitored_workflow(browser, workflow_name, workflow_func):
    """Execute workflow with error reporting."""
    try:
        result = await workflow_func()
        
        if not result.success:
            await report_error({
                "workflow": workflow_name,
                "type": "operation_failure",
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            })
        
        return result
        
    except Exception as e:
        await report_error({
            "workflow": workflow_name,
            "type": "exception",
            "error": str(e),
            "exception_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        })
        raise
```

## Next Steps

- [Multi-Page Workflows](multi-page-workflows.md) - Complex workflow patterns
- [Troubleshooting Guide](../advanced/troubleshooting.md) - Common issues and solutions
- [SDK Reference](../reference/sdk.md) - Complete API documentation
