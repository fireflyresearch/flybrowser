# Troubleshooting

This guide covers common issues and solutions when using FlyBrowser.

## Browser Issues

### Browser Won't Start

**Symptoms:**
- `BrowserError: Failed to launch browser`
- `Playwright browsers not installed`

**Solutions:**

1. Install Playwright browsers:
```bash
playwright install chromium
playwright install-deps chromium
```

2. Check system dependencies (Linux):
```bash
# Ubuntu/Debian
sudo apt-get install libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
  libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
  libxrandr2 libgbm1 libasound2
```

3. Use headless mode:
```python
browser = FlyBrowser(headless=True)
```

### Browser Crashes

**Symptoms:**
- `BrowserError: Browser process terminated`
- High memory usage before crash

**Solutions:**

1. Increase shared memory (Docker):
```bash
docker run --shm-size=2g ...
```

2. Reduce concurrent browsers:
```bash
FLYBROWSER_POOL__MAX_SIZE=5
```

3. Add swap space:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Timeout Errors

**Symptoms:**
- `TimeoutError: Navigation timeout exceeded`
- `TimeoutError: Timeout waiting for selector`

**Solutions:**

1. Increase timeouts:
```python
browser = FlyBrowser(
    timeout=60000,           # Default timeout
    navigation_timeout=60000, # Navigation timeout
)
```

2. Wait for specific elements:
```python
await browser.page.wait_for_selector("#content", timeout=30000)
```

3. Check network connectivity:
```python
try:
    await browser.goto("https://example.com")
except TimeoutError:
    print("Network may be slow or site unreachable")
```

## LLM Issues

### API Key Errors

**Symptoms:**
- `ConfigurationError: API key not found`
- `LLMError: Authentication failed`

**Solutions:**

1. Set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

2. Pass directly:
```python
browser = FlyBrowser(
    llm_provider="openai",
    llm_api_key="sk-...",
)
```

3. Check API key validity:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Rate Limiting

**Symptoms:**
- `LLMError: Rate limit exceeded`
- `429 Too Many Requests`

**Solutions:**

1. Add retry logic:
```python
from flybrowser.llm.config import RetryConfig

config = LLMProviderConfig(
    retry_config=RetryConfig(
        max_retries=5,
        initial_delay=2.0,
        max_delay=60.0,
    ),
)
```

2. Use rate limiting:
```python
from flybrowser.llm.config import RateLimitConfig

config = LLMProviderConfig(
    rate_limit_config=RateLimitConfig(
        requests_per_minute=30,
        tokens_per_minute=100000,
    ),
)
```

3. Switch to a different model or provider:
```python
browser = FlyBrowser(llm_provider="anthropic")
```

### Invalid Responses

**Symptoms:**
- `ValueError: Failed to parse JSON response`
- Agent produces unexpected results

**Solutions:**

1. Lower temperature for determinism:
```python
browser = FlyBrowser(temperature=0.1)
```

2. Use structured output:
```python
data = await browser.extract(
    "Get product info",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
        },
        "required": ["name", "price"],
    }
)
```

3. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Token Overflow / Context Length Errors

**Symptoms:**
- `LLMError: context_length_exceeded`
- `429 Rate limit exceeded` with TPM (tokens per minute) message
- Agent fails after extracting large amounts of data

**Solutions:**

1. FlyBrowser automatically handles most overflow scenarios:
```python
# The agent automatically:
# - Limits extractions to 25% of context budget
# - Truncates individual extractions to 32K chars
# - Prunes older history when context is full
```

2. Check content size before extraction:
```python
from flybrowser.llm.token_budget import TokenEstimator

# Estimate tokens for large content
large_content = await page.inner_text("body")
estimate = TokenEstimator.estimate(large_content)
print(f"Content: ~{estimate.tokens} tokens ({estimate.content_type.value})")

if estimate.tokens > 50000:
    # Use more targeted extraction
    data = await browser.extract("Get only the product names and prices")
```

3. Use ConversationManager for very large content:
```python
from flybrowser.llm.conversation import ConversationManager

manager = ConversationManager(llm_provider)
result = await manager.send_with_large_content(
    very_large_html,
    instruction="Extract all articles",
    schema={"type": "object", "properties": {...}}
)
# Automatically chunks and processes in multiple turns
```

4. Use more specific extraction instructions:
```python
# Instead of extracting entire page
await browser.extract("Get all text")  # May overflow

# Extract specific sections
await browser.extract("Get only the main article content")
```

5. Check model context window:
```python
info = llm_provider.get_model_info()
print(f"Context window: {info.context_window} tokens")
print(f"Max output: {info.max_output_tokens} tokens")
```

## Agent Issues

### Agent Gets Stuck

**Symptoms:**
- Agent repeats the same action
- `AgentError: Max steps exceeded`

**Solutions:**

1. Increase max steps:
```python
result = await browser.agent(
    task="Complete complex workflow",
    max_steps=50,
)
```

2. Add more specific instructions:
```python
# Too vague
await browser.agent("Fill the form")

# More specific
await browser.agent(
    "Fill the contact form: name='John', email='john@example.com', "
    "message='Hello', then click Submit"
)
```

3. Check for loop detection warnings in logs

### Agent Fails to Find Elements

**Symptoms:**
- `ElementNotFound: No element matches selector`
- Agent cannot interact with page

**Solutions:**

1. Wait for dynamic content:
```python
await browser.page.wait_for_load_state("networkidle")
```

2. Use natural language instead of selectors:
```python
# Instead of specific selectors
await browser.act("Click the blue Submit button at the bottom of the form")
```

3. Enable vision mode:
```python
browser = FlyBrowser(vision_enabled=True)
```

### Wrong Element Clicked

**Symptoms:**
- Agent clicks unexpected elements
- Actions don't have intended effect

**Solutions:**

1. Be more specific in instructions:
```python
await browser.act(
    "Click the 'Add to Cart' button next to the iPhone 15 product"
)
```

2. Use observe() to understand the page:
```python
observation = await browser.observe("What buttons are visible?")
print(observation)
```

3. Check for overlapping elements or iframes

### Modal/Popup Blocking Clicks

**Symptoms:**
- `ElementNotFound: Element intercepted by another element`
- Click fails with "covered by" error
- Newsletter or cookie popup appears mid-task

**Solutions:**

1. The agent auto-recovers from intercepted clicks:
```python
# Agent automatically detects and dismisses blocking popups
result = await browser.agent("Click the checkout button")
# If a popup appears, agent handles it and retries
```

2. Force obstacle detection before action:
```python
# Manually wait for potential dynamic content
await asyncio.sleep(2)  # Allow time for JavaScript popups to appear
result = await browser.act("Click the button")
```

3. Check logs for obstacle detection:
```python
import logging
logging.getLogger("flybrowser.agents").setLevel(logging.DEBUG)
# Look for: [DynamicObstacle] ✓ Handled X obstacle(s)
```

### Dynamic Popups Not Being Dismissed

**Symptoms:**
- Newsletter popups keep appearing
- Cookie banners not automatically handled
- JavaScript-triggered modals block interaction

**Solutions:**

1. Check if obstacle detection is enabled (default: on):
```python
# Obstacle detection happens automatically before each screenshot
# and on click failures - no configuration needed
```

2. Verify the popup type is supported:
```
# Supported newsletter tools: MailPoet, Mailchimp, HubSpot, Klaviyo
# Supported consent tools: OneTrust, CookieBot, Quantcast, Termly
# Supported modal frameworks: Bootstrap, MUI, React-Modal, Ant Design
```

3. Check detection confidence in logs:
```python
import logging
logging.getLogger("flybrowser.agents.obstacle_detector").setLevel(logging.DEBUG)
# Look for: [QuickCheck] confidence=X.XX
# Popups are handled if confidence > 0.3
```

4. If popup uses non-standard implementation:
```python
# Use explicit act() instruction
await browser.act("Click the X button to close the popup")
```

## Navigation Issues

### Page Not Loading

**Symptoms:**
- `NavigationError: Navigation failed`
- Blank page after navigation

**Solutions:**

1. Check URL format:
```python
# Include protocol
await browser.goto("https://example.com")  # Not just "example.com"
```

2. Handle redirects:
```python
await browser.goto(
    "https://example.com",
    wait_until="networkidle"
)
```

3. Check for JavaScript errors:
```python
browser.page.on("console", lambda msg: print(f"Console: {msg.text}"))
```

### SSL Certificate Errors

**Symptoms:**
- `NavigationError: SSL certificate error`
- `net::ERR_CERT_AUTHORITY_INVALID`

**Solutions:**

1. Ignore HTTPS errors (development only):
```python
browser = FlyBrowser(
    browser_options={"ignoreHTTPSErrors": True}
)
```

2. Check system certificates:
```bash
# Update CA certificates
sudo update-ca-certificates
```

### Cookie/Session Issues

**Symptoms:**
- Login state not persisted
- Unexpected logouts

**Solutions:**

1. Use persistent context:
```python
browser = FlyBrowser(
    user_data_dir="./browser_data"
)
```

2. Save and restore cookies:
```python
# Save
cookies = await browser.context.cookies()

# Restore
await browser.context.add_cookies(cookies)
```

## Server Issues

### Connection Refused

**Symptoms:**
- `ConnectionError: Connection refused`
- Cannot connect to FlyBrowser server

**Solutions:**

1. Check server is running:
```bash
curl http://localhost:8000/health
```

2. Check port binding:
```bash
netstat -tlnp | grep 8000
```

3. Check firewall rules:
```bash
sudo ufw allow 8000/tcp
```

### Memory Issues

**Symptoms:**
- `MemoryError: Out of memory`
- Server becomes unresponsive

**Solutions:**

1. Reduce pool size:
```bash
FLYBROWSER_POOL__MAX_SIZE=5
```

2. Reduce session timeout:
```bash
FLYBROWSER_SESSION_TIMEOUT=1800
```

3. Monitor memory:
```bash
docker stats flybrowser
```

### High CPU Usage

**Symptoms:**
- CPU consistently at 100%
- Slow response times

**Solutions:**

1. Use headless mode:
```bash
FLYBROWSER_POOL__HEADLESS=true
```

2. Limit concurrent operations:
```bash
FLYBROWSER_MAX_SESSIONS=10
```

3. Use resource limits:
```bash
docker run --cpus=2 --memory=4g ...
```

## Debug Techniques

### Enable Debug Logging

```python
import logging

# Enable all FlyBrowser logging
logging.basicConfig(level=logging.DEBUG)

# Or specific modules
logging.getLogger("flybrowser").setLevel(logging.DEBUG)
logging.getLogger("flybrowser.agents").setLevel(logging.DEBUG)
logging.getLogger("flybrowser.agents.obstacle_detector").setLevel(logging.DEBUG)
```

### Debug Vision Behavior

Check why screenshots are or aren't being captured:

```python
import logging
logging.getLogger("flybrowser.agents.react_agent").setLevel(logging.DEBUG)

# Look for these log messages:
# [VISION] Skipped: page is blank (about:blank)  - Expected on first iteration
# [VISION:MODE] Trigger: reason                   - Screenshot being captured
# [DynamicObstacle] ✓ Handled N obstacle(s)       - Popup dismissed
```

**Note on Blank Page Vision Skip:**
Screenshots are automatically skipped for `about:blank` pages to save resources.
This is normal behavior on the first iteration before navigation.

### Inspect Page State

```python
async with FlyBrowser() as browser:
    await browser.goto("https://example.com")
    
    # Get page HTML
    html = await browser.page.content()
    
    # Get page title
    title = await browser.page.title()
    
    # Get current URL
    url = browser.page.url
    
    # Take screenshot
    await browser.screenshot(path="debug.png")
    
    print(f"URL: {url}")
    print(f"Title: {title}")
```

### Monitor Network Requests

```python
async with FlyBrowser() as browser:
    # Log all requests
    browser.page.on("request", lambda req: print(f">> {req.method} {req.url}"))
    browser.page.on("response", lambda res: print(f"<< {res.status} {res.url}"))
    
    await browser.goto("https://example.com")
```

### Capture HAR File

```python
async with FlyBrowser() as browser:
    # Start HAR recording
    await browser.context.tracing.start(
        screenshots=True,
        snapshots=True,
    )
    
    await browser.goto("https://example.com")
    await browser.extract("Get data")
    
    # Save trace
    await browser.context.tracing.stop(path="trace.zip")
```

### Interactive Debugging

```python
async with FlyBrowser(headless=False) as browser:
    await browser.goto("https://example.com")
    
    # Pause for manual inspection
    input("Press Enter to continue...")
    
    await browser.extract("Get data")
```

## Getting Help

If you're still stuck:

1. Check the [GitHub Issues](https://github.com/flybrowser/flybrowser/issues)
2. Search existing discussions
3. Create a new issue with:
   - Python version
   - FlyBrowser version
   - Operating system
   - Full error traceback
   - Minimal reproduction code

## See Also

- [Performance Optimization](performance.md) - Speed up operations
- [Error Handling Guide](../guides/error-handling.md) - Handle errors gracefully
- [Configuration Reference](../reference/configuration.md) - All settings
