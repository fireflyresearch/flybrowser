# SDK Reference

This document provides complete reference documentation for the FlyBrowser Python SDK.

## FlyBrowser Class

The `FlyBrowser` class is the main entry point for embedded browser automation.

### Import

```python path=null start=null
from flybrowser import FlyBrowser
```

### Constructor

```python path=null start=null
FlyBrowser(
    endpoint: str = None,
    llm_provider: str = "openai",
    llm_model: str = None,  # Uses provider default if not specified
    api_key: str = None,
    base_url: str = None,
    headless: bool = True,
    browser_type: str = "chromium",
    recording_enabled: bool = False,
    pii_masking_enabled: bool = True,
    timeout: float = 30.0
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | `None` | Remote server endpoint (for client mode) |
| `llm_provider` | `str` | `"openai"` | LLM provider (see supported providers below) |
| `llm_model` | `str` | (provider default) | Model name for the LLM provider |
| `api_key` | `str` | `None` | API key (or use environment variable) |
| `base_url` | `str` | `None` | Custom endpoint URL (for local providers) |
| `headless` | `bool` | `True` | Run browser without visible window |
| `browser_type` | `str` | `"chromium"` | Browser engine: `"chromium"`, `"firefox"`, or `"webkit"` |
| `recording_enabled` | `bool` | `False` | Enable session recording |
| `pii_masking_enabled` | `bool` | `True` | Enable automatic PII masking |
| `timeout` | `float` | `30.0` | Default operation timeout in seconds |

#### Supported LLM Providers

| Provider | `llm_provider` value | Default Model | Environment Variable |
|----------|---------------------|---------------|---------------------|
| OpenAI | `"openai"` | `gpt-5.2` | `OPENAI_API_KEY` |
| Anthropic | `"anthropic"` | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Google Gemini | `"gemini"` or `"google"` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Ollama | `"ollama"` | `qwen3:8b` | `OLLAMA_HOST` (optional) |
| LM Studio | `"lm_studio"` | (user-defined) | - |
| LocalAI | `"localai"` | (user-defined) | - |
| vLLM | `"vllm"` | (user-defined) | - |

#### Examples

**OpenAI (default):**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",  # Or: gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4o
    headless=True
)
```

**Anthropic:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929"  # Or: claude-haiku-4-5-20251001, claude-opus-4-5-20251101
)
```

**Google Gemini:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="gemini",  # or "google"
    llm_model="gemini-2.0-flash"  # Or: gemini-2.0-pro, gemini-1.5-pro, gemini-1.5-flash
)
```

**Ollama (local):**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b"  # Or: gemma3:12b, llama3.2:3b, deepseek-r1:8b, phi4
)
```

**vLLM (high-throughput):**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="vllm",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000"
)
```

### Methods

#### start()

Initializes Playwright and launches the browser.

```python path=null start=null
async def start() -> None
```

**Returns**: None

**Raises**: 
- `RuntimeError`: If browser fails to start

**Example**:
```python path=null start=null
await browser.start()
```

#### stop()

Closes the browser and releases all resources.

```python path=null start=null
async def stop() -> None
```

**Returns**: None

**Example**:
```python path=null start=null
await browser.stop()
```

#### goto(url)

Navigates to a specific URL.

```python path=null start=null
async def goto(url: str) -> None
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | The URL to navigate to |

**Returns**: None

**Raises**:
- `TimeoutError`: If navigation times out
- `ConnectionError`: If the URL cannot be reached

**Example**:
```python path=null start=null
await browser.goto("https://example.com")
```

#### navigate(instruction)

Navigates using natural language instructions.

```python path=null start=null
async def navigate(instruction: str) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `instruction` | `str` | Natural language navigation instruction |

**Returns**: `dict` - Navigation result containing status and any extracted information

**Example**:
```python path=null start=null
result = await browser.navigate("Go to the login page")
result = await browser.navigate("Click on Products in the menu")
```

#### extract(query)

Extracts data from the current page using natural language.

```python path=null start=null
async def extract(query: str) -> str
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Natural language extraction query |

**Returns**: `str` - Extracted data based on the query

**Example**:
```python path=null start=null
title = await browser.extract("What is the page title?")
prices = await browser.extract("List all product prices")
```

#### act(command)

Performs an action on the page using natural language.

```python path=null start=null
async def act(command: str) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `str` | Natural language action command |

**Returns**: `dict` - Action result containing status and details

**Example**:
```python path=null start=null
await browser.act("Click the Submit button")
await browser.act("Type 'hello' in the search field")
await browser.act("Select 'Option 1' from the dropdown")
```

#### screenshot()

Captures a screenshot of the current page.

```python path=null start=null
async def screenshot() -> bytes
```

**Returns**: `bytes` - PNG image data

**Example**:
```python path=null start=null
image_data = await browser.screenshot()
with open("screenshot.png", "wb") as f:
    f.write(image_data)
```

#### run_workflow(workflow)

Executes a predefined workflow.

```python path=null start=null
async def run_workflow(workflow: dict) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `workflow` | `dict` | Workflow definition |

**Workflow Structure**:
```python path=null start=null
{
    "name": "workflow_name",
    "steps": [
        {"action": "goto", "url": "https://example.com"},
        {"action": "act", "command": "Click login"},
        {"action": "extract", "query": "Get the user name"}
    ]
}
```

**Step Actions**:
- `goto`: Navigate to URL (requires `url` field)
- `act`: Perform action (requires `command` field)
- `extract`: Extract data (requires `query` field)
- `screenshot`: Capture screenshot

**Returns**: `dict` - Workflow execution results

**Example**:
```python path=null start=null
workflow = {
    "name": "login_check",
    "steps": [
        {"action": "goto", "url": "https://example.com/login"},
        {"action": "act", "command": "Enter credentials"},
        {"action": "extract", "query": "Is login successful?"}
    ]
}
result = await browser.run_workflow(workflow)
```

#### monitor()

Returns the current browser status.

```python path=null start=null
async def monitor() -> dict
```

**Returns**: `dict` - Current browser state

**Response Structure**:
```python path=null start=null
{
    "url": "https://example.com/page",
    "title": "Page Title",
    "state": "ready"
}
```

**Example**:
```python path=null start=null
status = await browser.monitor()
print(f"Current URL: {status['url']}")
```

#### store_credential(name, value)

Stores a credential securely.

```python path=null start=null
async def store_credential(name: str, value: str) -> None
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Credential identifier |
| `value` | `str` | Credential value |

**Example**:
```python path=null start=null
await browser.store_credential("email", "user@example.com")
await browser.store_credential("password", "secret123")
```

#### secure_fill(field, credential_name)

Fills a form field using a stored credential.

```python path=null start=null
async def secure_fill(field: str, credential_name: str) -> None
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `field` | `str` | Field identifier or description |
| `credential_name` | `str` | Name of the stored credential |

**Example**:
```python path=null start=null
await browser.secure_fill("email", "email")
await browser.secure_fill("password", "password")
```

#### mask_pii(text)

Masks personally identifiable information in text.

```python path=null start=null
async def mask_pii(text: str) -> str
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text containing potential PII |

**Returns**: `str` - Text with PII masked

**Example**:
```python path=null start=null
masked = await browser.mask_pii("Contact: john@example.com, 555-1234")
# Result: "Contact: [EMAIL], [PHONE]"
```

### Context Manager

`FlyBrowser` supports async context manager protocol:

```python path=null start=null
async with FlyBrowser(llm_provider="openai", llm_model="gpt-5.2") as browser:
    await browser.goto("https://example.com")
    data = await browser.extract("Get the content")
```

## FlyBrowserClient Class

The `FlyBrowserClient` class provides a client interface to a remote FlyBrowser server.

### Import

```python path=null start=null
from flybrowser import FlyBrowserClient
```

### Constructor

```python path=null start=null
FlyBrowserClient(
    endpoint: str,
    api_key: str = None,
    timeout: float = 30.0
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | (required) | Server URL (e.g., `"http://localhost:8000"`) |
| `api_key` | `str` | `None` | API key for authentication |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

#### Example

```python path=null start=null
client = FlyBrowserClient(
    endpoint="http://localhost:8000",
    api_key="your-api-key"
)
```

### Methods

#### create_session(**kwargs)

Creates a new browser session on the server.

```python path=null start=null
async def create_session(
    llm_provider: str = "openai",
    llm_model: str = None,
    api_key: str = None,
    base_url: str = None,
    headless: bool = True,
    browser_type: str = "chromium"
) -> dict
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_provider` | `str` | `"openai"` | LLM provider (openai, anthropic, gemini, ollama, etc.) |
| `llm_model` | `str` | (provider default) | Model name |
| `api_key` | `str` | `None` | LLM API key |
| `base_url` | `str` | `None` | Custom endpoint for local providers |
| `headless` | `bool` | `True` | Headless mode |
| `browser_type` | `str` | `"chromium"` | Browser engine |

**Returns**: `dict` - Session information including `session_id`

**Example**:
```python path=null start=null
# With OpenAI
session = await client.create_session(
    llm_provider="openai",
    llm_model="gpt-5.2"
)

# With Gemini
session = await client.create_session(
    llm_provider="gemini",
    llm_model="gemini-2.0-flash"
)

# With local Ollama
session = await client.create_session(
    llm_provider="ollama",
    llm_model="qwen3:8b"
)

session_id = session["session_id"]
```

#### close_session(session_id)

Closes a browser session.

```python path=null start=null
async def close_session(session_id: str) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |

**Returns**: `dict` - Close confirmation

**Example**:
```python path=null start=null
await client.close_session("sess_abc123")
```

#### navigate(session_id, url)

Navigates to a URL.

```python path=null start=null
async def navigate(session_id: str, url: str) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |
| `url` | `str` | URL to navigate to |

**Returns**: `dict` - Navigation result

**Example**:
```python path=null start=null
await client.navigate("sess_abc123", "https://example.com")
```

#### extract(session_id, query)

Extracts data from the page.

```python path=null start=null
async def extract(session_id: str, query: str) -> str
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |
| `query` | `str` | Extraction query |

**Returns**: `str` - Extracted data

**Example**:
```python path=null start=null
result = await client.extract("sess_abc123", "Get the page title")
```

#### action(session_id, command)

Performs an action on the page.

```python path=null start=null
async def action(session_id: str, command: str) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |
| `command` | `str` | Action command |

**Returns**: `dict` - Action result

**Example**:
```python path=null start=null
await client.action("sess_abc123", "Click the Submit button")
```

#### screenshot(session_id)

Captures a screenshot.

```python path=null start=null
async def screenshot(session_id: str) -> bytes
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |

**Returns**: `bytes` - PNG image data

**Example**:
```python path=null start=null
image = await client.screenshot("sess_abc123")
```

#### run_workflow(session_id, workflow)

Executes a workflow.

```python path=null start=null
async def run_workflow(session_id: str, workflow: dict) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |
| `workflow` | `dict` | Workflow definition |

**Returns**: `dict` - Workflow results

**Example**:
```python path=null start=null
result = await client.run_workflow("sess_abc123", {
    "name": "example",
    "steps": [...]
})
```

#### monitor(session_id)

Gets session status.

```python path=null start=null
async def monitor(session_id: str) -> dict
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier |

**Returns**: `dict` - Session status

**Example**:
```python path=null start=null
status = await client.monitor("sess_abc123")
```

## Error Handling

### Exception Types

| Exception | Description |
|-----------|-------------|
| `TimeoutError` | Operation exceeded timeout |
| `ConnectionError` | Failed to connect to URL or server |
| `ValueError` | Invalid parameter value |
| `RuntimeError` | Browser or session error |

### Example

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(llm_provider="openai", llm_model="gpt-5.2")

try:
    await browser.start()
    await browser.goto("https://example.com")
    result = await browser.extract("Get data")
except TimeoutError:
    print("Operation timed out")
except ConnectionError:
    print("Connection failed")
except Exception as e:
    print(f"Error: {e}")
finally:
    await browser.stop()
```

## Type Definitions

### WorkflowStep

```python path=null start=null
class WorkflowStep(TypedDict):
    action: str  # "goto", "act", "extract", "screenshot"
    url: Optional[str]  # Required for "goto"
    command: Optional[str]  # Required for "act"
    query: Optional[str]  # Required for "extract"
```

### Workflow

```python path=null start=null
class Workflow(TypedDict):
    name: str
    steps: List[WorkflowStep]
```

### SessionInfo

```python path=null start=null
class SessionInfo(TypedDict):
    session_id: str
    status: str
    created_at: str
```

### MonitorStatus

```python path=null start=null
class MonitorStatus(TypedDict):
    url: str
    title: str
    state: str
```
