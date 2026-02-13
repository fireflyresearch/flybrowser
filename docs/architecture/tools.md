# Tools System

FlyBrowser uses ToolKits built on `fireflyframework_genai.tools.toolkit.ToolKit` for browser interactions. Tools are organized into six ToolKits that are registered with the `FireflyAgent`.

## Overview

Each ToolKit groups related browser tools. All ToolKits are created via `create_all_toolkits()` and passed to `FireflyAgent` during `BrowserAgent` initialization.

## ToolKit Architecture

```
BrowserAgent
    |
    +-- FireflyAgent(tools=toolkits)
            |
            +-- NavigationToolkit    (goto, back, forward, refresh)
            +-- InteractionToolkit   (click, type, scroll, hover, select, ...)
            +-- ExtractionToolkit    (extract_text, screenshot, get_page_state)
            +-- SystemToolkit        (complete, fail, wait, ask_user)
            +-- SearchToolkit        (web search)
            +-- CaptchaToolkit       (CAPTCHA solving)
```

## Creating ToolKits

All six ToolKits are created in `flybrowser/agents/toolkits/__init__.py`:

```python
from fireflyframework_genai.tools.toolkit import ToolKit

def create_all_toolkits(
    page: PageController,
    search_coordinator=None,
    captcha_solver=None,
    user_input_callback=None,
) -> List[ToolKit]:
    """Create all 6 browser ToolKits."""
    return [
        create_navigation_toolkit(page),
        create_interaction_toolkit(page),
        create_extraction_toolkit(page),
        create_system_toolkit(user_input_callback=user_input_callback),
        create_search_toolkit(search_coordinator),
        create_captcha_toolkit(page, captcha_solver),
    ]
```

## Built-in ToolKits

### Navigation ToolKit

| Tool | Description |
|------|-------------|
| `goto` | Navigate to a URL |
| `back` | Go back in history |
| `forward` | Go forward in history |
| `refresh` | Reload the page |

### Interaction ToolKit

| Tool | Description |
|------|-------------|
| `click` | Click an element |
| `type` | Type text into an element |
| `scroll` | Scroll the page |
| `hover` | Hover over an element |
| `press_key` | Press a keyboard key |
| `select` | Select from dropdown |
| `check` | Check/uncheck checkbox |

### Extraction ToolKit

| Tool | Description |
|------|-------------|
| `extract_text` | Extract text from page |
| `screenshot` | Take a screenshot |
| `get_page_state` | Get page structure and elements |

### System ToolKit

| Tool | Description |
|------|-------------|
| `complete` | Mark task complete with result |
| `fail` | Mark task failed with reason |
| `wait` | Wait for condition |
| `ask_user` | Request user input |

### Search ToolKit

| Tool | Description |
|------|-------------|
| `search` | Web search via configured provider |

### Captcha ToolKit

| Tool | Description |
|------|-------------|
| `solve_captcha` | Solve CAPTCHA challenges |

## How Tools Are Used

The `FireflyAgent` (from fireflyframework-genai) manages tool discovery and execution. When the LLM decides to call a tool:

1. The framework matches the tool name to a registered ToolKit tool
2. Parameters are validated
3. The tool executes against the `PageController`
4. The result is returned to the reasoning loop

## Tool Result

Tools return structured results:

```python
@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## See Also

- [Architecture Overview](overview.md) - System architecture
- [ReAct Framework](react.md) - How tools are used in reasoning
- [Custom Tools Guide](../advanced/custom-tools.md) - Building custom tools
