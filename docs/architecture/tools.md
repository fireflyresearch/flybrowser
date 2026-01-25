# Tools System

FlyBrowser uses a tool-based architecture for browser interactions. This document explains the tools system design and implementation.

## Overview

Tools are the primary mechanism for the agent to interact with the browser. Each tool:

- Performs a specific action (click, type, navigate, etc.)
- Has typed parameters with validation
- Returns structured results
- Includes metadata for documentation and safety

## Tool Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tool Registry                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  register() | get_tool() | list_tools() | filter()      │   │
│   └───────────────────────────┬─────────────────────────────┘   │
│                               │                                 │
│      ┌────────────────────────┼────────────────────┐            │
│      │                        │                    │            │
│      ▼                        ▼                    ▼            │
│ ┌──────────┐          ┌───────────┐          ┌──────────┐       │
│ │Navigation│          │Interaction│          │Extraction│       │
│ │  Tools   │          │  Tools    │          │  Tools   │       │
│ └──────────┘          └───────────┘          └──────────┘       │
│  goto                  click                 extract_text       │
│  navigate              type                  extract_structured │
│  back                  scroll                get_page_content   │
│  forward               hover                 screenshot         │
└─────────────────────────────────────────────────────────────────┘
```

## Base Tool Class

All tools inherit from `BaseTool`:

```python
class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools are the primary mechanism for the agent to take actions.
    Each tool must define its metadata and implement the execute method.
    """
    
    metadata: ToolMetadata  # Must be defined by subclass
    
    def __init__(self, page_controller: Optional[PageController] = None) -> None:
        """Initialize the tool with optional page controller."""
        self.page = page_controller
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters."""
        raise NotImplementedError("Subclasses must implement execute()")
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against the tool's JSON schema."""
        ...
```

## Tool Metadata

Tools are described with metadata:

```python
@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities."""
    
    name: str                           # Unique tool name
    description: str                    # Human-readable description
    category: ToolCategory              # Tool category
    safety_level: SafetyLevel           # Safety classification
    parameters: List[ToolParameter]     # Parameter definitions
    returns_description: str            # Return value description
    examples: List[str]                 # Usage examples
    requires_page: bool                 # Whether tool needs PageController
    timeout_seconds: float              # Execution timeout
    is_terminal: bool                   # Whether tool ends the task
    required_capabilities: List[ModelCapability]  # Required model capabilities
    optimal_capabilities: List[ModelCapability]   # Preferred capabilities
```

## Tool Categories

Tools are organized by category:

```python
class ToolCategory(str, Enum):
    """Categories of tools."""
    
    NAVIGATION = "navigation"       # goto, navigate, back, forward
    INTERACTION = "interaction"     # click, type, scroll, hover
    EXTRACTION = "extraction"       # extract, screenshot, get_content
    FORM = "form"                   # fill_form, submit, select
    SYSTEM = "system"               # wait, complete, fail
    UTILITY = "utility"             # log, debug
```

## Safety Levels

Tools have safety classifications:

```python
class SafetyLevel(str, Enum):
    """Safety level for tools."""
    
    SAFE = "safe"           # No side effects (read-only)
    LOW_RISK = "low_risk"   # Minor side effects (clicks)
    MEDIUM_RISK = "medium"  # Moderate effects (form submission)
    HIGH_RISK = "high"      # Significant effects (file operations)
    DANGEROUS = "dangerous" # Potentially harmful (authentication)
```

## Tool Parameters

Parameters are defined with type information:

```python
@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    
    name: str                           # Parameter name
    type: str                           # JSON type (string, number, etc.)
    description: str = ""               # Description
    required: bool = False              # Whether required
    default: Any = None                 # Default value
    enum: Optional[List[Any]] = None   # Allowed values
    items_type: Optional[str] = None   # For array types
```

## Tool Result

Tools return structured results:

```python
@dataclass
class ToolResult:
    """Result of a tool execution."""
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success_result(cls, data: Any, **metadata) -> "ToolResult":
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def failure_result(cls, error: str, **metadata) -> "ToolResult":
        return cls(success=False, error=error, metadata=metadata)
```

## Tool Registry

The registry manages tool discovery and instantiation:

```python
class ToolRegistry:
    """Registry for managing available tools."""
    
    def register(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool class."""
        ...
    
    def register_instance(self, tool_instance: BaseTool) -> None:
        """Register a pre-instantiated tool."""
        ...
    
    def get_tool(self, name: str, **kwargs) -> BaseTool:
        """Get a tool instance by name."""
        ...
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        ...
    
    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a category."""
        ...
    
    def get_filtered_registry(
        self,
        capabilities: List[ModelCapability],
        warn_suboptimal: bool = False,
    ) -> "ToolRegistry":
        """Get registry filtered by model capabilities."""
        ...
```

## Implementing a Tool

Example of a custom tool implementation:

```python
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel, ToolResult

class ClickTool(BaseTool):
    """Tool for clicking elements on the page."""
    
    metadata = ToolMetadata(
        name="click",
        description="Click on an element identified by selector",
        category=ToolCategory.INTERACTION,
        safety_level=SafetyLevel.LOW_RISK,
        parameters=[
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the element to click",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="number",
                description="Timeout in milliseconds",
                default=5000,
            ),
        ],
        returns_description="Confirmation of click action",
        examples=[
            'click(selector="#submit-btn")',
            'click(selector=".nav-link", timeout=10000)',
        ],
        requires_page=True,
        timeout_seconds=30.0,
    )
    
    async def execute(
        self,
        selector: str,
        timeout: int = 5000,
        **kwargs
    ) -> ToolResult:
        """Execute the click action."""
        try:
            element = await self.page.wait_for_selector(
                selector, 
                timeout=timeout
            )
            await element.click()
            
            return ToolResult.success_result({
                "clicked": selector,
                "success": True,
            })
        except Exception as e:
            return ToolResult.failure_result(
                f"Failed to click {selector}: {str(e)}"
            )
```

## Built-in Tools

FlyBrowser includes 32+ built-in tools:

### Navigation Tools

| Tool | Description |
|------|-------------|
| `goto` | Navigate to a URL |
| `navigate` | Navigate using natural language |
| `back` | Go back in history |
| `forward` | Go forward in history |
| `refresh` | Reload the page |

### Interaction Tools

| Tool | Description |
|------|-------------|
| `click` | Click an element |
| `type` | Type text into an element |
| `scroll` | Scroll the page |
| `hover` | Hover over an element |
| `press_key` | Press a keyboard key |
| `select` | Select from dropdown |
| `check` | Check/uncheck checkbox |

### Extraction Tools

| Tool | Description |
|------|-------------|
| `extract_text` | Extract text from page |
| `extract_structured` | Extract structured data |
| `get_page_content` | Get page HTML |
| `screenshot` | Take a screenshot |
| `get_element_info` | Get element details |

### System Tools

| Tool | Description |
|------|-------------|
| `wait` | Wait for condition |
| `complete` | Mark task complete |
| `fail` | Mark task failed |

## Capability-Based Filtering

Tools can require or benefit from model capabilities:

```python
class ScreenshotTool(BaseTool):
    metadata = ToolMetadata(
        name="screenshot",
        description="Capture screenshot for visual analysis",
        # This tool requires vision capability
        required_capabilities=[ModelCapability.VISION],
        optimal_capabilities=[ModelCapability.VISION],
        ...
    )
```

When the registry is filtered:

```python
# Filter tools based on model capabilities
filtered_registry = registry.get_filtered_registry(
    capabilities=model_info.capabilities,
    warn_suboptimal=True,  # Log warnings for suboptimal tools
)

# Text-only models won't see vision-required tools
if ModelCapability.VISION not in capabilities:
    # screenshot tool excluded
```

## JSON Schema Generation

Tools generate JSON schemas for LLM prompts:

```python
schema = tool.metadata.to_json_schema()
# {
#     "type": "object",
#     "properties": {
#         "selector": {
#             "type": "string",
#             "description": "CSS selector for the element to click"
#         },
#         "timeout": {
#             "type": "number",
#             "description": "Timeout in milliseconds",
#             "default": 5000
#         }
#     },
#     "required": ["selector"]
# }
```

## Tool Execution Flow

```
Agent decides to use tool
         │
         ▼
┌─────────────────┐
│ Get tool from   │
│    registry     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validate      │
│  parameters     │
└────────┬────────┘
         │
    Valid? ───No──► Return error
         │
        Yes
         │
         ▼
┌─────────────────┐
│    Execute      │
│   with timeout  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return result  │
│  (ToolResult)   │
└─────────────────┘
```

## Registering Custom Tools

Register custom tools for specialized automation:

```python
from flybrowser import FlyBrowser
from flybrowser.agents.tools.base import BaseTool, ToolMetadata
from flybrowser.agents.types import ToolResult

class MyCustomTool(BaseTool):
    metadata = ToolMetadata(
        name="my_custom_tool",
        description="Does something custom",
        ...
    )
    
    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
        ...

# Register before creating browser
async with FlyBrowser(...) as browser:
    browser.react_agent.tool_registry.register(MyCustomTool)
    
    # Now the agent can use my_custom_tool
    await browser.agent("Use my custom tool to...")
```

## See Also

- [Architecture Overview](overview.md) - System architecture
- [ReAct Framework](react.md) - How tools are used
- [Custom Tools Guide](../advanced/custom-tools.md) - Building custom tools
