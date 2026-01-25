# Custom Tools

FlyBrowser allows you to create custom tools that extend the agent's capabilities. This guide explains how to build, register, and use custom tools.

## Overview

Custom tools enable:
- Domain-specific functionality
- Integration with external services
- Specialized browser operations
- Custom data processing

## Creating a Custom Tool

### Basic Tool Structure

```python
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel, ToolResult

class MyCustomTool(BaseTool):
    """Custom tool implementation."""
    
    metadata = ToolMetadata(
        name="my_custom_tool",
        description="Description of what the tool does",
        category=ToolCategory.UTILITY,
        safety_level=SafetyLevel.SAFE,
        parameters=[
            ToolParameter(
                name="param1",
                type="string",
                description="Parameter description",
                required=True,
            ),
        ],
        returns_description="What the tool returns",
        examples=["my_custom_tool(param1='value')"],
        requires_page=False,
        timeout_seconds=30.0,
    )
    
    async def execute(self, param1: str, **kwargs) -> ToolResult:
        """Execute the tool."""
        try:
            # Implementation here
            result = f"Processed: {param1}"
            return ToolResult.success_result(result)
        except Exception as e:
            return ToolResult.failure_result(str(e))
```

### Tool Metadata

Configure tool behavior with `ToolMetadata`:

```python
@dataclass
class ToolMetadata:
    name: str                           # Unique tool name
    description: str                    # LLM-readable description
    category: ToolCategory              # Tool category
    safety_level: SafetyLevel           # Safety classification
    parameters: List[ToolParameter]     # Parameter definitions
    returns_description: str            # Return value description
    examples: List[str]                 # Usage examples
    requires_page: bool                 # Needs PageController?
    timeout_seconds: float              # Execution timeout
    is_terminal: bool = False           # Ends task?
    required_capabilities: List[ModelCapability] = field(default_factory=list)
    optimal_capabilities: List[ModelCapability] = field(default_factory=list)
```

### Tool Categories

Choose the appropriate category:

```python
class ToolCategory(str, Enum):
    NAVIGATION = "navigation"       # goto, back, forward
    INTERACTION = "interaction"     # click, type, scroll
    EXTRACTION = "extraction"       # extract, screenshot
    FORM = "form"                   # fill_form, submit
    SYSTEM = "system"               # wait, complete, fail
    UTILITY = "utility"             # custom/helper tools
```

### Safety Levels

Declare the safety level:

```python
class SafetyLevel(str, Enum):
    SAFE = "safe"           # Read-only, no side effects
    LOW_RISK = "low_risk"   # Minor side effects (clicks)
    MEDIUM_RISK = "medium"  # Moderate effects (form submission)
    HIGH_RISK = "high"      # Significant effects (file operations)
    DANGEROUS = "dangerous" # Potentially harmful
```

### Parameters

Define parameters with `ToolParameter`:

```python
ToolParameter(
    name="selector",
    type="string",              # string, number, integer, boolean, array, object
    description="CSS selector",
    required=True,
    default=None,
    enum=["option1", "option2"],  # Allowed values (optional)
    items_type="string",          # For array types
)
```

## Examples

### Page Interaction Tool

```python
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel, ToolResult

class HighlightElementTool(BaseTool):
    """Highlight an element on the page."""
    
    metadata = ToolMetadata(
        name="highlight_element",
        description="Highlight an element with a colored border",
        category=ToolCategory.INTERACTION,
        safety_level=SafetyLevel.LOW_RISK,
        parameters=[
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the element",
                required=True,
            ),
            ToolParameter(
                name="color",
                type="string",
                description="Border color",
                default="red",
            ),
            ToolParameter(
                name="duration_ms",
                type="integer",
                description="Highlight duration in milliseconds",
                default=2000,
            ),
        ],
        returns_description="Confirmation of highlight",
        examples=[
            'highlight_element(selector="#submit-btn")',
            'highlight_element(selector=".nav-link", color="blue")',
        ],
        requires_page=True,
        timeout_seconds=10.0,
    )
    
    async def execute(
        self,
        selector: str,
        color: str = "red",
        duration_ms: int = 2000,
        **kwargs
    ) -> ToolResult:
        """Execute the highlight action."""
        try:
            # Use the page controller
            await self.page.evaluate(f"""
                const el = document.querySelector('{selector}');
                if (el) {{
                    el.style.outline = '3px solid {color}';
                    setTimeout(() => el.style.outline = '', {duration_ms});
                }}
            """)
            
            return ToolResult.success_result({
                "highlighted": selector,
                "color": color,
                "duration_ms": duration_ms,
            })
        except Exception as e:
            return ToolResult.failure_result(f"Failed to highlight: {e}")
```

### External API Tool

```python
import httpx
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel, ToolResult

class WeatherTool(BaseTool):
    """Get weather information for a location."""
    
    metadata = ToolMetadata(
        name="get_weather",
        description="Get current weather for a city",
        category=ToolCategory.UTILITY,
        safety_level=SafetyLevel.SAFE,
        parameters=[
            ToolParameter(
                name="city",
                type="string",
                description="City name",
                required=True,
            ),
        ],
        returns_description="Weather information including temperature and conditions",
        examples=[
            'get_weather(city="San Francisco")',
        ],
        requires_page=False,
        timeout_seconds=30.0,
    )
    
    async def execute(self, city: str, **kwargs) -> ToolResult:
        """Fetch weather data."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.weather.example/v1/current",
                    params={"city": city},
                )
                data = response.json()
            
            return ToolResult.success_result({
                "city": city,
                "temperature": data["temp"],
                "conditions": data["conditions"],
            })
        except Exception as e:
            return ToolResult.failure_result(f"Weather fetch failed: {e}")
```

### Data Processing Tool

```python
import json
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel, ToolResult

class JsonValidatorTool(BaseTool):
    """Validate and format JSON data."""
    
    metadata = ToolMetadata(
        name="validate_json",
        description="Validate JSON and return formatted version",
        category=ToolCategory.UTILITY,
        safety_level=SafetyLevel.SAFE,
        parameters=[
            ToolParameter(
                name="json_string",
                type="string",
                description="JSON string to validate",
                required=True,
            ),
            ToolParameter(
                name="indent",
                type="integer",
                description="Indentation level for formatting",
                default=2,
            ),
        ],
        returns_description="Validation result and formatted JSON",
        examples=[
            'validate_json(json_string=\'{"key": "value"}\')',
        ],
        requires_page=False,
        timeout_seconds=5.0,
    )
    
    async def execute(
        self, 
        json_string: str, 
        indent: int = 2, 
        **kwargs
    ) -> ToolResult:
        """Validate and format JSON."""
        try:
            parsed = json.loads(json_string)
            formatted = json.dumps(parsed, indent=indent)
            
            return ToolResult.success_result({
                "valid": True,
                "formatted": formatted,
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
            })
        except json.JSONDecodeError as e:
            return ToolResult.failure_result(f"Invalid JSON: {e}")
```

### Vision-Required Tool

```python
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel, ToolResult
from flybrowser.llm.base import ModelCapability

class VisualComparisonTool(BaseTool):
    """Compare two screenshots for visual differences."""
    
    metadata = ToolMetadata(
        name="visual_compare",
        description="Compare current page with a reference screenshot",
        category=ToolCategory.EXTRACTION,
        safety_level=SafetyLevel.SAFE,
        parameters=[
            ToolParameter(
                name="reference_path",
                type="string",
                description="Path to reference screenshot",
                required=True,
            ),
            ToolParameter(
                name="threshold",
                type="number",
                description="Difference threshold (0-1)",
                default=0.1,
            ),
        ],
        returns_description="Comparison result with diff score",
        examples=[
            'visual_compare(reference_path="baseline.png")',
        ],
        requires_page=True,
        timeout_seconds=30.0,
        required_capabilities=[ModelCapability.VISION],
    )
    
    async def execute(
        self,
        reference_path: str,
        threshold: float = 0.1,
        **kwargs
    ) -> ToolResult:
        """Compare screenshots."""
        try:
            # Take current screenshot
            current = await self.page.screenshot()
            
            # Load reference
            with open(reference_path, "rb") as f:
                reference = f.read()
            
            # Compare (simplified example)
            diff_score = self._compare_images(current, reference)
            
            return ToolResult.success_result({
                "match": diff_score <= threshold,
                "diff_score": diff_score,
                "threshold": threshold,
            })
        except Exception as e:
            return ToolResult.failure_result(f"Comparison failed: {e}")
    
    def _compare_images(self, img1: bytes, img2: bytes) -> float:
        # Implement image comparison logic
        ...
```

## Registering Custom Tools

### During Browser Initialization

```python
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(...) as browser:
        # Register tool class
        browser.react_agent.tool_registry.register(MyCustomTool)
        
        # Register tool instance
        tool_instance = MyCustomTool()
        browser.react_agent.tool_registry.register_instance(tool_instance)
        
        # Now the agent can use the custom tool
        await browser.agent("Use my_custom_tool to process data")
```

### Creating a Tool Package

```python
# my_tools/__init__.py
from .highlight import HighlightElementTool
from .weather import WeatherTool

ALL_TOOLS = [HighlightElementTool, WeatherTool]

def register_all(registry):
    """Register all custom tools."""
    for tool_class in ALL_TOOLS:
        registry.register(tool_class)
```

```python
# Usage
from flybrowser import FlyBrowser
from my_tools import register_all

async with FlyBrowser(...) as browser:
    register_all(browser.react_agent.tool_registry)
```

## Tool Execution Context

### Accessing Page Controller

Tools with `requires_page=True` receive a page controller:

```python
class MyPageTool(BaseTool):
    metadata = ToolMetadata(
        ...
        requires_page=True,
    )
    
    async def execute(self, **kwargs) -> ToolResult:
        # Access page controller via self.page
        url = await self.page.url
        title = await self.page.title()
        
        # Use Playwright methods
        await self.page.click("#button")
        await self.page.fill("#input", "text")
        
        # Take screenshot
        screenshot = await self.page.screenshot()
        
        return ToolResult.success_result({"url": url})
```

### Tool Result

Return results using `ToolResult`:

```python
# Success
return ToolResult.success_result(
    data={"key": "value"},
    metadata={"duration_ms": 100}
)

# Failure
return ToolResult.failure_result(
    error="Error message",
    metadata={"attempted_action": "click"}
)

# Direct construction
return ToolResult(
    success=True,
    data=result_data,
    error=None,
    metadata=extra_info
)
```

## Testing Custom Tools

### Unit Testing

```python
import pytest
from my_tools import MyCustomTool

@pytest.mark.asyncio
async def test_my_custom_tool():
    tool = MyCustomTool()
    
    # Test successful execution
    result = await tool.execute(param1="test_value")
    assert result.success
    assert "Processed" in result.data
    
    # Test error handling
    result = await tool.execute(param1="")
    assert not result.success
```

### Integration Testing

```python
import pytest
from flybrowser import FlyBrowser
from my_tools import MyCustomTool

@pytest.mark.asyncio
async def test_tool_with_agent():
    async with FlyBrowser(headless=True) as browser:
        browser.react_agent.tool_registry.register(MyCustomTool)
        
        result = await browser.agent(
            "Use my_custom_tool with param1='test'"
        )
        
        assert result.success
```

## Best Practices

1. **Clear descriptions** - Write LLM-friendly descriptions
2. **Parameter validation** - Validate inputs in execute()
3. **Error handling** - Always catch exceptions, return ToolResult.failure_result()
4. **Timeout handling** - Respect timeout_seconds
5. **Safety levels** - Accurately classify tool safety
6. **Examples** - Provide clear usage examples
7. **Testing** - Write unit and integration tests

## See Also

- [Architecture: Tools System](../architecture/tools.md) - Tools architecture
- [Agent Feature](../features/agent.md) - Using the agent
- [ReAct Framework](../architecture/react.md) - How tools are used
