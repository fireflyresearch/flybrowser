# ReAct Framework

FlyBrowser implements the ReAct (Reasoning and Acting) paradigm for intelligent browser automation. The reasoning loop is provided by `fireflyframework_genai.reasoning.ReActPattern`, while FlyBrowser supplies browser-specific tools and memory.

## What is ReAct?

ReAct interleaves reasoning and acting. Instead of generating a single plan and executing it, the agent:

1. **Thinks** about what to do next
2. **Acts** by executing a tool
3. **Observes** the result
4. **Repeats** until the task is complete

## BrowserAgent

The main agent class in `flybrowser/agents/browser_agent.py`:

```python
from fireflyframework_genai.agents import FireflyAgent
from fireflyframework_genai.reasoning import ReActPattern

class BrowserAgent:
    def __init__(self, page_controller, config, ...):
        self._toolkits = create_all_toolkits(page=page_controller, ...)
        self._memory = BrowserMemoryManager()
        self._middleware = [
            ObstacleDetectionMiddleware(page_controller),
            ScreenshotOnErrorMiddleware(page_controller),
        ]
        self._agent = FireflyAgent(
            name="flybrowser",
            model=config.model,
            instructions=_SYSTEM_INSTRUCTIONS,
            tools=self._toolkits,
            middleware=self._middleware,
        )
        self._react = ReActPattern(max_steps=config.max_iterations)
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `act(instruction, context)` | Execute a browser action |
| `extract(query, schema, context)` | Extract data from the page |
| `observe(query, context)` | Find elements without interacting |
| `run_task(task, context)` | Full ReAct reasoning loop via `run_with_reasoning()` |
| `agent_stream(task)` | Stream reasoning tokens via SSE |

### How run_task Works

For complex autonomous tasks, `run_task` uses the framework's ReActPattern:

```python
async def run_task(self, task, context=None):
    memory_ctx = self._memory.format_for_prompt()
    full_prompt = f"{task}\n\nBrowser state:\n{memory_ctx}"
    result = await self._agent.run_with_reasoning(
        self._react, full_prompt, timeout=self._config.max_time,
    )
    return self._format_result(result, task)
```

The `ReActPattern` manages the thought-action-observation loop internally, calling tools from the registered ToolKits until the task is complete or `max_steps` is reached.

## Configuration

```python
@dataclass
class BrowserAgentConfig:
    model: str = "openai:gpt-4o"
    max_iterations: int = 50
    max_time: int = 1800
    budget_limit_usd: float = 5.0
    session_id: Optional[str] = None
```

## Middleware

Middleware runs around each tool execution:

- **ObstacleDetectionMiddleware**: Detects and dismisses popups, cookie banners, and modals before/after tool calls
- **ScreenshotOnErrorMiddleware**: Captures a screenshot when a tool fails, aiding debugging

## Streaming

The `agent_stream` method provides real-time reasoning events:

```python
async def agent_stream(self, task):
    async with await self._agent.run_stream(task, streaming_mode="incremental") as stream:
        async for token in stream.stream_tokens():
            yield {"type": "thought", "content": token, "timestamp": time.time()}
```

These events are formatted as SSE using `AgentStreamEvent` and `format_sse_event()` from `flybrowser/agents/streaming.py`.

## Execution States

```python
class ExecutionState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Tools System](tools.md) - ToolKit implementation
- [Memory System](memory.md) - BrowserMemoryManager
