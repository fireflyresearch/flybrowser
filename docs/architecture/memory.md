# Memory System

FlyBrowser uses two complementary memory systems: the legacy `AgentMemory` (retained for backward compatibility) and the new `BrowserMemoryManager` introduced with the fireflyframework-genai migration.

## BrowserMemoryManager

The primary memory system for `BrowserAgent`, defined in `flybrowser/agents/memory/browser_memory.py`.

### Data Structures

```python
@dataclass
class PageSnapshot:
    url: str
    title: str
    elements_summary: str
    timestamp: float = 0.0

@dataclass
class ObstacleInfo:
    obstacle_type: str
    resolution: str
```

### BrowserMemoryManager API

```python
class BrowserMemoryManager:
    # Page state tracking
    def record_page_state(self, url, title, elements_summary) -> None
    def get_current_page(self) -> Optional[PageSnapshot]

    # Navigation graph
    def record_navigation(self, from_url, to_url, method) -> None

    # Obstacle cache
    def record_obstacle(self, url, obstacle_type, resolution) -> None

    # URL tracking
    def has_visited_url(self, url) -> bool

    # Arbitrary facts
    def set_fact(self, key, value) -> None
    def get_fact(self, key, default=None) -> Any

    # Prompt formatting
    def format_for_prompt(self) -> str

    # Reset
    def clear(self) -> None
```

### How It Is Used

`BrowserAgent` creates a `BrowserMemoryManager` and includes its state in prompts:

```python
async def run_task(self, task, context=None):
    memory_ctx = self._memory.format_for_prompt()
    full_prompt = f"{task}\n\nBrowser state:\n{memory_ctx}"
    result = await self._agent.run_with_reasoning(
        self._react, full_prompt, timeout=self._config.max_time,
    )
```

### What format_for_prompt Returns

```
Current page: https://example.com/products - Products
Page elements: 20 product cards, navigation bar, search box
Pages visited: 5 (3 unique)
Recent history: example.com/products -> example.com/cart -> ...
Known obstacles: example.com/products
```

## Legacy AgentMemory

The `AgentMemory` and `WorkingMemory` classes in `flybrowser/agents/memory.py` are retained for backward compatibility. They provide a more complex multi-tier memory system (short-term, working, long-term, context store) that predates the framework migration.

These are still exported from `flybrowser.agents` but are not used by `BrowserAgent`.

## See Also

- [Architecture Overview](overview.md) - System architecture
- [ReAct Framework](react.md) - How memory integrates with reasoning
