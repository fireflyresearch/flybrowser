# Memory System

FlyBrowser uses two complementary memory systems: the legacy `AgentMemory` (retained for backward compatibility) and the new `BrowserMemoryManager` introduced with the fireflyframework-genai migration.

## BrowserMemoryManager

The primary memory system for `BrowserAgent`, defined in `flybrowser/agents/memory/browser_memory.py`. It delegates storage to the framework's `MemoryManager` while maintaining full backward compatibility with the original API.

### Framework Integration

`BrowserMemoryManager` implements a **dual-write pattern**: every mutation updates both a local Python cache and the framework's `WorkingMemory`. This provides fast, typed access for prompt formatting while enabling future migration to persistent backends with zero code changes.

```
BrowserMemoryManager
  |
  |-- Local cache (Python objects)
  |     PageSnapshot list, ObstacleInfo dict, visited URLs set, facts dict
  |     Fast typed access for get_current_page(), format_for_prompt(), etc.
  |
  |-- MemoryManager(store=InMemoryStore())
  |     Framework working memory (serializable key-value pairs)
  |     Enables backend pluggability (File, Postgres, MongoDB)
  |
  +-- ConversationMemory
        One conversation per session (conversation_id property)
        Tracks conversation turns for multi-step reasoning
```

### Initialization

On construction, `BrowserMemoryManager` creates a framework `MemoryManager` with an `InMemoryStore` and starts a new conversation:

```python
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager

memory = BrowserMemoryManager()
print(memory.conversation_id)  # "conv_abc123..."
```

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
    # Framework accessors
    @property
    def conversation_id(self) -> str          # Current conversation ID

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

### Dual-Write in Action

Every mutation writes to both the local cache and the framework working memory:

```python
# When you call:
memory.record_page_state("https://example.com", "Example", "3 links, 1 form")

# Internally, BrowserMemoryManager does:
# 1. Append PageSnapshot to self._page_history (local cache)
# 2. Add URL to self._visited_urls (local cache)
# 3. Update self._facts["current_page"] (local cache)
# 4. Sync "page_history" to framework working memory
# 5. Sync "visited_urls" to framework working memory
# 6. Sync "current_page" to framework working memory
```

Reads always come from the local cache for performance:

```python
page = memory.get_current_page()          # Reads from local _page_history
visited = memory.has_visited_url(url)      # Reads from local _visited_urls
context = memory.format_for_prompt()       # Formats from local caches
```

### Reserved Keys

The following keys are reserved for internal browser state and cannot be used with `set_fact()`:

- `page_history`
- `visited_urls`
- `current_page`
- `navigation_graph`
- `obstacle_cache`

Attempting to set a reserved key raises a `ValueError`.

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
Additional facts:
- login_status: authenticated
- cart_items: 3
```

The prompt context includes user-set facts (from `set_fact()`) but excludes reserved internal keys.

### Clearing Memory

`clear()` resets both the local cache and the framework state, starting a fresh conversation:

```python
memory.clear()
# Local caches are emptied
# Framework working memory is cleared
# A new conversation ID is generated
```

## Backend Pluggability

The framework `MemoryManager` supports multiple storage backends. By default, `BrowserMemoryManager` uses `InMemoryStore` (no persistence across restarts). To use a persistent backend:

| Store | Module | Description |
|-------|--------|-------------|
| `InMemoryStore` | `fireflyframework_genai.memory.store` | Default. Fast, no persistence. |
| `FileStore` | `fireflyframework_genai.memory.store` | JSON file persistence. Good for development. |
| `PostgresStore` | `fireflyframework_genai.memory.store` | PostgreSQL. Production distributed setups. |

Switching backends requires no changes to the `BrowserMemoryManager` public API -- only the store passed to `MemoryManager` changes.

## ConversationMemory Tracking

Each `BrowserMemoryManager` instance creates a conversation via the framework. This enables:

- **Turn tracking** -- The framework records each interaction as a conversation turn
- **History retrieval** -- Previous turns can be replayed for context
- **Multi-session support** -- Each browser session gets its own conversation ID

Access the conversation ID for debugging or logging:

```python
memory = BrowserMemoryManager()
print(f"Conversation: {memory.conversation_id}")
```

## Legacy AgentMemory

The `AgentMemory` and `WorkingMemory` classes in `flybrowser/agents/memory.py` are retained for backward compatibility. They provide a more complex multi-tier memory system (short-term, working, long-term, context store) that predates the framework migration.

These are still exported from `flybrowser.agents` but are not used by `BrowserAgent`.

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Framework Integration](framework-integration.md) - How memory fits into the framework
- [ReAct Framework](react.md) - How memory integrates with reasoning
