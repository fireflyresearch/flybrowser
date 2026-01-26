# Memory System

FlyBrowser's agent uses a sophisticated memory system for context management, failure avoidance, and learned pattern storage. This document explains the memory architecture.

## Overview

The memory system provides:

- Short-term memory: Current task execution context
- Working memory: Active reasoning and action cycles
- Long-term memory: Persistent patterns and learned knowledge
- Context window management: Token-aware memory pruning
- Relevance-based retrieval: Finding pertinent memories

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentMemory                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Unified Interface                      │  │
│  │  start_task() | record_cycle() | format_for_prompt()      │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           │                                     │
│   ┌───────────────────────┼───────────────────────┐             │
│   │           │           │           │           │             │
│   ▼           ▼           ▼           ▼           ▼             │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│ │ Short   │ │ Working │ │ Long    │ │ Context │ │ Sitemap │     │
│ │  Term   │ │ Memory  │ │  Term   │ │  Store  │ │  Graph  │     │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
│   Recent     Current     Learned      User        Site          │
│   Actions    Goal/Nav    Patterns    Context    Exploration     │
└─────────────────────────────────────────────────────────────────┘
```

## AgentMemory

The main memory class combining all memory subsystems:

```python
class AgentMemory:
    """Main memory class combining all memory types."""
    
    def __init__(
        self,
        context_window_budget: int = 64000,  # Modern LLMs have large context windows
        short_term_max_entries: int = 50,    # Keep more history
        long_term_max_patterns: int = 100,
        # Extraction data limits (prevent token overflow)
        max_extraction_budget_percent: float = 0.20,  # Max 20% of budget
        max_extraction_tokens: int = 4000,            # Hard cap on extraction tokens
        max_single_extraction_chars: int = 8000,      # Per-extraction char limit
    ) -> None:
        """Initialize agent memory system."""
        self.context_window_budget = context_window_budget
        
        # Store extraction limits
        self.max_extraction_budget_percent = max_extraction_budget_percent
        self.max_extraction_tokens = max_extraction_tokens
        self.max_single_extraction_chars = max_single_extraction_chars
        
        # Memory subsystems
        self.short_term = ShortTermMemory(max_entries=short_term_max_entries)
        self.working = WorkingMemory(token_budget=context_window_budget // 4)
        self.long_term = LongTermMemory(max_patterns=long_term_max_patterns)
        self.context_store = ContextStore()
        
        # Track visited URLs
        self._visited_urls: Set[str] = set()
```

### Key Methods

```python
# Start a new task
memory.start_task(
    goal="Extract product data from example.com",
    user_context={"category": "electronics"}
)

# Record a Thought-Action-Observation cycle
entry = memory.record_cycle(
    thought=thought,
    action=action,
    observation=observation,
    outcome=ExecutionOutcome.SUCCESS,
    priority=MemoryPriority.HIGH,
)

# Format memory for LLM prompt
prompt_context = memory.format_for_prompt(domain="example.com")

# Check for repeated failures
if memory.is_repeated_failure(action):
    # Try alternative approach
    ...

# Track visited URLs
memory.mark_url_visited("https://example.com/page1")
if memory.has_visited_url(url):
    # Skip already visited page
    ...
```

## Memory Entry

Each Thought-Action-Observation cycle is stored as a `MemoryEntry`:

```python
@dataclass
class MemoryEntry:
    """A single memory entry in the agent's memory system."""
    
    entry_id: str                       # Unique identifier
    timestamp: float                    # When recorded
    thought: Thought                    # Agent's reasoning
    action: Action                      # Action taken
    observation: Optional[Observation]  # Result observed
    outcome: ExecutionOutcome           # SUCCESS, FAILURE, etc.
    priority: MemoryPriority            # CRITICAL, HIGH, NORMAL, LOW
    
    # Context at time of action
    url: str = ""
    page_title: str = ""
    
    # Execution metadata
    duration_ms: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    
    # Token tracking
    estimated_tokens: int = 0
    
    # Tags for filtering
    tags: Set[str] = field(default_factory=set)
```

### Action Signature

Each entry generates a unique signature for deduplication and failure tracking:

```python
@property
def action_signature(self) -> str:
    """Generate a unique signature for this action."""
    sig_parts = [
        self.action.tool_name,
        str(sorted(self.action.parameters.items())),
        self.url,
    ]
    return hashlib.md5(":".join(sig_parts).encode()).hexdigest()[:16]
```

### Prompt Formatting

Entries format themselves for LLM prompts:

```python
def format_for_prompt(self, include_observation: bool = True) -> str:
    """Format this entry for inclusion in a prompt."""
    lines = []
    if self.thought and hasattr(self.thought, 'content'):
        lines.append(f"**Thought**: {self.thought.content}")
    lines.append(f"**Action**: {self.action.tool_name}({self.action.parameters})")
    if include_observation and self.observation:
        obs_content = self.observation.summary or self.observation.raw_output
        if len(obs_content) > 500:
            obs_content = obs_content[:500] + "..."
        lines.append(f"**Observation**: {obs_content}")
    return "\n".join(lines)
```

## Short-Term Memory

Stores recent TAO cycles for immediate decision-making:

```python
class ShortTermMemory:
    """Short-term memory for current task execution."""
    
    def __init__(self, max_entries: int = 50) -> None:
        self.max_entries = max_entries
        self._entries: Deque[MemoryEntry] = deque(maxlen=max_entries)
        self._failed_signatures: Set[str] = set()
        self._action_frequency: Dict[str, int] = {}
```

### Features

**Recent Entry Access:**
```python
# Get last 5 entries
recent = memory.short_term.get_recent(n=5)

# Get entries by outcome
failures = memory.short_term.get_by_outcome(ExecutionOutcome.FAILURE)
```

**Failure Tracking:**
```python
# Check if action has failed before
if memory.short_term.has_failed(action_signature):
    # Choose different approach
    ...
```

**Loop Detection:**
```python
# Get warnings about repeated actions
warnings = memory.short_term.get_loop_warnings(threshold=3)
# ["Warning: click(#submit) repeated 4 times in last 10 actions!"]

# Check action frequency
count = memory.short_term.get_action_frequency("scroll_page", {"direction": "down"})
```

**History Formatting:**
```python
history = memory.short_term.format_history(last_n=5)
# ## Recent Actions
# Warning: scroll_page repeated 3 times
# 
# 1. [Success] goto({"url": "https://example.com"})
#    Navigated to: https://example.com
# 2. [Success] click({"selector": "#menu"})
#    Result: Element clicked successfully
```

## Working Memory

Tracks active reasoning, current goal, and navigation path:

```python
class WorkingMemory:
    """Working memory for active reasoning."""
    
    def __init__(self, token_budget: int = 4000) -> None:
        self.token_budget = token_budget
        self._current_goal: Optional[str] = None
        self._active_thought: Optional[Thought] = None
        self._active_action: Optional[Action] = None
        self._navigation_path: List[StateSnapshot] = []
        self._scratch_pad: Dict[str, Any] = {}
        self._cycle_count: int = 0
```

### Goal Management

```python
# Set task goal
memory.working.set_goal("Find and extract all product prices")

# Track cycle count
print(f"Cycle: {memory.working.cycle_count}")

# Access current goal
goal = memory.working.current_goal
```

### Navigation Tracking

```python
# Record navigation state
snapshot = StateSnapshot(
    snapshot_id="snap_001",
    timestamp=time.time(),
    url="https://example.com/products",
    page_title="Products",
    visible_elements_summary="Product list with 20 items",
    scroll_position=(0, 500),
)
memory.working.record_navigation(snapshot)

# Get navigation history
history = memory.working.get_navigation_history()
```

### Scratch Pad

Temporary storage for working data:

```python
# Store temporary data
memory.working.set_scratch("extracted_items", [])
memory.working.set_scratch("current_page", 1)

# Retrieve data
items = memory.working.get_scratch("extracted_items", default=[])
```

### PageMap Storage

Working memory stores spatial page understanding:

```python
# Store PageMap for a URL
memory.store_page_map("https://example.com", page_map)

# Check if PageMap exists
if memory.has_page_map(url):
    page_map = memory.get_page_map(url)

# Get all stored PageMaps
all_maps = memory.get_all_page_maps()

# Get page summary
summary = memory.get_page_summary("https://example.com")
```

## Long-Term Memory

Stores learned patterns for reuse across tasks:

```python
class LongTermMemory:
    """Long-term memory for persistent patterns."""
    
    def __init__(self, max_patterns: int = 100) -> None:
        self.max_patterns = max_patterns
        self._patterns: Dict[str, LearnedPattern] = {}
        self._domain_index: Dict[str, List[str]] = {}
```

### Learned Patterns

```python
@dataclass
class LearnedPattern:
    """Site-specific pattern learned during execution."""
    
    pattern_id: str
    domain: str                     # e.g., "example.com"
    description: str                # Human-readable description
    pattern_type: str               # "login_flow", "navigation", "form_fill"
    steps: List[Dict[str, Any]]     # Sequence of actions
    success_count: int = 0
    failure_count: int = 0
    last_used: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
```

### Pattern Operations

```python
# Learn a new pattern
pattern = LearnedPattern(
    pattern_id="login_example_com",
    domain="example.com",
    description="Login flow for example.com",
    pattern_type="login_flow",
    steps=[
        {"tool": "click", "selector": "#login-btn"},
        {"tool": "type", "selector": "#email", "text": "{email}"},
        {"tool": "type", "selector": "#password", "text": "{password}"},
        {"tool": "click", "selector": "#submit"},
    ],
)
memory.learn_pattern(pattern)

# Get patterns for a domain
patterns = memory.get_patterns_for_domain("example.com")

# Record usage outcome
memory.long_term.record_usage("login_example_com", success=True)
```

### Automatic Pruning

Long-term memory prunes low-value patterns:

```python
def _prune_patterns(self) -> None:
    """Remove least useful patterns."""
    # Sort by success rate and recency
    sorted_patterns = sorted(
        self._patterns.values(),
        key=lambda p: (p.success_rate, p.last_used),
    )
    # Remove bottom 20%
    to_remove = len(self._patterns) - int(self.max_patterns * 0.8)
    for pattern in sorted_patterns[:to_remove]:
        self.remove_pattern(pattern.pattern_id)
```

## Context Store

Manages user-provided and inferred context:

```python
class ContextStore:
    """Manages user-provided and inferred context."""
    
    def __init__(self) -> None:
        self._user_context: Dict[str, Any] = {}
        self._inferred_context: Dict[str, Any] = {}
        self._context_history: List[Dict[str, Any]] = []
        self._extraction_cache: Dict[str, Any] = {}
```

### User Context

```python
# Set user-provided context
memory.context_store.set_user_context({
    "username": "user@example.com",
    "preferred_language": "English",
    "max_items": 100,
})

# Get context value
username = memory.context_store.get("username")

# Get all context
all_context = memory.context_store.get_all()
```

### Inferred Context

```python
# Add context learned during execution
memory.add_inferred_context(
    key="login_required",
    value=True,
    source="detected login form"
)

memory.add_inferred_context(
    key="pagination_type",
    value="infinite_scroll",
    source="page analysis"
)
```

### Relevance-Based Retrieval

```python
# Get context relevant to a specific task
relevant = memory.context_store.get_relevant_context(
    "login to the website and navigate to products"
)
# Returns context with matching keywords
```

### Extraction Cache

Cache extracted data with TTL:

```python
# Cache extraction result
memory.context_store.cache_extraction("product_list", products)

# Retrieve if still valid (5 min default TTL)
cached = memory.context_store.get_cached_extraction(
    "product_list",
    max_age_seconds=300
)
```

## State Snapshots

Capture browser state for backtracking:

```python
@dataclass
class StateSnapshot:
    """Snapshot of browser state for backtracking."""
    
    snapshot_id: str
    timestamp: float
    url: str
    page_title: str
    visible_elements_summary: str
    form_state: Dict[str, Any] = field(default_factory=dict)
    scroll_position: Tuple[int, int] = (0, 0)
    associated_entry_id: Optional[str] = None
```

## SitemapGraph Integration

For multi-page site exploration:

```python
# Initialize site exploration
graph = memory.init_sitemap_graph(
    homepage_url="https://example.com",
    limits=SitemapLimits(max_depth=2, max_pages=50)
)

# Add discovered links
memory.add_sitemap_links(
    parent_url="https://example.com",
    links=[
        {"url": "https://example.com/products", "text": "Products"},
        {"url": "https://example.com/about", "text": "About"},
    ],
    link_type="main_nav"
)

# Mark page as visited
memory.update_sitemap_visited(
    url="https://example.com/products",
    title="Products",
    summary="Product catalog with 50 items",
    section_count=5
)

# Get next page to visit
next_url = memory.get_next_page_to_visit()

# Check if exploration is complete
if memory.is_sitemap_exploration_complete():
    # Generate final summary
    summary = memory.format_site_exploration_summary()
```

## Memory Priorities

Entries have priority levels affecting retention:

```python
class MemoryPriority(str, Enum):
    """Priority levels for memory entries."""
    
    CRITICAL = "critical"   # Never prune (errors, key discoveries)
    HIGH = "high"           # Prune last (successful actions)
    NORMAL = "normal"       # Standard retention
    LOW = "low"             # Prune first (routine actions)
```

## Token Management

Memory tracks token usage for context window management, integrated with the `ConversationManager` for sophisticated token budget handling:

```python
# Get total tokens in short-term memory
total_tokens = memory.short_term.total_tokens()

# Format with budget awareness
context = memory.format_for_prompt(domain="example.com")
# Automatically fits within context_window_budget
```

### Token Estimation

The `TokenEstimator` class provides content-aware token estimation:

```python
from flybrowser.llm.token_budget import TokenEstimator, ContentType

# Estimate tokens for any content
estimate = TokenEstimator.estimate(text_content)
print(f"Tokens: {estimate.tokens}")
print(f"Type: {estimate.content_type.value}")  # TEXT, HTML, JSON, CODE
print(f"Confidence: {estimate.confidence}")

# Force specific content type
estimate = TokenEstimator.estimate(content, ContentType.HTML)

# Estimate for message arrays
tokens = TokenEstimator.estimate_messages(messages)
```

Content type affects estimation (chars per token):
- **TEXT**: ~4 chars/token
- **CODE**: ~3.5 chars/token
- **JSON**: ~3 chars/token
- **HTML**: ~2.5 chars/token

### Extraction Budget Management

The memory system automatically limits extraction data to prevent token overflow:

```python
# In format_for_prompt(), extraction data is limited:
# - Max 20% of remaining budget for all extractions (configurable)
# - Max 8000 chars per individual extraction (~2K tokens)
# - Smart head/tail truncation for large extractions

# Configuration in MemoryConfig:
max_extraction_budget_percent: float = 0.20  # Max 20% of budget
max_extraction_tokens: int = 4000  # Hard cap on total extraction tokens
max_single_extraction_chars: int = 8000  # Per-extraction limit
```

### Intelligent Compression

Large extractions are compressed using the `ContextCompressor` to preserve essential information:

```python
from flybrowser.llm import ContextCompressor, CompressionContentType

# Compression is triggered automatically when:
# - Data size exceeds compression_threshold_tokens (default 2000)
# - enable_intelligent_compression is True
# - Data is from a compressible tool (not URL-critical)

# URL-CRITICAL tools (NEVER compressed):
# - search, search_human, search_api, search_rank (contain navigation URLs)
# - get_page_state (contains navigation links, buttons, selectors)
# - get_attribute (contains href, src values)

# COMPRESSIBLE tools:
# - extract_text, extract_data, extract_structured_data
# - evaluate_javascript, get_page_content
```

**Content Types:**

The compressor auto-detects content type for optimal preservation:

| Content Type | Preservation Priority |
|-------------|----------------------|
| `search_results` | ALL URLs, titles, snippets |
| `page_state` | ALL links, buttons, selectors |
| `text_content` | Key facts, embedded URLs |
| `error_info` | Error type, message, recovery suggestions |
| `structured_data` | Field names, values, IDs |
| `form_data` | Field names, types, selectors |

**Compression Tiers:**

```python
# Tier 1 (Recent): Maximum detail preservation
# Tier 2 (Session): Compress text, preserve all URLs/selectors
# Tier 3 (Archival): Aggressive compression, facts only

compressed = await compressor.compress_extraction(
    large_data=extraction_result,
    content_type=ContentType.TEXT_CONTENT,
    compression_tier=2,  # Session tier
    validate_urls=True,  # Ensure URLs preserved
)
```

**URL Validation:**

```python
# After compression, URL preservation is validated:
is_valid, msg = compressed.validate_url_preservation(original_url_count)

# If URLs are lost, fallback extraction recovers them:
if not is_valid:
    fallback_urls = compressor._extract_all_urls(content)
    compressed.urls_mentioned.extend(fallback_urls)
```

**Accessing Compressed Data:**

```python
print(compressed.summary)  # One-line summary
print(compressed.key_facts)  # List of bullet points
print(compressed.urls_mentioned)  # Preserved URLs
print(compressed.navigation_targets)  # [{"label": ..., "url": ...}]
print(compressed.form_fields)  # [{"name": ..., "type": ...}]
print(compressed.compression_confidence)  # 0-1 confidence score

# Format for LLM prompt
context = compressed.format_for_prompt()
```

### Integration with ConversationManager

The ReActAgent integrates memory with `ConversationManager` for large context handling:

```python
# ConversationManager is initialized with model info
conversation = ConversationManager(
    llm_provider=llm_provider,
    model_info=model_info,
)

# During prompt building, context is checked
memory_context = memory.format_for_prompt()
memory_context = agent._check_and_handle_large_context(memory_context)

# If context is too large:
# 1. Essential sections (goal, current page) are preserved
# 2. Extraction data is truncated intelligently
# 3. Older history entries may be pruned
```

## Operation Mode Awareness

Memory adapts behavior based on operation mode:

```python
# Set operation mode
memory.set_operation_mode(OperationMode.SCRAPE)

# Get current mode
mode = memory.get_operation_mode()
```

This affects:
- Memory retention priorities
- Context formatting
- Pruning strategies

## Task Summary

Get execution summary:

```python
summary = memory.get_task_summary()
# Goal: Extract product data
# Actions: 15 total (12 successful, 3 failed)
# Current page: https://example.com/products
# Pages visited: 5
# 
# Explored pages (5 total):
#   - Products: Main product listing with categories...
#   - Electronics: Electronic products including...
# 
# Recent successful actions: click, extract_text, scroll
```

## Memory Lifecycle

```
Task Start
    │
    ▼
┌─────────────────┐
│  start_task()   │  Clear short-term, set goal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  record_cycle() │  Store TAO in short-term
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ format_prompt() │  Build context for LLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ learn_pattern() │  (Optional) Store in long-term
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  clear_task()   │  Clear short-term + working
└─────────────────┘
```

## Best Practices

1. **Set meaningful priorities** - Mark critical discoveries as HIGH/CRITICAL
2. **Use context store** - Store user preferences and inferred knowledge
3. **Learn patterns** - Save successful multi-step sequences
4. **Track visited URLs** - Avoid redundant navigation
5. **Use scratch pad** - Store intermediate results between cycles

## See Also

- [Architecture Overview](overview.md) - System architecture
- [ReAct Framework](react.md) - How memory integrates with reasoning
- [Agent Feature](../features/agent.md) - Using the agent
