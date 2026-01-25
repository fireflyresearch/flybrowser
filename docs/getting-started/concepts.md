# Core Concepts

Understanding these fundamental concepts will help you get the most out of FlyBrowser. This guide explains how the framework works under the hood and why it makes certain decisions.

## The ReAct Paradigm

FlyBrowser is built on the ReAct (Reasoning and Acting) paradigm, a methodology where an AI agent alternates between thinking about what to do and actually doing it. This cycle of thought, action, and observation continues until the task is complete.

### The Thought-Action-Observation Cycle

Every interaction follows this pattern:

1. **Thought** - The agent reasons about the current state and what to do next
2. **Action** - The agent executes a tool (like clicking or extracting data)  
3. **Observation** - The agent observes the result of the action
4. **Repeat** - The cycle continues until the task is complete

For example, when you ask FlyBrowser to "find the price of the first product":

```
Thought: I need to find the price. Let me look at the page structure.
Action: observe("Look for product prices or price-related elements")
Observation: Found element with class 'price' containing '$29.99'

Thought: I found the price. I should extract and return it.
Action: complete(result="$29.99")
Observation: Task completed successfully
```

This explicit reasoning process makes the agent's behavior predictable and debuggable.

## Operation Modes

FlyBrowser adapts its behavior based on the type of task you are performing. Each SDK method internally sets an operation mode that optimizes vision usage, exploration depth, and memory patterns.

### Navigate Mode

Used by `navigate()` for exploration-focused tasks:
- High-frequency vision (screenshots every 2 iterations)
- Comprehensive page exploration
- Detailed memory of site structure

### Execute Mode

Used by `act()` for fast, targeted interactions:
- Minimal vision (first iteration plus failures only)
- Viewport-only exploration for speed
- Focused on completing specific actions

### Scrape Mode

Used by `extract()` for data collection:
- Balanced vision for structure understanding
- Content-focused exploration
- Emphasis on data regions

### Research Mode

Used by `agent()` for autonomous tasks:
- Adaptive vision based on task needs
- Smart exploration scope
- Comprehensive memory for complex workflows

## The Tool System

FlyBrowser exposes browser capabilities through a tool registry. When the agent decides to take an action, it selects and invokes a tool from this registry.

### Core Tool Categories

**Navigation Tools**
- `navigate` / `goto` - Go to URLs
- `go_back` / `go_forward` - Browser history
- `refresh` - Reload the page

**Interaction Tools**
- `click` - Click on elements
- `type_text` - Enter text in fields
- `scroll` - Scroll the page
- `hover` - Hover over elements
- `select` - Choose dropdown options

**Extraction Tools**
- `extract_text` - Get text content
- `extract_table` - Get tabular data
- `get_attribute` - Get element attributes
- `get_page_state` - Get page structure

**System Tools**
- `complete` - Signal task completion with result
- `fail` - Signal task failure with reason
- `screenshot` - Capture page image
- `wait` - Wait for conditions

Each tool has metadata describing its parameters, safety level, and capabilities required to use it.

### Tool Results

Every tool returns a `ToolResult` object with a consistent structure:

```python
class ToolResult:
    success: bool          # Whether the tool executed successfully
    data: Any             # The result data (varies by tool)
    error: str | None     # Error message if failed
    error_code: str | None # Machine-readable error code
    metadata: dict        # Additional context
```

This standardized format makes it easy to handle results uniformly across different tools.

## Agent Memory

The agent maintains context about its execution through a multi-tier memory system. This allows it to build on previous actions, avoid repeating mistakes, and maintain coherent long-running sessions.

### Short-Term Memory

Holds recent ReAct cycles (thought-action-observation records). This gives the agent immediate context about what it just did:

- Last N successful actions
- Recent failures and their causes
- Current page state

Short-term memory is limited in size and older entries are pruned automatically.

### Working Memory

Stores task-relevant information being actively used:

- Current URL and page title
- Extracted data pending aggregation
- Intermediate results
- Scratch space for computations

Working memory is cleared between tasks.

### Long-Term Memory (PageMaps)

For vision-enabled models, FlyBrowser builds PageMaps that capture comprehensive page understanding:

- Page structure and layout
- Section analysis (navigation, content, forms)
- Element locations and purposes
- Screenshots for visual reference

PageMaps are indexed by URL and persist across navigation, allowing the agent to reference previously visited pages.

### Memory and Context Size

The memory system automatically manages context size to stay within LLM limits:

```python
# Memory formats its contents for the LLM prompt
context = memory.format_for_prompt()

# Automatic truncation keeps context manageable
# Recent actions prioritized over older ones
# Critical errors always preserved
```

## Vision System

When using vision-capable models (like GPT-4o), FlyBrowser can see the page through screenshots. This visual understanding complements DOM-based analysis.

### When Vision is Used

Vision usage is controlled by the operation mode:

- **First iteration** - Always captures initial page state
- **After navigation** - New pages require visual analysis
- **On failures** - Screenshots help diagnose problems
- **Periodic checks** - Frequency varies by mode (EXECUTE is minimal, NAVIGATE is frequent)

### Page Exploration

For complex pages, FlyBrowser performs systematic exploration:

1. Captures viewport screenshot
2. Scrolls and captures additional sections
3. Analyzes screenshots to identify regions
4. Builds a PageMap with structure understanding

This exploration depth adapts to the operation mode - full exploration for browsing, minimal for quick actions.

## Response Objects

All SDK methods return `AgentRequestResponse` objects with comprehensive information about the execution.

### Core Fields

```python
class AgentRequestResponse:
    success: bool         # Did the operation succeed?
    data: Any            # The extracted/returned data
    error: str | None    # Error message if failed
    operation: str       # What operation was performed
    query: str | None    # The original query/instruction
```

### Execution Information

```python
class ExecutionInfo:
    iterations: int           # How many ReAct cycles
    max_iterations: int       # The configured limit
    duration_seconds: float   # Total execution time
    actions_taken: list[str]  # What tools were invoked
    pages_scraped: int        # Pages visited
    success: bool             # Final outcome
    summary: str              # Human-readable summary
    history: list[dict]       # Detailed step-by-step history
```

### LLM Usage Statistics

```python
class LLMUsageInfo:
    prompt_tokens: int         # Tokens in prompts
    completion_tokens: int     # Tokens generated
    total_tokens: int          # Combined usage
    cost_usd: float           # Estimated cost
    model: str                # Model used
    calls_count: int          # Number of LLM calls
    cached_calls: int         # Cache hits (if applicable)
```

### Pretty Printing

For debugging and exploration, responses can be pretty-printed:

```python
result = await browser.extract("Get the page title")
result.pprint()  # Formatted output with all details
```

## Configuration System

FlyBrowser is highly configurable through the `AgentConfig` class and its sub-configurations.

### Configuration Hierarchy

```
AgentConfig
├── LLMConfig           # Model settings (temperature, tokens)
├── SafetyConfig        # Loop detection, failure limits
├── MemoryConfig        # Context sizes, retention policies
├── PageExplorationConfig    # Exploration depth and scope
├── ParallelExplorationConfig # Concurrent operations
├── ObstacleDetectorConfig   # Handling popups, obstacles
├── ElementInteractionConfig # Click and type behaviors
└── SearchToolConfig         # Web search settings
```

### Loading Configuration

Configuration can come from multiple sources:

```python
# Programmatic
config = AgentConfig(max_iterations=50)

# From YAML file
config = AgentConfig.from_yaml("config.yaml")

# From JSON file  
config = AgentConfig.from_json("config.json")

# Environment variables override file values
# FLYBROWSER_MAX_ITERATIONS=100 will override the file setting
```

### Key Configuration Options

**Execution Limits**
- `max_iterations` - Maximum ReAct cycles (default: 50)
- `timeout` - Overall task timeout

**LLM Settings**
- `reasoning_temperature` - Creativity vs determinism (0.0-1.0)
- `reasoning_max_tokens` - Output length limit
- `max_repair_attempts` - Retries on malformed responses

**Safety Settings**
- `max_consecutive_failures` - Stop after N failures
- `enable_loop_detection` - Detect repetitive actions
- `max_repeated_actions` - Tolerance for repeated actions

## Autonomous Planning

For complex multi-step tasks, the agent can create and execute plans automatically.

### When Planning Happens

The planner analyzes task complexity and creates structured plans for:
- Multi-page workflows
- Tasks with dependencies
- Complex data collection
- Site-wide exploration

Simple tasks execute directly without planning overhead.

### Plan Structure

```python
class ExecutionPlan:
    task: str                  # Original task
    phases: list[Phase]        # Ordered execution phases
    current_phase_index: int   # Progress tracking
    
class Phase:
    name: str                  # Phase identifier
    description: str           # What this phase does
    goals: list[Goal]          # Specific objectives
    status: PhaseStatus        # pending/in_progress/completed/failed
```

### Adaptive Replanning

When execution encounters obstacles, the planner can adapt:

1. Detect goal failure
2. Analyze cause
3. Generate alternative approach
4. Update plan and continue

This self-healing capability helps complex tasks recover from unexpected situations.

## Error Handling Philosophy

FlyBrowser distinguishes between different types of errors and handles them appropriately.

### Recoverable Errors

- Element not found → Wait and retry
- Page load timeout → Increase timeout, retry
- Click intercepted → Scroll into view, retry

These are handled automatically with configurable retry limits.

### Structural Errors

- Wrong page → Navigate back, find correct path
- Unexpected popup → Detect obstacle, handle it
- Page changed → Re-analyze, adapt plan

The agent adapts its strategy when the page does not match expectations.

### Terminal Errors

- Authentication required → Report and fail
- Site blocking automation → Report and fail
- Maximum iterations exceeded → Summarize progress, fail

Some situations cannot be recovered and require human intervention.

### Error Information

Failures include machine-readable error codes:

```python
result = await browser.act("click the nonexistent button")
if not result.success:
    print(result.error)      # Human-readable message
    # Execution info has detailed history
    print(result.execution.history[-1])  # What went wrong
```

## LLM Providers

FlyBrowser supports multiple LLM providers with a unified interface.

### Supported Providers

- **OpenAI** - GPT-4o, GPT-4o-mini, GPT-4 Turbo
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google** - Gemini 1.5 Pro, Gemini 1.5 Flash
- **Ollama** - Local models (Llama, Mistral, etc.)

### Model Capabilities

Different models have different capabilities:

```python
class ModelCapability(Enum):
    VISION = "vision"           # Can process images
    STRUCTURED_OUTPUT = "structured_output"  # JSON mode
    TOOL_CALLING = "tool_calling"   # Function calling
    STREAMING = "streaming"      # Stream responses
```

FlyBrowser automatically adapts its behavior based on available capabilities. For example:
- Vision-capable models get screenshots
- Text-only models get DOM descriptions
- Some tools are only available to models with certain capabilities

### Switching Providers

```python
# OpenAI
browser = FlyBrowser(llm_provider="openai", api_key="sk-...")

# Anthropic
browser = FlyBrowser(llm_provider="anthropic", api_key="sk-ant-...")

# Local Ollama
browser = FlyBrowser(llm_provider="ollama", llm_model="llama3")
```

## Asynchronous Execution

FlyBrowser is fully asynchronous, built on Python's asyncio.

### Why Async?

- Browser operations are I/O-bound (network, rendering)
- LLM calls take time
- Async allows concurrent operations where possible
- Better resource utilization

### Using Async/Await

```python
import asyncio

async def main():
    async with FlyBrowser(...) as browser:
        await browser.goto("https://example.com")
        result = await browser.extract("Get the title")
        print(result.data)

asyncio.run(main())
```

### Context Manager

The `async with` pattern ensures proper cleanup:

```python
async with FlyBrowser(...) as browser:
    # Browser is started and ready
    await browser.goto(...)
# Browser is automatically stopped and cleaned up
```

Manual start/stop is also available:

```python
browser = FlyBrowser(...)
await browser.start()
try:
    await browser.goto(...)
finally:
    await browser.stop()
```

## Next Steps

With these concepts understood, you can:

- [Basic Automation Guide](../guides/basic-automation.md) - Put concepts into practice
- [Data Extraction Guide](../guides/data-extraction.md) - Advanced extraction patterns
- [Configuration Reference](../reference/configuration.md) - Complete configuration options
- [Architecture Overview](../architecture/overview.md) - Deep dive into internals
