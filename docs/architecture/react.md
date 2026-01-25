# ReAct Framework

FlyBrowser implements the ReAct (Reasoning and Acting) paradigm for intelligent browser automation. This document explains the framework's design and implementation.

## What is ReAct?

ReAct is a prompting paradigm that interleaves reasoning and acting. Instead of generating a single plan and executing it, the agent:

1. **Thinks** about what to do next
2. **Acts** by executing a tool
3. **Observes** the result
4. **Repeats** until the task is complete

This approach allows the agent to:
- Adapt to unexpected situations
- Learn from failures
- Make context-aware decisions
- Handle dynamic web pages

## The ReAct Loop

```
┌──────────────────────────────────────────────────────────────┐
│                        ReAct Cycle                           │
│                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                │
│   │ THOUGHT │────►│ ACTION  │────►│OBSERV.  │                │
│   │         │     │         │     │         │                │
│   │ "I need │     │ click() │     │ Button  │                │
│   │  to..." │     │ type()  │     │ clicked │                │
│   └─────────┘     │ goto()  │     │ success │                │
│        ▲          └─────────┘     └────┬────┘                │
│        │                               │                     │
│        └───────────────────────────────┘                     │
│              (loop until complete)                           │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### ReActAgent

The main agent class in `agents/react_agent.py`:

```python
class ReActAgent:
    """
    Core ReAct agent implementing thought-action-observation cycles.
    
    Attributes:
        page: Browser page controller
        llm: LLM provider for reasoning
        tool_registry: Registry of available tools
        memory: Agent memory system
        config: Agent configuration
    """
    
    def __init__(
        self,
        page_controller: PageController,
        llm_provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        memory: Optional[AgentMemory] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        ...
    
    async def execute(
        self,
        task: str,
        operation_mode: Optional[OperationMode] = None,
        vision_override: Optional[bool] = None,
    ) -> AgentResult:
        """Execute a task using the ReAct loop."""
        ...
```

### Execution States

The agent transitions through these states:

```python
class ExecutionState(str, Enum):
    """Agent execution states."""
    
    IDLE = "idle"           # Not executing
    THINKING = "thinking"   # Generating thought/action
    ACTING = "acting"       # Executing tool
    OBSERVING = "observing" # Processing result
    COMPLETED = "completed" # Task successful
    FAILED = "failed"       # Task failed
    CANCELLED = "cancelled" # User cancelled
```

### ReActStep

Each iteration produces a step record:

```python
@dataclass
class ReActStep:
    """A single step in the ReAct cycle."""
    
    step_number: int
    thought: Thought       # LLM reasoning
    action: Action         # Tool to execute
    observation: Optional[Observation]  # Tool result
    timestamp: float = field(default_factory=time.time)
```

## Execution Flow

### 1. Task Reception

```python
async def execute(self, task: str, ...) -> AgentResult:
    # Reset state
    self._reset_state()
    
    # Initialize memory with task
    self.memory.start_task(task)
    
    # Check if planning is needed
    if self.planner.should_create_plan(task):
        return await self._execute_with_planning(task)
    else:
        return await self._execute_direct(task)
```

### 2. Planning Phase (Complex Tasks)

For complex tasks, the planner creates an execution plan:

```python
async def _execute_with_planning(self, task: str) -> AgentResult:
    # Create execution plan
    plan = await self.planner.create_plan(task)
    
    # Execute each phase
    for phase in plan.phases:
        result = await self._execute_phase(phase)
        if not result.success:
            # Adapt plan on failure
            plan = await self.planner.adapt_plan(plan, result.error)
```

### 3. Direct Execution (Simple Tasks)

Simple tasks execute without planning:

```python
async def _execute_direct(self, task: str) -> AgentResult:
    while not self._should_stop(iteration):
        iteration += 1
        
        # THINK: Generate thought and action
        self._state = ExecutionState.THINKING
        prompts = self._build_prompt(task)
        response = await self._generate_with_optional_vision(prompts)
        
        # Parse response
        parse_result = self.parser.parse(response.content)
        
        # ACT: Execute the action
        self._state = ExecutionState.ACTING
        observation = await self._execute_action(parse_result.action)
        
        # OBSERVE: Process result
        self._state = ExecutionState.OBSERVING
        self.memory.record_cycle(
            parse_result.thought,
            parse_result.action,
            observation
        )
        
        # Check completion
        if self._task_completed:
            break
```

### 4. Tool Execution

Actions are executed through the tool registry:

```python
async def _execute_action(self, action: Action) -> Observation:
    # Get tool from registry
    tool = self.tool_registry.get_tool(action.tool_name)
    
    # Validate parameters
    is_valid, error = tool.validate_parameters(action.parameters)
    if not is_valid:
        return Observation.error_result(error)
    
    # Execute with timeout
    try:
        result = await asyncio.wait_for(
            tool.execute(**action.parameters),
            timeout=tool.metadata.timeout_seconds
        )
        return Observation.from_tool_result(result)
    except asyncio.TimeoutError:
        return Observation.error_result("Tool execution timed out")
```

## Response Format

The LLM produces structured JSON responses:

```json
{
  "thought": {
    "content": "I need to click the login button to proceed",
    "reasoning_type": "action_selection"
  },
  "action": {
    "tool_name": "click",
    "parameters": {
      "selector": "#login-button"
    }
  },
  "is_complete": false,
  "result": null
}
```

### Response Schema

```python
REACT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "reasoning_type": {"type": "string"}
            },
            "required": ["content"]
        },
        "action": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "parameters": {"type": "object"}
            },
            "required": ["tool_name", "parameters"]
        },
        "is_complete": {"type": "boolean"},
        "result": {}
    },
    "required": ["thought", "action", "is_complete"]
}
```

## Operation Modes

Operation modes optimize agent behavior:

```python
class OperationMode(str, Enum):
    """Operation modes for the agent."""
    
    NAVIGATE = "navigate"   # Navigation-focused
    EXECUTE = "execute"     # Action-focused
    SCRAPE = "scrape"       # Extraction-focused
    RESEARCH = "research"   # Discovery-focused
    AUTO = "auto"           # Automatic planning
```

### Mode-Specific Behavior

| Mode | Planning | Vision | Tools |
|------|----------|--------|-------|
| NAVIGATE | Minimal | Optional | Navigation tools |
| EXECUTE | None | Enabled | Interaction tools |
| SCRAPE | None | Optional | Extraction tools |
| RESEARCH | None | Enabled | All tools |
| AUTO | Full | Enabled | All tools |

## Reasoning Strategies

The agent can use different reasoning approaches:

```python
class ReasoningStrategy(str, Enum):
    """Available reasoning strategies."""
    
    CHAIN_OF_THOUGHT = "cot"      # Step-by-step reasoning
    TREE_OF_THOUGHT = "tot"       # Explore multiple paths
    REACT_STANDARD = "react"      # Standard ReAct
    REACT_PLUS = "react_plus"     # Enhanced ReAct
```

## Memory Integration

Memory provides context for reasoning:

```python
def _build_prompt(self, task: str) -> Dict[str, str]:
    # Get memory context
    context = self.memory.get_reasoning_context()
    
    # Include recent history
    history = self.memory.get_recent_history(max_entries=10)
    
    # Build prompt with context
    return self.prompt_manager.build_react_prompt(
        task=task,
        context=context,
        history=history,
        available_tools=self.tool_registry.list_tools(),
    )
```

## Vision Support

Vision-enabled models can see the page:

```python
async def _generate_with_optional_vision(self, prompts: Dict) -> LLMResponse:
    # Check if vision should be used
    use_vision = self._should_use_vision()
    
    if use_vision:
        # Handle dynamic obstacles before capturing screenshot
        await self._check_and_handle_dynamic_obstacles()
        
        # Capture screenshot
        screenshot = await self.page.screenshot()
        
        # Include image in prompt
        return await self.llm.query(
            prompts["system"],
            prompts["user"],
            images=[screenshot],
        )
    else:
        # Text-only query
        return await self.llm.query(
            prompts["system"],
            prompts["user"],
        )
```

### Vision Optimization

Vision is skipped for blank pages to save resources:

```python
def _should_use_vision(self, iteration: int) -> bool:
    # Skip vision for blank pages - no useful content to capture
    # PageController wraps Playwright page - access underlying page.url
    # self.page is PageController, self.page.page is Playwright Page
    if hasattr(self.page, 'page') and hasattr(self.page.page, 'url'):
        current_url = self.page.page.url
    else:
        current_url = None
    
    if current_url and current_url in ('about:blank', 'about:blank#', ''):
        logger.debug(f"[VISION] Skipped: page is blank ({current_url})")
        return False
    
    # Continue with mode-specific logic...
```

**Key Architecture Notes:**
- `self.page` is the `PageController` wrapper class
- `self.page.page` is the underlying Playwright `Page` object
- `PageController` has async `get_url()` method (not sync `.url` property)
- Playwright `Page` has sync `.url` property for immediate access

## Dynamic Obstacle Detection

FlyBrowser implements state-of-the-art two-phase obstacle detection for handling modals, popups, and overlays that appear dynamically via JavaScript:

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Quick DOM Analysis (~10ms, no LLM)                    │
│  - Multi-point sampling (5 viewport positions)                  │
│  - ARIA role detection (dialog, alertdialog)                    │
│  - Framework modal detection (Bootstrap, MUI, etc.)             │
│  - Newsletter tool detection (MailPoet, Mailchimp, HubSpot)     │
│  - Consent tool detection (OneTrust, CookieBot, Quantcast)      │
│  - Confidence scoring with configurable threshold               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ (only if confidence > 0.3)
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Full VLM Analysis + Dismissal (~2-5s)                 │
│  - Screenshot capture and analysis                              │
│  - AI-driven strategy selection                                 │
│  - Multi-strategy dismissal with verification                   │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **Before Screenshot Capture**: Dynamic obstacles are detected and dismissed before every screenshot to ensure clean page state
2. **After Click Failures**: Auto-recovery when clicks are intercepted by modals
3. **Cooldown System**: 3-second cooldown after handling to prevent re-detection loops

```python
async def _check_and_handle_dynamic_obstacles(self) -> bool:
    # Skip for blank/empty pages
    if hasattr(self.page, 'page') and hasattr(self.page.page, 'url'):
        current_url = self.page.page.url
    else:
        current_url = None
    if not current_url or current_url in ('about:blank', 'about:blank#', ''):
        return False
    
    # Two-phase detection with intelligent throttling
    result = await detector.detect_and_handle_if_needed(
        cooldown_seconds=3.0,  # Prevent re-detection loop
        min_confidence=0.3,    # Tuned for low false-positive rate
    )
    return result is not None and result.obstacles_dismissed > 0
```

## Error Handling

The agent handles failures gracefully:

```python
# Consecutive failure tracking
self._consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 3

# On failure
if not observation.success:
    self._consecutive_failures += 1
    
    if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        # Trigger recovery or fail
        if self._current_plan:
            plan = await self.planner.adapt_plan(...)
        else:
            return AgentResult.failure_result(
                "Too many consecutive failures"
            )
else:
    self._consecutive_failures = 0  # Reset on success
```

## Completion Detection

The agent detects task completion:

```python
# From LLM response
if parse_result.is_complete:
    self._task_completed = True
    self._final_result = parse_result.result

# From tool result (terminal tools)
if tool.metadata.is_terminal:
    self._task_completed = True
    self._final_result = observation.data

# From iteration limit
if iteration >= self.config.max_iterations:
    return AgentResult.failure_result("Max iterations reached")
```

## Completion Page Data Flow

When running in non-headless mode, the SDK displays an interactive completion page after task execution. The data flows through multiple layers with robust validation:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. AgentResult                                                  │
│     - ReActStep objects with thought, action, observation        │
│     - LLM usage statistics from provider                         │
│     - Execution metadata (iterations, duration, success)         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. SDK._extract_completion_data()                               │
│     - Converts ReActStep objects to dicts                        │
│     - Handles both object and dict representations               │
│     - Extracts tools_used with name, duration, success           │
│     - Extracts reasoning_steps with thought, action              │
│     - Normalizes llm_usage with defaults for missing fields      │
│     - Builds metadata with session_id, strategy, stop_reason     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. TemplateRenderer.render_completion()                         │
│     - Validates lists are never None (empty list fallback)       │
│     - Ensures llm_usage has all required fields                  │
│     - Ensures metadata has all required fields                   │
│     - Formats duration (ms, seconds, or minutes)                 │
│     - Serializes result_data as JSON for tree view               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. completion.html (Jinja2 Template)                            │
│     - Defensive conditionals: {% if x and x | length > 0 %}      │
│     - Safe dict access: {{ tool.get('name', 'unknown') }}        │
│     - Default filters: {{ value | default(0) }}                  │
│     - Type checks: {% if step is mapping %}                      │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Multiple Validation Layers**: Each layer validates and transforms data
2. **Defensive Defaults**: Every field has a sensible default value
3. **Never Fail**: Template always renders, even with malformed data
4. **Type Handling**: Supports both ReActStep objects and dict serializations

## Configuration

Agent behavior is configurable:

```python
@dataclass
class AgentConfig:
    """Agent configuration."""
    
    # Execution limits
    max_iterations: int = 50
    timeout_seconds: float = 1800.0
    
    # Reasoning
    default_reasoning_strategy: str = "react_standard"
    
    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Memory
    memory: MemoryConfig = field(default_factory=MemoryConfig)
```

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Tools System](tools.md) - Tool implementation
- [Memory System](memory.md) - Memory architecture
