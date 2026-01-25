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
