# Architecture Overview

FlyBrowser is an LLM-powered browser automation framework. It delegates LLM orchestration to **fireflyframework-genai** (FireflyAgent, ReActPattern, ToolKit) and focuses on browser-specific logic.

## System Architecture

```
+------------------------------------------------------------------+
|                         FlyBrowser SDK                            |
|  +------------------------------------------------------------+  |
|  |                    FlyBrowser Class (sdk.py)                |  |
|  | goto() | navigate() | extract() | act() | agent() | observe|  |
|  +------------------------------------------------------------+  |
+-----------------------------+------------------------------------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
+---------------------------+  +---------------------------+
|     Embedded Mode         |  |      Server Mode          |
|   (Local Playwright)      |  | (REST via create_genai_app|
+-------------+-------------+  |  /health, /agents, RBAC)  |
              |                +-------------+-------------+
              |                              |
              v                              v
+------------------------------------------------------------------+
|                      BrowserAgent Layer                           |
|  +--------------+  +-------------------+  +-------------------+  |
|  | FireflyAgent |  | Reasoning Pattern |  | BrowserMemory     |  |
|  | (framework)  |<-| (configurable)    |->| Manager           |  |
|  +--------------+  +---------+---------+  | (-> MemoryManager)|  |
|                              |            +-------------------+  |
|  Reasoning patterns:         |                                   |
|  - ReActPattern (default)    |                                   |
|  - PlanAndExecutePattern     |                                   |
|  - ReflexionPattern          |                                   |
|                   +----------+----------+                        |
|                   | 6 ToolKits (32 tools)|                       |
|                   +----------+----------+                        |
|                                                                  |
|  Middleware Chain (in order):                                     |
|  +-------------------+  +-------------------+                    |
|  | LoggingMiddleware  |  | CostGuardMiddleware| (framework)      |
|  +-------------------+  +-------------------+                    |
|  +------------------------+  +-------------------+               |
|  | ExplainabilityMiddleware|  | ObstacleDetection | (flybrowser) |
|  +------------------------+  +-------------------+               |
|  +-------------------+                                           |
|  | ScreenshotOnError | (flybrowser)                              |
|  +-------------------+                                           |
|                                                                  |
|  Validation:                                                     |
|  +-------------------+                                           |
|  | OutputReviewer     | (schema-based extraction with retries)   |
|  +-------------------+                                           |
+-----------------------------+------------------------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
+---------------+  +---------------+  +---------------+
|  Navigation   |  | Interaction   |  |  Extraction   |
|   ToolKit     |  |   ToolKit     |  |   ToolKit     |
+-------+-------+  +-------+-------+  +-------+-------+
        |                   |                  |
        +-------------------+------------------+
                            v
+------------------------------------------------------------------+
|                      Browser Core Layer                           |
|  +--------------+  +--------------+  +--------------+            |
|  |    Page      |  |   Element    |  |   Browser    |            |
|  |  Controller  |  |   Detector   |  |   Manager    |            |
|  +--------------+  +--------------+  +--------------+            |
+-----------------------------+------------------------------------+
                              v
+------------------------------------------------------------------+
|                       Playwright (Browser)                        |
|               Chromium  |  Firefox  |  WebKit                    |
+------------------------------------------------------------------+

Cross-cutting concerns:
+------------------------------------------------------------------+
| Observability: FireflyTracer (OTLP) | FireflyMetrics (Prometheus)|
| Security: RBACAuthManager (JWT) | APIKeyManager (legacy keys)    |
| Cost: UsageTracker (per-session and global cost tracking)        |
+------------------------------------------------------------------+
```

## Core Components

### 1. SDK Layer (`flybrowser/sdk.py`)

The `FlyBrowser` class is the main entry point:

- **Unified Interface**: Same API for embedded and server modes
- **Mode Detection**: Automatically selects embedded or server mode based on `endpoint` parameter
- **Method Routing**: Routes SDK methods to the underlying `BrowserAgent`

### 2. BrowserAgent Layer (`flybrowser/agents/browser_agent.py`)

The intelligent automation core, built on fireflyframework-genai:

- **FireflyAgent**: Wraps Pydantic AI Agent for LLM orchestration
- **Configurable reasoning patterns**: ReActPattern (default), PlanAndExecutePattern, or ReflexionPattern, selected via `BrowserAgentConfig.reasoning_strategy`
- **BrowserMemoryManager**: Tracks page history, navigation graph, obstacle cache, visited URLs. Delegates to the framework's `MemoryManager` via a dual-write pattern.
- **Middleware chain**: LoggingMiddleware, CostGuardMiddleware, ExplainabilityMiddleware (framework), plus ObstacleDetectionMiddleware and ScreenshotOnErrorMiddleware (FlyBrowser-specific)
- **OutputReviewer**: Schema-based extraction validation with automatic retries (used when `extract()` is called with a schema)

### 3. ToolKit System (`flybrowser/agents/toolkits/`)

Six ToolKits built on `fireflyframework_genai.tools.toolkit.ToolKit`:

| ToolKit | Module | Purpose |
|---------|--------|---------|
| NavigationToolkit | `navigation.py` | goto, back, forward, refresh |
| InteractionToolkit | `interaction.py` | click, type, scroll, hover, select |
| ExtractionToolkit | `extraction.py` | extract_text, screenshot, get_page_state |
| SystemToolkit | `system.py` | complete, fail, wait, ask_user |
| SearchToolkit | `search.py` | web search integration |
| CaptchaToolkit | `captcha.py` | CAPTCHA solving |

All 6 toolkits are created via `create_all_toolkits()` and passed to `FireflyAgent`.

### 4. SSE Streaming (`flybrowser/agents/streaming.py`)

Real-time agent reasoning events via Server-Sent Events:

- **AgentStreamEvent**: Typed events (thought, action, observation, complete, error)
- **format_sse_event()**: Converts events to `data: {json}\n\n` format

### 5. Browser Core Layer

Low-level browser interaction:

- **PageController**: Manages page state and navigation
- **ElementDetector**: AI-powered element finding
- **BrowserManager**: Playwright browser lifecycle

## Deployment Modes

### Embedded Mode

Browser runs locally in the same process:

```python
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://example.com")
```

### Server Mode

Connects to a FlyBrowser server:

```python
async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    await browser.goto("https://example.com")
```

## Module Structure

```
flybrowser/
├── __init__.py         # Public API, version 26.02.01
├── sdk.py              # FlyBrowser class (unified SDK)
├── client.py           # HTTP client for server mode
├── agents/
│   ├── browser_agent.py    # BrowserAgent (FireflyAgent + reasoning patterns)
│   ├── toolkits/           # 6 ToolKits
│   │   ├── navigation.py
│   │   ├── interaction.py
│   │   ├── extraction.py
│   │   ├── system.py
│   │   ├── search.py
│   │   └── captcha.py
│   ├── middleware/          # Agent middleware
│   │   ├── obstacle.py     # ObstacleDetectionMiddleware
│   │   └── screenshot.py   # ScreenshotOnErrorMiddleware
│   ├── memory/
│   │   └── browser_memory.py  # BrowserMemoryManager (-> MemoryManager)
│   ├── streaming.py        # AgentStreamEvent, SSE formatting
│   ├── config.py           # AgentConfig
│   ├── types.py            # Action, ToolResult, ReasoningStrategy, etc.
│   ├── memory.py           # AgentMemory, WorkingMemory (legacy)
│   ├── response.py         # AgentRequestResponse
│   ├── context.py          # ContextBuilder, ActionContext
│   └── scope_validator.py  # BrowserScopeValidator
├── cli/
│   ├── main.py         # CLI entry point
│   ├── session.py      # Session management commands
│   ├── direct.py       # Direct SDK-like commands (goto, extract, act, ...)
│   ├── pipeline.py     # Pipeline/workflow execution (run)
│   ├── setup.py        # Setup wizard (quick, llm, server, observability, security)
│   ├── repl.py         # Interactive REPL
│   ├── serve.py        # API server start
│   └── output.py       # CLI output formatting
├── core/
│   ├── browser.py      # BrowserManager
│   ├── page.py         # PageController
│   └── element.py      # ElementDetector
├── llm/
│   ├── base.py         # LLM base types (ModelInfo, LLMResponse)
│   └── provider_status.py
├── observability/
│   ├── tracing.py      # FireflyTracer wrapper (OpenTelemetry)
│   └── metrics.py      # FireflyMetrics + UsageTracker wrappers
├── service/            # REST API service
│   ├── app.py          # create_genai_app() integration
│   ├── auth.py         # RBACAuthManager (JWT + API key auth)
│   ├── session_manager.py  # SessionManager (with UsageTracker)
│   └── config.py       # Service config
└── security/
    └── pii_handler.py  # PII masking
```

## Key Design Principles

### 1. Framework Delegation

LLM orchestration is handled by fireflyframework-genai. FlyBrowser provides browser-specific tools, memory, and middleware. The framework provides logging, cost guard, explainability middleware, reasoning patterns, RBAC security, and observability.

```python
# BrowserAgent wires everything together
self._middleware = [
    LoggingMiddleware(),                                   # Framework
    CostGuardMiddleware(budget_usd=config.budget_limit_usd),  # Framework
    ExplainabilityMiddleware(),                             # Framework
    ObstacleDetectionMiddleware(page_controller),           # FlyBrowser
    ScreenshotOnErrorMiddleware(page_controller),           # FlyBrowser
]
self._agent = FireflyAgent(
    name="flybrowser",
    model=config.model,
    instructions=_SYSTEM_INSTRUCTIONS,
    tools=self._toolkits,        # 6 ToolKits
    middleware=self._middleware,  # 5-layer middleware chain
)
# Reasoning pattern selected from config
self._react = self._create_reasoning_pattern(config)
# Could be ReActPattern, PlanAndExecutePattern, or ReflexionPattern
```

### 2. Transparent Mode Switching

The same code works in both embedded and server modes:

```python
# Embedded mode
browser = FlyBrowser(llm_provider="openai", api_key="sk-...")

# Server mode - same API
browser = FlyBrowser(endpoint="http://localhost:8000")
```

### 3. Memory-Augmented Reasoning

BrowserMemoryManager tracks browser-specific state:

```python
memory.record_page_state(url, title, elements_summary)
memory.record_navigation(from_url, to_url, method)
memory.record_obstacle(url, obstacle_type, resolution)
context = memory.format_for_prompt()
```

## See Also

- [Framework Integration](framework-integration.md) - Detailed framework subsystem integration
- [ReAct Framework](react.md) - How BrowserAgent uses reasoning patterns
- [Tools System](tools.md) - ToolKit architecture
- [Memory System](memory.md) - BrowserMemoryManager and framework MemoryManager
- [Security Architecture](security.md) - RBAC and JWT authentication
- [Observability](../features/observability.md) - Tracing, metrics, and cost tracking
- [CLI Reference](../reference/cli.md) - Session, direct, and pipeline commands
