# Framework Integration

FlyBrowser delegates core AI infrastructure to **fireflyframework-genai**, Firefly's shared GenAI framework. This document describes how each framework subsystem is integrated and how to configure it.

## Overview

The framework provides seven capabilities that FlyBrowser consumes:

| Capability | Framework Module | FlyBrowser Integration |
|-----------|-----------------|----------------------|
| Memory | `fireflyframework_genai.memory` | `BrowserMemoryManager` |
| Middleware | `fireflyframework_genai.agents.builtin_middleware` | `BrowserAgent.__init__` |
| Reasoning | `fireflyframework_genai.reasoning` | `BrowserAgentConfig.reasoning_strategy` |
| REST exposure | `fireflyframework_genai.exposure.rest` | `create_genai_app()` in `app.py` |
| Security | `fireflyframework_genai.security.rbac` | `RBACAuthManager` |
| Observability | `fireflyframework_genai.observability` | `tracing.py`, `metrics.py` |
| Validation | `fireflyframework_genai.validation` | `OutputReviewer` in `extract()` |

## Memory Layer

`BrowserMemoryManager` wraps the framework's `MemoryManager` with a dual-write pattern: every mutation updates both a local Python cache (for fast, typed access) and the framework's `WorkingMemory` (for persistence and pluggable backends).

### Architecture

```
BrowserMemoryManager
  |
  |-- Local cache (PageSnapshot, ObstacleInfo, facts)
  |       Fast, typed access for prompt formatting
  |
  |-- MemoryManager(store=InMemoryStore())
  |       Framework working memory for persistence
  |
  +-- ConversationMemory
          Tracks conversation turns per session
```

### How It Works

```python
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager

memory = BrowserMemoryManager()

# Every mutation writes to both local cache and framework
memory.record_page_state("https://example.com", "Example", "3 links, 1 form")
memory.record_navigation("https://google.com", "https://example.com", "click")
memory.set_fact("login_status", "authenticated")

# Reads come from the local cache (fast, typed)
page = memory.get_current_page()  # -> PageSnapshot
visited = memory.has_visited_url("https://example.com")  # -> True

# Prompt context merges both sources
context = memory.format_for_prompt()
```

### Backend Pluggability

The `MemoryManager` accepts any store that implements the framework's store interface:

| Store | Module | Use Case |
|-------|--------|----------|
| `InMemoryStore` | `fireflyframework_genai.memory.store` | Default, no persistence |
| `FileStore` | `fireflyframework_genai.memory.store` | Local file-based persistence |
| `PostgresStore` | `fireflyframework_genai.memory.store` | Production distributed persistence |

To switch backends, modify the `BrowserMemoryManager.__init__` to use a different store:

```python
from fireflyframework_genai.memory.store import PostgresStore

# In BrowserMemoryManager.__init__:
self._framework_memory = MemoryManager(store=PostgresStore(dsn="postgres://..."))
```

For more details, see [Memory System](memory.md).

## Middleware Chain

`BrowserAgent` assembles a middleware chain from both framework and FlyBrowser-specific middleware. Middleware runs around every agent action, providing cross-cutting concerns.

### Middleware Stack

The middleware executes in order for each agent step:

```
Request
  |
  v
1. LoggingMiddleware           (framework) - Structured logging of every step
2. CostGuardMiddleware         (framework) - Aborts if budget_usd is exceeded
3. ExplainabilityMiddleware    (framework) - Records reasoning trace
4. ObstacleDetectionMiddleware (flybrowser) - Detects and handles page obstacles
5. ScreenshotOnErrorMiddleware (flybrowser) - Captures screenshot on failures
  |
  v
Agent Action
```

### Configuration

```python
from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig

config = BrowserAgentConfig(
    model="openai:gpt-4o",
    budget_limit_usd=5.0,  # CostGuardMiddleware threshold
)

agent = BrowserAgent(page_controller=page, config=config)
```

The `CostGuardMiddleware` is configured via `BrowserAgentConfig.budget_limit_usd`. When accumulated cost exceeds this limit, the middleware raises an exception to prevent runaway spending.

### Adding Custom Middleware

Framework middleware must implement the `AbstractMiddleware` interface:

```python
from fireflyframework_genai.agents.middleware import AbstractMiddleware

class MyCustomMiddleware(AbstractMiddleware):
    async def before_action(self, context):
        # Runs before each agent action
        pass

    async def after_action(self, context, result):
        # Runs after each agent action
        return result
```

## Reasoning Patterns

`BrowserAgent` supports three reasoning patterns from the framework, selected via `BrowserAgentConfig.reasoning_strategy`:

| Strategy | Framework Class | Behavior |
|----------|----------------|----------|
| `REACT_STANDARD` | `ReActPattern` | Think-Act-Observe loop (default) |
| `PLAN_AND_SOLVE` | `PlanAndExecutePattern` | Generate plan first, then execute steps |
| `SELF_REFLECTION` | `ReflexionPattern` | Execute, reflect on results, self-correct |

### Configuration

```python
from flybrowser.agents.browser_agent import BrowserAgentConfig
from flybrowser.agents.types import ReasoningStrategy

# Default ReAct
config = BrowserAgentConfig(reasoning_strategy=ReasoningStrategy.REACT_STANDARD)

# Plan-and-Execute for complex multi-step tasks
config = BrowserAgentConfig(reasoning_strategy=ReasoningStrategy.PLAN_AND_SOLVE)

# Reflexion for tasks requiring self-correction
config = BrowserAgentConfig(reasoning_strategy=ReasoningStrategy.SELF_REFLECTION)
```

### When to Use Each Pattern

- **ReAct** -- Best for most browser automation tasks. Interleaves thinking and acting, adapting as the page changes.
- **PlanAndExecute** -- Best for well-defined multi-step workflows where you want a plan generated upfront (e.g., "fill this 10-field form").
- **Reflexion** -- Best for tasks where initial attempts may fail and the agent needs to learn from errors (e.g., navigating complex SPAs).

All patterns respect `max_iterations` from `BrowserAgentConfig`.

## REST Exposure

The API server uses `create_genai_app()` from the framework instead of plain FastAPI. This provides standard GenAI endpoints alongside FlyBrowser's custom routes.

### Framework-Provided Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/live` | GET | Liveness probe (Kubernetes) |
| `/health/ready` | GET | Readiness probe (Kubernetes) |
| `/agents` | GET | List registered agents |

### Integration

```python
# flybrowser/service/app.py
from fireflyframework_genai.exposure.rest import create_genai_app

app = create_genai_app(
    title="FlyBrowser API",
    version=__version__,
    lifespan=lifespan,
)

# FlyBrowser-specific routes are added to the same app
@app.post("/sessions")
async def create_session(...):
    ...
```

The `create_genai_app()` factory configures CORS, error handlers, and the standard health/agent endpoints. FlyBrowser adds its own session, navigation, extraction, and streaming routes on top.

## Security (RBAC)

The `RBACAuthManager` wraps the framework's `RBACManager` to provide JWT-based access control.

### Architecture

```
Client Request
  |
  +-- Bearer: <JWT token>    --> RBACAuthManager.validate_token()
  |                                --> RBACManager.validate_token()
  |                                --> RBACManager.has_permission()
  |
  +-- X-API-Key: <key>       --> APIKeyManager.validate_key()
                                    (backward compatibility)
```

### Roles and Permissions

| Role | Permissions |
|------|-------------|
| `admin` | All operations (`*`) |
| `operator` | Create/delete/use sessions, navigate, extract, act, stream, view recordings |
| `viewer` | List/view sessions, list/download recordings |

### Usage

```python
from flybrowser.service.auth import RBACAuthManager

mgr = RBACAuthManager(jwt_secret="your-secret-key")

# Create a token
token = mgr.create_token(user_id="alice", roles=["operator"])

# Validate a token
claims = mgr.validate_token(token)
# -> {"user_id": "alice", "roles": ["operator"], "exp": ...}

# Check permissions
mgr.has_permission("operator", "sessions.create")  # True
mgr.has_permission("viewer", "sessions.create")    # False
```

For complete security documentation, see [Security Architecture](security.md).

## Observability

FlyBrowser integrates the framework's `FireflyTracer`, `FireflyMetrics`, and `UsageTracker` for distributed tracing, metrics collection, and cost tracking.

### Tracing

```python
from flybrowser.observability.tracing import configure_tracing, get_tracer

# Configure at startup
configure_tracing(otlp_endpoint="http://localhost:4317", console=True)

# Use anywhere
tracer = get_tracer()
with tracer.custom_span("navigate"):
    await page.goto(url)
```

### Metrics

```python
from flybrowser.observability.metrics import record_operation, get_cost_summary

# Record an operation
record_operation("extract", latency_ms=1200, tokens=500, cost_usd=0.02)

# Get cumulative cost
summary = get_cost_summary()
# -> {"total_cost_usd": 1.42, "total_tokens": 35000}
```

For complete observability documentation, see [Observability](../features/observability.md).

## Validation (OutputReviewer)

The `OutputReviewer` from the framework validates and retries LLM extraction results against a schema. `BrowserAgent.extract()` uses it automatically when a `schema` parameter is provided.

### How It Works

```
extract(query="...", schema=ProductList)
  |
  v
OutputReviewer(output_type=ProductList, max_retries=3)
  |
  +-- Attempt 1: LLM generates output
  |     |
  |     +-- Validate against schema
  |     |     |
  |     |     +-- Pass? -> Return validated output
  |     |     +-- Fail? -> Feed validation errors back to LLM
  |     |
  +-- Attempt 2: LLM regenerates with error context
  |     ...
  +-- Attempt 3: Final attempt
```

### Usage in BrowserAgent

```python
from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    name: str
    price: float

class ProductList(BaseModel):
    products: List[Product]

# Schema-validated extraction with automatic retries
result = await agent.extract(
    query="Get all products with names and prices",
    schema=ProductList,
    max_retries=3,
)
```

When no schema is provided, `extract()` runs a standard LLM call without validation.

## Module Map

```
flybrowser/
  agents/
    browser_agent.py        # BrowserAgent (assembles all framework pieces)
    memory/
      browser_memory.py     # BrowserMemoryManager -> MemoryManager
    middleware/
      obstacle.py           # FlyBrowser-specific middleware
      screenshot.py         # FlyBrowser-specific middleware
    types.py                # ReasoningStrategy enum
  service/
    app.py                  # create_genai_app() integration
    auth.py                 # RBACAuthManager -> RBACManager
  observability/
    tracing.py              # FireflyTracer wrapper
    metrics.py              # FireflyMetrics + UsageTracker wrappers
```

## See Also

- [Architecture Overview](overview.md) -- High-level system architecture
- [Memory System](memory.md) -- BrowserMemoryManager details
- [Security Architecture](security.md) -- RBAC and JWT details
- [Observability](../features/observability.md) -- Tracing, metrics, and live view
- [ReAct Framework](react.md) -- Reasoning pattern details
