# FlyBrowser Comprehensive Overhaul Design

**Date:** 2026-02-13
**Status:** Approved
**Approach:** Layered Migration (Approach A)

## Problem Statement

FlyBrowser uses only ~5% of fireflyframework-genai's capabilities (FireflyAgent, ReActPattern, ToolKit, firefly_tool). The framework provides 20+ major subsystems that could replace custom implementations, improve reliability, and enable new features. Additionally, the CLI lacks session management commands, one-shot SDK-like operations, and pipeline execution.

## Goals

1. **Full framework leverage** — Migrate all applicable layers to use fireflyframework-genai
2. **Enhanced CLI** — Session management, direct commands, pipelines, enhanced wizard
3. **Both deployment modes first-class** — Embedded SDK and Server mode with equal priority
4. **State-of-art observability** — OpenTelemetry tracing, Prometheus metrics, cost tracking
5. **Comprehensive documentation** — Restructured, complete, with examples

## Non-Goals

- Rewriting browser core (Playwright integration stays as-is)
- Rewriting cluster layer (Raft consensus, discovery, load balancing stay custom)
- Changing the public SDK API signature (backward compatible)

---

## Section 1: Agent Layer Migration

### 1.1 Memory Migration

**Current:** Custom `BrowserMemoryManager` with page history, navigation graph, visited URLs, obstacle cache.

**Target:** Framework's `MemoryManager` with `ConversationMemory` + `WorkingMemory`.

**Design:**
- `ConversationMemory` tracks multi-turn browser interactions (agent reasoning history)
- `WorkingMemory` stores browser-specific state as scoped key-value pairs:
  - `page_history` — list of visited pages with timestamps
  - `navigation_graph` — URL transition graph
  - `obstacle_cache` — known obstacles per domain
  - `dom_snapshot` — current page state summary
- Backend selection via config: `InMemoryStore` (default), `FileStore` (persistence), `PostgresStore` (server mode)
- Session-scoped memory with fork support for delegation

**Migration path:** Create `BrowserMemoryAdapter` that wraps `MemoryManager` and exposes the same interface as current `BrowserMemoryManager` for backward compat during transition.

### 1.2 Middleware Adoption

**Current:** 2 custom middleware (ObstacleDetection, ScreenshotOnError).

**Target:** Framework built-in middleware + custom middleware.

**Middleware chain (ordered):**
1. `LoggingMiddleware` (framework) — structured logging of all agent operations
2. `CostGuardMiddleware` (framework) — enforce budget_limit_usd
3. `ExplainabilityMiddleware` (framework) — record decision traces
4. `ObstacleDetectionMiddleware` (custom) — detect/handle page obstacles
5. `ScreenshotOnErrorMiddleware` (custom) — capture on failure

### 1.3 Extraction Validation

**Current:** No validation on extracted data.

**Target:** Framework's `OutputReviewer` for quality assurance.

**Design:**
- When user provides a Pydantic schema to `extract()`, wrap with `OutputReviewer`
- `max_retries=3` for failed validations
- Confidence scoring on results
- Consistency checking when `quality="high"` is specified

### 1.4 Additional Reasoning Patterns

**Current:** ReAct only.

**Target:** Config-driven pattern selection.

**Design:**
- `ReActPattern` — default, good for most browser tasks
- `PlanAndExecutePattern` — for complex multi-page workflows (checkout, form sequences)
- `ReflexionPattern` — for self-correcting tasks (retry with reflection on failure)
- Selection via `AgentConfig.reasoning_strategy` enum
- Framework's `ReasoningPipeline` for chaining patterns

### 1.5 Agent Registration

**Current:** Agent created manually in `BrowserAgent.__init__`.

**Target:** Use `@firefly_agent` decorator + registry.

**Design:**
- Register BrowserAgent in framework's `AgentRegistry`
- Enable auto-discovery for REST exposure
- Support multiple agent variants (fast, thorough, extract-only)

---

## Section 2: Service Layer & REST Exposure

### 2.1 Framework REST Base

**Current:** Custom FastAPI app with manual endpoint definitions.

**Target:** Framework's `create_genai_app()` as base + custom browser routes.

**Design:**
```python
from fireflyframework_genai.exposure.rest import create_genai_app

# Framework provides: health, readiness, liveness, agent endpoints, CORS, rate limiting
app = create_genai_app(title="FlyBrowser", version=__version__)

# Custom browser-specific routes mounted alongside
app.include_router(session_router, prefix="/sessions")
app.include_router(browser_router, prefix="/browser")
```

### 2.2 RBAC Integration

**Current:** Basic API key auth in `service/auth.py`.

**Target:** Framework's `RBACManager` with JWT.

**Roles:**
- `admin` — full access (create/delete sessions, cluster management, config)
- `operator` — session lifecycle + execution (create sessions, run commands)
- `viewer` — read-only (list sessions, view results, screenshots)

### 2.3 Streaming Adoption

**Current:** Custom SSE implementation.

**Target:** Framework's `sse_stream()` + `sse_stream_incremental()`.

**Design:**
- Buffered mode for command results
- Incremental mode for agent reasoning (token-by-token)
- WebSocket support for bidirectional communication (live REPL via web)

### 2.4 Session Manager Enhancement

**Current:** In-memory session tracking with timeout cleanup.

**Target:** Enhanced with framework memory and usage tracking.

**Additions:**
- `ConversationMemory` per session (persisted reasoning history)
- `UsageTracker` per session (tokens, cost, latency)
- Session state serialization for resume-after-restart
- Session metadata (tags, labels, owner for multi-tenant)

---

## Section 3: CLI Overhaul

### 3.1 Session Management Commands

```
flybrowser session create [--provider openai] [--model gpt-4o] [--headless] [--name <name>]
flybrowser session list [--format table|json] [--status active|all]
flybrowser session info <session-id>
flybrowser session connect <session-id>      # Enters REPL attached to session
flybrowser session exec <session-id> <cmd>   # One-shot command on session
flybrowser session close <session-id>
flybrowser session close-all [--force]
```

**Behavior:**
- `create` starts a browser session (embedded or server-attached)
- `connect` drops into REPL with session context
- `exec` runs a single command and returns result
- Sessions persist across CLI invocations when using server mode

### 3.2 Direct SDK-like Commands

```
flybrowser goto <url> [--session <id>] [--wait-for <selector>]
flybrowser extract <query> [--session <id>] [--schema <file>] [--format json|csv|table]
flybrowser act <instruction> [--session <id>] [--screenshot-after]
flybrowser observe <query> [--session <id>] [--format json|table]
flybrowser screenshot [--session <id>] [--output <file>] [--full-page]
flybrowser agent <task> [--session <id>] [--max-iterations 50] [--stream]
```

**Behavior:**
- Auto-creates ephemeral session if none specified
- Reuses specified session, or last-used session with `--session last`
- Output format defaults to human-readable table, with `--format json` for scripting

### 3.3 Pipeline Execution

```
flybrowser run <workflow.yaml>
flybrowser run --inline "goto https://example.com && extract 'get all prices'"
```

**Workflow YAML format:**
```yaml
name: price-checker
sessions:
  main:
    provider: openai
    model: gpt-4o
    headless: true

steps:
  - name: navigate
    session: main
    action: goto
    url: https://example.com/products

  - name: extract-prices
    session: main
    action: extract
    query: "Get all product names and prices"
    schema: schemas/products.json
    output: prices.json

  - name: screenshot
    session: main
    action: screenshot
    output: products.png
```

Uses framework's `PipelineBuilder` for DAG execution with concurrent steps.

### 3.4 Enhanced Setup Wizard

```
flybrowser setup                     # Full interactive wizard
flybrowser setup quick               # 30-second quick start
flybrowser setup llm                 # LLM provider configuration
flybrowser setup browser             # Browser installation
flybrowser setup server              # Server mode configuration
flybrowser setup cluster             # Cluster setup
flybrowser setup observability       # Tracing/metrics setup
flybrowser setup security            # RBAC/auth configuration
flybrowser setup verify              # Full verification
```

**Full wizard flow:**
1. Welcome + version info
2. LLM provider selection + API key + model selection + connectivity test
3. Browser installation (Chromium/Firefox/WebKit) + verification
4. Deployment mode selection (embedded/server/cluster)
5. If server: port, host, workers, TLS
6. If cluster: node count, discovery method
7. Observability setup (OTLP endpoint, Prometheus port)
8. Security setup (enable RBAC, create admin token)
9. Performance preset selection
10. Final verification + summary

### 3.5 Enhanced REPL

**New features:**
- Streaming output (token-by-token for agent responses)
- Session switching: `session switch <id>`
- Cost display after each command: `[cost: $0.003 | tokens: 450 | 1.2s]`
- Pipeline mode: `pipeline start` / `pipeline add <step>` / `pipeline run` / `pipeline end`
- Tab completion for CSS selectors (via page state)
- Rich output with tables and colored diffs

### 3.6 Benchmark Commands

```
flybrowser benchmark run <test-suite.yaml>
flybrowser benchmark compare --models gpt-4o,claude-sonnet --task "extract prices"
flybrowser benchmark report [--format html|json|markdown]
```

Uses framework's `LabSession`, `Benchmark`, `EvalOrchestrator`.

---

## Section 4: Observability & Cross-Cutting

### 4.1 OpenTelemetry Tracing

- `FireflyTracer` for distributed tracing
- Agent spans, tool spans, reasoning spans
- Browser operation spans (navigation, click, extract)
- Session lifecycle spans
- Export to any OTLP collector (Jaeger, Datadog, etc.)

### 4.2 Prometheus Metrics

- `FireflyMetrics` for operational metrics
- Token usage counters (per model, per session)
- Latency histograms (per operation type)
- Error rate gauges
- Session count gauges
- Cost accumulator
- Prometheus `/metrics` endpoint on service

### 4.3 Cost Tracking

- `UsageTracker` integration per session and globally
- Per-command cost in CLI output
- Budget enforcement via `CostGuardMiddleware`
- Cost reports: `flybrowser admin costs [--period today|week|month]`
- Alert thresholds configurable

### 4.4 Explainability & Audit

- `TraceRecorder` on every agent execution
- `AuditTrail` for compliance (immutable, append-only)
- `ExplanationGenerator` for human-readable decision narratives
- Export: `flybrowser admin audit export [--format json|markdown]`

---

## Section 5: Documentation

### Structure

```
docs/
├── index.md
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   ├── setup-wizard.md
│   └── first-automation.md
├── sdk/
│   ├── overview.md
│   ├── embedded-mode.md
│   ├── server-mode.md
│   ├── sessions.md
│   ├── navigation.md
│   ├── actions.md
│   ├── extraction.md
│   ├── observation.md
│   ├── agent.md
│   ├── streaming.md
│   └── recording.md
├── cli/
│   ├── overview.md
│   ├── session-management.md
│   ├── direct-commands.md
│   ├── repl.md
│   ├── pipelines.md
│   ├── setup.md
│   └── admin.md
├── architecture/
│   ├── overview.md
│   ├── framework-integration.md
│   ├── agent-system.md
│   ├── memory.md
│   ├── service-layer.md
│   └── security.md
├── deployment/
│   ├── standalone.md
│   ├── cluster.md
│   ├── docker.md
│   └── kubernetes.md
├── advanced/
│   ├── custom-tools.md
│   ├── custom-providers.md
│   ├── observability.md
│   ├── performance.md
│   ├── stealth.md
│   └── troubleshooting.md
├── api-reference/
│   ├── sdk-reference.md
│   ├── rest-api.md
│   └── config-reference.md
└── plans/
```

### Documentation Standards

- Every doc follows: Purpose > Prerequisites > Guide > Examples > API Reference > Troubleshooting
- Code examples are runnable and tested
- Cross-references between related docs
- Version badge on each page
- Architecture diagrams where applicable

---

## Implementation Order

1. **Phase 1: Agent Layer** — Memory, middleware, validation, reasoning patterns
2. **Phase 2: Service Layer** — REST exposure, RBAC, streaming, session enhancements
3. **Phase 3: CLI** — Session commands, direct commands, pipeline, wizard, REPL
4. **Phase 4: Observability** — Tracing, metrics, cost, explainability
5. **Phase 5: Documentation** — Full restructure and rewrite

Each phase is independently deployable and backward compatible.
