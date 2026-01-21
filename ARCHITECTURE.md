# FlyBrowser Architecture

```
  _____.__         ___.
_/ ____\  | ___.__.\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/
```

This document explains how FlyBrowser is built, why we made certain design choices, and how the pieces fit together. Whether you're contributing code, debugging an issue, or just curious about the internals, this is your guide.

---

## The Big Picture

FlyBrowser translates natural language into browser automation. You say "click the login button," and the system figures out which element to click and executes the action. Here's how that works at a high level:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REQUEST FLOW                                       │
│                                                                              │
│   "Click the login button"                                                   │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │   FlyBrowser    │  Entry point - routes to appropriate agent             │
│   │   (sdk.py)      │                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐     ┌─────────────────┐                                │
│   │  ActionAgent    │────▶│   LLM Provider  │  "What element matches         │
│   │                 │◀────│   (OpenAI, etc) │   'login button'?"             │
│   └────────┬────────┘     └─────────────────┘                                │
│            │                                                                 │
│            │  Selector: button#login-btn                                     │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │  Playwright     │  Execute: page.click("button#login-btn")               │
│   │  (Browser)      │                                                        │
│   └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Modes

FlyBrowser runs in three configurations, each suited to different use cases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   EMBEDDED MODE              STANDALONE MODE           CLUSTER MODE         │
│   ──────────────             ───────────────           ────────────         │
│                                                                             │
│   ┌─────────────┐            ┌─────────────┐          ┌─────────────┐       │
│   │ Your Script │            │   Client    │          │   Client    │       │
│   │     +       │            └──────┬──────┘          └──────┬──────┘       │
│   │ FlyBrowser  │                   │                        │              │
│   │     +       │                   ▼                        ▼              │
│   │  Browser    │            ┌─────────────┐          ┌─────────────┐       │
│   └─────────────┘            │  REST API   │          │Load Balancer│       │
│                              │  + Browser  │          └──────┬──────┘       │
│   Everything in              └─────────────┘                 │              │
│   one process                                    ┌───────────┼───────────┐  │
│                              Shared service      │           │           │  │
│   Best for:                  for multiple       ┌▼─┐       ┌─▼┐       ┌─▼┐  │
│   • Scripts                  clients            │N1│◄─────►│N2│◄─────►│N3│  │
│   • Development                                 └──┘ Raft  └──┘       └──┘  │
│   • Quick tasks              Best for:                                      │
│                              • Team usage        High availability          │
│                              • Microservices     with automatic failover    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Embedded Mode**: The browser runs in your Python process. Simple, no network overhead, perfect for scripts and development.

**Standalone Mode**: A REST API server manages browser sessions. Multiple clients can share one server. Good for teams or when you want to separate browser management from your application.

**Cluster Mode**: Multiple nodes coordinate via Raft consensus. If a node dies, sessions migrate automatically. This is what you want for production workloads that can't afford downtime.

---

## Project Structure

Here's how the codebase is organized:

```
flybrowser/
│
├── sdk.py                 # Main entry point - the FlyBrowser class
├── client.py              # HTTP client for server mode
├── exceptions.py          # Custom exception types
│
├── agents/                # The brains - LLM-powered automation
│   ├── base_agent.py      # Shared agent functionality
│   ├── action_agent.py    # Clicks, types, form fills
│   ├── extraction_agent.py    # Pulls data from pages
│   ├── navigation_agent.py    # Follows links, navigates menus
│   ├── workflow_agent.py      # Multi-step sequences
│   └── monitoring_agent.py    # Waits for conditions
│
├── core/                  # Browser automation primitives
│   ├── browser.py         # Playwright wrapper
│   ├── browser_pool.py    # Connection pooling for server mode
│   ├── page.py            # Page interactions
│   ├── element.py         # Element detection
│   └── recording.py       # Screenshots and video
│
├── llm/                   # LLM provider abstraction
│   ├── factory.py         # Creates provider instances
│   ├── base.py            # Provider interface
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── ollama_provider.py
│   ├── cache.py           # Response caching
│   ├── cost_tracker.py    # Token/cost accounting
│   ├── rate_limiter.py    # Prevents quota exhaustion
│   └── retry.py           # Exponential backoff
│
├── prompts/               # Prompt engineering
│   ├── manager.py         # Prompt selection and rendering
│   ├── registry.py        # Template storage
│   └── templates/         # YAML prompt definitions
│
├── security/              # Credential protection
│   └── pii_handler.py     # Masks sensitive data from LLMs
│
├── service/               # REST API layer
│   ├── app.py             # Standalone FastAPI server
│   ├── ha_app.py          # Cluster-mode server
│   ├── auth.py            # API key validation
│   ├── session_manager.py # Browser session lifecycle
│   └── cluster/           # Distributed coordination
│       ├── ha_node.py     # Cluster node implementation
│       ├── load_balancer.py
│       ├── state_machine.py
│       └── raft/          # Consensus protocol
│
└── cli/                   # Command-line tools
    ├── serve.py           # flybrowser-serve
    ├── cluster.py         # flybrowser-cluster
    ├── admin.py           # flybrowser-admin
    └── setup.py           # flybrowser-setup
```

---

## Layer by Layer

The system is organized into layers, each with a specific responsibility. Data flows down through the layers, and each layer only talks to the one directly below it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   SERVICE LAYER (optional)                                                  │
│   ────────────────────────                                                  │
│   REST API endpoints, authentication, session management                    │
│   Only present when running as a server                                     │
│                                                                             │
│   Files: service/app.py, service/auth.py, service/session_manager.py        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SDK LAYER                                                                 │
│   ─────────                                                                 │
│   The FlyBrowser class - your main interface                                │
│   Works identically in embedded and server modes                            │
│                                                                             │
│   Files: sdk.py, client.py                                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   AGENT LAYER                                                               │
│   ───────────                                                               │
│   Specialized agents for different tasks                                    │
│   Each agent knows how to accomplish one type of goal                       │
│                                                                             │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│   │ ActionAgent  │ │ Navigation   │ │ Extraction   │ │ Workflow     │       │
│   │              │ │ Agent        │ │ Agent        │ │ Agent        │       │
│   │ clicks,types │ │ follows      │ │ pulls data   │ │ multi-step   │       │
│   │ fills forms  │ │ links        │ │ from pages   │ │ sequences    │       │
│   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│                                                                             │
│   Files: agents/*.py                                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LLM LAYER                                                                 │
│   ─────────                                                                 │
│   Abstracts away provider differences                                       │
│   Handles caching, retries, rate limiting, cost tracking                    │
│                                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ OpenAI   │  │Anthropic │  │ Ollama   │  │  Cache   │  │  Retry   │      │
│   │ Provider │  │ Provider │  │ Provider │  │          │  │ Handler  │      │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                             │
│   Files: llm/*.py                                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BROWSER LAYER                                                             │
│   ─────────────                                                             │
│   Playwright wrapper with additional capabilities                           │
│   Element detection, page control, recording                                │
│                                                                             │
│   Files: core/*.py                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deep Dives

### The Agent System

Agents are the heart of FlyBrowser's intelligence. Each agent specializes in one type of task and knows how to break it down into steps the browser can execute.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   AGENT RESPONSIBILITIES                                                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ ActionAgent                                                         │   │
│   │ ─────────────                                                       │   │
│   │ Handles: clicks, typing, form fills, scrolling                      │   │
│   │                                                                     │   │
│   │ Input:  "Fill in the email field with user@example.com"             │   │
│   │ Output: [                                                           │   │
│   │           { action: "click", selector: "#email" },                  │   │
│   │           { action: "type", text: "user@example.com" }              │   │
│   │         ]                                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ ExtractionAgent                                                     │   │
│   │ ────────────────                                                    │   │
│   │ Handles: pulling structured data from pages                         │   │
│   │                                                                     │   │
│   │ Input:  "Get all product names and prices"                          │   │
│   │ Output: [                                                           │   │
│   │           { name: "Widget A", price: 29.99 },                       │   │
│   │           { name: "Widget B", price: 49.99 }                        │   │
│   │         ]                                                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ NavigationAgent                                                     │   │
│   │ ───────────────                                                     │   │
│   │ Handles: following links, menu navigation, page transitions         │   │
│   │                                                                     │   │
│   │ Input:  "Go to the pricing page"                                    │   │
│   │ Output: { action: "click", selector: "a[href='/pricing']" }         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ WorkflowAgent                                                       │   │
│   │ ─────────────                                                       │   │
│   │ Handles: multi-step sequences with state management                 │   │
│   │                                                                     │   │
│   │ Input:  YAML workflow definition                                    │   │
│   │ Output: Execution results for each step                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The LLM Layer

The LLM layer abstracts away the differences between providers. Whether you're using OpenAI, Anthropic, or a local Ollama model, the rest of the system doesn't need to know.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LLM REQUEST FLOW                                                          │
│                                                                             │
│   Agent Request                                                             │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────┐                                                           │
│   │   Cache     │──── Hit? ──── Yes ────▶ Return cached response            │
│   └──────┬──────┘                                                           │
│          │ No                                                               │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │Rate Limiter │──── Quota? ──── No ────▶ Wait or fail                     │
│   └──────┬──────┘                                                           │
│          │ Yes                                                              │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │   Retry     │                                                           │
│   │   Handler   │◀─────────────────────────────────────┐                    │
│   └──────┬──────┘                                      │                    │
│          │                                             │                    │
│          ▼                                             │                    │
│   ┌─────────────┐                                      │                    │
│   │  Provider   │──── Error? ──── Retryable? ──── Yes ─┘                    │
│   │  (OpenAI,   │                     │                                     │
│   │  Anthropic, │                     No                                    │
│   │  Ollama)    │                     │                                     │
│   └──────┬──────┘                     ▼                                     │
│          │                       Raise error                                │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │Cost Tracker │──── Record tokens and cost                                │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │   Cache     │──── Store response                                        │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   Return response                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this design?**

- **Caching first**: LLM calls are expensive. Checking the cache before anything else saves money.
- **Rate limiting before retry**: No point retrying if you're out of quota.
- **Retry with backoff**: Transient errors (rate limits, timeouts) often resolve themselves.
- **Cost tracking last**: Only count successful calls.

### Cluster Consensus

In cluster mode, nodes coordinate using the Raft consensus protocol. This ensures that even if nodes fail, the cluster continues operating correctly.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   RAFT CONSENSUS                                                            │
│                                                                             │
│   Normal Operation:                                                         │
│   ─────────────────                                                         │
│                                                                             │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐                       │
│   │ Node 1  │◀───────▶│ Node 2  │◀───────▶│ Node 3  │                       │
│   │ LEADER  │         │FOLLOWER │         │FOLLOWER │                       │
│   └────┬────┘         └────┬────┘         └────┬────┘                       │
│        │                   │                   │                            │
│        │   Heartbeats      │                   │                            │
│        │──────────────────▶│                   │                            │
│        │──────────────────────────────────────▶│                            │
│        │                   │                   │                            │
│        │   Log Replication │                   │                            │
│        │══════════════════▶│                   │                            │
│        │══════════════════════════════════════▶│                            │
│                                                                             │
│   Leader Failure:                                                           │
│   ───────────────                                                           │
│                                                                             │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐                       │
│   │ Node 1  │    ✗    │ Node 2  │◀───────▶│ Node 3  │                       │
│   │  DEAD   │         │CANDIDATE│         │FOLLOWER │                       │
│   └─────────┘         └────┬────┘         └────┬────┘                       │
│                            │                   │                            │
│                            │   Vote Request    │                            │
│                            │──────────────────▶│                            │
│                            │◀──────────────────│                            │
│                            │   Vote Granted    │                            │
│                            │                   │                            │
│                            ▼                   │                            │
│                       ┌─────────┐              │                            │
│                       │ Node 2  │◀─────────────┘                            │
│                       │ LEADER  │  (new leader elected)                     │
│                       └─────────┘                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What gets replicated?**

- Session ownership (which node owns which session)
- Node health status
- Configuration changes

**What doesn't get replicated?**

- Browser state (too large, too fast-changing)
- LLM responses (cached locally)

---

## Design Decisions

### Why Layered Architecture?

Each layer has one job. The service layer handles HTTP. The SDK provides a clean API. Agents implement the intelligence. The LLM layer manages providers. The browser layer automates.

This separation means:
- You can test each layer independently
- You can swap implementations (different LLM provider, different browser engine)
- Dependencies flow one direction (down), making the code easier to reason about

### Why Multiple LLM Providers?

Different models excel at different tasks. OpenAI's GPT-5.2 is great for complex reasoning. Anthropic's Claude has strong vision capabilities. Ollama lets you run everything locally for privacy and cost savings.

Supporting multiple providers also gives you resilience. If one provider is down or rate-limited, you can fall back to another.

### Why Caching?

LLM API calls are expensive—both in money and time. A single extraction might cost $0.01 and take 2 seconds. If you're running the same extraction repeatedly, that adds up fast.

Our cache uses a hash of the prompt, model, and parameters as the key. Identical requests return cached responses instantly. The cache has a configurable TTL (default: 1 hour) and uses LRU eviction when full.

### Why Prompt Templates?

Prompts are code. They need version control, testing, and the ability to roll back. Our template system stores prompts as YAML files with:

- Version numbers
- Required variables (validated at runtime)
- Examples for few-shot learning
- Metadata for A/B testing

The A/B testing system uses Thompson sampling to automatically shift traffic toward better-performing prompts.

---

## Security Model

### PII Protection

Credentials never touch the LLM. When you store a credential, FlyBrowser replaces it with a placeholder before sending anything to the AI:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   CREDENTIAL FLOW                                                               │
│                                                                                 │
│   1. User stores credential                                                     │
│      browser.store_credential("password", "secret123")                          │
│                                                                                 │
│   2. User requests action                                                       │
│      browser.act("type the password into the password field")                   │
│                                                                                 │
│   3. Before sending to LLM, credentials are replaced                            │
│      "type {{CREDENTIAL:password}} into the password field"                     │
│                                                                                 │
│   4. LLM returns action plan                                                    │
│      { action: "type", selector: "#password", text: "{{CREDENTIAL:password}}" } │
│                                                                                 │
│   5. Before executing, placeholders are replaced with real values               │
│      { action: "type", selector: "#password", text: "secret123" }               │
│                                                                                 │
│   The LLM never sees "secret123"                                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### API Authentication

The REST API uses API keys passed in the `X-API-Key` header. Keys can have:
- Expiration dates
- Rate limits
- Scope restrictions

### Resource Limits

To prevent abuse:
- Maximum concurrent sessions per API key
- Session timeout (default: 1 hour)
- Request size limits
- Response size limits

---

## Performance Tuning

### Caching

| Setting | Default | Description |
|---------|---------|-------------|
| `cache.enabled` | `true` | Enable/disable caching |
| `cache.ttl_seconds` | `3600` | How long to keep cached responses |
| `cache.max_size` | `1000` | Maximum number of cached entries |

### Rate Limiting

| Setting | Default | Description |
|---------|---------|-------------|
| `rate_limit.requests_per_minute` | `60` | Max requests per minute |
| `rate_limit.tokens_per_minute` | `100000` | Max tokens per minute |
| `rate_limit.concurrent_requests` | `10` | Max concurrent requests |

### Retry

| Setting | Default | Description |
|---------|---------|-------------|
| `retry.max_retries` | `3` | Maximum retry attempts |
| `retry.initial_delay` | `1.0` | First retry delay (seconds) |
| `retry.max_delay` | `60.0` | Maximum retry delay (seconds) |
| `retry.exponential_base` | `2.0` | Backoff multiplier |

---

## Monitoring

### Metrics Endpoint

The `/metrics` endpoint returns:

```json
{
  "total_requests": 12345,
  "active_sessions": 5,
  "cache_stats": {
    "hits": 8000,
    "misses": 2000,
    "hit_rate": 0.80
  },
  "cost_stats": {
    "total_cost": 45.67,
    "total_tokens": 456789
  },
  "error_rates": {
    "llm_errors": 0.02,
    "browser_errors": 0.01
  }
}
```

### Health Checks

The `/health` endpoint checks:
- Service status
- Browser availability
- LLM provider connectivity
- Cluster status (in cluster mode)

### Logging

All logs are structured JSON for easy parsing:

```json
{
  "timestamp": "2026-01-21T10:30:00Z",
  "level": "INFO",
  "component": "extraction_agent",
  "session_id": "abc123",
  "message": "Extraction completed",
  "duration_ms": 1234,
  "tokens_used": 567
}
```

---

## Future Directions

We're actively working on:

1. **WebSocket streaming** - Real-time updates during long-running operations
2. **Distributed caching** - Redis integration for shared cache across nodes
3. **Background jobs** - Queue system for async workflow execution
4. **Multi-tenancy** - Isolated environments per customer
5. **Advanced analytics** - Usage patterns, cost optimization suggestions

---

<p align="center">
  <em>Questions? Open an issue or join our Discord.</em>
</p>
