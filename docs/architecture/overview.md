# Architecture Overview

FlyBrowser is an LLM-powered browser automation framework built on the ReAct (Reasoning and Acting) paradigm. This document provides a high-level overview of the system architecture.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         FlyBrowser SDK                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    FlyBrowser Class                         │ │
│  │ goto() | navigate() | extract() | act() | agent() | observe │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬──────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│    Embedded Mode        │  │     Server Mode         │
│  (Local Playwright)     │  │   (REST API Client)     │
└───────────┬─────────────┘  └───────────┬─────────────┘
            │                            │
            ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ReAct Agent Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Planner    │  │ ReAct Agent  │  │    Memory    │           │
│  │ (Task Plan)  │◄─┤ (Reasoning)  │──►│   System    │           │
│  └──────────────┘  └──────┬───────┘  └──────────────┘           │
│                           │                                     │
│                    ┌──────┴───────┐                             │
│                    │ Tool Registry │                            │
│                    │  (32+ Tools)  │                            │
│                    └──────┬───────┘                             │
└───────────────────────────┼─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Navigation  │  │  Interaction  │  │   Extraction  │
│     Tools     │  │     Tools     │  │     Tools     │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Browser Core Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Page      │  │   Element    │  │   Browser    │           │
│  │  Controller  │  │   Detector   │  │   Manager    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Playwright (Browser)                       │
│              Chromium  │  Firefox  │  WebKit                    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SDK Layer

The `FlyBrowser` class is the main entry point:

- **Unified Interface**: Same API for embedded and server modes
- **Mode Detection**: Automatically selects embedded or server mode based on configuration
- **Method Routing**: Routes SDK methods to appropriate operation modes

### 2. ReAct Agent Layer

The intelligent automation core:

- **ReActAgent**: Implements thought-action-observation cycles
- **TaskPlanner**: Creates execution plans for complex tasks
- **Memory System**: Manages context and execution history
- **Tool Registry**: Catalogs and manages available tools

### 3. Browser Core Layer

Low-level browser interaction:

- **PageController**: Manages page state and navigation
- **ElementDetector**: AI-powered element finding
- **BrowserManager**: Playwright browser lifecycle

### 4. LLM Integration

Multi-provider LLM support:

- **LLMProviderFactory**: Creates provider instances
- **BaseLLMProvider**: Abstract interface for all providers
- **Model Discovery**: Automatic capability detection

## Deployment Modes

### Embedded Mode

Browser runs locally in the same process:

```python
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://example.com")
```

- Direct Playwright integration
- No network overhead
- Single-user scenarios

### Server Mode

Connects to a FlyBrowser server:

```python
async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    await browser.goto("https://example.com")
```

- Multi-tenant support
- Horizontal scaling
- Session management

## Operation Modes

The SDK automatically sets operation modes based on method calls:

| SDK Method | Operation Mode | Optimization |
|------------|----------------|--------------|
| `goto()` | - | Direct navigation |
| `navigate()` | NAVIGATE | Navigation with element finding |
| `act()` | EXECUTE | Fast action execution |
| `extract()` | SCRAPE | Data extraction focus |
| `agent()` | AUTO | Full autonomous planning |
| `observe()` | RESEARCH | Element discovery |

## Data Flow

### ReAct Execution Cycle

```
Task Input
    │
    ▼
┌─────────────────┐
│  Task Planning  │ (for complex tasks)
│  (if needed)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    THINKING     │◄─────────────────────┐
│   (LLM Query)   │                      │
└────────┬────────┘                      │
         │                               │
         ▼                               │
┌─────────────────┐                      │
│    ACTING       │                      │
│ (Tool Execute)  │                      │
└────────┬────────┘                      │
         │                               │
         ▼                               │
┌─────────────────┐       ┌─────────────┐│
│   OBSERVING     │──────►│   Memory    ││
│  (Get Result)   │       │   Update    ││
└────────┬────────┘       └─────────────┘│
         │                               │
         ▼                               │
    Task Complete? ──No──────────────────┘
         │
        Yes
         │
         ▼
    Return Result
```

## Module Structure

```
flybrowser/
├── sdk.py              # Main FlyBrowser class
├── client.py           # HTTP client for server mode
├── agents/             # ReAct agent framework
│   ├── react_agent.py  # Core agent
│   ├── planner.py      # Task planning
│   ├── memory.py       # Memory system
│   ├── config.py       # Configuration
│   ├── types.py        # Type definitions
│   └── tools/          # Browser tools
│       ├── base.py     # Base tool class
│       ├── registry.py # Tool registry
│       ├── navigation.py
│       ├── interaction.py
│       └── extraction.py
├── core/               # Browser core
│   ├── browser.py      # Browser manager
│   ├── page.py         # Page controller
│   └── element.py      # Element detector
├── llm/                # LLM providers
│   ├── factory.py      # Provider factory
│   ├── base.py         # Base provider
│   ├── openai.py
│   ├── anthropic.py
│   └── ollama.py
├── service/            # REST API service
│   ├── app.py          # FastAPI app
│   └── config.py       # Service config
└── security/           # Security features
    └── pii_handler.py  # PII masking
```

## Key Design Principles

### 1. Transparent Mode Switching

The same code works in both embedded and server modes:

```python
# Embedded mode
browser = FlyBrowser(llm_provider="openai", api_key="sk-...")

# Server mode - same API
browser = FlyBrowser(endpoint="http://localhost:8000")
```

### 2. Capability-Aware Tooling

Tools are filtered based on model capabilities:

```python
# Vision-enabled models get screenshot tools
# Text-only models get text extraction tools
registry = tool_registry.get_filtered_registry(
    model_info.capabilities,
    warn_suboptimal=True,
)
```

### 3. Memory-Augmented Reasoning

The memory system provides context for decisions:

```python
# Memory tracks execution history
memory.record_cycle(thought, action, observation)

# Context for next reasoning step
context = memory.get_reasoning_context()
```

### 4. Structured Output

All LLM responses use JSON schemas for consistency:

```python
# Responses always parsed to structured format
response = llm.query(prompt, schema=REACT_RESPONSE_SCHEMA)
```

## See Also

- [ReAct Framework](react.md) - Detailed ReAct implementation
- [Tools System](tools.md) - Tool architecture
- [Memory System](memory.md) - Memory management
- [LLM Integration](llm-integration.md) - Provider architecture
