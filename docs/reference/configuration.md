# Configuration Reference

This document provides complete reference documentation for all FlyBrowser configuration options.

## Overview

FlyBrowser can be configured through:
1. Environment variables
2. Command-line arguments
3. Configuration files (YAML/JSON)
4. SDK constructor parameters

Configuration precedence (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values

## Environment Variables

### LLM Provider API Keys

| Variable | Provider | Required | Description |
|----------|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | Yes | OpenAI API key (starts with `sk-`) |
| `ANTHROPIC_API_KEY` | Anthropic | Yes | Anthropic API key (starts with `sk-ant-`) |
| `GOOGLE_API_KEY` | Google Gemini | Yes | Google AI API key (starts with `AIza`) |
| `OLLAMA_HOST` | Ollama | No | Ollama server URL (default: `http://localhost:11434`) |
| `LM_STUDIO_HOST` | LM Studio | No | LM Studio server URL (default: `http://localhost:1234`) |
| `LOCALAI_HOST` | LocalAI | No | LocalAI server URL (default: `http://localhost:8080`) |
| `VLLM_HOST` | vLLM | No | vLLM server URL (default: `http://localhost:8000`) |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_HOST` | `0.0.0.0` | Server bind address |
| `FLYBROWSER_PORT` | `8000` | HTTP server port |
| `FLYBROWSER_WORKERS` | `1` | Number of worker processes |
| `FLYBROWSER_API_KEY` | (none) | API key for authentication |
| `FLYBROWSER_MAX_SESSIONS` | `10` | Maximum concurrent sessions |
| `FLYBROWSER_DATA_DIR` | `./data` | Data storage directory |
| `FLYBROWSER_LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `FLYBROWSER_LOG_FILE` | (none) | Log file path (stdout if not set) |
| `FLYBROWSER_CONFIG` | (none) | Path to configuration file |

### Cluster Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_CLUSTER_ENABLED` | `false` | Enable cluster mode |
| `FLYBROWSER_NODE_ID` | (auto) | Unique node identifier (required in cluster mode) |
| `FLYBROWSER_RAFT_PORT` | `4321` | Raft consensus port |
| `FLYBROWSER_CLUSTER_PEERS` | (none) | Comma-separated peer addresses (host:raft_port) |
| `FLYBROWSER_API_HOST` | `0.0.0.0` | API host address in cluster mode |
| `FLYBROWSER_API_PORT` | `8000` | API port in cluster mode |

### Browser Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_HEADLESS` | `true` | Run browsers in headless mode |
| `FLYBROWSER_BROWSER_TYPE` | `chromium` | Default browser type |
| `FLYBROWSER_TIMEOUT` | `30000` | Default operation timeout (ms) |

### Security Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_PII_MASKING` | `false` | Enable PII masking |
| `FLYBROWSER_RECORDING` | `false` | Enable session recording |
| `FLYBROWSER_TLS_CERT` | (none) | TLS certificate path |
| `FLYBROWSER_TLS_KEY` | (none) | TLS private key path |

## Configuration File

FlyBrowser supports YAML and JSON configuration files.

### File Location

The configuration file is loaded from (in order):
1. Path specified by `--config` CLI argument
2. `FLYBROWSER_CONFIG` environment variable
3. `./flybrowser.yaml` or `./flybrowser.json`
4. `~/.config/flybrowser/config.yaml`
5. `/etc/flybrowser/config.yaml`

### YAML Configuration

```yaml path=null start=null
# flybrowser.yaml - Complete Configuration Example

# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_sessions: 50
  data_dir: "/var/lib/flybrowser"

# Logging
logging:
  level: "INFO"
  file: "/var/log/flybrowser/server.log"
  format: "json"  # "text" or "json"
  rotation:
    max_size: "100MB"
    max_files: 10
    compress: true

# Authentication
auth:
  api_key: "${FLYBROWSER_API_KEY}"  # Use environment variable

# Browser defaults
browser:
  headless: true
  type: "chromium"  # "chromium", "firefox", "webkit"
  timeout: 30000
  viewport:
    width: 1280
    height: 720

# LLM configuration - All supported providers
llm:
  default_provider: "openai"
  default_model: "gpt-5.2"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      # organization: "org-..."  # Optional
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
    gemini:
      api_key: "${GOOGLE_API_KEY}"
    ollama:
      base_url: "http://localhost:11434"
    lm_studio:
      base_url: "http://localhost:1234"
    localai:
      base_url: "http://localhost:8080"
    vllm:
      base_url: "http://localhost:8000"

# Advanced LLM settings
retry:
  max_retries: 3
  initial_delay: 1.0
  max_delay: 60.0
  exponential_base: 2.0
  jitter: true

rate_limit:
  requests_per_minute: 60
  tokens_per_minute: 100000
  concurrent_requests: 10

cache:
  enabled: true
  ttl_seconds: 3600
  max_size: 1000

cost_tracking:
  enabled: true
  track_tokens: true
  track_requests: true
  log_costs: true

# Security
security:
  pii_masking: true
  pii_patterns:
    - email
    - phone
    - ssn
    - credit_card
    - api_key
  recording:
    enabled: false
    video_enabled: true
    screenshot_interval: 5.0
  tls:
    enabled: false
    cert: "/etc/ssl/certs/flybrowser.crt"
    key: "/etc/ssl/private/flybrowser.key"

# Cluster configuration (High Availability)
cluster:
  enabled: false
  node_id: "${HOSTNAME}"
  raft_port: 4321
  peers:
    - "node1.internal:4321"
    - "node2.internal:4321"
    - "node3.internal:4321"
  election_timeout: 1000
  heartbeat_interval: 100
  rebalance_threshold: 0.2
  session_migration_timeout: 30

# Resource limits
resources:
  memory_limit: 4096      # MB per session
  total_memory_limit: 16384  # MB total
  cpu_limit: 1.0          # CPU cores per session
```

### JSON Configuration

```json path=null start=null
{
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "max_sessions": 20,
        "data_dir": "/var/lib/flybrowser"
    },
    "logging": {
        "level": "INFO",
        "file": "/var/log/flybrowser/server.log"
    },
    "browser": {
        "headless": true,
        "type": "chromium",
        "timeout": 30000
    },
    "llm": {
        "default_provider": "openai",
        "default_model": "gpt-4"
    },
    "cluster": {
        "enabled": false
    }
}
```

## SDK Configuration

### FlyBrowser Constructor

The `FlyBrowser` class supports all LLM providers:

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    # LLM Settings - choose any supported provider
    llm_provider="openai",       # "openai", "anthropic", "gemini", "google",
                                 # "ollama", "lm_studio", "localai", "vllm"
    llm_model="gpt-5.2",         # Model name (uses provider default if not set)
    api_key="sk-...",            # LLM API key (or use env vars)
    base_url=None,               # Custom endpoint for local providers
    
    # Browser Settings
    headless=True,               # Headless mode
    browser_type="chromium",     # "chromium", "firefox", "webkit"
    timeout=30.0,                # Timeout in seconds
    
    # Features
    recording_enabled=False,     # Session recording
    pii_masking_enabled=True,    # PII masking (enabled by default)
    
    # Remote Server (optional)
    endpoint=None                # Server URL for client mode
)
```

**Provider-specific examples:**

```python path=null start=null
# OpenAI (default)
browser = FlyBrowser(llm_provider="openai", llm_model="gpt-5.2")

# Anthropic
browser = FlyBrowser(llm_provider="anthropic", llm_model="claude-sonnet-4-5-20250929")

# Google Gemini
browser = FlyBrowser(llm_provider="gemini", llm_model="gemini-2.0-flash")

# Ollama (local)
browser = FlyBrowser(llm_provider="ollama", llm_model="qwen3:8b")

# vLLM (high-throughput)
browser = FlyBrowser(
    llm_provider="vllm",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000"
)
```

### FlyBrowserClient Constructor

```python path=null start=null
from flybrowser import FlyBrowserClient

client = FlyBrowserClient(
    endpoint="http://localhost:8000",  # Server URL
    api_key="your-api-key",            # Authentication key
    timeout=30.0                       # Request timeout in seconds
)
```

## Server Options Reference

### Host and Port

| Setting | CLI | Environment | Config File | Default |
|---------|-----|-------------|-------------|---------|
| Host | `--host` | `FLYBROWSER_HOST` | `server.host` | `0.0.0.0` |
| Port | `--port` | `FLYBROWSER_PORT` | `server.port` | `8000` |
| Workers | `--workers` | `FLYBROWSER_WORKERS` | `server.workers` | `1` |

### Session Management

| Setting | CLI | Environment | Config File | Default |
|---------|-----|-------------|-------------|---------|
| Max Sessions | `--max-sessions` | `FLYBROWSER_MAX_SESSIONS` | `server.max_sessions` | `10` |
| Session Timeout | `--session-timeout` | `FLYBROWSER_SESSION_TIMEOUT` | `server.session_timeout` | `3600` |

### Data Storage

| Setting | CLI | Environment | Config File | Default |
|---------|-----|-------------|-------------|---------|
| Data Directory | `--data-dir` | `FLYBROWSER_DATA_DIR` | `server.data_dir` | `./data` |

### Logging

| Setting | CLI | Environment | Config File | Default |
|---------|-----|-------------|-------------|---------|
| Log Level | `--log-level` | `FLYBROWSER_LOG_LEVEL` | `logging.level` | `INFO` |
| Log File | `--log-file` | `FLYBROWSER_LOG_FILE` | `logging.file` | (stdout) |
| Log Format | N/A | `FLYBROWSER_LOG_FORMAT` | `logging.format` | `text` |

### Cluster Options

| Setting | CLI | Environment | Config File | Default |
|---------|-----|-------------|-------------|---------|
| Cluster Mode | `--cluster` | `FLYBROWSER_CLUSTER_ENABLED` | `cluster.enabled` | `false` |
| Node ID | `--node-id` | `FLYBROWSER_NODE_ID` | `cluster.node_id` | (auto) |
| Raft Port | `--raft-port` | `FLYBROWSER_RAFT_PORT` | `cluster.raft_port` | `4321` |
| Peers | `--peers` | `FLYBROWSER_CLUSTER_PEERS` | `cluster.peers` | (none) |
| API Host | N/A | `FLYBROWSER_API_HOST` | `cluster.api_host` | `0.0.0.0` |
| API Port | N/A | `FLYBROWSER_API_PORT` | `cluster.api_port` | `8000` |

## Browser Configuration

### Browser Types

| Type | Description | Use Case |
|------|-------------|----------|
| `chromium` | Chromium-based browser | Best compatibility, default choice |
| `firefox` | Firefox browser | Privacy-focused testing |
| `webkit` | WebKit browser | Safari compatibility testing |

### Timeout Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `timeout` | Default operation timeout | `30000` ms |
| Navigation timeout | Page load timeout | Uses `timeout` |
| Action timeout | Element interaction timeout | Uses `timeout` |
| Extraction timeout | Data extraction timeout | Uses `timeout` |

## Security Configuration

### API Authentication

Enable API key authentication:

```bash path=null start=null
export FLYBROWSER_API_KEY="your-secure-key"
flybrowser-serve
```

Or in configuration:

```yaml path=null start=null
auth:
  api_key: "your-secure-key"
```

### TLS Configuration

Enable TLS for encrypted connections:

```yaml path=null start=null
security:
  tls:
    enabled: true
    cert: "/etc/ssl/certs/flybrowser.crt"
    key: "/etc/ssl/private/flybrowser.key"
```

### PII Masking

Configure automatic PII masking:

```yaml path=null start=null
security:
  pii_masking: true
  pii_patterns:
    - email
    - phone
    - ssn
    - credit_card
```

## LLM Provider Configuration

FlyBrowser supports multiple LLM providers for powering browser automation. Each provider has unique strengths and use cases.

### Supported Providers Overview

| Provider | Provider Key | API Key Required | Default Port | Best For |
|----------|--------------|------------------|--------------|----------|
| OpenAI | `openai` | Yes | N/A | General-purpose, coding, complex reasoning |
| Anthropic | `anthropic` | Yes | N/A | Safety-focused tasks, long context |
| Google Gemini | `gemini` or `google` | Yes | N/A | Multimodal, fast responses, large context |
| Ollama | `ollama` | No | 11434 | Local inference, privacy, offline use |
| LM Studio | `lm_studio` | No | 1234 | Local development, model experimentation |
| LocalAI | `localai` | No | 8080 | Self-hosted production deployments |
| vLLM | `vllm` | No | 8000 | High-throughput production inference |

### OpenAI

OpenAI provides state-of-the-art language models with excellent performance for browser automation.

**Supported Models:**

| Model | Description | Context Window | Best For |
|-------|-------------|----------------|----------|
| `gpt-5.2` | Latest flagship (default) | 256K | Complex automation, coding |
| `gpt-5-mini` | Fast, cost-efficient | 128K | Standard automation |
| `gpt-5-nano` | Fastest, most affordable | 64K | Simple tasks, high volume |
| `gpt-4.1` | Smartest non-reasoning | 128K | Balanced performance |
| `gpt-4o` | Fast, intelligent | 128K | General purpose |

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: openai
  default_model: gpt-5.2  # Default if not specified
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      organization: "org-..."       # Optional: organization ID
      # base_url: "https://..."     # Optional: custom endpoint (Azure OpenAI)
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",  # Optional, uses default
    api_key="sk-proj-your-key-here"  # Or use OPENAI_API_KEY env var
)
```

### Anthropic

Anthropic's Claude models excel at careful, nuanced tasks with excellent safety characteristics.

**Supported Models:**

| Model | Description | Context Window | Best For |
|-------|-------------|----------------|----------|
| `claude-sonnet-4-5-20250929` | Smart model for agents (default) | 200K | Complex automation, coding |
| `claude-haiku-4-5-20251001` | Fastest model | 200K | Quick responses, high volume |
| `claude-opus-4-5-20251101` | Most intelligent | 200K | Premium tasks, maximum quality |
| `claude-3-5-sonnet-20241022` | Previous generation | 200K | Legacy compatibility |

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: anthropic
  default_model: claude-sonnet-4-5-20250929
  providers:
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929",
    api_key="sk-ant-your-key-here"  # Or use ANTHROPIC_API_KEY env var
)
```

### Google Gemini

Google's Gemini models offer excellent multimodal capabilities and massive context windows, ideal for vision-heavy browser automation.

**Supported Models:**

| Model | Description | Context Window | Best For |
|-------|-------------|----------------|----------|
| `gemini-2.0-flash` | Fast, versatile (default) | 1M | General automation, vision |
| `gemini-2.0-pro` | Most capable | 1M | Complex tasks, reasoning |
| `gemini-1.5-pro` | Previous generation pro | 1M | Legacy compatibility |
| `gemini-1.5-flash` | Previous generation fast | 1M | Quick responses |

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: gemini  # or "google" - both work
  default_model: gemini-2.0-flash
  providers:
    gemini:
      api_key: "${GOOGLE_API_KEY}"
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

# Using "gemini" provider key
browser = FlyBrowser(
    llm_provider="gemini",
    llm_model="gemini-2.0-flash",
    api_key="AIza-your-google-api-key"  # Or use GOOGLE_API_KEY env var
)

# Using "google" alias (same provider)
browser = FlyBrowser(
    llm_provider="google",
    llm_model="gemini-2.0-pro"
)
```

**Note:** You can use either `gemini` or `google` as the provider key—they are aliases for the same provider.

### Ollama (Local LLM)

Ollama enables running open-source LLMs locally without API costs or data leaving your machine. Ideal for privacy-sensitive applications, offline use, or cost optimization.

**Prerequisites:**

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama service: `ollama serve`
3. Pull a model: `ollama pull qwen3:8b`

**Recommended Models:**

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| `qwen3:8b` | 8B | Alibaba's latest (default) | Balanced performance |
| `qwen3:32b` | 32B | Extended thinking | Complex reasoning |
| `gemma3:12b` | 12B | Google's efficient model | Fast, accurate |
| `llama3.2:3b` | 3B | Meta's small model | Quick responses |
| `deepseek-r1:8b` | 8B | Reasoning model | Logical tasks |
| `phi4` | 14B | Microsoft's model | Coding, reasoning |
| `gpt-oss:20b` | 20B | Open-source GPT | General purpose |

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: ollama
  default_model: qwen3:8b
  providers:
    ollama:
      base_url: "http://localhost:11434"  # Default
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

# Local Ollama with default URL
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b"
)

# Remote Ollama server
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="gemma3:12b",
    base_url="http://192.168.1.100:11434"
)
```

### LM Studio (Local)

LM Studio provides a user-friendly interface for running local LLMs with an OpenAI-compatible API.

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: lm_studio
  default_model: local-model  # Model loaded in LM Studio
  providers:
    lm_studio:
      base_url: "http://localhost:1234"  # Default LM Studio port
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="lm_studio",
    llm_model="local-model",
    base_url="http://localhost:1234"
)
```

### LocalAI (Self-Hosted)

LocalAI is a self-hosted, OpenAI-compatible API for running local LLMs in production environments.

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: localai
  default_model: your-model-name
  providers:
    localai:
      base_url: "http://localhost:8080"  # Default LocalAI port
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="localai",
    llm_model="your-model-name",
    base_url="http://localhost:8080"
)
```

### vLLM (High-Throughput)

vLLM is optimized for high-throughput production LLM inference with efficient memory management.

**Configuration:**

```yaml path=null start=null
llm:
  default_provider: vllm
  default_model: meta-llama/Llama-2-7b-chat-hf
  providers:
    vllm:
      base_url: "http://localhost:8000"  # Default vLLM port
```

**Python SDK:**

```python path=null start=null
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="vllm",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000"
)
```

## Advanced LLM Configuration

FlyBrowser provides advanced configuration options for fine-tuning LLM behavior.

### Retry Configuration

Handle transient failures with automatic retries:

```yaml path=null start=null
retry:
  max_retries: 3              # Maximum retry attempts
  initial_delay: 1.0          # Initial delay in seconds
  max_delay: 60.0             # Maximum delay in seconds
  exponential_base: 2.0       # Exponential backoff base
  jitter: true                # Add randomness to delays
```

### Rate Limiting

Avoid API throttling with built-in rate limiting:

```yaml path=null start=null
rate_limit:
  requests_per_minute: 60     # Max requests per minute
  tokens_per_minute: 100000   # Max tokens per minute
  concurrent_requests: 10     # Max concurrent requests
```

### Response Caching

Improve performance with response caching:

```yaml path=null start=null
cache:
  enabled: true               # Enable response caching
  ttl_seconds: 3600           # Cache time-to-live (1 hour)
  max_size: 1000              # Maximum cached entries
  cache_key_prefix: "flybrowser:llm"  # Key prefix
```

### Cost Tracking

Track LLM usage and costs for budget management:

```yaml path=null start=null
cost_tracking:
  enabled: true               # Enable cost tracking
  track_tokens: true          # Track token usage
  track_requests: true        # Track request counts
  log_costs: true             # Log costs to file/console
```

## Resource Limits

### Memory Limits

```yaml path=null start=null
resources:
  memory_limit: 4096  # MB per session
  total_memory_limit: 16384  # MB total
```

### CPU Limits

```yaml path=null start=null
resources:
  cpu_limit: 1.0  # CPU cores per session
```

## Logging Configuration

### Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | Detailed debugging information |
| `INFO` | General operational information |
| `WARNING` | Warning messages |
| `ERROR` | Error messages only |

### Log Format

#### Text Format

```
2024-01-15 10:30:00 INFO [flybrowser.server] Server started on 0.0.0.0:8080
```

#### JSON Format

```json path=null start=null
{"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "logger": "flybrowser.server", "message": "Server started on 0.0.0.0:8080"}
```

### Log Rotation

Configure log rotation:

```yaml path=null start=null
logging:
  file: "/var/log/flybrowser/server.log"
  rotation:
    max_size: "100MB"
    max_files: 10
    compress: true
```

## Example Configurations

### Development

```yaml path=null start=null
server:
  host: "127.0.0.1"
  port: 8080
  max_sessions: 5

logging:
  level: "DEBUG"

browser:
  headless: false  # Show browser for debugging
  timeout: 60000   # Longer timeout for debugging
```

### Production (Standalone)

```yaml path=null start=null
server:
  host: "0.0.0.0"
  port: 8080
  max_sessions: 50
  data_dir: "/var/lib/flybrowser"

logging:
  level: "INFO"
  file: "/var/log/flybrowser/server.log"
  format: "json"

auth:
  api_key: "${FLYBROWSER_API_KEY}"

browser:
  headless: true
  timeout: 30000

security:
  pii_masking: true
```

### Production (Cluster)

```yaml path=null start=null
server:
  host: "0.0.0.0"
  port: 8080
  max_sessions: 100
  data_dir: "/var/lib/flybrowser"

cluster:
  enabled: true
  node_id: "${HOSTNAME}"
  raft_port: 5000
  peers:
    - "node1.internal:5000"
    - "node2.internal:5000"
    - "node3.internal:5000"
  election_timeout: 1500
  heartbeat_interval: 150

logging:
  level: "INFO"
  file: "/var/log/flybrowser/server.log"
  format: "json"

auth:
  api_key: "${FLYBROWSER_API_KEY}"
```

## Validation

FlyBrowser validates configuration on startup. Invalid configurations result in clear error messages:

```
Error: Invalid configuration
  - server.port: must be between 1 and 65535
  - cluster.node_id: required when cluster.enabled is true
```

Use the `--validate` flag to check configuration without starting:

```bash path=null start=null
flybrowser-serve --config /etc/flybrowser/config.yaml --validate
```
