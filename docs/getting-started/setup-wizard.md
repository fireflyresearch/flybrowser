# Setup Wizard

FlyBrowser provides an enhanced setup wizard with targeted subcommands for configuring individual aspects of your installation. You can run the full interactive wizard or focus on a specific area.

## Quick Reference

```bash
# Full interactive wizard (walks through everything)
flybrowser setup

# Targeted subcommands
flybrowser setup quick           # 30-second start
flybrowser setup llm             # LLM provider configuration
flybrowser setup server          # Server mode configuration
flybrowser setup observability   # Tracing and metrics
flybrowser setup security        # RBAC and authentication
flybrowser setup verify          # Verify installation
```

## Quick Start (`flybrowser setup quick`)

The fastest path from zero to a working FlyBrowser installation. This subcommand auto-detects your LLM provider from environment variables, installs the Chromium browser, and verifies the setup.

```bash
flybrowser setup quick
```

**What happens:**

1. **Auto-detect LLM provider** -- Checks for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and `DASHSCOPE_API_KEY` in your environment. If found, the corresponding provider is selected automatically.
2. **Interactive fallback** -- If no API key is found in the environment, you are prompted to choose a provider and enter a key.
3. **Install Chromium** -- Runs `playwright install chromium`.
4. **Verify** -- Confirms that FlyBrowser, Playwright, and the browser binary are all working.

**Example session:**

```
  _____.__         ___.
_/ ____\  | ___.__.\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/

[QUICK START] 30-second FlyBrowser setup

[OK] Detected openai from $OPENAI_API_KEY

[INSTALL] Installing Chromium browser...
[OK] Installed chromium

[CHECK] Verifying installation...
[OK] FlyBrowser 26.02.06 installed
[OK] Playwright installed
[OK] Chromium browser available

[OK] Quick setup complete!
```

## Full Wizard (`flybrowser setup`)

The full interactive wizard walks through every configuration area step by step:

1. **Deployment mode** -- Standalone or cluster
2. **Service configuration** -- Host, port, environment, log level
3. **Browser pool** -- Min/max instances, max sessions, headless mode
4. **Cluster settings** -- Node role, coordinator host/port (cluster mode only)
5. **LLM providers** -- Default provider, API keys, model selection
6. **Save configuration** -- Writes a `.env` file with all settings

```bash
flybrowser setup
```

The wizard generates a `.env` file you can place in your project root or in `~/.flybrowser/`. FlyBrowser reads this file automatically on startup.

## LLM Provider Configuration (`flybrowser setup llm`)

Configure which LLM provider and model FlyBrowser uses for AI-powered automation.

```bash
flybrowser setup llm
```

**Supported providers:**

| Provider | Models | API Key Env Var |
|----------|--------|-----------------|
| OpenAI | gpt-5.2, gpt-5-mini, gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022 | `ANTHROPIC_API_KEY` |
| Google Gemini | gemini-2.0-flash, gemini-1.5-pro | `GOOGLE_API_KEY` |
| Qwen | qwen-plus, qwen-turbo, qwen-max, qwen-vl-max | `DASHSCOPE_API_KEY` |
| Ollama | Any locally installed model | N/A (local) |

**Example session:**

```
============================================================
LLM PROVIDER CONFIGURATION
============================================================

Select LLM provider:
  > 1. OpenAI (GPT-5.2, GPT-4o)
    2. Anthropic (Claude)
    3. Google (Gemini)
    4. Qwen (Alibaba Cloud)
    5. Ollama (Local)
Enter choice [1-5] (default: 1): 1

OpenAI API Key: sk-...

Select model:
  > 1. gpt-5.2
    2. gpt-5-mini
    3. gpt-4o
    4. gpt-4o-mini
Enter choice [1-4] (default: 1): 3

Run connectivity test? [y/N]: y
[TEST] Testing openai connectivity...

[OK] LLM provider configured: openai
```

The optional connectivity test sends a minimal prompt to confirm the API key and model are reachable.

## Server Configuration (`flybrowser setup server`)

Configure FlyBrowser to run as an API server, including network binding, worker count, and optional TLS.

```bash
flybrowser setup server
```

**Configuration options:**

| Setting | Default | Description |
|---------|---------|-------------|
| Host | `0.0.0.0` | Network interface to bind |
| Port | `8000` | TCP port |
| Workers | `4` | Number of uvicorn worker processes |
| TLS | Disabled | Enable HTTPS with certificate and key paths |

**Example session:**

```
============================================================
SERVER CONFIGURATION
============================================================

Host address [0.0.0.0]: 0.0.0.0
Port [8000]: 8000
Number of workers [4]: 4
Enable TLS? [y/N]: y
Path to TLS certificate: /etc/ssl/certs/flybrowser.pem
Path to TLS private key: /etc/ssl/private/flybrowser.key

[OK] Server configured: 0.0.0.0:8000 (4 workers, TLS=on)
```

After configuration, start the server with:

```bash
flybrowser serve --host 0.0.0.0 --port 8000 --workers 4
```

## Observability Configuration (`flybrowser setup observability`)

Configure OpenTelemetry tracing, Prometheus metrics, and log levels for production monitoring.

```bash
flybrowser setup observability
```

**Configuration options:**

| Setting | Default | Description |
|---------|---------|-------------|
| OTLP endpoint | `http://localhost:4317` | gRPC endpoint for the OpenTelemetry collector |
| Prometheus metrics | Enabled | Expose a `/metrics` endpoint for Prometheus scraping |
| Prometheus port | `9090` | Port for the metrics endpoint |
| Log level | `INFO` | Minimum log severity |

**Example session:**

```
============================================================
OBSERVABILITY CONFIGURATION
============================================================

OTLP collector endpoint [http://localhost:4317]: http://otel-collector:4317
Enable Prometheus metrics? [Y/n]: y
Prometheus metrics port [9090]: 9090

Log level:
  > 1. DEBUG
    2. INFO
    3. WARNING
    4. ERROR
Enter choice [1-4] (default: 2): 2

[OK] Observability configured: OTLP=http://otel-collector:4317, log_level=INFO
```

For more details on what is traced and measured, see [Observability](../features/observability.md).

## Security Configuration (`flybrowser setup security`)

Configure role-based access control (RBAC) and JWT authentication for the API server.

```bash
flybrowser setup security
```

**Configuration options:**

| Setting | Default | Description |
|---------|---------|-------------|
| RBAC enabled | Yes | Enable role-based access control |
| JWT secret | Auto-generated | Secret key for signing JWT tokens |
| Admin token | Auto-generated | Initial admin access token |

**Example session:**

```
============================================================
SECURITY CONFIGURATION
============================================================

Enable RBAC (role-based access control)? [Y/n]: y
JWT secret (leave empty to auto-generate):
[OK] Generated JWT secret: aBcDeFgH...
[OK] Generated admin token: xYzWvUtS...
[IMPORTANT] Save the admin token securely -- it will not be shown again.

[OK] Security configured: RBAC=enabled
```

The generated admin token grants full access to the API. Use it to create operator and viewer tokens for other users. See [Security Architecture](../architecture/security.md) for the full permission model.

## Verify Installation (`flybrowser setup verify`)

Run a verification check to confirm everything is properly installed.

```bash
flybrowser setup verify
```

**Checks performed:**

1. FlyBrowser Python package can be imported
2. Playwright library is installed
3. Chromium browser binary is available

```
[CHECK] Verifying installation...
[OK] FlyBrowser 26.02.06 installed
[OK] Playwright installed
[OK] Chromium browser available
```

## Environment File

All wizard subcommands that produce configuration write a `.env` file. A typical generated file looks like this:

```bash
# FlyBrowser Configuration
# Generated by flybrowser-setup

# Service Settings
FLYBROWSER_HOST=0.0.0.0
FLYBROWSER_PORT=8000
FLYBROWSER_ENV=production
FLYBROWSER_LOG_LEVEL=INFO

# Session Settings
FLYBROWSER_MAX_SESSIONS=100
FLYBROWSER_SESSION_TIMEOUT=3600

# Browser Pool Settings
FLYBROWSER_POOL__MIN_SIZE=1
FLYBROWSER_POOL__MAX_SIZE=10
FLYBROWSER_POOL__HEADLESS=true

# Deployment Mode
FLYBROWSER_DEPLOYMENT_MODE=standalone

# LLM Provider Configuration
FLYBROWSER_LLM_PROVIDER=openai
FLYBROWSER_LLM_MODEL=gpt-4o

# LLM Provider API Keys
OPENAI_API_KEY=sk-...
```

FlyBrowser searches for `.env` files in this order:

1. `.env` in the current working directory
2. `~/.flybrowser/config`
3. `~/.config/flybrowser/config`

## See Also

- [Installation](installation.md) -- Package installation instructions
- [Quickstart](quickstart.md) -- Your first automation script
- [CLI Reference](../reference/cli.md) -- Full CLI command reference
- [Security Architecture](../architecture/security.md) -- RBAC and authentication details
- [Observability](../features/observability.md) -- Tracing and metrics
