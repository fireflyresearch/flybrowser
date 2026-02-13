# CLI Reference

FlyBrowser provides a comprehensive command-line interface for installation, configuration, and operation.

## Quick Reference

```bash
# Interactive mode (default)
flybrowser                    # Launch REPL

# Core commands
flybrowser repl               # Launch interactive REPL
flybrowser serve              # Start API service
flybrowser doctor             # Diagnose installation
flybrowser version            # Show version

# Setup and management
flybrowser setup              # Installation wizard
flybrowser cluster            # Cluster management
flybrowser admin              # Administrative commands
flybrowser uninstall          # Uninstall FlyBrowser

# Media management
flybrowser stream             # Manage live streams
flybrowser recordings         # Manage recordings
```

## Global Options

Options available for all commands:

| Option | Description |
|--------|-------------|
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO) |
| `--human-readable` | Use human-readable log format instead of JSON |
| `--help` | Show help message |

## Commands

### flybrowser (default)

When run without arguments, launches the interactive REPL.

```bash
flybrowser
```

### flybrowser repl

Launch the interactive REPL (Read-Eval-Print Loop) for browser automation.

```bash
flybrowser repl [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | openai | LLM provider (openai, anthropic, ollama, gemini) |
| `--model` | `-m` | (provider default) | LLM model to use |
| `--headless` | | True | Run browser in headless mode |
| `--no-headless` | | | Run browser with visible UI |
| `--api-key` | | (env var) | LLM API key |
| `--verbosity` | `-v` | normal | Log verbosity: silent, minimal, normal, verbose, debug |

**Examples:**

```bash
# Default (OpenAI, headless)
flybrowser repl

# Use Anthropic Claude
flybrowser repl -p anthropic -m claude-sonnet-4-5-20250929

# Visible browser
flybrowser repl --no-headless

# Local Ollama
flybrowser repl -p ollama -m qwen3:8b

# Debug mode
flybrowser repl -v debug
```

### flybrowser serve

Start the FlyBrowser REST API service.

```bash
flybrowser serve [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 8000 | Port to bind |
| `--workers` | 1 | Number of worker processes |
| `--reload` | False | Auto-reload on code changes |

**Examples:**

```bash
# Default settings
flybrowser serve

# Custom port
flybrowser serve --port 9000

# Development mode with auto-reload
flybrowser serve --reload

# Production with multiple workers
flybrowser serve --workers 4
```

### flybrowser doctor

Run installation diagnostics to verify FlyBrowser is properly set up.

```bash
flybrowser doctor
```

**Checks performed:**

1. Python version (requires 3.9+)
2. FlyBrowser installation
3. Playwright installation
4. Browser availability (Chromium)
5. LLM provider status (dynamic discovery)
6. Configuration files

**Example output:**

```
╔═══════════════════════════════════════════════════════════════╗
║  FlyBrowser v1.26.1                                           ║
╚═══════════════════════════════════════════════════════════════╝

System Requirements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OK] Python 3.11.5
[OK] FlyBrowser 1.26.1 installed
[OK] Playwright installed
[OK] Chromium browser available

LLM Provider Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OK]   OpenAI: API key configured
[OK]   Anthropic: API key configured
[INFO] Ollama: Available at localhost:11434
[WARN] Google Gemini: No API key configured

Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SUCCESS] All checks passed!
```

### flybrowser version

Show version information.

```bash
flybrowser version [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--json` | `-j` | Output in JSON format |

**Examples:**

```bash
# Simple output
flybrowser version
# Output: FlyBrowser 1.26.1

# JSON output
flybrowser version --json
# Output:
# {
#   "flybrowser": "1.26.1",
#   "python": "3.11.5",
#   "platform": "Darwin",
#   "architecture": "arm64"
# }
```

### flybrowser setup

Installation and configuration wizard with targeted subcommands.

```bash
flybrowser setup [COMMAND]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| (none) | Full interactive setup wizard |
| `quick` | 30-second quick start (auto-detect LLM, install browser, verify) |
| `llm` | LLM provider configuration (provider, model, API key) |
| `server` | Server mode configuration (host, port, workers, TLS) |
| `observability` | Observability setup (OTLP endpoint, Prometheus, log level) |
| `security` | Security setup (RBAC enable, JWT secret, admin token) |
| `verify` | Verify FlyBrowser installation |
| `configure` | Interactive configuration wizard (legacy) |
| `install` | Install dependencies |
| `browsers` | Install Playwright browsers |
| `jupyter` | Manage Jupyter kernel (install, uninstall, status, fix) |

**Examples:**

```bash
# Full interactive wizard
flybrowser setup

# 30-second quick start
flybrowser setup quick

# Configure just the LLM provider
flybrowser setup llm

# Configure server settings
flybrowser setup server

# Set up observability (tracing + metrics)
flybrowser setup observability

# Configure RBAC security
flybrowser setup security

# Verify everything works
flybrowser setup verify

# Install browsers
flybrowser setup browsers install
```

See [Setup Wizard Guide](../getting-started/setup-wizard.md) for detailed walkthroughs of each subcommand.

### flybrowser cluster

Cluster management commands.

```bash
flybrowser cluster [COMMAND]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `status` | Show cluster status |
| `join` | Join a cluster |
| `leave` | Leave the cluster |
| `nodes` | List cluster nodes |

**Examples:**

```bash
# Check cluster status
flybrowser cluster status

# Join a cluster
flybrowser cluster join --coordinator http://coordinator:8000

# List nodes
flybrowser cluster nodes
```

### flybrowser admin

Administrative commands.

```bash
flybrowser admin [COMMAND]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `sessions` | Manage sessions |
| `cache` | Manage cache |
| `logs` | View logs |

### flybrowser stream

Manage live streams.

```bash
flybrowser stream [COMMAND]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `start` | Start a stream |
| `stop` | Stop a stream |
| `status` | Get stream status |
| `url` | Get stream URL |
| `play` | Play stream in local player |
| `web` | Open stream in browser |

**Examples:**

```bash
# Start HLS stream
flybrowser stream start <session_id> --protocol hls --quality high

# Stop stream
flybrowser stream stop <session_id>

# Get status
flybrowser stream status <session_id>

# Open in browser
flybrowser stream web <session_id>

# Play with VLC
flybrowser stream play <session_id> --player vlc
```

**Stream start options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--protocol` | hls | Streaming protocol: hls, dash, rtmp |
| `--quality` | medium | Quality: low_bandwidth, medium, high |
| `--codec` | h264 | Video codec: h264, h265, vp9 |
| `--endpoint` | http://localhost:8000 | API endpoint |

### flybrowser recordings

Manage session recordings.

```bash
flybrowser recordings [COMMAND]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `list` | List recordings |
| `download` | Download a recording |
| `delete` | Delete a recording |
| `clean` | Clean old recordings |

**Examples:**

```bash
# List all recordings
flybrowser recordings list

# List recordings for a session
flybrowser recordings list --session-id abc123

# Download a recording
flybrowser recordings download rec_123 -o my_recording.mp4

# Delete a recording
flybrowser recordings delete rec_123

# Clean recordings older than 30 days
flybrowser recordings clean --older-than 30d
```

### flybrowser session

Manage browser sessions (create, list, inspect, execute commands, close).

```bash
flybrowser session <command> [OPTIONS]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `create` | Create a new browser session |
| `list` | List active sessions |
| `info <id>` | Get session details |
| `connect <id>` | Show connection info for a session |
| `exec <id> <cmd>` | Run a command on a session |
| `close <id>` | Close a session |
| `close-all` | Close all sessions |

**Examples:**

```bash
# Create a session
flybrowser session create --provider openai --model gpt-4o

# List sessions
flybrowser session list --format table

# Run a command on a session
flybrowser session exec sess_abc123 "extract the page title"

# Close all sessions
flybrowser session close-all
```

See [Session Management CLI](../cli/session-management.md) for complete documentation.

### flybrowser goto

Navigate to a URL (direct command, auto-creates ephemeral session).

```bash
flybrowser goto <url> [--session <id>] [--wait-for <selector>]
```

### flybrowser extract

Extract data from the current page using natural language.

```bash
flybrowser extract <query> [--session <id>] [--schema <file>] [--format json|csv|table]
```

### flybrowser act

Perform a browser action described in natural language.

```bash
flybrowser act <instruction> [--session <id>]
```

### flybrowser screenshot

Capture a screenshot of the current page.

```bash
flybrowser screenshot [--session <id>] [--output <file>] [--full-page]
```

### flybrowser agent

Run an autonomous agent task.

```bash
flybrowser agent <task> [--session <id>] [--max-iterations 50] [--stream]
```

See [Direct Commands](../cli/direct-commands.md) for complete documentation of all direct commands.

### flybrowser run

Run a multi-step browser workflow from a YAML file or inline commands.

```bash
flybrowser run <workflow.yaml>
flybrowser run --inline "goto https://example.com && extract 'get the title'"
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `workflow` | | Path to a YAML workflow file |
| `--inline` | `-i` | Inline commands separated by `&&` |

See [Pipelines](../cli/pipelines.md) for complete documentation.

### flybrowser uninstall

Uninstall FlyBrowser.

```bash
flybrowser uninstall [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--keep-config` | Keep configuration files |
| `--keep-cache` | Keep cache files |
| `--yes` | Skip confirmation |

## Environment Variables

The CLI respects these environment variables:

| Variable | Description |
|----------|-------------|
| `FLYBROWSER_LOG_LEVEL` | Default log level |
| `FLYBROWSER_LOG_FORMAT` | Log format (json or human) |
| `FLYBROWSER_LLM_PROVIDER` | Default LLM provider |
| `FLYBROWSER_LLM_MODEL` | Default LLM model |
| `FLYBROWSER_LOG_VERBOSITY` | Default verbosity |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google Gemini API key |

## Configuration Files

The CLI looks for configuration in these locations:

1. `.env` in current directory
2. `~/.flybrowser/config`
3. `~/.config/flybrowser/config`

Run `flybrowser setup configure` to create a configuration file.

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error or check failed |
| 130 | Interrupted (Ctrl+C) |

## Interactive REPL Commands

When using `flybrowser repl`, these commands are available:

| Command | Description |
|---------|-------------|
| `goto <url>` | Navigate to URL |
| `extract <query>` | Extract data |
| `act <instruction>` | Perform action |
| `screenshot` | Take screenshot |
| `help` | Show available commands |
| `exit` or `quit` | Exit REPL |

## See Also

- [SDK Reference](sdk.md) - Python SDK documentation
- [REST API Reference](rest-api.md) - HTTP API documentation
- [Configuration](configuration.md) - Configuration options
- [Session Management CLI](../cli/session-management.md) - Detailed session commands
- [Direct Commands](../cli/direct-commands.md) - One-shot SDK-like commands
- [Pipelines](../cli/pipelines.md) - YAML workflow execution
- [Setup Wizard](../getting-started/setup-wizard.md) - Setup subcommand walkthroughs
