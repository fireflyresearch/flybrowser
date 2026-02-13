# Session Management CLI

FlyBrowser provides a complete set of CLI commands for managing browser sessions. Sessions are long-lived browser instances that you can create, inspect, send commands to, and close.

## Quick Reference

```bash
# Create a new session
flybrowser session create [--provider openai] [--model gpt-4o] [--headless]

# List active sessions
flybrowser session list [--format table|json] [--status active|all]

# Get session details
flybrowser session info <session-id>

# Connect to an existing session
flybrowser session connect <session-id>

# Run a command on a session
flybrowser session exec <session-id> <command>

# Close a session
flybrowser session close <session-id>

# Close all sessions
flybrowser session close-all [--force]
```

## Embedded vs Server Mode

Session commands work in two modes:

| Mode | When | How |
|------|------|-----|
| **Embedded** | No `--endpoint` specified | Launches a local Playwright browser in-process |
| **Server** | `--endpoint` specified or `FLYBROWSER_ENDPOINT` set | Sends requests to a running FlyBrowser API server |

Most commands accept an `--endpoint` / `-e` flag. If you set the `FLYBROWSER_ENDPOINT` environment variable, all commands default to server mode:

```bash
export FLYBROWSER_ENDPOINT=http://localhost:8000
```

## Commands

### session create

Create a new browser session.

```bash
flybrowser session create [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | openai | LLM provider (openai, anthropic, gemini, ollama) |
| `--model` | `-m` | (provider default) | LLM model to use |
| `--headless` | | True | Run browser in headless mode |
| `--no-headless` | | | Run browser with visible UI |
| `--name` | `-n` | (auto) | Human-readable session name |
| `--api-key` | | (env var) | LLM API key |
| `--endpoint` | `-e` | (env var) | FlyBrowser server endpoint |

**Embedded mode example:**

```bash
$ flybrowser session create --provider anthropic --model claude-sonnet-4-5-20250929 --no-headless

Session Created
  Session ID: embedded-a1b2c3d4e5f6
  Provider:   anthropic
  Model:      claude-sonnet-4-5-20250929
  Mode:       embedded
  Status:     active
```

**Server mode example:**

```bash
$ flybrowser session create -e http://localhost:8000 -p openai -m gpt-4o

Session Created
  Session ID: sess_abc123
  Provider:   openai
  Model:      gpt-4o
  Mode:       server
  Status:     active
```

### session list

List active sessions on a FlyBrowser server.

```bash
flybrowser session list [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | table | Output format: `table` or `json` |
| `--status` | `-s` | active | Filter: `active` or `all` |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Table output:**

```bash
$ flybrowser session list

Sessions (3)
Session ID           | Status | Provider | Model  | Created
------------------------------------------------------------------
sess_abc123          | active | openai   | gpt-4o | 2026-02-13T10:30:00Z
sess_def456          | active | openai   | gpt-4o | 2026-02-13T10:35:00Z
sess_ghi789          | active | anthropic|        | 2026-02-13T10:40:00Z
```

**JSON output:**

```bash
$ flybrowser session list --format json

{
  "sessions": [
    {
      "session_id": "sess_abc123",
      "status": "active",
      "provider": "openai",
      "model": "gpt-4o",
      "created_at": "2026-02-13T10:30:00Z"
    }
  ],
  "total": 1
}
```

### session info

Get detailed information about a specific session.

```bash
flybrowser session info <session-id> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Example:**

```bash
$ flybrowser session info sess_abc123

Session: sess_abc123
  Status:   active
  Provider: openai
  Model:    gpt-4o
  Created:  2026-02-13T10:30:00Z
  Node:     node-1
```

### session connect

Display connection info for an existing session. This prints the session ID and shows how to run commands against it.

```bash
flybrowser session connect <session-id> [OPTIONS]
```

**Example:**

```bash
$ flybrowser session connect sess_abc123

Connecting to session sess_abc123...
Endpoint: http://localhost:8000

Use 'flybrowser session exec' to run commands on this session.
  flybrowser session exec sess_abc123 "navigate to https://example.com"
```

### session exec

Run a natural language command on an existing session.

```bash
flybrowser session exec <session-id> <command> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Examples:**

```bash
# Navigate to a page
$ flybrowser session exec sess_abc123 "navigate to https://news.ycombinator.com"

# Extract data
$ flybrowser session exec sess_abc123 "extract the top 5 story titles"

# Perform an action
$ flybrowser session exec sess_abc123 "click on the first story link"
```

The result is printed as JSON:

```json
{
  "success": true,
  "result": "Navigated to https://news.ycombinator.com",
  "task": "navigate to https://news.ycombinator.com"
}
```

### session close

Close a specific session and release its resources.

```bash
flybrowser session close <session-id> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Example:**

```bash
$ flybrowser session close sess_abc123
Session sess_abc123 closed.
```

### session close-all

Close all active sessions.

```bash
flybrowser session close-all [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--force` | | False | Skip confirmation |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Example:**

```bash
$ flybrowser session close-all
Closed 3 session(s).
```

## Common Workflows

### Create, use, and close a session

```bash
# Create a session
flybrowser session create -e http://localhost:8000 -p openai
# Output: Session ID: sess_abc123

# Navigate
flybrowser session exec sess_abc123 "navigate to https://news.ycombinator.com"

# Extract data
flybrowser session exec sess_abc123 "extract the title of the top story"

# Done -- close it
flybrowser session close sess_abc123
```

### Scripted session management

```bash
#!/bin/bash
set -e

export FLYBROWSER_ENDPOINT=http://localhost:8000

# Create session and capture ID (using JSON output for parsing)
SESSION_ID=$(flybrowser session create --provider openai | grep "Session ID" | awk '{print $NF}')

# Run commands
flybrowser session exec "$SESSION_ID" "navigate to https://example.com"
flybrowser session exec "$SESSION_ID" "extract the main heading"

# Clean up
flybrowser session close "$SESSION_ID"
```

### Monitor active sessions

```bash
# Watch sessions in real time
watch -n 5 flybrowser session list --format table
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FLYBROWSER_ENDPOINT` | Default server endpoint for all session commands |
| `FLYBROWSER_LLM_PROVIDER` | Default LLM provider for `session create` |
| `FLYBROWSER_LLM_MODEL` | Default LLM model for `session create` |
| `OPENAI_API_KEY` | OpenAI API key (auto-detected) |
| `ANTHROPIC_API_KEY` | Anthropic API key (auto-detected) |

## See Also

- [Direct Commands](direct-commands.md) -- One-shot commands that auto-create ephemeral sessions
- [Pipelines](pipelines.md) -- Multi-step workflow execution
- [CLI Reference](../reference/cli.md) -- Full CLI command reference
- [REST API Reference](../reference/rest-api.md) -- HTTP API for session management
