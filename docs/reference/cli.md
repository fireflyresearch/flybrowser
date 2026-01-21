# CLI Reference

This document provides complete reference documentation for FlyBrowser command-line tools.

## Overview

FlyBrowser provides three CLI commands:

| Command | Description |
|---------|-------------|
| `flybrowser-serve` | Start the FlyBrowser HTTP server |
| `flybrowser-cluster` | Cluster management and monitoring |
| `flybrowser-admin` | Administrative operations |

## flybrowser-serve

Starts the FlyBrowser HTTP server in standalone or cluster mode.

### Synopsis

```bash path=null start=null
flybrowser-serve [OPTIONS]
```

### Options

#### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind |
| `--port` | `8000` | HTTP port number |
| `--workers` | `1` | Number of worker processes |
| `--max-sessions` | `10` | Maximum concurrent browser sessions |
| `--data-dir` | `./data` | Directory for persistent data |
| `--reload` | `false` | Enable auto-reload for development |

#### Cluster Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster` | `false` | Enable cluster mode |
| `--node-id` | (auto-generated) | Unique identifier for this node |
| `--raft-port` | `4321` | Port for Raft consensus protocol |
| `--peers` | (none) | Comma-separated list of peer addresses (`host:raft_port`) |

#### Logging Options

| Option | Default | Description |
|--------|---------|-------------|
| `--log-level` | `info` | Log level: `debug`, `info`, `warning`, `error` |
| `--log-file` | (none) | Path to log file (stdout if not specified) |

### Examples

#### Start in Standalone Mode

```bash path=null start=null
# Basic startup (starts on 0.0.0.0:8000)
flybrowser-serve

# Custom host and port
flybrowser-serve --host 0.0.0.0 --port 8000

# With session limit and workers
flybrowser-serve --workers 4 --max-sessions 50 --data-dir /var/lib/flybrowser
```

#### Start in Cluster Mode

```bash path=null start=null
# First node (bootstrap)
flybrowser-serve --cluster --node-id node1 --port 8001 --raft-port 4321

# Additional node
flybrowser-serve --cluster --node-id node2 --port 8002 --raft-port 4322 \
    --peers node1:4321

# Third node
flybrowser-serve --cluster --node-id node3 --port 8003 --raft-port 4323 \
    --peers node1:4321,node2:4322
```

#### Production Configuration

```bash path=null start=null
flybrowser-serve \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --max-sessions 100 \
    --data-dir /var/lib/flybrowser \
    --log-level info \
    --log-file /var/log/flybrowser/server.log
```

## flybrowser-cluster

Cluster management and monitoring tool.

### Synopsis

```bash path=null start=null
flybrowser-cluster <COMMAND> [OPTIONS]
```

### Commands

#### status

Display cluster status.

```bash path=null start=null
flybrowser-cluster status [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--json` | `false` | Output in JSON format |

**Example**:
```bash path=null start=null
flybrowser-cluster status --endpoint http://localhost:8001

# Output:
# Cluster Status
# ==============
# Leader: node1
# Term: 5
# Nodes: 3
#
# Node Details:
#   node1: leader (192.168.1.10:4321)
#   node2: follower (192.168.1.11:4322)
#   node3: follower (192.168.1.12:4323)
#
# Cluster health: HEALTHY
```

#### nodes

List cluster nodes.

```bash path=null start=null
flybrowser-cluster nodes [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--json` | `false` | Output in JSON format |

**Example**:
```bash path=null start=null
flybrowser-cluster nodes --endpoint http://localhost:8001

# Output:
# Cluster Nodes
# =============
# node1 (leader)
#   Address: 192.168.1.10:4321
#   HTTP: 192.168.1.10:8001
#   Status: healthy
#   Sessions: 5
```

#### sessions

List sessions across cluster.

```bash path=null start=null
flybrowser-cluster sessions [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--json` | `false` | Output in JSON format |

**Example**:
```bash path=null start=null
flybrowser-cluster sessions --endpoint http://localhost:8001

# Output:
# Cluster Sessions
# ================
# Session ID        Node     Status   Created
# sess_abc123       node1    active   2024-01-15 10:30
# sess_def456       node2    active   2024-01-15 11:00
```

## flybrowser-admin

Administrative operations for session and node management.

### Synopsis

```bash path=null start=null
flybrowser-admin <COMMAND> <SUBCOMMAND> [OPTIONS]
```

### Commands

#### sessions

Manage browser sessions.

##### sessions list

List all sessions.

```bash path=null start=null
flybrowser-admin sessions list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--json` | `false` | Output in JSON format |

**Example**:
```bash path=null start=null
flybrowser-admin sessions list --endpoint http://localhost:8000

# Output:
# Sessions
# ========
# Session ID        Status   Created              URL
# sess_abc123       active   2024-01-15 10:30     https://example.com
# sess_def456       idle     2024-01-15 11:00     https://google.com
```

##### sessions kill

Terminate a session.

```bash path=null start=null
flybrowser-admin sessions kill <SESSION_ID> [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `SESSION_ID` | Session identifier to terminate |

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--force` | `false` | Force termination without cleanup |

**Example**:
```bash path=null start=null
flybrowser-admin sessions kill sess_abc123 --endpoint http://localhost:8000
# Session sess_abc123 terminated
```

##### sessions kill-all

Terminate all sessions.

```bash path=null start=null
flybrowser-admin sessions kill-all [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--force` | `false` | Skip confirmation |

**Example**:
```bash path=null start=null
flybrowser-admin sessions kill-all --endpoint http://localhost:8000 --force
# Terminated 5 sessions
```

#### nodes

Manage cluster nodes.

##### nodes list

List cluster nodes.

```bash path=null start=null
flybrowser-admin nodes list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--json` | `false` | Output in JSON format |

**Example**:
```bash path=null start=null
flybrowser-admin nodes list --endpoint http://localhost:8001
```

##### nodes drain

Drain a node of all sessions before removal.

```bash path=null start=null
flybrowser-admin nodes drain <NODE_ID> [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `NODE_ID` | Node identifier to drain |

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--force` | `false` | Force drain without waiting |

**Example**:
```bash path=null start=null
flybrowser-admin nodes drain node3 --endpoint http://localhost:8001
# Draining node3...
# Moved 4 sessions to other nodes
# Node node3 drained successfully
```

#### cluster

Cluster-level administrative operations.

##### cluster rebalance

Rebalance sessions across cluster nodes.

```bash path=null start=null
flybrowser-cluster rebalance [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--target` | (none) | Target node for sessions |

##### cluster step-down

Force the current leader to step down and trigger a new election.

```bash path=null start=null
flybrowser-cluster step-down [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |

**Example**:
```bash path=null start=null
flybrowser-admin cluster step-down --endpoint http://localhost:8001
# Leader node1 stepped down
# New leader: node2
```

#### backup

Backup and restore operations.

##### backup create

Create a backup of the cluster state.

```bash path=null start=null
flybrowser-admin backup create [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000` | Server endpoint |
| `--output` | `flybrowser-backup.tar.gz` | Output file path |

**Example**:
```bash path=null start=null
flybrowser-admin backup create \
    --endpoint http://localhost:8001 \
    --output /backup/flybrowser-$(date +%Y%m%d).tar.gz
# Backup created: /backup/flybrowser-20240115.tar.gz
# Size: 15.2 MB
```

##### backup restore

Restore from a backup file.

```bash path=null start=null
flybrowser-admin backup restore [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input backup file path |
| `--data-dir` | `./data` | Target data directory |
| `--force` | `false` | Overwrite existing data |

**Example**:
```bash path=null start=null
flybrowser-admin backup restore \
    --input /backup/flybrowser-20240115.tar.gz \
    --data-dir /var/lib/flybrowser \
    --force
# Restored backup to /var/lib/flybrowser
```

## Environment Variables

CLI tools respect the following environment variables:

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `FLYBROWSER_HOST` | `--host` | Server host (default: 0.0.0.0) |
| `FLYBROWSER_PORT` | `--port` | Server port (default: 8000) |
| `FLYBROWSER_WORKERS` | `--workers` | Number of workers (default: 1) |
| `FLYBROWSER_API_KEY` | N/A | API key for authentication |
| `FLYBROWSER_LOG_LEVEL` | `--log-level` | Logging level (default: info) |
| `FLYBROWSER_DATA_DIR` | `--data-dir` | Data directory (default: ./data) |
| `FLYBROWSER_CLUSTER_ENABLED` | `--cluster` | Enable cluster mode |
| `FLYBROWSER_NODE_ID` | `--node-id` | Cluster node ID |
| `FLYBROWSER_RAFT_PORT` | `--raft-port` | Raft port (default: 4321) |
| `FLYBROWSER_CLUSTER_PEERS` | `--peers` | Cluster peers |

Command-line options take precedence over environment variables.

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Connection error |
| `4` | Authentication error |
| `5` | Resource not found |
| `6` | Timeout |

## Output Formats

Most commands support multiple output formats:

### Table Format (default)

Human-readable tabular output:

```bash path=null start=null
flybrowser-admin sessions list --format table
```

### JSON Format

Machine-readable JSON output:

```bash path=null start=null
flybrowser-admin sessions list --format json
```

```json path=null start=null
{
    "sessions": [
        {
            "session_id": "sess_abc123",
            "status": "active",
            "created_at": "2024-01-15T10:30:00Z"
        }
    ],
    "total": 1
}
```

## Shell Completion

Generate shell completion scripts:

### Bash

```bash path=null start=null
flybrowser-serve --completion bash > /etc/bash_completion.d/flybrowser
```

### Zsh

```bash path=null start=null
flybrowser-serve --completion zsh > ~/.zsh/completions/_flybrowser
```

### Fish

```bash path=null start=null
flybrowser-serve --completion fish > ~/.config/fish/completions/flybrowser.fish
```
