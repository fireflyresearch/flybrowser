# REST API Reference

This document provides complete reference documentation for the FlyBrowser REST API.

## Base URL

```
http://{host}:{port}
```

Default: `http://localhost:8000`

## Authentication

When API key authentication is enabled, include the key in the `Authorization` header:

```
Authorization: Bearer {api_key}
```

Or as a query parameter:

```
?api_key={api_key}
```

## Endpoints

### Health Check

#### GET /health

Returns the server health status.

**Request**:
```bash path=null start=null
curl http://localhost:8000/health
```

**Response** (200 OK):
```json path=null start=null
{
    "status": "healthy",
    "version": "1.0.0"
}
```

**Cluster Response** (200 OK):
```json path=null start=null
{
    "status": "healthy",
    "version": "1.0.0",
    "cluster": {
        "node_id": "node1",
        "role": "leader",
        "term": 5
    }
}
```

### Session Management

#### POST /sessions

Creates a new browser session.

**Request Body** (`SessionCreateRequest`):
```json path=null start=null
{
    "llm_provider": "openai",
    "llm_model": "gpt-5.2",
    "headless": true,
    "browser_type": "chromium"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `llm_provider` | string | No | `"openai"` | LLM provider (openai, anthropic, gemini, ollama, etc.) |
| `llm_model` | string | No | (provider default) | Model name |
| `api_key` | string | No | null | LLM API key (or use env vars) |
| `base_url` | string | No | null | Custom endpoint for local providers |
| `headless` | boolean | No | `true` | Headless mode |
| `browser_type` | string | No | `"chromium"` | Browser type |

**Response** (201 Created):
```json path=null start=null
{
    "session_id": "sess_abc123def456",
    "status": "created"
}
```

**Errors**:
- `400 Bad Request`: Invalid parameters
- `429 Too Many Requests`: Maximum sessions reached

#### GET /sessions

Lists all active sessions.

**Request**:
```bash path=null start=null
curl http://localhost:8000/sessions
```

**Response** (200 OK):
```json path=null start=null
{
    "sessions": [
        {
            "session_id": "sess_abc123",
            "status": "active",
            "created_at": "2024-01-15T10:30:00Z"
        },
        {
            "session_id": "sess_def456",
            "status": "active",
            "created_at": "2024-01-15T11:00:00Z"
        }
    ],
    "total": 2
}
```

#### GET /sessions/{session_id}

Gets details for a specific session.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Response** (200 OK):
```json path=null start=null
{
    "session_id": "sess_abc123",
    "status": "active",
    "created_at": "2024-01-15T10:30:00Z",
    "browser_type": "chromium",
    "current_url": "https://example.com"
}
```

**Errors**:
- `404 Not Found`: Session not found

#### DELETE /sessions/{session_id}

Closes and deletes a session.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Response** (200 OK):
```json path=null start=null
{
    "status": "closed",
    "session_id": "sess_abc123"
}
```

**Errors**:
- `404 Not Found`: Session not found

### Navigation

#### POST /sessions/{session_id}/navigate

Navigates to a URL.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Request Body** (`NavigateRequest`):
```json path=null start=null
{
    "url": "https://example.com"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | URL to navigate to |

**Response** (200 OK):
```json path=null start=null
{
    "status": "success",
    "url": "https://example.com"
}
```

**Errors**:
- `400 Bad Request`: Invalid URL
- `404 Not Found`: Session not found
- `408 Request Timeout`: Navigation timeout

### Data Extraction

#### POST /sessions/{session_id}/extract

Extracts data from the current page.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Request Body** (`ExtractRequest`):
```json path=null start=null
{
    "query": "Extract the main heading and first paragraph"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language extraction query |

**Response** (200 OK):
```json path=null start=null
{
    "result": "Heading: Example Domain\nParagraph: This domain is for use in illustrative examples..."
}
```

**Errors**:
- `400 Bad Request`: Missing query
- `404 Not Found`: Session not found
- `408 Request Timeout`: Extraction timeout

### Actions

#### POST /sessions/{session_id}/action

Performs an action on the page.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Request Body** (`ActionRequest`):
```json path=null start=null
{
    "command": "Click the Sign In button"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | Yes | Natural language action command |

**Response** (200 OK):
```json path=null start=null
{
    "status": "success",
    "action": "click",
    "element": "Sign In button"
}
```

**Errors**:
- `400 Bad Request`: Missing command
- `404 Not Found`: Session not found
- `408 Request Timeout`: Action timeout

### Screenshots

#### GET /sessions/{session_id}/screenshot

Captures a screenshot of the current page.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Response** (200 OK):
- Content-Type: `image/png`
- Body: PNG image binary data

**Errors**:
- `404 Not Found`: Session not found

### Workflows

#### POST /sessions/{session_id}/workflow

Executes a workflow.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Request Body** (`WorkflowRequest`):
```json path=null start=null
{
    "workflow": {
        "name": "login_workflow",
        "steps": [
            {
                "action": "goto",
                "url": "https://example.com/login"
            },
            {
                "action": "act",
                "command": "Enter 'user@example.com' in email field"
            },
            {
                "action": "act",
                "command": "Enter 'password' in password field"
            },
            {
                "action": "act",
                "command": "Click Login button"
            },
            {
                "action": "extract",
                "query": "Confirm login success"
            }
        ]
    }
}
```

**Step Types**:

| Action | Required Fields | Description |
|--------|-----------------|-------------|
| `goto` | `url` | Navigate to URL |
| `act` | `command` | Perform action |
| `extract` | `query` | Extract data |
| `screenshot` | (none) | Capture screenshot |

**Response** (200 OK):
```json path=null start=null
{
    "status": "completed",
    "workflow": "login_workflow",
    "results": [
        {"step": 0, "action": "goto", "status": "success"},
        {"step": 1, "action": "act", "status": "success"},
        {"step": 2, "action": "act", "status": "success"},
        {"step": 3, "action": "act", "status": "success"},
        {"step": 4, "action": "extract", "status": "success", "result": "Login successful"}
    ]
}
```

**Errors**:
- `400 Bad Request`: Invalid workflow definition
- `404 Not Found`: Session not found

### Monitoring

#### GET /sessions/{session_id}/monitor

Gets the current session status.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session identifier |

**Response** (200 OK):
```json path=null start=null
{
    "url": "https://example.com/page",
    "title": "Page Title",
    "state": "ready"
}
```

**States**:
- `ready`: Browser is ready for commands
- `loading`: Page is loading
- `error`: An error occurred

### Cluster Operations

#### GET /cluster/status

Gets the cluster status (cluster mode only).

**Response** (200 OK):
```json path=null start=null
{
    "leader": "node1",
    "term": 5,
    "nodes": [
        {
            "node_id": "node1",
            "role": "leader",
            "address": "192.168.1.10:4321",
            "status": "healthy"
        },
        {
            "node_id": "node2",
            "role": "follower",
            "address": "192.168.1.11:4322",
            "status": "healthy"
        },
        {
            "node_id": "node3",
            "role": "follower",
            "address": "192.168.1.12:4323",
            "status": "healthy"
        }
    ],
    "health": "HEALTHY"
}
```

#### GET /cluster/nodes

Lists all cluster nodes.

**Response** (200 OK):
```json path=null start=null
{
    "nodes": [
        {
            "node_id": "node1",
            "role": "leader",
            "address": "192.168.1.10:4321",
            "http_address": "192.168.1.10:8001",
            "sessions": 5,
            "status": "healthy"
        }
    ]
}
```

#### GET /cluster/sessions

Lists sessions across all cluster nodes.

**Response** (200 OK):
```json path=null start=null
{
    "sessions": [
        {
            "session_id": "sess_abc123",
            "node_id": "node1",
            "status": "active"
        }
    ],
    "total": 1
}
```

## Request/Response Schemas

### SessionCreateRequest

```json path=null start=null
{
    "type": "object",
    "properties": {
        "llm_provider": {
            "type": "string",
            "enum": ["openai", "anthropic", "gemini", "google", "ollama", "lm_studio", "localai", "vllm"],
            "default": "openai"
        },
        "llm_model": {
            "type": "string",
            "description": "Uses provider default if not specified"
        },
        "base_url": {
            "type": "string",
            "description": "Custom endpoint for local providers"
        },
        "api_key": {
            "type": "string"
        },
        "headless": {
            "type": "boolean",
            "default": true
        },
        "browser_type": {
            "type": "string",
            "enum": ["chromium", "firefox", "webkit"],
            "default": "chromium"
        }
    }
}
```

### NavigateRequest

```json path=null start=null
{
    "type": "object",
    "required": ["url"],
    "properties": {
        "url": {
            "type": "string",
            "format": "uri"
        }
    }
}
```

### ExtractRequest

```json path=null start=null
{
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string"
        }
    }
}
```

### ActionRequest

```json path=null start=null
{
    "type": "object",
    "required": ["command"],
    "properties": {
        "command": {
            "type": "string"
        }
    }
}
```

### WorkflowRequest

```json path=null start=null
{
    "type": "object",
    "required": ["workflow"],
    "properties": {
        "workflow": {
            "type": "object",
            "required": ["name", "steps"],
            "properties": {
                "name": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["goto", "act", "extract", "screenshot"]
                            },
                            "url": {"type": "string"},
                            "command": {"type": "string"},
                            "query": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}
```

## Error Responses

All error responses follow this format:

```json path=null start=null
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable error message"
    }
}
```

### Error Codes

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | `BAD_REQUEST` | Invalid request parameters |
| 401 | `UNAUTHORIZED` | Missing or invalid API key |
| 404 | `NOT_FOUND` | Resource not found |
| 408 | `TIMEOUT` | Operation timed out |
| 429 | `TOO_MANY_REQUESTS` | Rate limit exceeded |
| 500 | `INTERNAL_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

## Rate Limiting

When rate limiting is enabled, responses include these headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

Exceeded rate limits return `429 Too Many Requests`.

## Pagination

List endpoints support pagination:

```
GET /sessions?offset=0&limit=10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset` | integer | 0 | Number of items to skip |
| `limit` | integer | 100 | Maximum items to return |

## WebSocket API

For real-time updates, connect to the WebSocket endpoint:

```
ws://{host}:{port}/ws/sessions/{session_id}
```

### Event Types

**Browser Events**:
```json path=null start=null
{
    "type": "navigation",
    "url": "https://example.com/page"
}
```

```json path=null start=null
{
    "type": "action",
    "action": "click",
    "element": "button"
}
```

**Error Events**:
```json path=null start=null
{
    "type": "error",
    "message": "Navigation failed"
}
```
