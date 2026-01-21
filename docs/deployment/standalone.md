# Standalone Mode Deployment

Standalone mode runs FlyBrowser as an HTTP service, exposing a REST API for browser automation. This deployment mode is suitable for microservice architectures, language-agnostic integration, and scenarios requiring process isolation.

## Overview

In standalone mode, FlyBrowser runs as a separate process or container, accepting HTTP requests and managing browser sessions independently. Clients interact with the service through a REST API.

**Advantages:**
- Language-agnostic access via HTTP
- Process isolation from client applications
- Suitable for containerized deployments
- Session management across multiple clients

**Considerations:**
- Network latency between client and service
- Single point of failure without clustering
- Requires separate deployment and management

## Installation

Install FlyBrowser from source:

```bash path=null start=null
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh
```

## Starting the Server

### Basic Startup

Start the server with default settings:

```bash path=null start=null
flybrowser-serve
```

This starts the server on `http://0.0.0.0:8000`.

### Custom Host and Port

```bash path=null start=null
flybrowser-serve --host 0.0.0.0 --port 8000
```

### Full Configuration

```bash path=null start=null
flybrowser-serve \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --max-sessions 50 \
    --data-dir /var/lib/flybrowser \
    --log-level info
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind |
| `--port` | `8000` | Port number |
| `--workers` | `1` | Number of worker processes |
| `--max-sessions` | `10` | Maximum concurrent sessions |
| `--data-dir` | `./data` | Directory for persistent data |
| `--log-level` | `info` | Log level (debug, info, warning, error) |
| `--reload` | (flag) | Enable auto-reload for development |

## Python Client

The `FlyBrowserClient` class provides a Python interface to the standalone server:

```python path=null start=null
from flybrowser import FlyBrowserClient

# Connect to the server
client = FlyBrowserClient(
    endpoint="http://localhost:8000",
    api_key="your-api-key"  # Optional, for authenticated servers
)

# Create a session with any supported provider
session = await client.create_session(
    llm_provider="openai",      # or: anthropic, gemini, ollama, vllm, etc.
    llm_model="gpt-5.2",        # uses provider default if not set
    headless=True,
    browser_type="chromium"
)

session_id = session["session_id"]

# Navigate
await client.navigate(session_id, "https://example.com")

# Extract data
result = await client.extract(session_id, "Extract the page title")

# Close the session
await client.close_session(session_id)
```

### Complete Client Example

```python path=null start=null
import asyncio
from flybrowser import FlyBrowserClient

async def main():
    client = FlyBrowserClient(endpoint="http://localhost:8000")
    
    try:
        # Create a new browser session
        session = await client.create_session(
            llm_provider="openai",
            llm_model="gpt-5.2",
            headless=True
        )
        session_id = session["session_id"]
        
        # Perform navigation
        await client.navigate(session_id, "https://example.com")
        
        # Extract data
        result = await client.extract(
            session_id,
            "What is the main heading on this page?"
        )
        print(f"Extracted: {result}")
        
        # Perform actions
        await client.action(session_id, "Click the About link")
        
        # Take a screenshot
        screenshot = await client.screenshot(session_id)
        with open("screenshot.png", "wb") as f:
            f.write(screenshot)
            
    finally:
        # Always close the session
        await client.close_session(session_id)

asyncio.run(main())
```

## REST API Usage

### Create Session

```bash path=null start=null
# With OpenAI
curl -X POST http://localhost:8000/sessions \
    -H "Content-Type: application/json" \
    -d '{
        "llm_provider": "openai",
        "llm_model": "gpt-5.2",
        "headless": true,
        "browser_type": "chromium"
    }'

# With Google Gemini
curl -X POST http://localhost:8000/sessions \
    -H "Content-Type: application/json" \
    -d '{
        "llm_provider": "gemini",
        "llm_model": "gemini-2.0-flash",
        "headless": true
    }'

# With local Ollama
curl -X POST http://localhost:8000/sessions \
    -H "Content-Type: application/json" \
    -d '{
        "llm_provider": "ollama",
        "llm_model": "qwen3:8b",
        "headless": true
    }'
```

Response:

```json path=null start=null
{
    "session_id": "sess_abc123",
    "status": "created"
}
```

### Navigate

```bash path=null start=null
curl -X POST http://localhost:8000/sessions/sess_abc123/navigate \
    -H "Content-Type: application/json" \
    -d '{
        "url": "https://example.com"
    }'
```

### Extract Data

```bash path=null start=null
curl -X POST http://localhost:8000/sessions/sess_abc123/extract \
    -H "Content-Type: application/json" \
    -d '{
        "query": "Extract the main heading"
    }'
```

Response:

```json path=null start=null
{
    "result": "Example Domain"
}
```

### Perform Action

```bash path=null start=null
curl -X POST http://localhost:8000/sessions/sess_abc123/action \
    -H "Content-Type: application/json" \
    -d '{
        "command": "Click the More information link"
    }'
```

### Screenshot

```bash path=null start=null
curl -X GET http://localhost:8000/sessions/sess_abc123/screenshot \
    --output screenshot.png
```

### Close Session

```bash path=null start=null
curl -X DELETE http://localhost:8000/sessions/sess_abc123
```

### List Sessions

```bash path=null start=null
curl -X GET http://localhost:8000/sessions
```

Response:

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

### Run Workflow

```bash path=null start=null
curl -X POST http://localhost:8000/sessions/sess_abc123/workflow \
    -H "Content-Type: application/json" \
    -d '{
        "workflow": {
            "name": "example_workflow",
            "steps": [
                {"action": "goto", "url": "https://example.com"},
                {"action": "extract", "query": "Get the page title"}
            ]
        }
    }'
```

### Health Check

```bash path=null start=null
curl -X GET http://localhost:8000/health
```

## Docker Deployment

### Dockerfile

```dockerfile path=null start=null
FROM python:3.11-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install FlyBrowser from source
COPY . /app
RUN pip install -e .

# Install Playwright browsers
RUN playwright install chromium

# Expose the service port
EXPOSE 8000

# Run the server
CMD ["flybrowser-serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash path=null start=null
# Build the image
docker build -t flybrowser-server .

# Run the container
docker run -d \
    --name flybrowser \
    -p 8000:8000 \
    -e OPENAI_API_KEY=your-key \
    -e GOOGLE_API_KEY=your-google-key \
    -e ANTHROPIC_API_KEY=your-anthropic-key \
    flybrowser-server
```

### Docker Compose

```yaml path=null start=null
version: "3.8"

services:
  flybrowser:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - flybrowser-data:/var/lib/flybrowser
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  flybrowser-data:
```

## Kubernetes Deployment

### Deployment Manifest

```yaml path=null start=null
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flybrowser
  labels:
    app: flybrowser
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flybrowser
  template:
    metadata:
      labels:
        app: flybrowser
    spec:
      containers:
        - name: flybrowser
          image: flybrowser-server:latest
          ports:
        - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: flybrowser-secrets
                  key: openai-api-key
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: flybrowser
spec:
  selector:
    app: flybrowser
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### Create Secrets

```bash path=null start=null
kubectl create secret generic flybrowser-secrets \
    --from-literal=openai-api-key=your-key
```

### Deploy

```bash path=null start=null
kubectl apply -f flybrowser-deployment.yaml
```

## Session Management

### Session Lifecycle

1. **Creation**: Client requests a new session with configuration
2. **Active**: Session accepts commands (navigate, extract, action)
3. **Idle**: Session remains open but inactive
4. **Closed**: Session is explicitly closed or times out

### Session Timeouts

Sessions have configurable idle timeouts:

```bash path=null start=null
flybrowser-serve --session-timeout 3600  # 1 hour timeout
```

### Managing Sessions

List all sessions:

```python path=null start=null
sessions = await client.list_sessions()
for session in sessions["sessions"]:
    print(f"Session {session['session_id']}: {session['status']}")
```

Close inactive sessions:

```bash path=null start=null
# Using the admin CLI
flybrowser-admin sessions list
flybrowser-admin sessions kill sess_abc123
```

## Security

### API Authentication

Enable API key authentication:

```bash path=null start=null
export FLYBROWSER_API_KEY="your-secure-api-key"
flybrowser-serve --host 0.0.0.0 --port 8080
```

Clients must include the API key:

```python path=null start=null
client = FlyBrowserClient(
    endpoint="http://localhost:8080",
    api_key="your-secure-api-key"
)
```

Or via header:

```bash path=null start=null
curl -X POST http://localhost:8080/sessions \
    -H "Authorization: Bearer your-secure-api-key" \
    -H "Content-Type: application/json" \
-d '{"llm_provider": "openai", "llm_model": "gpt-5.2"}'
```

### TLS/HTTPS

For production deployments, use a reverse proxy (nginx, traefik) to terminate TLS:

```nginx path=null start=null
server {
    listen 443 ssl;
    server_name flybrowser.example.com;
    
    ssl_certificate /etc/ssl/certs/flybrowser.crt;
    ssl_certificate_key /etc/ssl/private/flybrowser.key;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Health Endpoint

```bash path=null start=null
curl http://localhost:8000/health
```

Response:

```json path=null start=null
{
    "status": "healthy",
    "version": "1.0.0"
}
```

### Metrics

Monitor key metrics:
- Active session count
- Request latency
- Error rates
- Resource utilization

### Logging

Configure logging level:

```bash path=null start=null
export FLYBROWSER_LOG_LEVEL=INFO
flybrowser-serve
```

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## Error Handling

### Client-Side Error Handling

```python path=null start=null
from flybrowser import FlyBrowserClient

client = FlyBrowserClient(endpoint="http://localhost:8080")

try:
    session = await client.create_session(
        llm_provider="openai",
    llm_model="gpt-5.2"
    )
    session_id = session["session_id"]
    
    await client.navigate(session_id, "https://example.com")
    result = await client.extract(session_id, "Get data")
    
except ConnectionError:
    print("Failed to connect to FlyBrowser server")
except TimeoutError:
    print("Request timed out")
except Exception as e:
    print(f"Error: {e}")
finally:
    if session_id:
        await client.close_session(session_id)
```

### HTTP Error Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Session created |
| `400` | Bad request (invalid parameters) |
| `401` | Unauthorized (invalid API key) |
| `404` | Session not found |
| `429` | Too many requests |
| `500` | Internal server error |

## Performance Tuning

### Connection Pooling

Configure client connection pooling:

```python path=null start=null
import aiohttp
from flybrowser import FlyBrowserClient

# Use custom connector with connection pooling
connector = aiohttp.TCPConnector(
    limit=100,  # Total connections
    limit_per_host=20  # Per-host limit
)

client = FlyBrowserClient(
    endpoint="http://localhost:8080",
    connector=connector
)
```

### Resource Limits

Set appropriate resource limits:

```bash path=null start=null
flybrowser-serve \
    --max-sessions 20 \
    --memory-limit 4096  # MB
```

## Next Steps

- [Cluster Mode](cluster.md) - Deploy a distributed cluster for high availability
- [REST API Reference](../reference/api.md) - Complete API documentation
- [CLI Reference](../reference/cli.md) - Command-line interface documentation
