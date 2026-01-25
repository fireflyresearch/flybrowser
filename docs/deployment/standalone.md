# Standalone Deployment

Standalone deployment runs FlyBrowser as a separate server process that clients connect to via REST API or the Python client. This is ideal for multi-client scenarios and production single-server deployments.

## Overview

In standalone mode:
- FlyBrowser runs as a dedicated service
- Clients connect via HTTP/REST API
- Browser pool manages multiple concurrent sessions
- Supports multiple simultaneous clients
- Process isolation for better stability

## When to Use

Standalone deployment is best for:

- Multi-client applications
- Microservice architectures
- Single-server production deployments
- Separation of concerns (browser service vs application)
- Language-agnostic access (any language can use REST API)

## Starting the Server

### Using CLI

```bash
# Basic start
flybrowser-serve

# Custom host and port
flybrowser-serve --host 0.0.0.0 --port 8000

# Development mode with auto-reload
flybrowser-serve --reload

# Production mode with multiple workers
flybrowser-serve --workers 4

# Custom log level
flybrowser-serve --log-level debug
```

### Using Python

```python
import uvicorn
from flybrowser.service.app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
    )
```

### Using Docker

```bash
docker run -p 8000:8000 flybrowser/flybrowser:latest
```

## Configuration

### Server Configuration

Configure via environment variables:

```bash
# Server settings
export FLYBROWSER_HOST=0.0.0.0
export FLYBROWSER_PORT=8000
export FLYBROWSER_WORKERS=4
export FLYBROWSER_LOG_LEVEL=INFO

# Deployment mode
export FLYBROWSER_DEPLOYMENT_MODE=standalone

# Session limits
export FLYBROWSER_MAX_SESSIONS=100
export FLYBROWSER_SESSION_TIMEOUT=3600
```

### Browser Pool Configuration

```bash
# Pool settings
export FLYBROWSER_POOL__MIN_SIZE=1
export FLYBROWSER_POOL__MAX_SIZE=10
export FLYBROWSER_POOL__IDLE_TIMEOUT_SECONDS=300
export FLYBROWSER_POOL__MAX_SESSION_AGE_SECONDS=3600
export FLYBROWSER_POOL__HEADLESS=true
export FLYBROWSER_POOL__BROWSER_TYPE=chromium
```

### Recording Configuration

```bash
# Recording settings
export FLYBROWSER_RECORDING_ENABLED=true
export FLYBROWSER_RECORDING_OUTPUT_DIR=/data/recordings
export FLYBROWSER_RECORDING_RETENTION_DAYS=7
```

### CORS Configuration

```bash
# Allow specific origins
export FLYBROWSER_CORS_ORIGINS='["https://app.example.com"]'

# Allow all origins (development only)
export FLYBROWSER_CORS_ORIGINS='["*"]'
```

## Client Connection

### Python Client

```python
from flybrowser import FlyBrowserClient

# Connect to server
client = FlyBrowserClient("http://localhost:8000")

# Create session
async with client.session() as browser:
    await browser.goto("https://example.com")
    data = await browser.extract("Get the title")
    print(data)
```

### REST API

```bash
# Create session
SESSION_ID=$(curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" | jq -r '.session_id')

# Navigate
curl -X POST "http://localhost:8000/api/v1/sessions/$SESSION_ID/navigate" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Extract
curl -X POST "http://localhost:8000/api/v1/sessions/$SESSION_ID/extract" \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Get the page title"}'

# Close session
curl -X DELETE "http://localhost:8000/api/v1/sessions/$SESSION_ID"
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sessions` | POST | Create new session |
| `/api/v1/sessions/{id}` | GET | Get session info |
| `/api/v1/sessions/{id}` | DELETE | Close session |
| `/api/v1/sessions/{id}/navigate` | POST | Navigate to URL |
| `/api/v1/sessions/{id}/act` | POST | Perform action |
| `/api/v1/sessions/{id}/extract` | POST | Extract data |
| `/api/v1/sessions/{id}/agent` | POST | Run agent task |
| `/api/v1/sessions/{id}/screenshot` | GET | Take screenshot |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Health Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
# {"status": "healthy", "sessions": 5, "pool_available": 5}
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
# Prometheus-format metrics
```

### Monitoring Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'flybrowser'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
```

## Production Considerations

### Process Management

Use a process manager like systemd or supervisord:

**systemd service:**

```ini
# /etc/systemd/system/flybrowser.service
[Unit]
Description=FlyBrowser Service
After=network.target

[Service]
Type=simple
User=flybrowser
WorkingDirectory=/opt/flybrowser
ExecStart=/usr/local/bin/flybrowser-serve --workers 4
Restart=always
RestartSec=5
Environment=FLYBROWSER_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable flybrowser
sudo systemctl start flybrowser
```

**supervisord:**

```ini
# /etc/supervisor/conf.d/flybrowser.conf
[program:flybrowser]
command=/usr/local/bin/flybrowser-serve --workers 4
directory=/opt/flybrowser
user=flybrowser
autostart=true
autorestart=true
stderr_logfile=/var/log/flybrowser/err.log
stdout_logfile=/var/log/flybrowser/out.log
```

### Reverse Proxy

**nginx configuration:**

```nginx
upstream flybrowser {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name flybrowser.example.com;

    ssl_certificate /etc/ssl/certs/flybrowser.crt;
    ssl_certificate_key /etc/ssl/private/flybrowser.key;

    location / {
        proxy_pass http://flybrowser;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

### Resource Limits

Set appropriate resource limits:

```bash
# Memory limit (via systemd)
MemoryMax=4G

# CPU limit
CPUQuota=200%  # 2 cores

# File descriptors (browsers need many)
LimitNOFILE=65535
```

### Security

1. **Authentication** - Enable API key authentication:
```bash
export FLYBROWSER_AUTH_ENABLED=true
export FLYBROWSER_API_KEYS='["key1","key2"]'
```

2. **Network** - Run behind firewall, only expose through reverse proxy

3. **TLS** - Always use HTTPS in production

4. **Rate Limiting** - Configure request limits:
```bash
export FLYBROWSER_RATE_LIMIT_RPM=60
```

## Scaling

Standalone mode scales vertically:

- **CPU**: More workers = more concurrent operations
- **Memory**: More pool size = more concurrent browsers
- **Disk**: More storage = longer recording retention

For horizontal scaling across multiple servers, see [Cluster Deployment](cluster.md).

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Browser startup failures:**
```bash
# Check Playwright dependencies
playwright install-deps chromium

# Verify browsers installed
playwright install chromium
```

**Memory issues:**
```bash
# Reduce pool size
export FLYBROWSER_POOL__MAX_SIZE=5

# Reduce session timeout
export FLYBROWSER_SESSION_TIMEOUT=1800
```

### Logs

```bash
# View logs
journalctl -u flybrowser -f

# Debug logging
export FLYBROWSER_LOG_LEVEL=DEBUG
flybrowser-serve
```

## See Also

- [Embedded Deployment](embedded.md) - In-process deployment
- [Cluster Deployment](cluster.md) - High-availability deployment
- [REST API Reference](../reference/rest-api.md) - API documentation
- [CLI Reference](../reference/cli.md) - Command line options
