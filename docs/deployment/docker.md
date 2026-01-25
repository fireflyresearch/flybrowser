# Docker Deployment

Deploy FlyBrowser using Docker containers for easy portability and consistent environments.

## Quick Start

### Pull and Run

```bash
# Pull the latest image
docker pull flybrowser/flybrowser:latest

# Run standalone server
docker run -d \
  --name flybrowser \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  flybrowser/flybrowser:latest
```

### Verify

```bash
# Check container is running
docker ps

# Check health
curl http://localhost:8000/health

# View logs
docker logs -f flybrowser
```

## Docker Images

### Available Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest stable release |
| `x.y.z` | Specific version |
| `slim` | Minimal image without dev tools |
| `dev` | Development image with debug tools |

### Image Sizes

| Image | Size |
|-------|------|
| `flybrowser:latest` | ~1.5GB |
| `flybrowser:slim` | ~1.2GB |

The large size is due to browser binaries (Chromium, Firefox, WebKit).

## Configuration

### Environment Variables

```bash
docker run -d \
  --name flybrowser \
  -p 8000:8000 \
  # LLM Configuration
  -e OPENAI_API_KEY="sk-..." \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  # Server Configuration
  -e FLYBROWSER_HOST=0.0.0.0 \
  -e FLYBROWSER_PORT=8000 \
  -e FLYBROWSER_LOG_LEVEL=INFO \
  # Browser Pool
  -e FLYBROWSER_POOL__MAX_SIZE=10 \
  -e FLYBROWSER_POOL__HEADLESS=true \
  # Session Limits
  -e FLYBROWSER_MAX_SESSIONS=100 \
  flybrowser/flybrowser:latest
```

### Volume Mounts

```bash
docker run -d \
  --name flybrowser \
  -p 8000:8000 \
  # Recordings directory
  -v /data/recordings:/app/recordings \
  # Configuration file
  -v /path/to/config.yaml:/app/config.yaml \
  # Persistent data
  -v flybrowser-data:/app/data \
  flybrowser/flybrowser:latest
```

### Resource Limits

```bash
docker run -d \
  --name flybrowser \
  -p 8000:8000 \
  # Memory limit (browsers need significant memory)
  --memory=4g \
  --memory-swap=4g \
  # CPU limit
  --cpus=2 \
  # Shared memory (required for Chrome)
  --shm-size=2g \
  flybrowser/flybrowser:latest
```

## Docker Compose

### Basic Standalone

```yaml
# docker-compose.yml
version: '3.8'

services:
  flybrowser:
    image: flybrowser/flybrowser:latest
    ports:
      - "8000:8000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      FLYBROWSER_LOG_LEVEL: INFO
      FLYBROWSER_POOL__MAX_SIZE: 10
    volumes:
      - recordings:/app/recordings
    shm_size: 2gb
    restart: unless-stopped

volumes:
  recordings:
```

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f
```

### With Reverse Proxy

```yaml
version: '3.8'

services:
  flybrowser:
    image: flybrowser/flybrowser:latest
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      FLYBROWSER_LOG_LEVEL: INFO
    volumes:
      - recordings:/app/recordings
    shm_size: 2gb
    restart: unless-stopped
    networks:
      - internal

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - flybrowser
    restart: unless-stopped
    networks:
      - internal

networks:
  internal:

volumes:
  recordings:
```

### With Monitoring

```yaml
version: '3.8'

services:
  flybrowser:
    image: flybrowser/flybrowser:latest
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    volumes:
      - recordings:/app/recordings
    shm_size: 2gb
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - monitoring

networks:
  monitoring:

volumes:
  recordings:
  prometheus-data:
  grafana-data:
```

## Building Custom Images

### Dockerfile

```dockerfile
FROM flybrowser/flybrowser:latest

# Add custom configuration
COPY config.yaml /app/config.yaml

# Install additional dependencies
RUN pip install custom-package

# Set custom environment
ENV FLYBROWSER_CUSTOM_SETTING=value

# Custom entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
```

### Build and Push

```bash
# Build
docker build -t myorg/flybrowser:custom .

# Test locally
docker run -it --rm myorg/flybrowser:custom

# Push to registry
docker push myorg/flybrowser:custom
```

## Security Considerations

### Non-Root User

The official image runs as non-root by default. To verify:

```bash
docker run --rm flybrowser/flybrowser:latest id
# uid=1000(flybrowser) gid=1000(flybrowser)
```

### Read-Only Filesystem

For enhanced security:

```bash
docker run -d \
  --name flybrowser \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/recordings \
  -v flybrowser-data:/app/data:rw \
  flybrowser/flybrowser:latest
```

### Network Isolation

```bash
# Create isolated network
docker network create --internal flybrowser-internal

# Run with limited network access
docker run -d \
  --name flybrowser \
  --network flybrowser-internal \
  flybrowser/flybrowser:latest
```

### Secrets Management

```yaml
# docker-compose.yml with secrets
version: '3.8'

services:
  flybrowser:
    image: flybrowser/flybrowser:latest
    environment:
      OPENAI_API_KEY_FILE: /run/secrets/openai_api_key
    secrets:
      - openai_api_key

secrets:
  openai_api_key:
    file: ./secrets/openai_api_key.txt
```

## Health Checks

### Docker Health Check

The image includes a health check. Configure it:

```yaml
services:
  flybrowser:
    image: flybrowser/flybrowser:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### Custom Health Check

```bash
docker run -d \
  --name flybrowser \
  --health-cmd='curl -f http://localhost:8000/health || exit 1' \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  flybrowser/flybrowser:latest
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs flybrowser

# Run interactively
docker run -it --rm flybrowser/flybrowser:latest bash

# Check browser installation
docker run -it --rm flybrowser/flybrowser:latest \
  playwright install --check
```

### Browser Crashes

Increase shared memory:

```bash
docker run -d \
  --shm-size=2g \
  flybrowser/flybrowser:latest
```

### Permission Denied

```bash
# Fix volume permissions
docker run -it --rm \
  -v /data/recordings:/app/recordings \
  flybrowser/flybrowser:latest \
  chown -R 1000:1000 /app/recordings
```

### Out of Memory

```bash
# Monitor memory usage
docker stats flybrowser

# Reduce pool size
docker run -d \
  -e FLYBROWSER_POOL__MAX_SIZE=5 \
  --memory=2g \
  flybrowser/flybrowser:latest
```

## See Also

- [Standalone Deployment](standalone.md) - Non-containerized deployment
- [Cluster Deployment](cluster.md) - Multi-node deployment
- [Kubernetes Deployment](kubernetes.md) - Orchestrated containers
