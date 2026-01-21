# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
#
# FlyBrowser Docker Image
#
# This Dockerfile supports both standalone and cluster modes.
# The mode is controlled via environment variables.
#
# Build:
#   docker build -t flybrowser/flybrowser:latest -f docker/Dockerfile .
#
# Run Standalone:
#   docker run -p 8000:8000 flybrowser/flybrowser:latest
#
# Run Cluster Node:
#   docker run -p 8000:8000 \
#     -e FLYBROWSER_CLUSTER_ENABLED=true \
#     -e FLYBROWSER_NODE_ID=node1 \
#     -e FLYBROWSER_CLUSTER_PEERS=node2:4321,node3:4321 \
#     flybrowser/flybrowser:latest

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY flybrowser/ ./flybrowser/

# Build wheel
RUN pip install --no-cache-dir build && python -m build --wheel

# Stage 2: Runtime
FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="FlyBrowser"
LABEL org.opencontainers.image.description="Browser automation powered by LLM agents"
LABEL org.opencontainers.image.vendor="Firefly Software Solutions Inc"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install runtime dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install FlyBrowser from wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Install Playwright browsers
RUN playwright install chromium && playwright install-deps chromium

# Create non-root user
RUN useradd -m -u 1000 flybrowser && \
    mkdir -p /data /home/flybrowser/.flybrowser && \
    chown -R flybrowser:flybrowser /app /data /home/flybrowser

USER flybrowser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLYBROWSER_ENV=production
ENV FLYBROWSER_HOST=0.0.0.0
ENV FLYBROWSER_PORT=8000
ENV FLYBROWSER_DATA_DIR=/data

# Cluster mode (disabled by default)
ENV FLYBROWSER_CLUSTER_ENABLED=false
ENV FLYBROWSER_NODE_ID=""
ENV FLYBROWSER_CLUSTER_PEERS=""
ENV FLYBROWSER_RAFT_PORT=4321

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${FLYBROWSER_PORT}/health || exit 1

# Expose ports: API (8000) and Raft (4321)
EXPOSE 8000 4321

# Default command
CMD ["python", "-m", "flybrowser.cli.serve"]

