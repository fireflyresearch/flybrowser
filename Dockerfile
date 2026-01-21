# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
#
# FlyBrowser Docker Image - Root Wrapper
#
# This is a convenience wrapper that includes the main Dockerfile.
# The canonical Dockerfile is at docker/Dockerfile.
#
# Build:
#   docker build -t flybrowser/flybrowser:latest .
#
# Or use the docker/ directory directly:
#   docker build -t flybrowser/flybrowser:latest -f docker/Dockerfile .
#
# Run Standalone:
#   docker run -p 8000:8000 flybrowser/flybrowser:latest
#
# Run Cluster Node:
#   docker run -p 8000:8000 -p 4321:4321 \
#     -e FLYBROWSER_CLUSTER_ENABLED=true \
#     -e FLYBROWSER_NODE_ID=node1 \
#     -e FLYBROWSER_CLUSTER_PEERS=node2:4321,node3:4321 \
#     flybrowser/flybrowser:latest

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY flybrowser/ ./flybrowser/

RUN pip install --no-cache-dir build && python -m build --wheel

# Stage 2: Runtime
FROM python:3.11-slim

LABEL org.opencontainers.image.title="FlyBrowser"
LABEL org.opencontainers.image.description="Browser automation powered by LLM agents"
LABEL org.opencontainers.image.vendor="Firefly Software Solutions Inc"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libdbus-1-3 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
RUN playwright install chromium && playwright install-deps chromium

RUN useradd -m -u 1000 flybrowser && \
    mkdir -p /data /home/flybrowser/.flybrowser && \
    chown -R flybrowser:flybrowser /app /data /home/flybrowser

USER flybrowser

ENV PYTHONUNBUFFERED=1
ENV FLYBROWSER_ENV=production
ENV FLYBROWSER_HOST=0.0.0.0
ENV FLYBROWSER_PORT=8000
ENV FLYBROWSER_DATA_DIR=/data
ENV FLYBROWSER_CLUSTER_ENABLED=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${FLYBROWSER_PORT}/health || exit 1

EXPOSE 8000 4321

CMD ["python", "-m", "flybrowser.cli.serve"]

