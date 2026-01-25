# Cluster Deployment

Cluster deployment runs FlyBrowser across multiple nodes for high availability, fault tolerance, and horizontal scaling. It uses the Raft consensus algorithm to maintain consistency across nodes.

## Overview

In cluster mode:
- Multiple FlyBrowser nodes form a distributed cluster
- Raft consensus ensures leader election and state replication
- Automatic failover when nodes fail
- Load balancing distributes requests across nodes
- Shared state for session management

## When to Use

Cluster deployment is best for:

- High-availability requirements
- Large-scale production deployments
- Horizontal scaling needs
- Mission-critical applications
- Multi-region deployments

## Architecture

```
                    ┌─────────────┐
                    │   Clients   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Load Balancer│
                    └──────┬──────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
│   Node 1    │     │   Node 2    │     │   Node 3    │
│  (Leader)   │◄───►│ (Follower)  │◄───►│ (Follower)  │
│             │     │             │     │             │
│ ┌─────────┐ │     │ ┌─────────┐ │     │ ┌─────────┐ │
│ │ Browser │ │     │ │ Browser │ │     │ │ Browser │ │
│ │  Pool   │ │     │ │  Pool   │ │     │ │  Pool   │ │
│ └─────────┘ │     │ └─────────┘ │     │ └─────────┘ │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Quick Start

### Starting a 3-Node Cluster

**Node 1 (Bootstrap):**
```bash
flybrowser-serve --cluster \
  --node-id node1 \
  --port 8001 \
  --raft-port 4321 \
  --data-dir ./data/node1
```

**Node 2:**
```bash
flybrowser-serve --cluster \
  --node-id node2 \
  --port 8002 \
  --raft-port 4322 \
  --peers node1:4321:8001 \
  --data-dir ./data/node2
```

**Node 3:**
```bash
flybrowser-serve --cluster \
  --node-id node3 \
  --port 8003 \
  --raft-port 4323 \
  --peers node1:4321:8001,node2:4322:8002 \
  --data-dir ./data/node3
```

### Using Environment Variables

```bash
# Node 1
export FLYBROWSER_CLUSTER_ENABLED=true
export FLYBROWSER_NODE_ID=node1
export FLYBROWSER_PORT=8001
export FLYBROWSER_RAFT_PORT=4321
export FLYBROWSER_DATA_DIR=./data/node1
flybrowser-serve

# Node 2
export FLYBROWSER_CLUSTER_ENABLED=true
export FLYBROWSER_NODE_ID=node2
export FLYBROWSER_PORT=8002
export FLYBROWSER_RAFT_PORT=4322
export FLYBROWSER_CLUSTER_PEERS=node1:4321:8001
export FLYBROWSER_DATA_DIR=./data/node2
flybrowser-serve
```

## Configuration

### Cluster Configuration

```bash
# Enable cluster mode
export FLYBROWSER_CLUSTER_ENABLED=true

# Node identification
export FLYBROWSER_NODE_ID=node1              # Unique node identifier

# Network ports
export FLYBROWSER_PORT=8000                  # API port
export FLYBROWSER_RAFT_PORT=4321             # Raft consensus port

# Peer discovery
export FLYBROWSER_CLUSTER_PEERS=node2:4321:8002,node3:4321:8003

# Data directory (for Raft logs and state)
export FLYBROWSER_DATA_DIR=/data/flybrowser

# Max sessions per node
export FLYBROWSER_MAX_SESSIONS=10
```

### Raft Configuration

```bash
# Raft timing (milliseconds)
export FLYBROWSER_RAFT_ELECTION_TIMEOUT_MIN=300
export FLYBROWSER_RAFT_ELECTION_TIMEOUT_MAX=500
export FLYBROWSER_RAFT_HEARTBEAT_INTERVAL=100

# Raft data directory
export FLYBROWSER_RAFT_DATA_DIR=/data/raft
```

### Load Balancing

```bash
# Load balancing strategy
export FLYBROWSER_CLUSTER_LOAD_BALANCING_STRATEGY=least_load

# Options:
# - least_load: Route to node with fewest active sessions
# - round_robin: Distribute evenly across nodes
# - random: Random node selection
```

### Discovery Methods

```bash
# Static peer list (default)
export FLYBROWSER_CLUSTER_DISCOVERY_METHOD=static
export FLYBROWSER_CLUSTER_PEERS=node1:4321:8001,node2:4321:8002

# DNS-based discovery
export FLYBROWSER_CLUSTER_DISCOVERY_METHOD=dns
export FLYBROWSER_CLUSTER_DNS_NAME=flybrowser.local

# Kubernetes service discovery
export FLYBROWSER_CLUSTER_DISCOVERY_METHOD=kubernetes
export FLYBROWSER_CLUSTER_K8S_NAMESPACE=default
export FLYBROWSER_CLUSTER_K8S_SERVICE=flybrowser
```

## Cluster Management

### CLI Commands

```bash
# Check cluster status
flybrowser cluster status

# List nodes
flybrowser cluster nodes

# Get leader info
flybrowser cluster leader

# Add a node
flybrowser cluster add-node node4:4324:8004

# Remove a node
flybrowser cluster remove-node node4

# Force leader election (emergency)
flybrowser cluster force-election
```

### Cluster Status API

```bash
# Get cluster status
curl http://localhost:8000/api/v1/cluster/status
# {
#   "cluster_id": "flybrowser-cluster-1",
#   "leader": "node1",
#   "nodes": [
#     {"id": "node1", "state": "leader", "api": "http://node1:8001"},
#     {"id": "node2", "state": "follower", "api": "http://node2:8002"},
#     {"id": "node3", "state": "follower", "api": "http://node3:8003"}
#   ],
#   "healthy": true
# }

# Get specific node info
curl http://localhost:8000/api/v1/cluster/nodes/node1
```

## High Availability

### Leader Election

- Nodes automatically elect a leader using Raft consensus
- Leader handles all write operations
- Followers replicate state from leader
- Automatic re-election if leader fails (within election timeout)

### Failover Behavior

1. **Leader failure**: Followers detect missing heartbeats, trigger election
2. **Follower failure**: Cluster continues with remaining nodes
3. **Network partition**: Majority partition elects new leader, minority becomes unavailable

### Quorum Requirements

For a cluster of N nodes:
- Requires (N/2)+1 nodes for quorum
- 3-node cluster: tolerates 1 failure
- 5-node cluster: tolerates 2 failures

## Session Management

### Session Affinity

Sessions are bound to specific nodes:

```bash
# Create session (routed to optimal node)
curl -X POST http://load-balancer:8000/api/v1/sessions
# {"session_id": "abc123", "node_id": "node2"}

# Subsequent requests go to the same node
curl http://load-balancer:8000/api/v1/sessions/abc123/navigate \
  -H "X-Node-ID: node2"
```

### Session Migration

If a node fails, sessions can be migrated:

```bash
# Automatic migration (if session state is persisted)
# Sessions with in-flight operations are lost

# Manual migration
curl -X POST http://node2:8002/api/v1/sessions/abc123/migrate \
  -d '{"target_node": "node3"}'
```

## Storage Configuration

### Local Storage

```bash
# Local storage (default)
export FLYBROWSER_RECORDING_STORAGE=local
export FLYBROWSER_RECORDING_OUTPUT_DIR=/data/recordings
```

### Shared Storage (S3)

For clusters, use shared storage:

```bash
# S3 storage
export FLYBROWSER_RECORDING_STORAGE=s3
export FLYBROWSER_S3_BUCKET=flybrowser-recordings
export FLYBROWSER_S3_REGION=us-east-1
export FLYBROWSER_S3_ACCESS_KEY=AKIA...
export FLYBROWSER_S3_SECRET_KEY=...

# S3-compatible (MinIO)
export FLYBROWSER_S3_ENDPOINT_URL=http://minio:9000
```

## Docker Compose Example

```yaml
version: '3.8'

services:
  flybrowser-node1:
    image: flybrowser/flybrowser:latest
    environment:
      FLYBROWSER_CLUSTER_ENABLED: "true"
      FLYBROWSER_NODE_ID: node1
      FLYBROWSER_PORT: "8001"
      FLYBROWSER_RAFT_PORT: "4321"
      FLYBROWSER_DATA_DIR: /data
    volumes:
      - node1-data:/data
    ports:
      - "8001:8001"
    networks:
      - flybrowser-net

  flybrowser-node2:
    image: flybrowser/flybrowser:latest
    environment:
      FLYBROWSER_CLUSTER_ENABLED: "true"
      FLYBROWSER_NODE_ID: node2
      FLYBROWSER_PORT: "8002"
      FLYBROWSER_RAFT_PORT: "4322"
      FLYBROWSER_CLUSTER_PEERS: node1:4321:8001
      FLYBROWSER_DATA_DIR: /data
    volumes:
      - node2-data:/data
    ports:
      - "8002:8002"
    networks:
      - flybrowser-net
    depends_on:
      - flybrowser-node1

  flybrowser-node3:
    image: flybrowser/flybrowser:latest
    environment:
      FLYBROWSER_CLUSTER_ENABLED: "true"
      FLYBROWSER_NODE_ID: node3
      FLYBROWSER_PORT: "8003"
      FLYBROWSER_RAFT_PORT: "4323"
      FLYBROWSER_CLUSTER_PEERS: node1:4321:8001,node2:4322:8002
      FLYBROWSER_DATA_DIR: /data
    volumes:
      - node3-data:/data
    ports:
      - "8003:8003"
    networks:
      - flybrowser-net
    depends_on:
      - flybrowser-node1
      - flybrowser-node2

  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - flybrowser-net
    depends_on:
      - flybrowser-node1
      - flybrowser-node2
      - flybrowser-node3

networks:
  flybrowser-net:

volumes:
  node1-data:
  node2-data:
  node3-data:
```

**nginx.conf for load balancing:**

```nginx
events {
    worker_connections 1024;
}

http {
    upstream flybrowser {
        least_conn;
        server flybrowser-node1:8001;
        server flybrowser-node2:8002;
        server flybrowser-node3:8003;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://flybrowser;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 300s;
        }
    }
}
```

## Monitoring

### Cluster Metrics

```bash
# Prometheus metrics include:
# - flybrowser_cluster_leader (gauge)
# - flybrowser_cluster_nodes_total (gauge)
# - flybrowser_cluster_healthy (gauge)
# - flybrowser_raft_term (counter)
# - flybrowser_raft_commit_index (counter)
```

### Alerting

```yaml
# prometheus-alerts.yml
groups:
  - name: flybrowser-cluster
    rules:
      - alert: ClusterLeaderLost
        expr: sum(flybrowser_cluster_leader) == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "FlyBrowser cluster has no leader"

      - alert: ClusterNodeDown
        expr: flybrowser_cluster_nodes_total < 3
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "FlyBrowser cluster node count below 3"
```

## Troubleshooting

### Common Issues

**Nodes not joining:**
```bash
# Check network connectivity
ping node2

# Check Raft port is accessible
nc -zv node2 4321

# Check logs
journalctl -u flybrowser -f
```

**Split brain:**
```bash
# Force cluster to specific leader
flybrowser cluster force-election --preferred-leader node1

# Restart minority partition nodes
systemctl restart flybrowser  # on minority nodes
```

**State desync:**
```bash
# Force state resync from leader
flybrowser cluster resync --node node3

# Or restart the desync'd node
systemctl restart flybrowser
```

### Debug Mode

```bash
# Enable verbose Raft logging
export FLYBROWSER_RAFT_DEBUG=true
export FLYBROWSER_LOG_LEVEL=DEBUG
flybrowser-serve --cluster
```

## See Also

- [Embedded Deployment](embedded.md) - In-process deployment
- [Standalone Deployment](standalone.md) - Single-server deployment
- [Docker Deployment](docker.md) - Container deployment
- [Kubernetes Deployment](kubernetes.md) - Orchestrated deployment
