# Cluster Mode Deployment

Cluster mode deploys FlyBrowser as a distributed system with multiple nodes coordinated through Raft consensus. This deployment mode provides high availability, fault tolerance, and horizontal scaling for production workloads.

## Overview

In cluster mode, multiple FlyBrowser nodes form a distributed cluster. The Raft consensus protocol ensures data consistency and provides automatic leader election. Sessions are replicated across nodes, allowing the cluster to survive node failures.

**Advantages:**
- High availability through automatic failover
- Horizontal scaling for increased throughput
- Session replication across nodes
- No single point of failure

**Considerations:**
- More complex deployment and operations
- Network latency between nodes
- Requires odd number of nodes for proper quorum (3, 5, 7)

## Architecture

### Cluster Components

A FlyBrowser cluster consists of:

- **Leader Node**: Handles all write operations and coordinates replication
- **Follower Nodes**: Replicate state from the leader and serve read operations
- **Raft Protocol**: Manages consensus and leader election

### Quorum Requirements

The cluster requires a majority of nodes (quorum) to operate:

| Nodes | Quorum | Tolerated Failures |
|-------|--------|-------------------|
| 3     | 2      | 1                 |
| 5     | 3      | 2                 |
| 7     | 4      | 3                 |

For production deployments, a minimum of 3 nodes is recommended.

## Prerequisites

Before deploying a cluster:

1. **Network Configuration**: Nodes must be able to communicate on both HTTP and Raft ports
2. **Time Synchronization**: All nodes should have synchronized clocks (NTP recommended)
3. **Persistent Storage**: Each node needs dedicated storage for Raft logs
4. **Firewall Rules**: Allow traffic on HTTP ports (default 8000) and Raft ports (default 4321)

## Step-by-Step: 3-Node Cluster Setup

This section provides a complete walkthrough for deploying a 3-node FlyBrowser cluster.

### Step 1: Prepare the Environment

On each node, install FlyBrowser from source:

```bash path=null start=null
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh
```

Create data directories:

```bash path=null start=null
# On each node
mkdir -p /var/lib/flybrowser/data
mkdir -p /var/lib/flybrowser/raft
```

### Step 2: Network Planning

Plan your network configuration:

| Node | Hostname | IP Address | HTTP Port | Raft Port |
|------|----------|------------|-----------|-----------|
| Node 1 | node1.example.com | 192.168.1.10 | 8001 | 4321 |
| Node 2 | node2.example.com | 192.168.1.11 | 8002 | 4322 |
| Node 3 | node3.example.com | 192.168.1.12 | 8003 | 4323 |

Ensure the following ports are open between all nodes:
- HTTP ports (8001, 8002, 8003) for API traffic
- Raft ports (4321, 4322, 4323) for cluster communication

### Step 3: Start Node 1 (Initial Leader)

Start the first node. This node will bootstrap the cluster:

```bash path=null start=null
flybrowser-serve \
    --cluster \
    --node-id node1 \
    --host 0.0.0.0 \
    --port 8001 \
    --raft-port 4321 \
    --data-dir /var/lib/flybrowser
```

Wait for the node to start. You should see log output indicating the node has become leader:

```
INFO: Starting FlyBrowser in cluster mode
INFO: Node ID: node1
INFO: Raft port: 4321
INFO: HTTP port: 8001
INFO: Node node1 elected as leader
```

Verify the node is running:

```bash path=null start=null
curl http://192.168.1.10:8001/health
```

Expected response:

```json path=null start=null
{
    "status": "healthy",
    "cluster": {
        "node_id": "node1",
        "role": "leader",
        "term": 1
    }
}
```

### Step 4: Start Node 2

Start the second node, pointing it to Node 1 as a peer:

```bash path=null start=null
flybrowser-serve \
    --cluster \
    --node-id node2 \
    --host 0.0.0.0 \
    --port 8002 \
    --raft-port 4322 \
    --data-dir /var/lib/flybrowser \
    --peers node1:4321
```

The `--peers` argument uses the format `hostname:raft-port` or `ip:raft-port`.

Wait for Node 2 to join the cluster. You should see:

```
INFO: Starting FlyBrowser in cluster mode
INFO: Node ID: node2
INFO: Connecting to peer: node1:4321
INFO: Successfully joined cluster
INFO: Current leader: node1
```

Verify Node 2 is a follower:

```bash path=null start=null
curl http://192.168.1.11:8002/health
```

Expected response:

```json path=null start=null
{
    "status": "healthy",
    "cluster": {
        "node_id": "node2",
        "role": "follower",
        "leader": "node1",
        "term": 1
    }
}
```

### Step 5: Start Node 3

Start the third node, pointing it to both existing nodes:

```bash path=null start=null
flybrowser-serve \
    --cluster \
    --node-id node3 \
    --host 0.0.0.0 \
    --port 8003 \
    --raft-port 4323 \
    --data-dir /var/lib/flybrowser \
    --peers node1:4321,node2:4322
```

Wait for Node 3 to join:

```
INFO: Starting FlyBrowser in cluster mode
INFO: Node ID: node3
INFO: Connecting to peers: node1:4321, node2:4322
INFO: Successfully joined cluster
INFO: Current leader: node1
```

### Step 6: Verify Cluster Status

Check the cluster status from any node:

```bash path=null start=null
flybrowser-cluster status --endpoint http://192.168.1.10:8001
```

Expected output:

```
Cluster Status
==============
Leader: node1
Term: 1
Nodes: 3

Node Details:
  node1: leader (192.168.1.10:4321)
  node2: follower (192.168.1.11:4322)
  node3: follower (192.168.1.12:4323)

Cluster health: HEALTHY
```

Or use the API:

```bash path=null start=null
curl http://192.168.1.10:8001/cluster/status
```

### Step 7: Configure Load Balancer (Recommended)

For production deployments, configure a load balancer to distribute traffic:

#### nginx Configuration

```nginx path=null start=null
upstream flybrowser_cluster {
    least_conn;
    server 192.168.1.10:8001;
    server 192.168.1.11:8002;
    server 192.168.1.12:8003;
}

server {
    listen 80;
    server_name flybrowser.example.com;

    location / {
        proxy_pass http://flybrowser_cluster;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 10s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://flybrowser_cluster;
        proxy_connect_timeout 5s;
        proxy_read_timeout 5s;
    }
}
```

#### HAProxy Configuration

```haproxy path=null start=null
frontend flybrowser_front
    bind *:80
    default_backend flybrowser_back

backend flybrowser_back
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    server node1 192.168.1.10:8001 check
    server node2 192.168.1.11:8002 check
    server node3 192.168.1.12:8003 check
```

## Docker Compose Cluster

For development or testing, deploy a 3-node cluster using Docker Compose:

```yaml path=null start=null
version: "3.8"

services:
  node1:
    image: flybrowser-server:latest
    container_name: flybrowser-node1
    command: >
      flybrowser-serve
        --cluster
        --node-id node1
        --host 0.0.0.0
        --port 8000
        --raft-port 4321
        --data-dir /data
    ports:
      - "8001:8000"
      - "4321:4321"
    volumes:
      - node1-data:/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - flybrowser-net

  node2:
    image: flybrowser-server:latest
    container_name: flybrowser-node2
    command: >
      flybrowser-serve
        --cluster
        --node-id node2
        --host 0.0.0.0
        --port 8000
        --raft-port 4321
        --data-dir /data
        --peers node1:4321
    ports:
      - "8002:8000"
      - "4322:4321"
    volumes:
      - node2-data:/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - node1
    networks:
      - flybrowser-net

  node3:
    image: flybrowser-server:latest
    container_name: flybrowser-node3
    command: >
      flybrowser-serve
        --cluster
        --node-id node3
        --host 0.0.0.0
        --port 8000
        --raft-port 4321
        --data-dir /data
        --peers node1:4321,node2:4321
    ports:
      - "8003:8000"
      - "4323:4321"
    volumes:
      - node3-data:/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - node1
      - node2
    networks:
      - flybrowser-net

  loadbalancer:
    image: nginx:alpine
    container_name: flybrowser-lb
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - node1
      - node2
      - node3
    networks:
      - flybrowser-net

networks:
  flybrowser-net:
    driver: bridge

volumes:
  node1-data:
  node2-data:
  node3-data:
```

Start the cluster:

```bash path=null start=null
docker-compose up -d
```

## Kubernetes Cluster Deployment

### StatefulSet Configuration

```yaml path=null start=null
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: flybrowser
spec:
  serviceName: flybrowser-headless
  replicas: 3
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
              name: http
            - containerPort: 4321
              name: raft
          env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: flybrowser-secrets
                  key: openai-api-key
          command:
            - /bin/sh
            - -c
            - |
              ORDINAL=${POD_NAME##*-}
              PEERS=""
              for i in $(seq 0 $((ORDINAL - 1))); do
                if [ -n "$PEERS" ]; then
                  PEERS="${PEERS},"
                fi
                PEERS="${PEERS}flybrowser-${i}.flybrowser-headless:4321"
              done
              
              flybrowser-serve \
                --cluster \
                --node-id ${POD_NAME} \
                --host 0.0.0.0 \
                --port 8000 \
                --raft-port 4321 \
                --data-dir /data \
                ${PEERS:+--peers $PEERS}
          volumeMounts:
            - name: data
              mountPath: /data
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
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: flybrowser-headless
spec:
  clusterIP: None
  selector:
    app: flybrowser
  ports:
    - port: 8000
      name: http
    - port: 4321
      name: raft
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
      targetPort: 8080
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash path=null start=null
kubectl apply -f flybrowser-statefulset.yaml
```

## Scaling the Cluster

### Adding Nodes

To add a new node to an existing cluster:

1. **Prepare the new node** with FlyBrowser installed

2. **Start the new node** with peers pointing to existing cluster members:

```bash path=null start=null
flybrowser-serve \
    --cluster \
    --node-id node4 \
    --host 0.0.0.0 \
    --port 8004 \
    --raft-port 4324 \
    --data-dir /var/lib/flybrowser \
    --peers node1:4321,node2:4322,node3:4323
```

3. **Verify the node joined**:

```bash path=null start=null
flybrowser-cluster nodes --endpoint http://192.168.1.10:8001
```

4. **Update the load balancer** to include the new node

### Removing Nodes

To gracefully remove a node:

1. **Drain the node** to move sessions:

```bash path=null start=null
flybrowser-admin nodes drain node3 --endpoint http://192.168.1.10:8001
```

2. **Wait for sessions to migrate** (monitor with cluster status)

3. **Stop the node**:

```bash path=null start=null
# On node3
systemctl stop flybrowser
```

4. **Update the load balancer** to remove the node

### Scaling in Kubernetes

```bash path=null start=null
# Scale to 5 nodes
kubectl scale statefulset flybrowser --replicas=5

# Scale back to 3 nodes
kubectl scale statefulset flybrowser --replicas=3
```

## Cluster Operations

### Checking Cluster Health

```bash path=null start=null
# Using CLI
flybrowser-cluster status --endpoint http://localhost:8001

# Using API
curl http://localhost:8001/cluster/status
```

### Viewing Node List

```bash path=null start=null
flybrowser-cluster nodes --endpoint http://localhost:8001
```

Output:

```
Cluster Nodes
=============
node1 (leader)
  Address: 192.168.1.10:4321
  HTTP: 192.168.1.10:8001
  Status: healthy
  Sessions: 5

node2 (follower)
  Address: 192.168.1.11:4322
  HTTP: 192.168.1.11:8002
  Status: healthy
  Sessions: 3

node3 (follower)
  Address: 192.168.1.12:4323
  HTTP: 192.168.1.12:8003
  Status: healthy
  Sessions: 4
```

### Viewing Session Distribution

```bash path=null start=null
flybrowser-cluster sessions --endpoint http://localhost:8001
```

### Forcing Leader Election

In rare cases, you may need to trigger a new leader election:

```bash path=null start=null
flybrowser-admin cluster step-down --endpoint http://localhost:8001
```

## Fault Tolerance

### Leader Failure

When the leader fails:

1. Remaining nodes detect the failure through heartbeat timeout
2. A new election is triggered automatically
3. One of the followers becomes the new leader
4. Clients are redirected to the new leader

**Recovery time**: Typically 1-5 seconds depending on configuration

### Follower Failure

When a follower fails:

1. The leader detects the failure
2. The node is marked as unavailable
3. Cluster continues operating with remaining nodes
4. When the node recovers, it automatically rejoins and syncs

### Network Partition

If the network partitions:

1. The partition with majority (quorum) continues operating
2. The minority partition becomes read-only
3. When the partition heals, the minority syncs with the majority

## Backup and Recovery

### Creating Backups

Create a backup from any node:

```bash path=null start=null
flybrowser-admin backup create \
    --output /backup/flybrowser-$(date +%Y%m%d).tar.gz \
    --endpoint http://localhost:8001
```

### Restoring from Backup

To restore a cluster from backup:

1. **Stop all nodes**

2. **Restore data on each node**:

```bash path=null start=null
flybrowser-admin backup restore \
    --input /backup/flybrowser-20240115.tar.gz \
    --data-dir /var/lib/flybrowser
```

3. **Start nodes normally**

## Monitoring

### Key Metrics

Monitor these metrics for cluster health:

- **Raft term**: Indicates leader changes
- **Commit index**: Shows replication progress
- **Log entries**: Raft log size
- **RPC latency**: Inter-node communication time
- **Session count per node**: Distribution of workload

### Prometheus Metrics

FlyBrowser exposes metrics at `/metrics`:

```bash path=null start=null
curl http://localhost:8001/metrics
```

Example metrics:

```
flybrowser_cluster_term 5
flybrowser_cluster_commit_index 1234
flybrowser_cluster_role{node="node1"} 1
flybrowser_sessions_active{node="node1"} 5
flybrowser_raft_rpc_duration_seconds_bucket{le="0.01"} 100
```

### Alerting

Configure alerts for:

- Leader changes (more than expected frequency)
- Node unavailability
- Replication lag
- Session count imbalance
- Disk space on data directories

## Troubleshooting

### Node Fails to Join Cluster

**Symptoms**: Node starts but doesn't join the cluster

**Possible causes**:
1. Network connectivity issues
2. Incorrect peer addresses
3. Firewall blocking Raft ports

**Resolution**:
```bash path=null start=null
# Check network connectivity
ping node1
telnet node1 5001

# Verify peer configuration
flybrowser-serve --cluster --node-id node2 --peers node1:5001 --debug

# Check firewall
iptables -L -n | grep 5001
```

### Split Brain

**Symptoms**: Multiple nodes claiming to be leader

**Possible causes**:
1. Network partition
2. Clock skew between nodes

**Resolution**:
1. Check network connectivity between all nodes
2. Verify NTP synchronization
3. If persistent, stop minority partition nodes and let majority stabilize

### Slow Replication

**Symptoms**: High replication lag, slow writes

**Possible causes**:
1. Network latency
2. Disk I/O bottleneck
3. Large log entries

**Resolution**:
1. Check network latency between nodes
2. Monitor disk I/O
3. Consider increasing batch size for replication

### Data Inconsistency

**Symptoms**: Different data on different nodes

**Possible causes**:
1. Incomplete recovery after failure
2. Bug (rare)

**Resolution**:
1. Identify the authoritative node (usually leader)
2. Stop affected nodes
3. Remove corrupted data directory
4. Restart node to sync from leader

## Configuration Reference

### Cluster-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster` | false | Enable cluster mode |
| `--node-id` | (auto) | Unique identifier for this node |
| `--raft-port` | 4321 | Port for Raft communication |
| `--peers` | (none) | Comma-separated list of peer addresses |
| `--data-dir` | ./data | Directory for persistent data |
| `--election-timeout` | 1000 | Election timeout in milliseconds |
| `--heartbeat-interval` | 100 | Heartbeat interval in milliseconds |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FLYBROWSER_NODE_ID` | Node identifier |
| `FLYBROWSER_RAFT_PORT` | Raft communication port |
| `FLYBROWSER_PEERS` | Comma-separated peer addresses |
| `FLYBROWSER_DATA_DIR` | Data directory path |

## Next Steps

- [SDK Reference](../reference/sdk.md) - Client SDK documentation
- [REST API Reference](../reference/api.md) - Complete API documentation
- [CLI Reference](../reference/cli.md) - Command-line tools documentation
- [Configuration Reference](../reference/configuration.md) - All configuration options
