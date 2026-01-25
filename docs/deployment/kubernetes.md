# Kubernetes Deployment

Deploy FlyBrowser on Kubernetes for production-grade orchestration, scaling, and management.

## Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.x (optional but recommended)

### Basic Deployment

```bash
# Apply manifests
kubectl apply -f https://raw.githubusercontent.com/flybrowser/flybrowser/main/deploy/kubernetes/deployment.yaml

# Verify
kubectl get pods -l app=flybrowser
kubectl get svc flybrowser
```

## Deployment Manifests

### Basic Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flybrowser
  labels:
    app: flybrowser
spec:
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
        image: flybrowser/flybrowser:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: flybrowser-secrets
              key: openai-api-key
        - name: FLYBROWSER_LOG_LEVEL
          value: "INFO"
        - name: FLYBROWSER_POOL__MAX_SIZE
          value: "10"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: recordings
          mountPath: /app/recordings
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
      - name: recordings
        persistentVolumeClaim:
          claimName: flybrowser-recordings
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: flybrowser
  labels:
    app: flybrowser
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  selector:
    app: flybrowser
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: flybrowser
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - flybrowser.example.com
    secretName: flybrowser-tls
  rules:
  - host: flybrowser.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: flybrowser
            port:
              number: 8000
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: flybrowser-secrets
type: Opaque
stringData:
  openai-api-key: "sk-..."
  anthropic-api-key: "sk-ant-..."
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flybrowser-config
data:
  FLYBROWSER_LOG_LEVEL: "INFO"
  FLYBROWSER_POOL__MAX_SIZE: "10"
  FLYBROWSER_POOL__HEADLESS: "true"
  FLYBROWSER_MAX_SESSIONS: "100"
```

### PersistentVolumeClaim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flybrowser-recordings
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs
```

## Helm Chart

### Install via Helm

```bash
# Add repository
helm repo add flybrowser https://charts.flybrowser.dev
helm repo update

# Install
helm install flybrowser flybrowser/flybrowser \
  --namespace flybrowser \
  --create-namespace \
  --set openaiApiKey="sk-..." \
  --set replicaCount=3

# Upgrade
helm upgrade flybrowser flybrowser/flybrowser \
  --set replicaCount=5

# Uninstall
helm uninstall flybrowser -n flybrowser
```

### Helm Values

```yaml
# values.yaml
replicaCount: 3

image:
  repository: flybrowser/flybrowser
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: flybrowser.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: flybrowser-tls
      hosts:
        - flybrowser.example.com

resources:
  requests:
    memory: 2Gi
    cpu: 1
  limits:
    memory: 4Gi
    cpu: 2

secrets:
  openaiApiKey: ""
  anthropicApiKey: ""

config:
  logLevel: INFO
  poolMaxSize: 10
  maxSessions: 100

persistence:
  recordings:
    enabled: true
    size: 100Gi
    storageClass: nfs

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilization: 70
  targetMemoryUtilization: 80
```

## Cluster Mode (HA)

### StatefulSet for Cluster Mode

```yaml
# statefulset.yaml
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
        image: flybrowser/flybrowser:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 4321
          name: raft
        env:
        - name: FLYBROWSER_CLUSTER_ENABLED
          value: "true"
        - name: FLYBROWSER_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: FLYBROWSER_PORT
          value: "8000"
        - name: FLYBROWSER_RAFT_PORT
          value: "4321"
        - name: FLYBROWSER_CLUSTER_DISCOVERY_METHOD
          value: "kubernetes"
        - name: FLYBROWSER_CLUSTER_K8S_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: FLYBROWSER_CLUSTER_K8S_SERVICE
          value: "flybrowser-headless"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: flybrowser-secrets
              key: openai-api-key
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
# Headless service for StatefulSet
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
```

## Autoscaling

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flybrowser
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flybrowser
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### Pod Disruption Budget

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: flybrowser
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: flybrowser
```

## Monitoring

### ServiceMonitor (Prometheus Operator)

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: flybrowser
  labels:
    app: flybrowser
spec:
  selector:
    matchLabels:
      app: flybrowser
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### PrometheusRule (Alerts)

```yaml
# prometheusrule.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: flybrowser-alerts
spec:
  groups:
  - name: flybrowser
    rules:
    - alert: FlyBrowserDown
      expr: up{job="flybrowser"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "FlyBrowser instance is down"
    - alert: FlyBrowserHighMemory
      expr: container_memory_usage_bytes{container="flybrowser"} / container_spec_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "FlyBrowser memory usage above 90%"
```

## Network Policies

```yaml
# networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: flybrowser
spec:
  podSelector:
    matchLabels:
      app: flybrowser
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: flybrowser
    ports:
    - protocol: TCP
      port: 4321
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 80
```

## Security

### Pod Security Policy

```yaml
# podsecuritypolicy.yaml (pre-K8s 1.25)
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: flybrowser
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
  - configMap
  - emptyDir
  - persistentVolumeClaim
  - secret
```

### Security Context

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: flybrowser
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL
      readOnlyRootFilesystem: true
```

## Troubleshooting

### Pod Stuck in Pending

```bash
# Check events
kubectl describe pod flybrowser-xxx

# Check node resources
kubectl describe nodes

# Check PVC status
kubectl get pvc
```

### Pod CrashLoopBackOff

```bash
# Check logs
kubectl logs flybrowser-xxx --previous

# Check events
kubectl describe pod flybrowser-xxx

# Exec into running pod
kubectl exec -it flybrowser-xxx -- bash
```

### Network Issues

```bash
# Test DNS
kubectl run -it --rm debug --image=busybox -- nslookup flybrowser

# Test connectivity
kubectl run -it --rm debug --image=curlimages/curl -- curl http://flybrowser:8000/health
```

## See Also

- [Docker Deployment](docker.md) - Container basics
- [Cluster Deployment](cluster.md) - HA configuration
- [Configuration Reference](../reference/configuration.md) - All options
