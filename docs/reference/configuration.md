# Configuration Reference

FlyBrowser can be configured through environment variables, configuration files, or programmatically through the SDK.

## Configuration Methods

### 1. Environment Variables

Environment variables are the primary configuration method. Variables are prefixed with `FLYBROWSER_` and use uppercase.

```bash
export FLYBROWSER_HOST=0.0.0.0
export FLYBROWSER_PORT=8000
export FLYBROWSER_LOG_LEVEL=INFO
```

### 2. Configuration Files

Configuration files in YAML or JSON format:

```yaml
# config.yaml
host: "0.0.0.0"
port: 8000
log_level: "INFO"
pool:
  max_size: 20
  headless: true
```

Load with:
```python
from flybrowser.service.config import load_config_from_file
config = load_config_from_file("config.yaml")
```

### 3. SDK Configuration

Configure directly when creating FlyBrowser:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4o",
    api_key="sk-...",
    headless=True,
    speed_preset="balanced",
    log_verbosity="normal",
)
```

## Environment Variables Reference

### Service Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_HOST` | `0.0.0.0` | Service host address |
| `FLYBROWSER_PORT` | `8000` | Service port |
| `FLYBROWSER_WORKERS` | `1` | Number of worker processes |
| `FLYBROWSER_ENV` | `development` | Environment name |
| `FLYBROWSER_DEBUG` | `false` | Enable debug mode |
| `FLYBROWSER_LOG_LEVEL` | `INFO` | Logging level |
| `FLYBROWSER_LOG_FORMAT` | `json` | Log format: json or human |
| `FLYBROWSER_LOG_VERBOSITY` | `normal` | Execution verbosity: silent, minimal, normal, verbose, debug |

### Session Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_MAX_SESSIONS` | `100` | Maximum concurrent sessions |
| `FLYBROWSER_SESSION_TIMEOUT` | `3600` | Session timeout in seconds |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_LLM_PROVIDER` | `openai` | Default LLM provider |
| `FLYBROWSER_LLM_MODEL` | (provider default) | Default LLM model |
| `OPENAI_API_KEY` | | OpenAI API key |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `GOOGLE_API_KEY` | | Google Gemini API key |

### Rate Limit Retry Configuration

These variables control automatic retry behavior for HTTP 429 rate limit errors.
The retry applies at the framework level (`FireflyAgent.run()`) with exponential
backoff and jitter.

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREFLY_GENAI_QUOTA_ENABLED` | `true` | Enable adaptive backoff and quota management |
| `FIREFLY_GENAI_RATE_LIMIT_MAX_RETRIES` | `3` | Maximum retry attempts for 429 errors |
| `FIREFLY_GENAI_RATE_LIMIT_BASE_DELAY` | `1.0` | Base delay (seconds) for exponential backoff |
| `FIREFLY_GENAI_RATE_LIMIT_MAX_DELAY` | `60.0` | Maximum delay (seconds) between retries |

SDK-level configuration:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    max_retries=5,           # Retry up to 5 times on 429
    retry_base_delay=2.0,    # Start with 2s delay
    rate_limit_max_delay=120.0,  # Cap at 2 minutes
)
```

### Browser Pool Configuration

Nested variables use double underscore (`__`) as separator.

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_POOL__MIN_SIZE` | `1` | Minimum pool size |
| `FLYBROWSER_POOL__MAX_SIZE` | `10` | Maximum pool size |
| `FLYBROWSER_POOL__IDLE_TIMEOUT_SECONDS` | `300` | Idle browser timeout |
| `FLYBROWSER_POOL__MAX_SESSION_AGE_SECONDS` | `3600` | Maximum session age |
| `FLYBROWSER_POOL__STARTUP_TIMEOUT_SECONDS` | `30` | Browser startup timeout |
| `FLYBROWSER_POOL__HEADLESS` | `true` | Run browsers in headless mode |
| `FLYBROWSER_POOL__BROWSER_TYPE` | `chromium` | Default browser type |

### Recording Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_RECORDING_ENABLED` | `true` | Enable recording features |
| `FLYBROWSER_RECORDING_OUTPUT_DIR` | `./recordings` | Recording output directory |
| `FLYBROWSER_RECORDING_STORAGE` | `local` | Storage backend: local, s3, shared |
| `FLYBROWSER_RECORDING_RETENTION_DAYS` | `7` | Recording retention period |

### FFmpeg Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_FFMPEG_PATH` | (auto-detect) | Path to ffmpeg binary |
| `FLYBROWSER_FFMPEG_ENABLE_HW_ACCEL` | `true` | Enable hardware acceleration |

### Streaming Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_STREAMING_ENABLED` | `true` | Enable streaming features |
| `FLYBROWSER_STREAMING_BASE_URL` | (auto) | Base URL for HLS/DASH streams |

### S3 Storage Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_S3_BUCKET` | | S3 bucket name |
| `FLYBROWSER_S3_REGION` | `us-east-1` | S3 region |
| `FLYBROWSER_S3_ENDPOINT_URL` | | S3 endpoint URL (for MinIO) |
| `FLYBROWSER_S3_ACCESS_KEY` | | S3 access key |
| `FLYBROWSER_S3_SECRET_KEY` | | S3 secret key |
| `FLYBROWSER_S3_PREFIX` | `recordings/` | S3 key prefix |

### Cluster Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_DEPLOYMENT_MODE` | `standalone` | Mode: standalone or cluster |
| `FLYBROWSER_CLUSTER__ENABLED` | `false` | Enable cluster mode |
| `FLYBROWSER_CLUSTER__MODE` | `ha` | Cluster mode: ha (Raft) or legacy |
| `FLYBROWSER_CLUSTER__PEERS` | | Comma-separated peer list |
| `FLYBROWSER_CLUSTER__DISCOVERY_METHOD` | `static` | Discovery: static, dns, kubernetes |
| `FLYBROWSER_CLUSTER__LOAD_BALANCING_STRATEGY` | `least_load` | Load balancing strategy |

### Raft Configuration (HA Cluster)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_CLUSTER__RAFT__BIND_HOST` | `0.0.0.0` | Raft bind host |
| `FLYBROWSER_CLUSTER__RAFT__BIND_PORT` | `4321` | Raft bind port |
| `FLYBROWSER_CLUSTER__RAFT__DATA_DIR` | `./data/raft` | Raft data directory |

### PII Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_PII_MASKING_ENABLED` | `true` | Enable PII masking |

### Search Configuration

Search providers require API keys set via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `SERPER_API_KEY` | | Serper.dev API key (recommended) |
| `GOOGLE_CUSTOM_SEARCH_API_KEY` | | Google Custom Search API key |
| `GOOGLE_CUSTOM_SEARCH_CX` | | Google Custom Search engine ID |
| `BING_SEARCH_API_KEY` | | Bing Web Search API key |

## SDK Configuration Options

### FlyBrowser Constructor

```python
FlyBrowser(
    # Connection mode
    endpoint: Optional[str] = None,  # Server endpoint (None = embedded)
    
    # LLM Configuration
    llm_provider: str = "openai",    # openai, anthropic, ollama, gemini
    llm_model: Optional[str] = None, # Model name (uses provider default)
    api_key: Optional[str] = None,   # API key for LLM provider
    vision_enabled: Optional[bool] = None,  # Override vision capability
    model_config: Optional[Dict] = None,    # Additional model config
    
    # Browser Configuration
    headless: bool = True,           # Run headless
    browser_type: str = "chromium",  # chromium, firefox, webkit
    
    # Features
    recording_enabled: bool = False, # Enable recording
    pii_masking_enabled: bool = True, # Enable PII masking
    
    # Performance
    timeout: float = 30.0,           # Request timeout (server mode)
    speed_preset: str = "balanced",  # fast, balanced, thorough
    
    # Logging
    pretty_logs: bool = True,        # Human-readable logs
    log_verbosity: str = "normal",   # silent, minimal, normal, verbose, debug
    
    # Agent Configuration
    agent_config: Optional[AgentConfig] = None,  # Custom agent config
    config_file: Optional[str] = None,           # Path to config file
    
    # Search Configuration
    search_provider: Optional[str] = None,       # serper, google, bing, auto
    search_api_key: Optional[str] = None,        # API key for search provider
)
```

### Speed Presets

| Preset | Description |
|--------|-------------|
| `fast` | Optimized for speed, shorter timeouts, fewer retries |
| `balanced` | Good balance of speed and reliability (default) |
| `thorough` | More thorough, longer timeouts for complex pages |

### Log Verbosity Levels

| Level | Description |
|-------|-------------|
| `silent` | Errors only, no LLM logging |
| `minimal` | Errors + warnings, no LLM logging |
| `normal` | Standard info logs, no LLM logging |
| `verbose` | Detailed execution + basic LLM timing |
| `debug` | Full technical details + detailed LLM prompts/responses |

## Agent Configuration

### AgentConfig

For advanced agent customization:

```python
from flybrowser.agents.config import AgentConfig

config = AgentConfig(
    # Execution limits
    max_iterations=50,
    timeout_seconds=1800.0,
    
    # Reasoning
    default_reasoning_strategy="react_standard",
    
    # Safety
    default_safety_level="standard",
    
    # LLM settings
    llm=LLMConfig(
        reasoning_temperature=0.4,
        reasoning_max_tokens=4096,
        enable_dynamic_tokens=True,
    ),
    
    # Obstacle detection
    obstacle_detector=ObstacleDetectorConfig(
        enabled=True,
        aggressive_mode=False,
    ),
    
    # Memory
    memory=MemoryConfig(
        max_entries=100,
        relevance_threshold=0.6,
    ),
    
    # Search providers
    search_providers=SearchProviderConfig(
        default_provider="auto",       # auto, serper, google, bing
        enable_ranking=True,
        ranking_weights={
            "bm25": 0.35,              # Keyword relevance
            "freshness": 0.20,         # Recency
            "domain_authority": 0.15,  # Source quality
            "position": 0.30,          # Original ranking
        },
        cache_ttl_seconds=300,
    ),
)

browser = FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    agent_config=config,
)
```

### Loading from YAML

```yaml
# agent_config.yaml
max_iterations: 100
timeout_seconds: 3600

llm:
  reasoning_temperature: 0.3
  reasoning_max_tokens: 8192
  enable_dynamic_tokens: true

obstacle_detector:
  enabled: true
  aggressive_mode: false
  min_confidence_threshold: 0.8

memory:
  max_entries: 200
  relevance_threshold: 0.7
```

```python
browser = FlyBrowser(
    llm_provider="openai",
    config_file="agent_config.yaml",
)
```

## Service Configuration (YAML)

Complete service configuration example:

```yaml
# service_config.yaml
host: "0.0.0.0"
port: 8000
workers: 4
env: "production"
debug: false
log_level: "INFO"

max_sessions: 50
session_timeout: 3600

deployment_mode: "standalone"

pool:
  min_size: 2
  max_size: 20
  idle_timeout_seconds: 300
  max_session_age_seconds: 3600
  headless: true
  browser_type: "chromium"

recording:
  enabled: true
  output_dir: "/var/flybrowser/recordings"
  storage: "local"
  retention_days: 14

streaming:
  enabled: true
  base_url: "https://stream.example.com"

pii_masking_enabled: true

cors_origins:
  - "https://app.example.com"
  - "https://admin.example.com"
```

## Environment-Specific Configuration

### Development

```bash
export FLYBROWSER_ENV=development
export FLYBROWSER_DEBUG=true
export FLYBROWSER_LOG_LEVEL=DEBUG
export FLYBROWSER_LOG_VERBOSITY=verbose
export FLYBROWSER_POOL__MAX_SIZE=5
```

### Production

```bash
export FLYBROWSER_ENV=production
export FLYBROWSER_DEBUG=false
export FLYBROWSER_LOG_LEVEL=WARNING
export FLYBROWSER_LOG_VERBOSITY=minimal
export FLYBROWSER_WORKERS=4
export FLYBROWSER_POOL__MAX_SIZE=50
```

### Cluster Mode

```bash
export FLYBROWSER_DEPLOYMENT_MODE=cluster
export FLYBROWSER_CLUSTER__ENABLED=true
export FLYBROWSER_CLUSTER__MODE=ha
export FLYBROWSER_CLUSTER__PEERS="node1:4321:8000,node2:4321:8000,node3:4321:8000"
export FLYBROWSER_CLUSTER__LOAD_BALANCING_STRATEGY=least_load
```

## Stealth Configuration

### StealthConfig

Configure fingerprint generation, CAPTCHA solving, and proxy network:

```python
from flybrowser.stealth import StealthConfig

config = StealthConfig(
    # Fingerprint
    fingerprint_enabled=True,
    os="windows",                    # windows, macos, linux
    browser="chrome",                # chrome, firefox, safari, edge
    geolocation="us-west",           # us-west, uk, germany, japan, etc.
    
    # CAPTCHA
    captcha_enabled=True,
    captcha_provider="2captcha",     # 2captcha, anticaptcha, capsolver
    captcha_api_key="your-key",
    captcha_auto_solve=True,
    
    # Proxy
    proxy_enabled=True,
    
    # Human behavior
    simulate_human=True,
    typing_delay_min=50,
    typing_delay_max=150,
    simulate_mouse_movement=True,
    action_delay_min=100,
    action_delay_max=500,
)
```

#### StealthConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fingerprint_enabled` | `bool` | `False` | Enable fingerprint generation |
| `fingerprint` | `str` | `"auto"` | Fingerprint profile: auto, macos_chrome, etc. |
| `os` | `str` | `None` | OS override: windows, macos, linux |
| `browser` | `str` | `None` | Browser override: chrome, firefox, safari |
| `geolocation` | `str` | `"us-west"` | Geolocation for consistency |
| `captcha_enabled` | `bool` | `False` | Enable CAPTCHA solving |
| `captcha_provider` | `str` | `None` | Provider: 2captcha, anticaptcha, capsolver |
| `captcha_api_key` | `str` | `None` | CAPTCHA provider API key |
| `captcha_auto_solve` | `bool` | `True` | Auto-solve during agent execution |
| `proxy_enabled` | `bool` | `False` | Enable proxy network |
| `stealth_mode` | `bool` | `True` | Enable anti-detection scripts |
| `simulate_human` | `bool` | `True` | Human-like behavior simulation |
| `typing_delay_min` | `int` | `50` | Min typing delay (ms) |
| `typing_delay_max` | `int` | `150` | Max typing delay (ms) |
| `simulate_mouse_movement` | `bool` | `True` | Mouse movement simulation |
| `action_delay_min` | `int` | `100` | Min delay between actions (ms) |
| `action_delay_max` | `int` | `500` | Max delay between actions (ms) |

### Stealth Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_CAPTCHA_PROVIDER` | | CAPTCHA provider |
| `FLYBROWSER_CAPTCHA_API_KEY` | | CAPTCHA API key |
| `FLYBROWSER_CAPTCHA_AUTO_SOLVE` | `true` | Auto-solve CAPTCHAs |
| `BRIGHT_DATA_USERNAME` | | Bright Data proxy username |
| `BRIGHT_DATA_PASSWORD` | | Bright Data proxy password |
| `OXYLABS_USERNAME` | | Oxylabs proxy username |
| `OXYLABS_PASSWORD` | | Oxylabs proxy password |

## Observability Configuration

### ObservabilityConfig

Configure command logging, source capture, and live view:

```python
from flybrowser.observability import ObservabilityConfig, StreamQuality, ControlMode

config = ObservabilityConfig(
    # Command logging
    enable_command_logging=True,
    log_llm_prompts=True,
    log_llm_responses=True,
    max_log_entries=10000,
    auto_export_logs=False,
    log_export_path="./logs/",
    
    # Source capture
    enable_source_capture=True,
    capture_resources=True,
    max_resource_size_bytes=5*1024*1024,
    max_snapshots=100,
    auto_capture_on_navigation=False,
    capture_har=True,
    
    # Live view
    enable_live_view=True,
    live_view_host="0.0.0.0",
    live_view_port=8765,
    live_view_quality=StreamQuality.HIGH,
    live_view_control_mode=ControlMode.VIEW_ONLY,
    live_view_require_auth=True,
    live_view_auth_token="secret-token",
    live_view_max_viewers=10,
)
```

#### ObservabilityConfig Options

**Command Logging:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_command_logging` | `bool` | `True` | Enable command logging |
| `log_llm_prompts` | `bool` | `False` | Log LLM prompts |
| `log_llm_responses` | `bool` | `False` | Log LLM responses |
| `max_log_entries` | `int` | `10000` | Max log entries to keep |
| `auto_export_logs` | `bool` | `False` | Auto-export on session end |
| `log_export_path` | `str` | `None` | Export path for logs |

**Source Capture:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_source_capture` | `bool` | `False` | Enable source capture |
| `capture_resources` | `bool` | `False` | Capture page resources |
| `max_resource_size_bytes` | `int` | `5MB` | Max resource size |
| `max_snapshots` | `int` | `100` | Max snapshots to keep |
| `auto_capture_on_navigation` | `bool` | `False` | Auto-capture on page load |
| `capture_har` | `bool` | `False` | Enable HAR logging |

**Live View:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_live_view` | `bool` | `False` | Enable live view |
| `live_view_host` | `str` | `"0.0.0.0"` | Server host |
| `live_view_port` | `int` | `8765` | Server port |
| `live_view_quality` | `StreamQuality` | `MEDIUM` | Stream quality |
| `live_view_control_mode` | `ControlMode` | `VIEW_ONLY` | Viewer control mode |
| `live_view_require_auth` | `bool` | `False` | Require authentication |
| `live_view_auth_token` | `str` | `None` | Auth token |
| `live_view_max_viewers` | `int` | `10` | Max concurrent viewers |

### Observability Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYBROWSER_OBSERVABILITY_LOGGING` | `true` | Enable command logging |
| `FLYBROWSER_OBSERVABILITY_LOG_LLM` | `false` | Log LLM prompts/responses |
| `FLYBROWSER_OBSERVABILITY_MAX_ENTRIES` | `10000` | Max log entries |
| `FLYBROWSER_OBSERVABILITY_CAPTURE` | `false` | Enable source capture |
| `FLYBROWSER_OBSERVABILITY_CAPTURE_HAR` | `false` | Enable HAR logging |
| `FLYBROWSER_LIVE_VIEW_ENABLED` | `false` | Enable live view |
| `FLYBROWSER_LIVE_VIEW_PORT` | `8765` | Live view port |
| `FLYBROWSER_LIVE_VIEW_AUTH` | `false` | Require authentication |
| `FLYBROWSER_LIVE_VIEW_TOKEN` | | Auth token |

## See Also

- [CLI Reference](cli.md) - Command-line configuration
- [Deployment Guide](../deployment/standalone.md) - Deployment options
- [Stealth Mode](../features/stealth.md) - Stealth feature guide
- [Observability](../features/observability.md) - Observability feature guide
