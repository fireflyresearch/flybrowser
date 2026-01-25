# Stealth Mode

FlyBrowser provides state-of-the-art stealth capabilities to keep your automations running smoothly against bot detection, CAPTCHAs, and IP blocks.

## Overview

Modern websites employ sophisticated bot detection systems. FlyBrowser's stealth layer provides:

- **Fingerprint Generation**: Dynamic browser fingerprints that pass detection
- **CAPTCHA Solving**: Automatic CAPTCHA solving with multiple providers
- **Proxy Network**: Intelligent proxy selection with residential IPs

## Quick Start

```python
from flybrowser import FlyBrowser
from flybrowser.stealth import StealthConfig

async with FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    stealth_config=StealthConfig(
        fingerprint_enabled=True,
        captcha_enabled=True,
        captcha_provider="2captcha",
        captcha_api_key="your-api-key",
        proxy_enabled=True,
    ),
) as browser:
    await browser.goto("https://protected-site.com")
    result = await browser.extract("Get the data")
```

## Fingerprint Generation

Browser fingerprinting is a technique used by websites to identify and track users. FlyBrowser generates consistent, realistic fingerprints that pass detection.

### Basic Usage

```python
from flybrowser.stealth import StealthConfig

# Auto-generate fingerprint
config = StealthConfig(fingerprint_enabled=True)

# Specify OS and browser
config = StealthConfig(
    fingerprint_enabled=True,
    os="macos",
    browser="chrome",
)

# Use profile string
config = StealthConfig(
    fingerprint="macos_chrome",  # Auto-enables fingerprint
)
```

### Fingerprint Components

FlyBrowser generates fingerprints including:

- **User Agent**: Browser and OS identification
- **Screen Resolution**: Display dimensions and color depth
- **Timezone**: Consistent with geolocation
- **Language**: Browser language preferences
- **WebGL**: Graphics card and renderer info
- **Canvas**: Canvas fingerprint hash
- **Audio**: Audio context fingerprint
- **Fonts**: Installed font detection evasion
- **Plugins**: Browser plugin simulation
- **Hardware**: CPU cores, device memory

### Advanced Fingerprint Control

```python
from flybrowser.stealth import FingerprintGenerator, FingerprintProfile

# Generate a custom profile
generator = FingerprintGenerator()
profile = generator.generate(
    os="windows",
    browser="chrome",
    geolocation="us-east",
)

# Use the profile
config = StealthConfig(fingerprint=profile)
```

### Geolocation Consistency

Fingerprints are automatically consistent with geolocation:

```python
config = StealthConfig(
    fingerprint_enabled=True,
    geolocation="japan",  # Timezone, language, locale all match
)
```

Available geolocations:
- `us-west`, `us-east`, `us-central`
- `uk`, `germany`, `france`, `spain`, `italy`
- `japan`, `korea`, `australia`, `brazil`, `india`

## CAPTCHA Solving

FlyBrowser automatically detects and solves CAPTCHAs during automation.

### Supported CAPTCHA Types

- **reCAPTCHA v2**: Checkbox and invisible
- **reCAPTCHA v3**: Score-based
- **hCaptcha**: Image challenges
- **Cloudflare Turnstile**: Bot verification
- **FunCaptcha**: Rotating puzzles
- **Image CAPTCHAs**: Text recognition

### CAPTCHA Providers

FlyBrowser integrates with leading CAPTCHA solving services:

| Provider | Website | Best For |
|----------|---------|----------|
| 2Captcha | 2captcha.com | Cost-effective, wide support |
| Anti-Captcha | anti-captcha.com | Fast solving, enterprise |
| CapSolver | capsolver.com | Modern CAPTCHAs, hCaptcha |

### Configuration

```python
from flybrowser.stealth import StealthConfig

# Simple configuration
config = StealthConfig(
    captcha_enabled=True,
    captcha_provider="2captcha",
    captcha_api_key="your-api-key",
    captcha_auto_solve=True,  # Solve when detected
)

# Advanced configuration
from flybrowser.stealth import CaptchaConfig

config = StealthConfig(
    captcha_solver=CaptchaConfig(
        provider="capsolver",
        api_key="your-api-key",
        auto_solve=True,
        max_retries=3,
        timeout_seconds=120,
        min_score=0.7,  # For reCAPTCHA v3
    ),
)
```

### ReAct Agent Integration

CAPTCHAs are automatically handled during agent execution:

```python
async with FlyBrowser(
    stealth_config=StealthConfig(
        captcha_enabled=True,
        captcha_provider="2captcha",
        captcha_api_key="your-key",
    ),
) as browser:
    # Agent automatically solves CAPTCHAs encountered
    result = await browser.agent(
        "Submit the contact form",
        context={"name": "John", "email": "john@example.com"},
    )
```

### Manual CAPTCHA Tools

The ReAct agent has access to CAPTCHA tools:

- `detect_captcha`: Detect CAPTCHA presence on page
- `solve_captcha`: Solve detected CAPTCHA
- `wait_for_captcha_resolved`: Wait for CAPTCHA completion

## Proxy Network

FlyBrowser's proxy network intelligently selects the best proxy for your target.

### Proxy Types

| Type | Description | Use Case |
|------|-------------|----------|
| Residential | Real user IPs | High-protection sites |
| Datacenter | Fast, stable IPs | Speed-critical tasks |
| Mobile | Mobile carrier IPs | Mobile-only content |
| ISP | Static residential | Long sessions |

### Supported Providers

FlyBrowser integrates with premium proxy providers:

| Provider | Type | Features |
|----------|------|----------|
| Bright Data | All types | Largest network, geo-targeting |
| Oxylabs | Residential, DC | Fast, reliable |
| Smartproxy | Residential | Cost-effective |

### Basic Configuration

```python
from flybrowser.stealth import StealthConfig

config = StealthConfig(
    proxy_enabled=True,
)
```

### Provider Configuration

```python
from flybrowser.stealth import StealthConfig, ProxyNetworkConfig, ProviderConfig

config = StealthConfig(
    proxy_network=ProxyNetworkConfig(
        providers=[
            ProviderConfig(
                provider="bright_data",
                username="your-username",
                password="your-password",
                zone="residential",
            ),
            ProviderConfig(
                provider="oxylabs",
                username="your-username",
                password="your-password",
            ),
        ],
        strategy="smart",  # Intelligent selection
        fallback_enabled=True,
        health_check_interval_seconds=60,
    ),
)
```

### Selection Strategies

| Strategy | Description |
|----------|-------------|
| `smart` | AI-powered selection based on target (default) |
| `round_robin` | Rotate through providers |
| `fastest` | Select lowest latency proxy |
| `geo_match` | Match proxy to target geo |
| `sticky` | Same IP for session duration |

### Geolocation Targeting

```python
config = StealthConfig(
    geolocation="germany",  # Proxy will be from Germany
    proxy_enabled=True,
)
```

### Fingerprint-Proxy Consistency

FlyBrowser ensures your fingerprint matches proxy location:

```python
# Fingerprint timezone, language, and locale
# automatically match the proxy's geolocation
config = StealthConfig(
    fingerprint_enabled=True,
    proxy_enabled=True,
    geolocation="japan",
)
```

## Human-Like Behavior

FlyBrowser simulates human behavior to avoid detection:

```python
config = StealthConfig(
    simulate_human=True,
    
    # Typing speed variation
    typing_delay_min=50,   # 50ms minimum
    typing_delay_max=150,  # 150ms maximum
    
    # Mouse movement
    simulate_mouse_movement=True,
    
    # Random delays between actions
    action_delay_min=100,
    action_delay_max=500,
)
```

## Full Configuration Example

```python
from flybrowser import FlyBrowser
from flybrowser.stealth import (
    StealthConfig,
    CaptchaConfig,
    ProxyNetworkConfig,
    ProviderConfig,
)

config = StealthConfig(
    # Fingerprint
    fingerprint_enabled=True,
    os="windows",
    browser="chrome",
    geolocation="us-west",
    
    # CAPTCHA
    captcha_solver=CaptchaConfig(
        provider="2captcha",
        api_key="your-key",
        auto_solve=True,
        max_retries=3,
    ),
    
    # Proxy
    proxy_network=ProxyNetworkConfig(
        providers=[
            ProviderConfig(
                provider="bright_data",
                username="user",
                password="pass",
                zone="residential",
            ),
        ],
        strategy="smart",
    ),
    
    # Human behavior
    simulate_human=True,
    typing_delay_min=30,
    typing_delay_max=100,
    simulate_mouse_movement=True,
)

async with FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    stealth_config=config,
) as browser:
    await browser.goto("https://heavily-protected-site.com")
    result = await browser.agent("Complete the signup process")
```

## Environment Variables

Configure stealth via environment variables:

```bash
# CAPTCHA
FLYBROWSER_CAPTCHA_PROVIDER=2captcha
FLYBROWSER_CAPTCHA_API_KEY=your-key
FLYBROWSER_CAPTCHA_AUTO_SOLVE=true

# Proxy providers
BRIGHT_DATA_USERNAME=user
BRIGHT_DATA_PASSWORD=pass
OXYLABS_USERNAME=user
OXYLABS_PASSWORD=pass
```

## Best Practices

### 1. Match Fingerprint to Proxy

Always ensure fingerprint geolocation matches proxy location:

```python
# Good - consistent
config = StealthConfig(
    fingerprint_enabled=True,
    proxy_enabled=True,
    geolocation="uk",  # Both match
)

# Bad - inconsistent (may be detected)
# Fingerprint says US, proxy from Japan
```

### 2. Use Appropriate Proxy Types

- **Residential**: Protected sites, account creation
- **Datacenter**: Scraping, high volume
- **Mobile**: Mobile apps, carrier-specific content

### 3. Enable Human Simulation

For heavily protected sites:

```python
config = StealthConfig(
    simulate_human=True,
    action_delay_min=200,
    action_delay_max=800,
)
```

### 4. Handle CAPTCHA Costs

CAPTCHA solving has per-solve costs. Configure wisely:

```python
config = StealthConfig(
    captcha_enabled=True,
    captcha_auto_solve=True,  # Only when needed
)
```

## Troubleshooting

### Detection Still Occurring

1. Enable human simulation
2. Use residential proxies
3. Increase action delays
4. Check fingerprint-proxy consistency

### CAPTCHA Solve Failures

1. Check API key balance
2. Increase timeout
3. Try different provider
4. Verify site URL is correct

### Proxy Connection Issues

1. Verify credentials
2. Check provider status
3. Enable fallback providers
4. Review health check logs

## See Also

- [SDK Reference](../reference/sdk.md) - Complete API
- [Configuration](../reference/configuration.md) - All options
- [PII Protection](pii-protection.md) - Secure credentials
