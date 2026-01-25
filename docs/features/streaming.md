# Streaming

FlyBrowser supports live streaming of browser sessions for monitoring, collaboration, and debugging purposes.

## Overview

Streaming enables real-time observation of browser automation in progress. This is useful for:

- Remote debugging and troubleshooting
- Team collaboration on automation development
- Monitoring long-running workflows
- Recording automation sessions for review
- Broadcasting to external platforms

## Starting a Stream

### Basic Stream

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        # Start HLS stream
        stream = await browser.start_stream()
        
        print(f"Stream URL: {stream.get('stream_url')}")
        print(f"Player URL: {stream.get('player_url')}")
        
        # Perform actions (viewers see these in real-time)
        await browser.goto("https://example.com")
        await browser.act("click the navigation menu")
        
        # Stop streaming when done
        await browser.stop_stream()

asyncio.run(main())
```

### Method Signature

```python
async def start_stream(
    protocol: str = "hls",
    quality: str = "high",
    codec: str = "h264",
    width: Optional[int] = None,
    height: Optional[int] = None,
    frame_rate: Optional[int] = None,
    rtmp_url: Optional[str] = None,
    rtmp_key: Optional[str] = None,
) -> Dict[str, Any]
```

**Parameters:**

- `protocol` (str, default: "hls") - Streaming protocol. Options: "hls", "dash", "rtmp"
- `quality` (str, default: "high") - Quality profile preset (see Quality Profiles below)
- `codec` (str, default: "h264") - Video codec. Options: "h264", "h265", "vp9"
- `width` (int, optional) - Video width in pixels. If None, uses profile default
- `height` (int, optional) - Video height in pixels. If None, uses profile default
- `frame_rate` (int, optional) - Frames per second. If None, uses profile default
- `rtmp_url` (str, optional) - RTMP destination URL (required for RTMP protocol)
- `rtmp_key` (str, optional) - RTMP stream key

### Returns

Dictionary containing:
- `stream_id` - Unique stream identifier
- `session_id` - Session ID
- `protocol` - Active protocol ("hls", "dash", or "rtmp")
- `quality` - Quality profile used
- `codec` - Video codec
- `status` - Stream status
- `hls_url` - HLS playlist URL (for HLS/DASH)
- `dash_url` - DASH manifest URL (for DASH)
- `rtmp_url` - RTMP URL (for RTMP)
- `player_url` - URL to embedded web player
- `stream_url` - Primary stream URL (hls_url, dash_url, or rtmp_url)
- `local_server_port` - Port of local HTTP server (embedded mode)

## Streaming Protocols

### HLS Streaming

HLS (HTTP Live Streaming) is the default protocol. It works in most browsers and provides adaptive bitrate streaming.

```python
stream = await browser.start_stream(protocol="hls")

# Access HLS stream URL
hls_url = stream.get("hls_url")

# Access embedded web player
player_url = stream.get("player_url")
print(f"Open in browser: {player_url}")
```

To view an HLS stream:
- Use the embedded player URL (`player_url`) - works in any browser
- Open the HLS URL in VLC, Safari, or Chrome with extensions
- Embed in a web page using hls.js

### DASH Streaming

DASH (Dynamic Adaptive Streaming over HTTP) provides similar functionality to HLS:

```python
stream = await browser.start_stream(protocol="dash")
dash_url = stream.get("dash_url")
```

### RTMP Streaming

RTMP (Real-Time Messaging Protocol) is used for streaming to platforms like Twitch, YouTube, or custom servers.

```python
stream = await browser.start_stream(
    protocol="rtmp",
    rtmp_url="rtmp://live.twitch.tv/app",
    rtmp_key="your_stream_key"
)
```

Supported platforms:
- Twitch: `rtmp://live.twitch.tv/app`
- YouTube Live: `rtmp://a.rtmp.youtube.com/live2`
- Facebook Live: `rtmps://live-api-s.facebook.com:443/rtmp/`
- Custom RTMP servers

## Quality Profiles

FlyBrowser provides several quality presets optimized for different use cases:

### Available Profiles

```python
# Ultra low latency - for real-time monitoring
stream = await browser.start_stream(quality="ultra_low_latency")

# Low bandwidth - 500kbps, for slow connections
stream = await browser.start_stream(quality="low_bandwidth")

# Medium - 1.5Mbps, balanced
stream = await browser.start_stream(quality="medium")

# High - 3Mbps, high quality (default)
stream = await browser.start_stream(quality="high")

# Ultra high - 6Mbps, maximum quality
stream = await browser.start_stream(quality="ultra_high")

# Local high - 12Mbps, optimized for localhost/LAN
stream = await browser.start_stream(quality="local_high")

# Local 4K - 25Mbps, 4K quality for localhost
stream = await browser.start_stream(quality="local_4k")

# Studio - 50Mbps, near-lossless production quality
stream = await browser.start_stream(quality="studio")

# Lossless - maximum quality, large file size
stream = await browser.start_stream(quality="lossless")
```

### Custom Resolution

Override the profile defaults with custom resolution:

```python
stream = await browser.start_stream(
    quality="high",
    width=1920,
    height=1080,
    frame_rate=60
)
```

### 4K Streaming Example

```python
# 4K streaming for localhost
stream = await browser.start_stream(
    quality="local_4k",
    width=3840,
    height=2160
)
```

### Codec Selection

```python
# H.264 (widely compatible, default)
stream = await browser.start_stream(codec="h264")

# H.265/HEVC (better compression, less compatible)
stream = await browser.start_stream(codec="h265")

# VP9 (good compression, requires compatible player)
stream = await browser.start_stream(codec="vp9")
```

## Managing Streams

### Stop Streaming

```python
await browser.stop_stream()
```

### Check Stream Status

```python
status = await browser.get_stream_status()

print(f"Active: {status.get('active')}")
print(f"Stream ID: {status.get('stream_id')}")
```

### Status Response

For embedded mode:

```python
{
    "stream_id": "stream_abc123",
    "active": True,
    "status": {
        # Full stream info dict
    }
}
```

If no stream is active:

```python
{
    "active": False,
    "error": "No active stream"
}
```

## Use Cases

### Remote Debugging

Share a live view of automation for debugging:

```python
async def debug_workflow(browser, task):
    # Start stream for remote debugging
    stream = await browser.start_stream(quality="high")
    
    print(f"Share this URL with your team: {stream['stream_url']}")
    
    try:
        # Execute workflow (team can watch live)
        result = await browser.agent(task)
        return result
    finally:
        await browser.stop_stream()
```

### Monitoring Dashboard

Stream automation status to a monitoring dashboard:

```python
async def monitored_automation(browser, tasks):
    # Start stream for monitoring
    stream = await browser.start_stream(
        protocol="hls",
        quality="medium"
    )
    
    # Send stream URL to monitoring system
    await notify_monitoring_system(stream["stream_url"])
    
    # Execute tasks
    for task in tasks:
        await browser.agent(task)
    
    await browser.stop_stream()
```

### Broadcasting Sessions

Stream automation to an external platform:

```python
async def broadcast_demo(browser):
    # Stream to YouTube
    stream = await browser.start_stream(
        protocol="rtmp",
        rtmp_url="rtmp://a.rtmp.youtube.com/live2",
        rtmp_key="your-stream-key",
        quality="high"
    )
    
    # Perform demonstration
    await browser.goto("https://demo.example.com")
    await browser.agent("demonstrate the key features of the application")
    
    await browser.stop_stream()
```

## Combining with Recording

Streaming and recording can be used together:

```python
async def stream_and_record(browser, task):
    # Start both streaming and recording
    stream = await browser.start_stream()
    await browser.start_recording()
    
    print(f"Live at: {stream['stream_url']}")
    
    try:
        result = await browser.agent(task)
        return result
    finally:
        # Save recording and stop stream
        recording = await browser.stop_recording()
        await browser.stop_stream()
        
        # Recording is available for later review
        save_recording(recording)
```

## Performance Considerations

### Quality Profile Bandwidth Guidelines

- `ultra_low_latency` - Minimal bandwidth, real-time monitoring
- `low_bandwidth` - ~500 kbps, slow connections
- `medium` - ~1.5 Mbps, balanced
- `high` - ~3 Mbps, high quality (default)
- `ultra_high` - ~6 Mbps, maximum quality
- `local_high` - ~12 Mbps, localhost/LAN
- `local_4k` - ~25 Mbps, 4K localhost
- `studio` - ~50 Mbps, production quality

### Resource Usage

Streaming adds some overhead:
- CPU: 5-15% additional usage for video encoding (FFmpeg)
- Memory: ~100-200 MB for stream buffer
- Network: Continuous upload bandwidth based on quality profile

### Optimization Tips

```python
# For low-bandwidth environments
stream = await browser.start_stream(
    quality="low_bandwidth",
    frame_rate=15
)

# For local network streaming with high quality
stream = await browser.start_stream(
    quality="local_high",
    frame_rate=60
)

# Ultra-low latency for real-time monitoring
stream = await browser.start_stream(
    quality="ultra_low_latency"
)
```

## Error Handling

```python
async def safe_stream(browser, task):
    try:
        stream = await browser.start_stream()
        
        if not stream.get("stream_url"):
            print("Failed to start stream")
            return
        
        result = await browser.agent(task)
        return result
        
    except Exception as e:
        print(f"Streaming error: {e}")
        # Ensure stream is stopped on error
        try:
            await browser.stop_stream()
        except:
            pass
        raise
    finally:
        await browser.stop_stream()
```

## Best Practices

### Always Stop Streams

```python
async with FlyBrowser(...) as browser:
    stream = await browser.start_stream()
    try:
        # Perform automation
        pass
    finally:
        await browser.stop_stream()
```

### Use Appropriate Quality

- Local debugging: high quality
- Remote teams: medium quality
- Production monitoring: low quality

### Secure Stream Keys

```python
import os

# Load stream key from environment
rtmp_key = os.environ.get("RTMP_STREAM_KEY")

stream = await browser.start_stream(
    protocol="rtmp",
    rtmp_url="rtmp://live.twitch.tv/app",
    rtmp_key=rtmp_key
)
```

### Monitor Stream Health

```python
import asyncio

async def monitor_stream_health(browser):
    while True:
        status = await browser.get_stream_status()
        
        if not status.get("is_streaming"):
            print("Stream ended")
            break
        
        if status.get("dropped_frames", 0) > 100:
            print("Warning: High frame drop rate")
        
        await asyncio.sleep(5)
```

## Related Features

- [Screenshots and Recording](screenshots.md) - Capture capabilities
- [Agent Mode](agent.md) - Autonomous task execution

## See Also

- [SDK Reference](../reference/sdk.md) - Complete API documentation
