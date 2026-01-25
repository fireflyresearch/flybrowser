# Screenshots and Recording

FlyBrowser provides comprehensive screenshot and recording capabilities for debugging, documentation, and monitoring.

## Taking Screenshots

### Basic Screenshot

```python
import asyncio
import base64
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com")
        
        # Take a screenshot
        screenshot = await browser.screenshot()
        
        # Save to file
        image_data = base64.b64decode(screenshot["data_base64"])
        with open("screenshot.png", "wb") as f:
            f.write(image_data)

asyncio.run(main())
```

### Method Signature

```python
async def screenshot(
    full_page: bool = False,
    mask_pii: bool = True
) -> Dict[str, Any]
```

**Parameters:**

- `full_page` (bool, default: False) - Capture the full scrollable page instead of just the viewport
- `mask_pii` (bool, default: True) - Apply PII masking to the screenshot (note: this parameter is accepted but masking behavior depends on pii_masking_enabled setting)

### Returns

Dictionary containing:
- `screenshot_id` - Unique identifier from the Screenshot object
- `format` - Image format value (e.g., "png")
- `width` - Image width in pixels
- `height` - Image height in pixels
- `data_base64` - Base64-encoded image data
- `url` - Page URL when captured
- `timestamp` - Capture timestamp

## Full Page Screenshots

Capture the entire scrollable page:

```python
# Capture full page
screenshot = await browser.screenshot(full_page=True)

print(f"Full page size: {screenshot['width']}x{screenshot['height']}")
```

## PII Masking

Screenshots can automatically mask sensitive information:

```python
# With PII masking (default)
screenshot = await browser.screenshot(mask_pii=True)

# Without PII masking
screenshot = await browser.screenshot(mask_pii=False)
```

Masked PII types include:
- Credit card numbers
- Social security numbers
- Phone numbers
- Email addresses
- Passwords (in form fields)

## Saving Screenshots

### Save to File

```python
import base64

screenshot = await browser.screenshot()

# Decode and save
image_data = base64.b64decode(screenshot["data_base64"])
with open("screenshot.png", "wb") as f:
    f.write(image_data)
```

### Save with Timestamp

```python
from datetime import datetime

screenshot = await browser.screenshot()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"screenshot_{timestamp}.png"

image_data = base64.b64decode(screenshot["data_base64"])
with open(filename, "wb") as f:
    f.write(image_data)
```

## Session Recording

### Starting a Recording

```python
async with FlyBrowser(...) as browser:
    # Start recording
    await browser.start_recording()
    
    # Perform actions (all will be recorded)
    await browser.goto("https://example.com")
    await browser.act("scroll down to see more content")
    await browser.act("click the first article link")
    
    # Stop and get recording
    recording = await browser.stop_recording()
    
    print(f"Recording ID: {recording.get('session_id')}")
```

### Recording Output

The recording contains:
- Screenshots at each step
- Video (if supported)
- Session metadata

```python
recording = await browser.stop_recording()

# Access screenshots
for screenshot in recording.get("screenshots", []):
    print(f"Screenshot at {screenshot.get('timestamp')}")

# Access video
video_data = recording.get("video")
if video_data:
    with open("recording.webm", "wb") as f:
        f.write(base64.b64decode(video_data))
```

## Debugging with Screenshots

### Screenshot on Error

```python
async def operation_with_debug(browser, operation):
    try:
        return await operation()
    except Exception as e:
        # Save debug screenshot
        screenshot = await browser.screenshot(full_page=True)
        image_data = base64.b64decode(screenshot["data_base64"])
        with open("error_screenshot.png", "wb") as f:
            f.write(image_data)
        raise
```

### Screenshot Before and After

```python
async def documented_action(browser, action):
    # Before screenshot
    before = await browser.screenshot()
    save_screenshot(before, "before.png")
    
    # Perform action
    result = await browser.act(action)
    
    # After screenshot
    after = await browser.screenshot()
    save_screenshot(after, "after.png")
    
    return result

def save_screenshot(screenshot, filename):
    image_data = base64.b64decode(screenshot["data_base64"])
    with open(filename, "wb") as f:
        f.write(image_data)
```

## Live Streaming

### Start a Stream

```python
# Start HLS stream
stream = await browser.start_stream(
    protocol="hls",
    quality="high"
)

print(f"Stream URL: {stream.get('stream_url')}")
```

### Stream Options

```python
stream = await browser.start_stream(
    protocol="hls",      # hls or rtmp
    quality="high",      # low, medium, high
    codec="h264",        # h264 or vp9
    width=1920,          # Custom width
    height=1080,         # Custom height
    frame_rate=30,       # Frames per second
)
```

### RTMP Streaming

For streaming to platforms like Twitch or YouTube:

```python
stream = await browser.start_stream(
    protocol="rtmp",
    rtmp_url="rtmp://live.twitch.tv/app",
    rtmp_key="your_stream_key"
)
```

### Stop Streaming

```python
await browser.stop_stream()
```

### Check Stream Status

```python
status = await browser.get_stream_status()
print(f"Streaming: {status.get('is_streaming')}")
print(f"Duration: {status.get('duration_seconds')}s")
```

## Best Practices

### Use Full Page for Documentation

```python
# For documentation purposes
screenshot = await browser.screenshot(full_page=True)
```

### Always Mask PII in Production

```python
# Always mask in production
screenshot = await browser.screenshot(mask_pii=True)
```

### Organize Screenshots

```python
import os
from datetime import datetime

async def save_organized_screenshot(browser, category, name):
    # Create directory structure
    date_dir = datetime.now().strftime("%Y-%m-%d")
    dir_path = f"screenshots/{category}/{date_dir}"
    os.makedirs(dir_path, exist_ok=True)
    
    # Take and save screenshot
    screenshot = await browser.screenshot()
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{dir_path}/{name}_{timestamp}.png"
    
    image_data = base64.b64decode(screenshot["data_base64"])
    with open(filename, "wb") as f:
        f.write(image_data)
    
    return filename
```

### Record Critical Workflows

```python
async def recorded_checkout(browser, order_data):
    await browser.start_recording()
    
    try:
        await browser.goto("https://shop.example.com/cart")
        await browser.act("click Checkout")
        # ... complete checkout
        return await browser.stop_recording()
    except Exception:
        # Still get partial recording on error
        return await browser.stop_recording()
```

## Related Features

- [PII Masking](pii.md) - Data protection features
- [Error Handling Guide](../guides/error-handling.md) - Debug screenshots

## See Also

- [SDK Reference](../reference/sdk.md) - Complete API documentation
