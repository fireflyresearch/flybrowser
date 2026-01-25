"""
Example: Basic Streaming

Demonstrates live HLS/DASH streaming of browser sessions.
This matches the streaming examples from the README.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
import webbrowser as wb
from flybrowser import FlyBrowser


async def hls_streaming_example():
    """Demonstrate HLS streaming with embedded web player."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Start live HLS stream
        stream = await browser.start_stream(
            protocol="hls",      # or "dash", "rtmp"
            quality="medium",    # low_bandwidth, medium, high
            codec="h265"         # 40% bandwidth savings vs h264
        )
        
        print(f"Watch at: {stream['stream_url']}")
        print(f"Web player: {stream['player_url']}")
        print("Works in ALL modes: embedded, standalone, cluster")
        
        # Open embedded web player in browser (no external software needed)
        print("\nOpening web player in browser...")
        wb.open(stream['player_url'])
        
        # Navigate and interact while streaming
        await browser.goto("https://news.ycombinator.com")
        await browser.act("scroll down slowly")
        await asyncio.sleep(5)
        
        # Monitor stream health (nested structure - safe access required)
        status = await browser.get_stream_status()
        if status.get('active'):
            stream_data = status.get('status', {})  # First level of nesting
            metrics = stream_data.get('metrics', {})  # Second level for metrics
            print(f"\nStream Health:")
            print(f"  FPS: {metrics.get('current_fps', 0):.1f}")
            print(f"  Health: {stream_data.get('health', 'unknown')}")
            print(f"  Bitrate: {metrics.get('current_bitrate', 0):.0f} bps")
        
        # Stop stream
        await asyncio.sleep(10)
        await browser.stop_stream()
        print("\nStream stopped")


async def dash_streaming_example():
    """Demonstrate DASH streaming (alternative to HLS)."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Start DASH stream
        stream = await browser.start_stream(
            protocol="dash",
            quality="high",
            codec="h264"
        )
        
        print(f"DASH Stream URL: {stream['stream_url']}")
        print(f"Player URL: {stream['player_url']}")
        
        # Navigate to content
        await browser.goto("https://example.com")
        await asyncio.sleep(10)
        
        await browser.stop_stream()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Basic Streaming Examples")
    print("=" * 60)
    
    print("\n--- HLS Streaming with Web Player ---")
    await hls_streaming_example()
    
    # Uncomment to try DASH
    # print("\n--- DASH Streaming ---")
    # await dash_streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
