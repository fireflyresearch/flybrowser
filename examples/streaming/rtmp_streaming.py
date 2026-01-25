"""
Example: RTMP Streaming to Twitch/YouTube

Demonstrates streaming browser sessions to Twitch, YouTube, or other RTMP platforms.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
- export TWITCH_STREAM_KEY="live_..." (or YOUTUBE_STREAM_KEY)
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def stream_to_twitch():
    """Stream to Twitch using RTMP."""
    stream_key = os.getenv("TWITCH_STREAM_KEY")
    
    if not stream_key:
        print("Error: TWITCH_STREAM_KEY environment variable not set")
        print("Get your stream key from: https://dashboard.twitch.tv/settings/stream")
        return
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Stream to Twitch
        stream = await browser.start_stream(
            protocol="rtmp",
            rtmp_url="rtmp://live.twitch.tv/app",
            rtmp_key=stream_key
        )
        
        print(f"Streaming to Twitch!")
        print(f"Stream URL: {stream.get('rtmp_url', 'N/A')}")
        print(f"Status: {stream.get('status', 'unknown')}")
        
        # Navigate and interact - this is what viewers will see
        await browser.goto("https://news.ycombinator.com")
        
        # Perform actions
        await browser.act("scroll down to see more posts")
        await asyncio.sleep(5)
        
        posts = await browser.extract("Get the top 5 post titles")
        print(f"\nExtracted posts: {posts.data}")
        
        # Continue streaming for a while
        print("\nStreaming for 30 seconds...")
        await asyncio.sleep(30)
        
        # Stop stream
        await browser.stop_stream()
        print("Stream stopped")


async def stream_to_youtube():
    """Stream to YouTube using RTMP."""
    stream_key = os.getenv("YOUTUBE_STREAM_KEY")
    
    if not stream_key:
        print("Error: YOUTUBE_STREAM_KEY environment variable not set")
        print("Get your stream key from: https://studio.youtube.com/")
        return
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Stream to YouTube
        stream = await browser.start_stream(
            protocol="rtmp",
            rtmp_url="rtmp://a.rtmp.youtube.com/live2",
            rtmp_key=stream_key,
            quality="high",
            codec="h264"
        )
        
        print(f"Streaming to YouTube!")
        print(f"Stream status: {stream.get('status', 'unknown')}")
        
        # Navigate to content
        await browser.goto("https://example.com")
        
        # Perform demo actions
        await browser.act("scroll through the page")
        await asyncio.sleep(10)
        
        # Continue streaming
        print("\nStreaming for 60 seconds...")
        await asyncio.sleep(60)
        
        # Stop stream
        await browser.stop_stream()
        print("Stream stopped")


async def stream_with_monitoring():
    """Stream with real-time monitoring and health checks."""
    stream_key = os.getenv("TWITCH_STREAM_KEY")
    
    if not stream_key:
        print("Using demo mode (no actual stream)")
        stream_key = "demo_key_only"
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Start stream
        stream = await browser.start_stream(
            protocol="rtmp",
            rtmp_url="rtmp://live.twitch.tv/app",
            rtmp_key=stream_key
        )
        
        print("Stream started with monitoring")
        
        # Navigate to content
        await browser.goto("https://news.ycombinator.com")
        
        # Monitor stream health periodically
        for i in range(5):
            await asyncio.sleep(5)
            
            status = await browser.get_stream_status()
            if status.get('active'):
                stream_data = status.get('status', {})
                metrics = stream_data.get('metrics', {})
                
                print(f"\n[Check {i+1}] Stream Health:")
                print(f"  Active: {status.get('active')}")
                print(f"  FPS: {metrics.get('current_fps', 0):.1f}")
                print(f"  Bitrate: {metrics.get('current_bitrate', 0):.0f} bps")
                print(f"  Health: {stream_data.get('health', 'unknown')}")
                
                # Perform an action
                await browser.act("scroll down a bit")
        
        # Stop stream
        await browser.stop_stream()
        print("\nStream stopped")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("RTMP Streaming Examples")
    print("=" * 60)
    
    # Check which stream keys are available
    has_twitch = bool(os.getenv("TWITCH_STREAM_KEY"))
    has_youtube = bool(os.getenv("YOUTUBE_STREAM_KEY"))
    
    if has_twitch:
        print("\n--- Streaming to Twitch ---")
        await stream_to_twitch()
    elif has_youtube:
        print("\n--- Streaming to YouTube ---")
        await stream_to_youtube()
    else:
        print("\n--- Stream with Monitoring (Demo) ---")
        print("Note: Set TWITCH_STREAM_KEY or YOUTUBE_STREAM_KEY to stream to platforms")
        await stream_with_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
