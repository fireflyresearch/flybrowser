#!/usr/bin/env python3
"""
Test script for FlyBrowser streaming functionality.

Tests basic streaming operations in embedded mode.
"""

import asyncio
import sys
from flybrowser import FlyBrowser


async def test_streaming():
    """Test basic streaming functionality."""
    print("🧪 Testing FlyBrowser Streaming...")
    print("=" * 60)
    
    # Create browser
    print("\n1. Creating browser instance...")
    browser = FlyBrowser(
        llm_provider="ollama",
        llm_model="llama2",
        log_level="info"  # Changed from debug to reduce noise
    )
    
    try:
        # Start browser
        print("2. Starting browser...")
        await browser.start()
        print("   ✅ Browser started")
        
        # Navigate
        print("\n3. Navigating to example.com...")
        await browser.goto("https://example.com")
        print("   ✅ Navigation complete")
        
        # Test streaming
        print("\n4. Starting HLS stream with H.265 codec...")
        stream = await browser.start_stream(
            protocol="hls",
            quality="medium",
            codec="h265"
        )
        
        print(f"   ✅ Stream started successfully!")
        print(f"   📺 Stream ID: {stream.get('stream_id')}")
        print(f"   🌐 Stream URL: {stream.get('stream_url')}")
        print(f"   📊 Status: {stream.get('status')}")
        print(f"   🎬 Protocol: {stream.get('protocol')}")
        print(f"   🎨 Codec: {stream.get('codec')}")
        
        # Wait a bit
        print("\n5. Streaming for 3 seconds...")
        await asyncio.sleep(3)
        
        # Get stream status
        print("\n6. Checking stream status...")
        status = await browser.get_stream_status()
        print(f"   Active: {status.get('active')}")
        if status.get('status'):
            print(f"   Stream ID: {status['status'].get('stream_id')}")
        
        # Stop stream
        print("\n7. Stopping stream...")
        stop_result = await browser.stop_stream()
        print(f"   ✅ Stream stopped")
        print(f"   Success: {stop_result.get('success')}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        print("\n8. Cleaning up...")
        try:
            await browser.stop()
            print("   ✅ Browser stopped")
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")


async def main():
    """Main entry point."""
    success = await test_streaming()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
