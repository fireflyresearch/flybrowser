# Copyright 2026 Firefly Software Solutions Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example: Basic HLS Streaming

Demonstrates live HLS streaming of a FlyBrowser session. Starts a stream
with h264 codec, navigates to real sites while streaming, monitors stream
status, and gracefully stops.

NOTE: start_stream / stop_stream may not work in headless mode or in
environments without display/media support. All streaming calls are
wrapped in try/except for graceful degradation.

Prerequisites:
    pip install flybrowser
    export ANTHROPIC_API_KEY="sk-ant-..."

Environment Variables:
    ANTHROPIC_API_KEY          - Required. Your LLM provider API key.
    FLYBROWSER_LLM_PROVIDER    - Optional. Defaults to "anthropic".
    FLYBROWSER_LLM_MODEL       - Optional. Defaults to "claude-sonnet-4-5-20250929".
"""

import asyncio
import os
import sys

from flybrowser import FlyBrowser


async def start_hls_stream(browser: FlyBrowser) -> dict | None:
    """Attempt to start an HLS stream. Returns stream info or None on failure."""
    try:
        stream_info = await browser.start_stream(
            protocol="hls",
            quality="high",
            codec="h264",
        )
        print("  Stream started successfully.")
        print(f"    Protocol : hls")
        print(f"    Codec    : h264")
        print(f"    Quality  : high")
        for key, value in stream_info.items():
            print(f"    {key}: {value}")
        return stream_info
    except Exception as exc:
        print(f"  WARNING: start_stream() failed: {exc}")
        print("  This is expected in headless mode or environments without media support.")
        return None


async def check_stream_status(browser: FlyBrowser) -> dict | None:
    """Query the stream status. Returns the status dict or None."""
    try:
        status = await browser.get_stream_status()
        print("  Stream status:")
        for key, value in status.items():
            print(f"    {key}: {value}")
        return status
    except Exception as exc:
        print(f"  WARNING: get_stream_status() failed: {exc}")
        return None


async def stop_hls_stream(browser: FlyBrowser) -> dict | None:
    """Attempt to stop the stream. Returns stop info or None."""
    try:
        stop_info = await browser.stop_stream()
        print("  Stream stopped successfully.")
        for key, value in stop_info.items():
            print(f"    {key}: {value}")
        return stop_info
    except Exception as exc:
        print(f"  WARNING: stop_stream() failed: {exc}")
        return None


async def navigate_and_interact(browser: FlyBrowser) -> None:
    """Perform navigation and extraction while the stream is active."""
    sites = [
        ("https://news.ycombinator.com", "What is the top story headline on Hacker News?"),
        ("https://example.com", "What is the main heading on this page?"),
    ]

    for url, query in sites:
        print(f"\n  Navigating to {url}")
        try:
            await browser.goto(url)
            result = await browser.extract(query)
            if result.success:
                print(f"    Extracted: {str(result.data)[:120]}")
            else:
                print(f"    Extraction failed: {result.error}")
        except Exception as exc:
            print(f"    Navigation/extraction error: {exc}")

        # Pause between navigations to allow stream to capture
        await asyncio.sleep(2)


async def demo_hls_streaming() -> None:
    """Full HLS streaming demonstration."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser HLS Streaming Demo")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 65)

    stream_active = False

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Phase 1: Start stream
        print("\n[Phase 1] Starting HLS stream")
        stream_info = await start_hls_stream(browser)
        stream_active = stream_info is not None

        # Phase 2: Navigate and interact while streaming
        print("\n[Phase 2] Navigating while streaming")
        await navigate_and_interact(browser)

        # Phase 3: Check stream status mid-session
        if stream_active:
            print("\n[Phase 3] Checking stream status")
            await check_stream_status(browser)

        # Phase 4: One more interaction
        print("\n[Phase 4] Additional interaction")
        try:
            await browser.goto("https://en.wikipedia.org/wiki/Main_Page")
            act_result = await browser.act("Scroll down to see content below the fold.")
            if act_result.success:
                print("  Scrolled Wikipedia page successfully.")
            else:
                print(f"  Scroll action issue: {act_result.error}")
        except Exception as exc:
            print(f"  Interaction error: {exc}")

        await asyncio.sleep(2)

        # Phase 5: Stop stream
        if stream_active:
            print("\n[Phase 5] Stopping HLS stream")
            await stop_hls_stream(browser)
        else:
            print("\n[Phase 5] Skipping stop (stream was not started)")

        # Usage summary
        usage = browser.get_usage_summary()
        print(f"\n  LLM Usage: {usage}")

    print("\n  HLS streaming demo complete.")


if __name__ == "__main__":
    asyncio.run(demo_hls_streaming())
