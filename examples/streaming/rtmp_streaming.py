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
Example: RTMP Streaming Concept

Demonstrates the concept of RTMP streaming (e.g., to Twitch or YouTube)
using FlyBrowser's streaming API. Since RTMP requires external infrastructure
and stream keys, this example focuses on graceful handling: attempting to
start a stream, navigating and interacting, monitoring status, and stopping.

All streaming calls are wrapped in try/except because RTMP streaming may
not be available in headless mode or without proper infrastructure.

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


async def attempt_rtmp_stream(browser: FlyBrowser) -> dict | None:
    """
    Attempt to start an RTMP stream.

    RTMP streaming requires external infrastructure (Twitch, YouTube, custom
    RTMP server). This function demonstrates the API call pattern with
    graceful error handling for environments where RTMP is not available.
    """
    try:
        stream_info = await browser.start_stream(
            protocol="hls",       # Use HLS as fallback-safe protocol
            quality="high",
            codec="h264",
        )
        print("  Stream started successfully.")
        for key, value in stream_info.items():
            print(f"    {key}: {value}")
        return stream_info
    except Exception as exc:
        print(f"  Stream start failed (expected in most environments): {exc}")
        print("  Continuing with navigation-only demo.")
        return None


async def monitor_stream(browser: FlyBrowser, checks: int = 3, interval: float = 3.0) -> None:
    """Periodically check stream health."""
    for i in range(1, checks + 1):
        await asyncio.sleep(interval)
        try:
            status = await browser.get_stream_status()
            active = status.get("active", False)
            print(f"  [Health Check {i}/{checks}] Active: {active}")
            for key, value in status.items():
                if key != "active":
                    print(f"    {key}: {value}")
        except Exception as exc:
            print(f"  [Health Check {i}/{checks}] Status unavailable: {exc}")


async def stop_stream_safe(browser: FlyBrowser) -> dict | None:
    """Stop the stream gracefully."""
    try:
        result = await browser.stop_stream()
        print("  Stream stopped.")
        for key, value in result.items():
            print(f"    {key}: {value}")
        return result
    except Exception as exc:
        print(f"  Stop stream failed: {exc}")
        return None


async def navigate_with_extraction(browser: FlyBrowser) -> list[dict]:
    """Navigate to several sites and extract data, simulating a live broadcast."""
    results: list[dict] = []

    sites = [
        {
            "url": "https://news.ycombinator.com",
            "query": "What are the top 3 stories on Hacker News right now?",
            "label": "Hacker News top stories",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "query": "What is the featured article on Wikipedia today?",
            "label": "Wikipedia featured article",
        },
        {
            "url": "https://books.toscrape.com",
            "query": "What are the first 3 book titles and their prices?",
            "label": "Books to Scrape catalogue",
        },
    ]

    for site in sites:
        print(f"\n  -> {site['label']}")
        entry = {"label": site["label"], "url": site["url"], "success": False}
        try:
            await browser.goto(site["url"])
            result = await browser.extract(site["query"])
            if result.success:
                print(f"     Extracted: {str(result.data)[:140]}")
                entry["success"] = True
                entry["data"] = str(result.data)[:200]
            else:
                print(f"     Extraction error: {result.error}")
                entry["error"] = result.error
        except Exception as exc:
            print(f"     Exception: {exc}")
            entry["error"] = str(exc)

        results.append(entry)
        await asyncio.sleep(2)

    return results


async def perform_interactive_actions(browser: FlyBrowser) -> None:
    """Perform some interactive actions that would be visible in a stream."""
    print("\n  Interactive actions (simulating live broadcast):")

    actions = [
        ("https://news.ycombinator.com", "Click on the first story link to read it."),
        (None, "Scroll down slowly through the article or page content."),
    ]

    for url, instruction in actions:
        try:
            if url:
                await browser.goto(url)
            act_result = await browser.act(instruction)
            if act_result.success:
                print(f"    [OK] {instruction[:80]}")
            else:
                print(f"    [FAIL] {instruction[:60]}: {act_result.error}")
        except Exception as exc:
            print(f"    [ERROR] {instruction[:60]}: {exc}")
        await asyncio.sleep(2)

    # Observe the resulting page
    try:
        obs = await browser.observe("What interactive elements are visible on this page?")
        if obs.success:
            print(f"    Observed: {str(obs.data)[:140]}")
    except Exception as exc:
        print(f"    Observe error: {exc}")


async def demo_rtmp_streaming() -> None:
    """Full RTMP streaming concept demonstration."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser RTMP Streaming Concept Demo")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("  NOTE: RTMP requires external infrastructure. This demo uses")
    print("  HLS as a fallback and handles all errors gracefully.")
    print("=" * 65)

    stream_active = False

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Phase 1: Attempt to start stream
        print("\n[Phase 1] Attempting to start stream")
        stream_info = await attempt_rtmp_stream(browser)
        stream_active = stream_info is not None

        # Phase 2: Navigate and extract (works regardless of stream status)
        print("\n[Phase 2] Navigating and extracting data")
        extraction_results = await navigate_with_extraction(browser)

        # Phase 3: Monitor stream if active
        if stream_active:
            print("\n[Phase 3] Monitoring stream health")
            await monitor_stream(browser, checks=2, interval=2.0)
        else:
            print("\n[Phase 3] Skipping stream monitoring (stream not active)")

        # Phase 4: Interactive actions
        print("\n[Phase 4] Interactive actions")
        await perform_interactive_actions(browser)

        # Phase 5: Stop stream
        if stream_active:
            print("\n[Phase 5] Stopping stream")
            await stop_stream_safe(browser)
        else:
            print("\n[Phase 5] No stream to stop")

        # Summary
        successful = sum(1 for r in extraction_results if r.get("success"))
        total = len(extraction_results)
        usage = browser.get_usage_summary()

        print("\n" + "=" * 65)
        print("  RTMP STREAMING DEMO SUMMARY")
        print("=" * 65)
        print(f"    Stream attempted : Yes")
        print(f"    Stream active    : {stream_active}")
        print(f"    Sites visited    : {total}")
        print(f"    Extractions OK   : {successful}/{total}")
        print(f"    LLM Usage        : {usage}")
        print("=" * 65)

    print("\n  RTMP streaming demo complete.")


if __name__ == "__main__":
    asyncio.run(demo_rtmp_streaming())
