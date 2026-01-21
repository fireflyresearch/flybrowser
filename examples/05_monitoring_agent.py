#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
MonitoringAgent Example

This example demonstrates the MonitoringAgent capabilities:
- Natural language condition monitoring
- Waiting for elements to appear/disappear
- Content change detection
- Configurable timeout and polling

Run with: python examples/05_monitoring_agent.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate MonitoringAgent capabilities."""
    print("=" * 60)
    print("FlyBrowser MonitoringAgent Example")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Navigate to a page
        print("\n[1] Navigating to Python.org...")
        await browser.goto("https://www.python.org")

        # Monitor for an element that exists
        print("\n[2] Monitoring for existing element...")
        result = await browser.monitor(
            "wait for the Python logo to be visible",
            timeout=10.0,
            poll_interval=0.5,
        )
        print(f"    Condition met: {result.get('condition_met')}")
        print(f"    Elapsed time: {result.get('elapsed_time'):.2f}s")

        # Monitor for page content
        print("\n[3] Monitoring for page content...")
        result = await browser.monitor(
            "wait for the page to contain 'Python'",
            timeout=5.0,
        )
        print(f"    Condition met: {result.get('condition_met')}")

        # Navigate to a page with dynamic content
        print("\n[4] Navigating to a page with dynamic content...")
        await browser.goto("https://www.google.com")

        # Monitor for search box
        print("\n[5] Monitoring for search box...")
        result = await browser.monitor(
            "wait for the search input field to be visible",
            timeout=10.0,
        )
        print(f"    Condition met: {result.get('condition_met')}")
        print(f"    Details: {result.get('details', {})}")

        # Demonstrate timeout behavior
        print("\n[6] Demonstrating timeout (waiting for non-existent element)...")
        result = await browser.monitor(
            "wait for an element with text 'This does not exist'",
            timeout=3.0,  # Short timeout for demo
        )
        print(f"    Condition met: {result.get('condition_met')}")
        print(f"    Elapsed time: {result.get('elapsed_time'):.2f}s")
        print(f"    (Expected: condition not met due to timeout)")

    print("\n" + "=" * 60)
    print("MonitoringAgent example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

