#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
Server Mode Example

This example demonstrates using FlyBrowser in server mode:
- Connecting to a standalone FlyBrowser server
- All SDK methods work identically to embedded mode
- Automatic session management

Prerequisites:
1. Start the FlyBrowser server:
   uvicorn flybrowser.service.app:app --host 0.0.0.0 --port 8000

2. Run this example:
   python examples/07_server_mode.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate server mode usage."""
    print("=" * 60)
    print("FlyBrowser Server Mode Example")
    print("=" * 60)

    # Connect to a FlyBrowser server
    # The endpoint is the ONLY difference from embedded mode!
    async with FlyBrowser(
        endpoint="http://localhost:8000",  # Server endpoint
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        print(f"\n[1] Connected to server in {browser.mode} mode")
        print(f"    Session ID: {browser.session_id}")

        # All operations work exactly the same as embedded mode
        print("\n[2] Navigating to example.com...")
        await browser.goto("https://example.com")

        print("\n[3] Extracting data...")
        data = await browser.extract("What is the main heading?")
        print(f"    Extracted: {data}")

        print("\n[4] Taking screenshot...")
        screenshot = await browser.screenshot()
        print(f"    Screenshot: {screenshot['width']}x{screenshot['height']}")

        print("\n[5] Performing action...")
        result = await browser.act("click on the 'More information' link")
        print(f"    Action result: {result}")

        # Server mode also supports all new features
        print("\n[6] Using navigate() with natural language...")
        result = await browser.navigate("go back to the previous page")
        print(f"    Navigation: {result}")

        print("\n[7] Using monitor()...")
        result = await browser.monitor(
            "wait for the page to contain 'Example'",
            timeout=5.0,
        )
        print(f"    Monitor result: {result}")

    print("\n" + "=" * 60)
    print("Server Mode example completed!")
    print("=" * 60)
    print("\nNote: The same code works for:")
    print("- Standalone server (single node)")
    print("- Cluster mode (multiple nodes with HA)")
    print("Just change the endpoint URL!")


if __name__ == "__main__":
    asyncio.run(main())

