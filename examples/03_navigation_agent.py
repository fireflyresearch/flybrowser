#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
NavigationAgent Example

This example demonstrates the NavigationAgent capabilities:
- Natural language navigation commands
- Smart waiting for page loads
- Link and button detection
- Navigation type detection (URL, click, form)

Run with: python examples/03_navigation_agent.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate NavigationAgent capabilities."""
    print("=" * 60)
    print("FlyBrowser NavigationAgent Example")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Start at a known page
        print("\n[1] Starting at Python.org...")
        await browser.goto("https://www.python.org")

        # Natural language navigation - click-based
        print("\n[2] Navigating using natural language (click)...")
        result = await browser.navigate("go to the documentation page")
        print(f"    Success: {result.get('success')}")
        print(f"    URL: {result.get('url')}")
        print(f"    Navigation type: {result.get('navigation_type')}")

        # Natural language navigation - menu item
        print("\n[3] Navigating to a menu item...")
        result = await browser.navigate("click on the 'Downloads' menu item")
        print(f"    Success: {result.get('success')}")
        print(f"    Title: {result.get('title')}")

        # Navigate back
        print("\n[4] Going back to the homepage...")
        result = await browser.navigate("go back to the Python homepage")
        print(f"    Success: {result.get('success')}")
        print(f"    URL: {result.get('url')}")

        # Navigate to a specific section
        print("\n[5] Navigating to a specific section...")
        result = await browser.navigate("find and click on the 'About' section")
        print(f"    Success: {result.get('success')}")
        print(f"    Navigation type: {result.get('navigation_type')}")

        # Complex navigation instruction
        print("\n[6] Complex navigation...")
        result = await browser.navigate(
            "look for a link about Python's history and click on it"
        )
        print(f"    Success: {result.get('success')}")
        print(f"    Final URL: {result.get('url')}")

    print("\n" + "=" * 60)
    print("NavigationAgent example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

