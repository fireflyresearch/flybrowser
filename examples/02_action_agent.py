#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
ActionAgent Example

This example demonstrates the ActionAgent capabilities:
- Multi-step action planning
- Natural language action execution
- Click, type, and form interactions
- Automatic retry on failure

Run with: python examples/02_action_agent.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate ActionAgent capabilities."""
    print("=" * 60)
    print("FlyBrowser ActionAgent Example")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Navigate to a page with interactive elements
        print("\n[1] Navigating to a demo page...")
        await browser.goto("https://www.python.org")

        # Simple click action
        print("\n[2] Performing click action...")
        result = await browser.act("click on the 'About' link in the navigation")
        print(f"    Result: success={result.get('success')}, steps={result.get('steps_completed')}")

        # Multi-step action (the agent will plan and execute multiple steps)
        print("\n[3] Performing multi-step action...")
        result = await browser.act(
            "go back to the homepage and then click on the 'Downloads' link"
        )
        print(f"    Result: success={result.get('success')}, steps={result.get('steps_completed')}")

        # Type action
        print("\n[4] Navigating to a search page...")
        await browser.goto("https://www.google.com")

        print("\n[5] Performing type action...")
        result = await browser.act("type 'Python programming' into the search box")
        print(f"    Result: success={result.get('success')}")

        # Complex interaction
        print("\n[6] Performing complex interaction...")
        result = await browser.act(
            "clear the search box and type 'FlyBrowser automation'"
        )
        print(f"    Result: success={result.get('success')}")

    print("\n" + "=" * 60)
    print("ActionAgent example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

