#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
Basic FlyBrowser Usage Example

This example demonstrates the fundamental operations of FlyBrowser:
- Initializing the browser in embedded mode
- Navigating to URLs
- Extracting data using natural language
- Taking screenshots
- Performing actions

Run with: python examples/01_basic_usage.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate basic FlyBrowser usage."""
    print("=" * 60)
    print("FlyBrowser Basic Usage Example")
    print("=" * 60)

    # Initialize FlyBrowser in embedded mode (local browser)
    # Requires an OpenAI API key for LLM-powered features
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,  # Set to False to see the browser
    ) as browser:
        print("\n[1] Navigating to example.com...")
        await browser.goto("https://example.com")
        print("    Navigation complete!")

        # Extract data using natural language
        print("\n[2] Extracting page information...")
        data = await browser.extract("What is the main heading and paragraph text?")
        print(f"    Extracted data: {data}")

        # Take a screenshot
        print("\n[3] Taking a screenshot...")
        screenshot = await browser.screenshot(full_page=True)
        print(f"    Screenshot captured: {screenshot['width']}x{screenshot['height']} pixels")

        # Navigate to a more complex page
        print("\n[4] Navigating to Python.org...")
        await browser.goto("https://www.python.org")

        # Extract structured data
        print("\n[5] Extracting Python.org information...")
        data = await browser.extract(
            "What is the latest Python version mentioned on the page?",
            schema={
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "release_date": {"type": "string"},
                },
            },
        )
        print(f"    Python info: {data}")

        # Perform an action using natural language
        print("\n[6] Performing action: clicking on Downloads...")
        try:
            result = await browser.act("click on the Downloads menu item")
            print(f"    Action result: {result}")
        except Exception as e:
            print(f"    Action skipped (demo mode): {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

