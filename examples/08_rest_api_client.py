#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
REST API Client Example

This example demonstrates using the FlyBrowser REST API directly:
- Creating sessions via HTTP
- All API endpoints including new workflow and monitor
- Proper error handling

Prerequisites:
1. Start the FlyBrowser server:
   uvicorn flybrowser.service.app:app --host 0.0.0.0 --port 8000

2. Run this example:
   python examples/08_rest_api_client.py
"""

import asyncio
import os

from flybrowser.client import FlyBrowserClient


async def main():
    """Demonstrate REST API client usage."""
    print("=" * 60)
    print("FlyBrowser REST API Client Example")
    print("=" * 60)

    async with FlyBrowserClient(
        endpoint="http://localhost:8000",
        api_key="flybrowser_dev_key",  # Service API key
    ) as client:
        # Health check
        print("\n[1] Checking server health...")
        is_healthy = await client.health_check()
        print(f"    Server healthy: {is_healthy}")

        # Create a session
        print("\n[2] Creating session...")
        session = await client.create_session(
            llm_provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            headless=True,
        )
        session_id = session.get("session_id")
        print(f"    Session ID: {session_id}")

        try:
            # Navigate
            print("\n[3] Navigating...")
            result = await client.navigate(session_id, "https://example.com")
            print(f"    URL: {result.get('url')}")

            # Extract data
            print("\n[4] Extracting data...")
            result = await client.extract(session_id, "What is the page title?")
            print(f"    Data: {result.get('data')}")

            # Perform action
            print("\n[5] Performing action...")
            result = await client.action(session_id, "click on the More information link")
            print(f"    Success: {result.get('success')}")

            # Take screenshot
            print("\n[6] Taking screenshot...")
            result = await client.screenshot(session_id, full_page=True)
            print(f"    Size: {result.get('width')}x{result.get('height')}")

            # Natural language navigation
            print("\n[7] Natural language navigation...")
            result = await client.navigate_nl(session_id, "go back to the previous page")
            print(f"    Result: {result}")

            # Monitor for condition
            print("\n[8] Monitoring for condition...")
            result = await client.monitor(
                session_id,
                "wait for the page to contain 'Example'",
                timeout=5.0,
            )
            print(f"    Condition met: {result.get('condition_met')}")

            # Store credential
            print("\n[9] Storing credential...")
            result = await client.store_credential(
                session_id,
                name="test_email",
                value="test@example.com",
                pii_type="email",
            )
            print(f"    Credential ID: {result.get('credential_id')}")

            # Run workflow
            print("\n[10] Running workflow...")
            workflow = {
                "name": "simple_workflow",
                "steps": [
                    {"name": "nav", "action": "goto", "url": "https://example.com"},
                    {"name": "wait", "action": "wait", "timeout": 1000},
                ],
            }
            result = await client.run_workflow(session_id, workflow)
            print(f"    Steps completed: {result.get('steps_completed')}")

        finally:
            # Always close the session
            print("\n[11] Closing session...")
            await client.close_session(session_id)
            print("    Session closed")

    print("\n" + "=" * 60)
    print("REST API Client example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

