#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
Integrated Example - All Agents Working Together

This example demonstrates a real-world scenario using all agents:
- NavigationAgent for intelligent navigation
- ActionAgent for form interactions
- WorkflowAgent for multi-step automation
- MonitoringAgent for waiting conditions
- PIIHandler for secure credential handling

Scenario: Automated website exploration with data extraction

Run with: python examples/10_integrated_example.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate integrated agent usage."""
    print("=" * 60)
    print("FlyBrowser Integrated Example")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        pii_masking_enabled=True,
    ) as browser:
        # Store credentials for later use
        print("\n[1] Setting up secure credentials...")
        browser.store_credential("demo_email", "demo@example.com", pii_type="email")
        print("    Credentials stored securely")

        # Phase 1: Navigation
        print("\n[2] Phase 1: Intelligent Navigation")
        await browser.goto("https://www.python.org")

        result = await browser.navigate("go to the documentation section")
        print(f"    Navigated to: {result.get('url')}")

        # Phase 2: Monitoring
        print("\n[3] Phase 2: Wait for Page Load")
        result = await browser.monitor(
            "wait for the documentation page to fully load",
            timeout=10.0,
        )
        print(f"    Page ready: {result.get('condition_met')}")

        # Phase 3: Data Extraction
        print("\n[4] Phase 3: Extract Information")
        data = await browser.extract(
            "What are the main documentation sections available?",
            schema={
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        )
        print(f"    Documentation sections: {data}")

        # Phase 4: Workflow Execution
        print("\n[5] Phase 4: Execute Multi-Step Workflow")
        workflow = {
            "name": "explore_python_org",
            "steps": [
                {
                    "name": "go_home",
                    "action": "goto",
                    "url": "https://www.python.org",
                },
                {
                    "name": "extract_news",
                    "action": "extract",
                    "query": "What are the latest Python news items?",
                    "output_variable": "news",
                },
                {
                    "name": "click_downloads",
                    "action": "click",
                    "selector": "a[href*='downloads']",
                },
                {
                    "name": "extract_versions",
                    "action": "extract",
                    "query": "What Python versions are available for download?",
                    "output_variable": "versions",
                },
            ],
        }

        result = await browser.run_workflow(workflow)
        print(f"    Workflow completed: {result.get('steps_completed')}/{result.get('total_steps')} steps")
        print(f"    Extracted variables: {list(result.get('variables', {}).keys())}")

        # Phase 5: Action Execution
        print("\n[6] Phase 5: Perform Actions")
        result = await browser.act("scroll down to see more content")
        print(f"    Action completed: {result.get('success')}")

        # Final screenshot
        print("\n[7] Capturing final state...")
        screenshot = await browser.screenshot(full_page=True)
        print(f"    Screenshot: {screenshot['width']}x{screenshot['height']}")

    print("\n" + "=" * 60)
    print("Integrated example completed!")
    print("=" * 60)
    print("\nThis example demonstrated:")
    print("- NavigationAgent: Natural language navigation")
    print("- MonitoringAgent: Waiting for conditions")
    print("- ExtractionAgent: Structured data extraction")
    print("- WorkflowAgent: Multi-step automation")
    print("- ActionAgent: Browser interactions")
    print("- PIIHandler: Secure credential storage")


if __name__ == "__main__":
    asyncio.run(main())

