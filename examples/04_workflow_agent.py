#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
WorkflowAgent Example

This example demonstrates the WorkflowAgent capabilities:
- Multi-step workflow execution
- Variable substitution
- State management between steps
- Error handling and recovery

Run with: python examples/04_workflow_agent.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate WorkflowAgent capabilities."""
    print("=" * 60)
    print("FlyBrowser WorkflowAgent Example")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Define a simple workflow
        print("\n[1] Defining a search workflow...")
        search_workflow = {
            "name": "search_workflow",
            "description": "Search for a term and extract results",
            "steps": [
                {
                    "name": "navigate",
                    "action": "goto",
                    "url": "https://www.google.com",
                },
                {
                    "name": "search",
                    "action": "type",
                    "selector": "textarea[name='q']",
                    "value": "{{search_term}}",
                },
                {
                    "name": "submit",
                    "action": "press",
                    "key": "Enter",
                },
                {
                    "name": "wait",
                    "action": "wait",
                    "timeout": 2000,
                },
            ],
        }

        # Execute the workflow with variables
        print("\n[2] Executing search workflow...")
        result = await browser.run_workflow(
            search_workflow,
            variables={"search_term": "Python automation"},
        )
        print(f"    Success: {result.get('success')}")
        print(f"    Steps completed: {result.get('steps_completed')}/{result.get('total_steps')}")

        # Define a data extraction workflow
        print("\n[3] Defining a data extraction workflow...")
        extraction_workflow = {
            "name": "extract_python_info",
            "steps": [
                {
                    "name": "navigate",
                    "action": "goto",
                    "url": "https://www.python.org",
                },
                {
                    "name": "extract_version",
                    "action": "extract",
                    "query": "What is the latest Python version?",
                    "output_variable": "python_version",
                },
                {
                    "name": "click_downloads",
                    "action": "click",
                    "selector": "a[href*='downloads']",
                },
                {
                    "name": "extract_downloads",
                    "action": "extract",
                    "query": "List the available download options",
                    "output_variable": "downloads",
                },
            ],
        }

        print("\n[4] Executing extraction workflow...")
        result = await browser.run_workflow(extraction_workflow)
        print(f"    Success: {result.get('success')}")
        print(f"    Steps completed: {result.get('steps_completed')}")
        print(f"    Variables: {result.get('variables', {})}")

    print("\n" + "=" * 60)
    print("WorkflowAgent example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

