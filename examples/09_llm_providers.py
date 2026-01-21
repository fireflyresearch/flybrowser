#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
LLM Providers Example

This example demonstrates using different LLM providers:
- OpenAI (GPT-5.2, GPT-4.1)
- Anthropic (Claude Sonnet 4.5, Claude Opus 4.5)
- Google Gemini (Gemini 2.5 Pro)
- Ollama (local models: Qwen3, Gemma 3, Llama 3.2)

Run with: python examples/09_llm_providers.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def demo_openai():
    """Demonstrate OpenAI provider."""
    print("\n--- OpenAI Provider ---")
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2",  # Latest flagship model
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(f"OpenAI extracted: {data}")


async def demo_anthropic():
    """Demonstrate Anthropic provider."""
    print("\n--- Anthropic Provider ---")
    async with FlyBrowser(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250929",  # Latest Sonnet
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(f"Anthropic extracted: {data}")


async def demo_gemini():
    """Demonstrate Google Gemini provider."""
    print("\n--- Google Gemini Provider ---")
    async with FlyBrowser(
        llm_provider="gemini",
        llm_model="gemini-2.5-pro",  # Latest Gemini Pro
        api_key=os.getenv("GOOGLE_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(f"Gemini extracted: {data}")


async def demo_ollama():
    """Demonstrate Ollama provider (local LLMs)."""
    print("\n--- Ollama Provider (Local) ---")
    print("Note: Requires Ollama running locally with a model pulled")
    async with FlyBrowser(
        llm_provider="ollama",
        llm_model="qwen3:8b",  # Or: gemma3:12b, llama3.2:3b, deepseek-r1:8b
        # No API key needed for local models!
        headless=True,
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(f"Ollama extracted: {data}")


async def main():
    """Demonstrate all LLM providers."""
    print("=" * 60)
    print("FlyBrowser LLM Providers Example")
    print("=" * 60)

    # Run demos based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        await demo_openai()
    else:
        print("\nSkipping OpenAI (OPENAI_API_KEY not set)")

    if os.getenv("ANTHROPIC_API_KEY"):
        await demo_anthropic()
    else:
        print("\nSkipping Anthropic (ANTHROPIC_API_KEY not set)")

    if os.getenv("GOOGLE_API_KEY"):
        await demo_gemini()
    else:
        print("\nSkipping Gemini (GOOGLE_API_KEY not set)")

    # Ollama doesn't need an API key
    print("\nNote: For Ollama, ensure the service is running:")
    print("  ollama serve")
    print("  ollama pull qwen3:8b")

    print("\n" + "=" * 60)
    print("LLM Providers example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

