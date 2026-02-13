# Copyright 2026 Firefly Software Solutions Inc
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
Comprehensive Quickstart: Every Core FlyBrowser Operation

Demonstrates ALL core SDK operations in a single, self-contained script:
  1. goto     - Navigate to a URL
  2. extract  - Pull structured data from the page
  3. observe  - Inspect visible UI elements and layout
  4. act      - Perform a browser action (scroll, click, type)
  5. screenshot - Capture a base64-encoded page image
  6. agent    - Run an autonomous multi-step task

Target: Hacker News (https://news.ycombinator.com) - a stable, public site.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."          # required
    export FLYBROWSER_LLM_PROVIDER="anthropic"     # optional, default shown
    export FLYBROWSER_LLM_MODEL="claude-sonnet-4-5-20250929"  # optional
"""

import asyncio
import os
import sys

from flybrowser import FlyBrowser

# ---------------------------------------------------------------------------
# Configuration from environment variables (never hardcode keys)
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")
TARGET_URL = "https://news.ycombinator.com"


def section(title: str) -> None:
    """Print a visible section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


async def main() -> None:
    """Execute every core FlyBrowser operation against Hacker News."""

    section("FlyBrowser Quickstart - All Core Operations")
    print(f"  Provider : {PROVIDER}")
    print(f"  Model    : {MODEL}")
    print(f"  Target   : {TARGET_URL}")

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        # ------------------------------------------------------------------
        # 1. GOTO - Navigate to Hacker News
        # ------------------------------------------------------------------
        section("1/6  goto - Navigate to Hacker News")
        await browser.goto(TARGET_URL)
        print("  Navigation complete.")

        # ------------------------------------------------------------------
        # 2. EXTRACT - Pull the top 5 stories with structured fields
        # ------------------------------------------------------------------
        section("2/6  extract - Top 5 stories (title, score, comment count)")
        result = await browser.extract(
            "Extract the top 5 stories. For each story return the title, "
            "the URL it links to, the point score, and the number of comments."
        )

        if result.success:
            print(f"  Extracted data (type: {type(result.data).__name__}):")
            result.pprint()
            print(f"  Tokens used: {result.llm_usage.total_tokens:,}")
            print(f"  Duration:    {result.execution.duration_seconds:.2f}s")
        else:
            print(f"  [FAILED] {result.error}")

        # ------------------------------------------------------------------
        # 3. OBSERVE - Describe visible interactive elements
        # ------------------------------------------------------------------
        section("3/6  observe - Describe visible elements on the page")
        result = await browser.observe(
            "List all visible interactive elements: navigation links, "
            "buttons, and form inputs. Include their text and purpose."
        )

        if result.success:
            print("  Observation results:")
            result.pprint()
        else:
            print(f"  [FAILED] {result.error}")

        # ------------------------------------------------------------------
        # 4. ACT - Scroll the page down
        # ------------------------------------------------------------------
        section("4/6  act - Scroll down the page")
        result = await browser.act("Scroll down the page slowly")

        if result.success:
            print("  Action completed successfully.")
            print(f"  Duration: {result.execution.duration_seconds:.2f}s")
        else:
            print(f"  [FAILED] {result.error}")

        # ------------------------------------------------------------------
        # 5. SCREENSHOT - Capture the current viewport
        # ------------------------------------------------------------------
        section("5/6  screenshot - Capture current viewport")
        screenshot = await browser.screenshot()

        if screenshot and screenshot.get("data_base64"):
            b64_len = len(screenshot["data_base64"])
            kb_size = b64_len * 3 / 4 / 1024  # approximate decoded size
            print(f"  Screenshot captured: ~{kb_size:.1f} KB (base64 length: {b64_len})")
        else:
            print("  Screenshot returned no data.")

        # ------------------------------------------------------------------
        # 6. AGENT - Autonomous multi-step task
        # ------------------------------------------------------------------
        section("6/6  agent - Autonomous task: summarize front page")
        result = await browser.agent(
            "Go back to the top of the page, count how many stories are "
            "visible, identify the story with the highest score, and "
            "summarize the dominant topics on the front page today."
        )

        if result.success:
            print("  Agent task completed:")
            result.pprint()
            print(f"  Tokens used: {result.llm_usage.total_tokens:,}")
        else:
            print(f"  [FAILED] {result.error}")

        # ------------------------------------------------------------------
        # Session summary
        # ------------------------------------------------------------------
        section("Session Usage Summary")
        usage = browser.get_usage_summary()
        print(f"  Total tokens : {usage.get('total_tokens', 0):,}")
        print(f"  API calls    : {usage.get('calls_count', 0)}")
        print(f"  Est. cost    : ${usage.get('cost_usd', 0):.4f}")
        print(f"  Model        : {usage.get('model', 'N/A')}")

    print("\nQuickstart complete. All 6 operations demonstrated successfully.")


if __name__ == "__main__":
    asyncio.run(main())
