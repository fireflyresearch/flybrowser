# Copyright 2026 Firefly Software Solutions Inc.
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
Example: Browser Session Recording

Demonstrates recording browser sessions using FlyBrowser's recording API.
Starts a recording (no parameters), navigates to real sites, performs
interactions with act() and extract(), then stops the recording and
inspects the result.

NOTE: start_recording() takes NO arguments. stop_recording() returns
a dict with session_id, screenshots, and duration_seconds.

Prerequisites:
    pip install flybrowser
    export ANTHROPIC_API_KEY="sk-ant-..."

Environment Variables:
    ANTHROPIC_API_KEY          - Required. Your LLM provider API key.
    FLYBROWSER_LLM_PROVIDER    - Optional. Defaults to "anthropic".
    FLYBROWSER_LLM_MODEL       - Optional. Defaults to "claude-sonnet-4-5-20250929".
"""

import asyncio
import os
import sys

from flybrowser import FlyBrowser


async def start_recording_safe(browser: FlyBrowser) -> dict | None:
    """Start a recording session. Returns recording info or None on failure."""
    try:
        # start_recording() takes NO arguments
        recording_info = await browser.start_recording()
        print("  Recording started successfully.")
        for key, value in recording_info.items():
            print(f"    {key}: {value}")
        return recording_info
    except Exception as exc:
        print(f"  WARNING: start_recording() failed: {exc}")
        print("  Recording may not be supported in this environment.")
        return None


async def stop_recording_safe(browser: FlyBrowser) -> dict | None:
    """Stop the recording and return session metadata."""
    try:
        recording_result = await browser.stop_recording()
        print("  Recording stopped successfully.")
        print("  Recording metadata:")
        for key, value in recording_result.items():
            if key == "screenshots" and isinstance(value, list):
                print(f"    screenshots: {len(value)} captured")
            else:
                print(f"    {key}: {value}")
        return recording_result
    except Exception as exc:
        print(f"  WARNING: stop_recording() failed: {exc}")
        return None


async def perform_recorded_actions(browser: FlyBrowser) -> list[dict]:
    """Navigate and interact while recording. Returns a log of actions taken."""
    action_log: list[dict] = []

    # Action 1: Navigate to Hacker News
    print("\n  [Action 1] Navigate to Hacker News")
    try:
        await browser.goto("https://news.ycombinator.com")
        result = await browser.extract(
            "What are the titles of the top 3 stories on the front page?"
        )
        if result.success:
            print(f"    Extracted top stories: {str(result.data)[:150]}")
            action_log.append({"action": "extract_hn_stories", "success": True})
        else:
            print(f"    Extraction failed: {result.error}")
            action_log.append({"action": "extract_hn_stories", "success": False})
    except Exception as exc:
        print(f"    Error: {exc}")
        action_log.append({"action": "extract_hn_stories", "success": False})

    await asyncio.sleep(1)

    # Action 2: Scroll down
    print("\n  [Action 2] Scroll Hacker News page")
    try:
        act_result = await browser.act("Scroll down to see more stories on the page.")
        if act_result.success:
            print("    Scrolled successfully.")
            action_log.append({"action": "scroll_hn", "success": True})
        else:
            print(f"    Scroll failed: {act_result.error}")
            action_log.append({"action": "scroll_hn", "success": False})
    except Exception as exc:
        print(f"    Error: {exc}")
        action_log.append({"action": "scroll_hn", "success": False})

    await asyncio.sleep(1)

    # Action 3: Navigate to Wikipedia
    print("\n  [Action 3] Navigate to Wikipedia")
    try:
        await browser.goto("https://en.wikipedia.org/wiki/Main_Page")
        result = await browser.extract(
            "What is the featured article title on the Wikipedia main page?"
        )
        if result.success:
            print(f"    Featured article: {str(result.data)[:150]}")
            action_log.append({"action": "extract_wikipedia", "success": True})
        else:
            print(f"    Extraction failed: {result.error}")
            action_log.append({"action": "extract_wikipedia", "success": False})
    except Exception as exc:
        print(f"    Error: {exc}")
        action_log.append({"action": "extract_wikipedia", "success": False})

    await asyncio.sleep(1)

    # Action 4: Click a link on Wikipedia
    print("\n  [Action 4] Interact with Wikipedia")
    try:
        act_result = await browser.act(
            "Click on the 'Featured article' link or the first article link visible."
        )
        if act_result.success:
            print("    Clicked article link.")
            action_log.append({"action": "click_wiki_article", "success": True})
        else:
            print(f"    Click failed: {act_result.error}")
            action_log.append({"action": "click_wiki_article", "success": False})

        await asyncio.sleep(1)

        verify = await browser.extract("What is the title of the article I am now viewing?")
        if verify.success:
            print(f"    Now viewing: {str(verify.data)[:120]}")
            verify.pprint()
    except Exception as exc:
        print(f"    Error: {exc}")
        action_log.append({"action": "click_wiki_article", "success": False})

    # Action 5: Navigate to books.toscrape.com
    print("\n  [Action 5] Navigate to Books to Scrape")
    try:
        await browser.goto("https://books.toscrape.com")
        result = await browser.extract("How many books are shown on the front page?")
        if result.success:
            print(f"    Books visible: {str(result.data)[:120]}")
            action_log.append({"action": "extract_books", "success": True})
        else:
            print(f"    Extraction failed: {result.error}")
            action_log.append({"action": "extract_books", "success": False})
    except Exception as exc:
        print(f"    Error: {exc}")
        action_log.append({"action": "extract_books", "success": False})

    return action_log


async def demo_recording() -> None:
    """Full recording demonstration."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser Recording Demo")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 65)

    recording_active = False

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Phase 1: Start recording
        print("\n[Phase 1] Starting recording")
        recording_info = await start_recording_safe(browser)
        recording_active = recording_info is not None

        # Phase 2: Perform actions while recording
        print("\n[Phase 2] Performing recorded actions")
        action_log = await perform_recorded_actions(browser)

        # Phase 3: Stop recording
        if recording_active:
            print("\n[Phase 3] Stopping recording")
            recording_result = await stop_recording_safe(browser)

            if recording_result:
                session_id = recording_result.get("session_id", "N/A")
                screenshots = recording_result.get("screenshots", [])
                duration = recording_result.get("duration_seconds", "N/A")
                print(f"\n  Session ID      : {session_id}")
                print(f"  Screenshots     : {len(screenshots) if isinstance(screenshots, list) else screenshots}")
                print(f"  Duration (sec)  : {duration}")
        else:
            print("\n[Phase 3] Skipping stop (recording was not started)")

        # Summary
        successful_actions = sum(1 for a in action_log if a.get("success"))
        total_actions = len(action_log)
        print(f"\n  Actions performed: {total_actions}")
        print(f"  Successful       : {successful_actions}")
        print(f"  Failed           : {total_actions - successful_actions}")

        # Usage summary
        usage = browser.get_usage_summary()
        print(f"  LLM Usage        : {usage}")

    print("\n  Recording demo complete.")


if __name__ == "__main__":
    asyncio.run(demo_recording())
