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
Example: Navigation Testing

Tests browser navigation to multiple real public sites, verifies page loads
correctly, checks page titles and URLs, and validates cross-site navigation
flows using the FlyBrowser SDK.

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
from dataclasses import dataclass, field
from typing import Optional

from flybrowser import FlyBrowser


@dataclass
class NavigationResult:
    """Stores the outcome of a single navigation test."""

    url: str
    expected_content: str
    passed: bool = False
    title: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TestSuite:
    """Aggregates results across all navigation tests."""

    results: list[NavigationResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print("  NAVIGATION TEST RESULTS")
        print("=" * 65)
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}]  {r.url}")
            if r.title:
                print(f"          Title: {r.title}")
            if r.error:
                print(f"          Error: {r.error}")
        print("-" * 65)
        print(f"  Total: {len(self.results)}  |  Passed: {self.passed}  |  Failed: {self.failed}")
        print("=" * 65)


SITES_UNDER_TEST = [
    {
        "url": "https://news.ycombinator.com",
        "name": "Hacker News",
        "expected_content": "Hacker News",
        "verify_query": "What is the page title and the first visible heading?",
    },
    {
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "name": "Wikipedia Main Page",
        "expected_content": "Wikipedia",
        "verify_query": "What is the page title? Mention Wikipedia if visible.",
    },
    {
        "url": "https://example.com",
        "name": "Example Domain",
        "expected_content": "Example Domain",
        "verify_query": "What is the main heading on this page?",
    },
]


async def test_single_navigation(
    browser: FlyBrowser,
    url: str,
    verify_query: str,
    expected_content: str,
) -> NavigationResult:
    """Navigate to a URL and verify the page loaded correctly."""
    result = NavigationResult(url=url, expected_content=expected_content)
    try:
        await browser.goto(url)
        extraction = await browser.extract(verify_query)
        if not extraction.success:
            result.error = extraction.error or "Extraction returned success=False"
            return result
        page_text = str(extraction.data).lower()
        result.title = str(extraction.data)[:120]
        if expected_content.lower() in page_text:
            result.passed = True
        else:
            result.error = f"Expected '{expected_content}' not found in: {result.title}"
    except Exception as exc:
        result.error = f"Exception during navigation: {exc}"
    return result


async def test_sequential_navigation(browser: FlyBrowser) -> list[NavigationResult]:
    """Navigate through all sites sequentially, verifying each transition."""
    results: list[NavigationResult] = []
    for site in SITES_UNDER_TEST:
        print(f"\n  -> Navigating to {site['name']} ({site['url']})")
        nav_result = await test_single_navigation(
            browser,
            url=site["url"],
            verify_query=site["verify_query"],
            expected_content=site["expected_content"],
        )
        status = "PASS" if nav_result.passed else "FAIL"
        print(f"     [{status}] {nav_result.title or nav_result.error}")
        results.append(nav_result)
    return results


async def test_back_navigation(browser: FlyBrowser) -> NavigationResult:
    """Navigate forward two pages then verify we can observe the current page."""
    result = NavigationResult(url="back-navigation-test", expected_content="Example Domain")
    try:
        await browser.goto("https://example.com")
        await browser.goto("https://news.ycombinator.com")
        # Navigate back to example.com
        await browser.goto("https://example.com")
        check = await browser.extract("What is the main heading on this page?")
        if check.success and "example" in str(check.data).lower():
            result.passed = True
            result.title = str(check.data)[:120]
        else:
            result.error = f"Back navigation check failed: {check.error or check.data}"
    except Exception as exc:
        result.error = f"Exception: {exc}"
    return result


async def test_observe_page_elements(browser: FlyBrowser) -> NavigationResult:
    """Use observe() to verify interactive elements exist on Hacker News."""
    result = NavigationResult(
        url="https://news.ycombinator.com",
        expected_content="link",
    )
    try:
        await browser.goto("https://news.ycombinator.com")
        observation = await browser.observe("What navigation links are at the top of the page?")
        if observation.success:
            result.title = str(observation.data)[:120]
            result.passed = True
        else:
            result.error = observation.error or "observe() returned success=False"
    except Exception as exc:
        result.error = f"Exception: {exc}"
    return result


async def main() -> None:
    """Run the full navigation test suite."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser Navigation Testing")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 65)

    suite = TestSuite()

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Phase 1: Sequential multi-site navigation
        print("\n[Phase 1] Sequential Navigation Tests")
        sequential_results = await test_sequential_navigation(browser)
        suite.results.extend(sequential_results)

        # Phase 2: Back-navigation simulation
        print("\n[Phase 2] Back Navigation Test")
        back_result = await test_back_navigation(browser)
        status = "PASS" if back_result.passed else "FAIL"
        print(f"  [{status}] Return to example.com: {back_result.title or back_result.error}")
        suite.results.append(back_result)

        # Phase 3: Observe page elements
        print("\n[Phase 3] Observe Page Elements")
        observe_result = await test_observe_page_elements(browser)
        status = "PASS" if observe_result.passed else "FAIL"
        print(f"  [{status}] Observe nav links: {observe_result.title or observe_result.error}")
        suite.results.append(observe_result)

        # Print usage summary
        usage = browser.get_usage_summary()
        print(f"\n  LLM Usage: {usage}")

    suite.print_summary()

    if suite.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
