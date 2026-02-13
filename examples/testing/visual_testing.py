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
Example: Visual Testing

Takes screenshots of multiple real public pages, verifies screenshots were
captured correctly, compares sizes, and demonstrates PII masking and
full-page captures. Uses the FlyBrowser screenshot() API which returns
a dict with data_base64, width, and height.

Prerequisites:
    pip install flybrowser
    export ANTHROPIC_API_KEY="sk-ant-..."

Environment Variables:
    ANTHROPIC_API_KEY          - Required. Your LLM provider API key.
    FLYBROWSER_LLM_PROVIDER    - Optional. Defaults to "anthropic".
    FLYBROWSER_LLM_MODEL       - Optional. Defaults to "claude-sonnet-4-5-20250929".
"""

import asyncio
import base64
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from flybrowser import FlyBrowser

PAGES_TO_TEST = [
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com",
    },
    {
        "name": "Wikipedia Main Page",
        "url": "https://en.wikipedia.org/wiki/Main_Page",
    },
    {
        "name": "Example Domain",
        "url": "https://example.com",
    },
]


@dataclass
class ScreenshotResult:
    """Metadata about a captured screenshot."""

    page_name: str
    url: str
    width: int = 0
    height: int = 0
    data_size_bytes: int = 0
    captured: bool = False
    error: Optional[str] = None


@dataclass
class VisualTestReport:
    """Aggregated visual test results."""

    screenshots: list[ScreenshotResult] = field(default_factory=list)

    def add(self, result: ScreenshotResult) -> None:
        self.screenshots.append(result)
        tag = "PASS" if result.captured else "FAIL"
        if result.captured:
            msg = f"{result.width}x{result.height}, {result.data_size_bytes:,} bytes"
        else:
            msg = result.error or "Unknown error"
        print(f"    [{tag}] {result.page_name}: {msg}")

    def print_summary(self) -> None:
        captured = sum(1 for s in self.screenshots if s.captured)
        total = len(self.screenshots)
        print("\n" + "=" * 65)
        print("  VISUAL TESTING REPORT")
        print("=" * 65)
        for s in self.screenshots:
            tag = "PASS" if s.captured else "FAIL"
            print(f"    [{tag}] {s.page_name} ({s.url})")
            if s.captured:
                print(f"           Dimensions: {s.width}x{s.height}")
                print(f"           Size: {s.data_size_bytes:,} bytes")
            if s.error:
                print(f"           Error: {s.error}")
        print("-" * 65)
        print(f"  Total: {total}  |  Captured: {captured}  |  Failed: {total - captured}")
        print("=" * 65)


def parse_screenshot(raw: dict, page_name: str, url: str) -> ScreenshotResult:
    """Parse a screenshot dict into a ScreenshotResult."""
    result = ScreenshotResult(page_name=page_name, url=url)
    try:
        data_b64 = raw.get("data_base64", "")
        if not data_b64:
            result.error = "No data_base64 in screenshot response"
            return result
        image_bytes = base64.b64decode(data_b64)
        result.data_size_bytes = len(image_bytes)
        result.width = raw.get("width", 0)
        result.height = raw.get("height", 0)
        result.captured = result.data_size_bytes > 0
    except Exception as exc:
        result.error = f"Failed to decode screenshot: {exc}"
    return result


async def capture_page_screenshots(browser: FlyBrowser, report: VisualTestReport) -> None:
    """Navigate to each page and capture a screenshot."""
    for page in PAGES_TO_TEST:
        try:
            await browser.goto(page["url"])
            screenshot = await browser.screenshot(full_page=False, mask_pii=True)
            sr = parse_screenshot(screenshot, page["name"], page["url"])
        except Exception as exc:
            sr = ScreenshotResult(
                page_name=page["name"],
                url=page["url"],
                error=str(exc),
            )
        report.add(sr)


async def capture_full_page(browser: FlyBrowser, report: VisualTestReport) -> None:
    """Capture a full-page screenshot of Hacker News for comparison."""
    page_name = "Hacker News (full page)"
    url = "https://news.ycombinator.com"
    try:
        await browser.goto(url)
        screenshot = await browser.screenshot(full_page=True, mask_pii=False)
        sr = parse_screenshot(screenshot, page_name, url)
    except Exception as exc:
        sr = ScreenshotResult(page_name=page_name, url=url, error=str(exc))
    report.add(sr)


async def compare_viewport_vs_fullpage(report: VisualTestReport) -> None:
    """Compare the viewport-only and full-page screenshots of Hacker News."""
    hn_viewport = None
    hn_full = None
    for s in report.screenshots:
        if s.page_name == "Hacker News" and s.captured:
            hn_viewport = s
        if s.page_name == "Hacker News (full page)" and s.captured:
            hn_full = s

    print("\n  --- Viewport vs Full-Page Comparison ---")
    if hn_viewport and hn_full:
        print(f"    Viewport: {hn_viewport.width}x{hn_viewport.height} "
              f"({hn_viewport.data_size_bytes:,} bytes)")
        print(f"    Full page: {hn_full.width}x{hn_full.height} "
              f"({hn_full.data_size_bytes:,} bytes)")
        if hn_full.height > hn_viewport.height:
            print("    [PASS] Full-page screenshot is taller than viewport screenshot.")
        elif hn_full.data_size_bytes > hn_viewport.data_size_bytes:
            print("    [PASS] Full-page screenshot is larger in bytes.")
        else:
            print("    [INFO] Full-page and viewport sizes are similar "
                  "(page may be short enough to fit in viewport).")
    else:
        print("    [SKIP] Could not compare -- one or both screenshots missing.")


async def test_pii_masking(browser: FlyBrowser, report: VisualTestReport) -> None:
    """Capture with PII masking enabled and verify the screenshot is valid."""
    page_name = "Wikipedia (PII masked)"
    url = "https://en.wikipedia.org/wiki/Main_Page"
    try:
        await browser.goto(url)
        screenshot = await browser.screenshot(full_page=False, mask_pii=True)
        sr = parse_screenshot(screenshot, page_name, url)
    except Exception as exc:
        sr = ScreenshotResult(page_name=page_name, url=url, error=str(exc))
    report.add(sr)


async def test_screenshot_after_interaction(browser: FlyBrowser, report: VisualTestReport) -> None:
    """Scroll a page then capture to verify dynamic content."""
    page_name = "Hacker News (after scroll)"
    url = "https://news.ycombinator.com"
    try:
        await browser.goto(url)
        act_result = await browser.act("Scroll down to see more stories.")
        if not act_result.success:
            sr = ScreenshotResult(page_name=page_name, url=url, error="Scroll action failed")
            report.add(sr)
            return
        await asyncio.sleep(1)
        screenshot = await browser.screenshot(full_page=False, mask_pii=True)
        sr = parse_screenshot(screenshot, page_name, url)
    except Exception as exc:
        sr = ScreenshotResult(page_name=page_name, url=url, error=str(exc))
    report.add(sr)


async def main() -> None:
    """Run the full visual testing suite."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser Visual Testing")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 65)

    report = VisualTestReport()

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Phase 1: Capture screenshots of all pages
        print("\n[Phase 1] Capture page screenshots (viewport, PII masked)")
        await capture_page_screenshots(browser, report)

        # Phase 2: Full-page capture
        print("\n[Phase 2] Full-page screenshot")
        await capture_full_page(browser, report)

        # Phase 3: Compare viewport vs full-page
        await compare_viewport_vs_fullpage(report)

        # Phase 4: PII masking test
        print("\n[Phase 4] PII-masked screenshot")
        await test_pii_masking(browser, report)

        # Phase 5: Screenshot after interaction
        print("\n[Phase 5] Screenshot after scrolling")
        await test_screenshot_after_interaction(browser, report)

        # Usage summary
        usage = browser.get_usage_summary()
        print(f"\n  LLM Usage: {usage}")

    report.print_summary()

    failed = sum(1 for s in report.screenshots if not s.captured)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
