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
Website Health Monitor
======================

Checks the health of multiple public websites (github.com, wikipedia.org,
news.ycombinator.com), extracts page status information, takes screenshots,
and produces a consolidated health report.  Demonstrates observe, extract,
screenshot, and error handling across unreliable network conditions.

Environment variables:
    ANTHROPIC_API_KEY          - API key for the configured LLM provider
    FLYBROWSER_LLM_PROVIDER   - LLM provider (default: "anthropic")
    FLYBROWSER_LLM_MODEL      - LLM model   (default: "claude-sonnet-4-5-20250929")
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from flybrowser import FlyBrowser


SITES_TO_MONITOR = [
    {
        "name": "GitHub",
        "url": "https://github.com",
        "expect_text": "Sign up",
    },
    {
        "name": "Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "expect_text": "Featured article",
    },
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com",
        "expect_text": "Hacker News",
    },
]


async def check_site(browser: FlyBrowser, site: dict) -> dict:
    """
    Run a full health check on a single site: navigate, observe key elements,
    extract status details, and capture a screenshot.
    """
    report: dict = {
        "name": site["name"],
        "url": site["url"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reachable": False,
        "has_expected_content": False,
        "page_title": None,
        "screenshot_bytes": 0,
        "issues": [],
    }

    try:
        # Step 1 -- Navigate
        await browser.goto(site["url"])
        report["reachable"] = True

        # Step 2 -- Observe expected element
        obs = await browser.observe(
            f"Is the text '{site['expect_text']}' visible on the page?"
        )
        if obs.success:
            report["has_expected_content"] = True
            print(f"    Observe OK  ({obs.llm_usage.total_tokens} tokens)")
        else:
            report["issues"].append(f"Expected text not found: {site['expect_text']}")

        # Step 3 -- Extract page health details
        ext = await browser.extract(
            "Extract the page title and report whether the page appears to be "
            "loading correctly. Note any error banners, outage notices, or "
            "missing content."
        )
        if ext.success:
            report["page_title"] = ext.data if isinstance(ext.data, str) else str(ext.data)
            print(f"    Extract OK  (duration {ext.execution.duration_seconds:.2f}s)")
        else:
            report["issues"].append(f"Extraction error: {ext.error}")

        # Step 4 -- Screenshot
        screenshot = await browser.screenshot()
        report["screenshot_bytes"] = len(screenshot.get("data_base64", ""))

    except Exception as exc:
        report["issues"].append(f"Unexpected error: {exc}")

    return report


def print_health_dashboard(reports: list[dict]) -> None:
    """Render a summary dashboard to the terminal."""
    print("\n" + "=" * 64)
    print("WEBSITE HEALTH DASHBOARD")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 64)

    for r in reports:
        status = "HEALTHY" if r["reachable"] and not r["issues"] else "DEGRADED"
        icon = "[OK]" if status == "HEALTHY" else "[!!]"
        print(f"\n  {icon}  {r['name']}  ({r['url']})")
        print(f"       Reachable: {r['reachable']}")
        print(f"       Expected content present: {r['has_expected_content']}")
        print(f"       Screenshot size: {r['screenshot_bytes']} bytes")
        if r["issues"]:
            for issue in r["issues"]:
                print(f"       Issue: {issue}")

    healthy = sum(1 for r in reports if r["reachable"] and not r["issues"])
    print(f"\n  Overall: {healthy}/{len(reports)} sites healthy")
    print("=" * 64)


async def main() -> None:
    """Run health checks against all configured sites."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Website Health Monitor")
    print(f"Provider: {provider}  |  Model: {model}")
    print(f"Sites: {len(SITES_TO_MONITOR)}")
    print("=" * 64)

    reports: list[dict] = []

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        for site in SITES_TO_MONITOR:
            print(f"\n  Checking {site['name']}...")
            report = await check_site(browser, site)
            reports.append(report)

        usage = browser.get_usage_summary()

    # Display dashboard
    print_health_dashboard(reports)

    # Persist results
    output_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    serializable = [{k: v for k, v in r.items()} for r in reports]
    with open(output_file, "w") as fh:
        json.dump(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "sites": serializable},
            fh,
            indent=2,
        )
    print(f"\nFull report saved to: {output_file}")
    print(f"Total LLM tokens used: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
