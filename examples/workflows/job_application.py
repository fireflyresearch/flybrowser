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
Job Search Workflow
===================

Navigates to Hacker News "Who is Hiring?" threads, extracts job listings,
filters them by criteria (remote, language, seniority), and produces a
curated shortlist.  Demonstrates agent-driven navigation, act for
interaction, extract for data pulling, observe for verification, and
detailed execution / LLM usage reporting.

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


HN_JOBS_URL = "https://news.ycombinator.com/jobs"

FILTER_CRITERIA = {
    "keywords": ["Python", "backend", "remote"],
    "exclude": ["crypto", "blockchain"],
    "seniority": "senior",
}


async def navigate_to_jobs(browser: FlyBrowser) -> bool:
    """Navigate to the Hacker News jobs page."""
    print("  Navigating to HN Jobs...")
    await browser.goto(HN_JOBS_URL)

    obs = await browser.observe("Is this the Hacker News jobs listing page?")
    if obs.success:
        print(f"    Page confirmed ({obs.llm_usage.total_tokens} tokens)")
        return True

    print(f"    Could not verify page: {obs.error}")
    return False


async def extract_job_listings(browser: FlyBrowser) -> list[dict]:
    """Extract job listings from the current page."""
    print("  Extracting job listings...")
    result = await browser.extract(
        "Extract up to 15 job listings from this page. For each listing "
        "include: title, company name, location (or 'Remote' if remote), "
        "and the URL if visible. If any tags like language requirements "
        "or seniority are mentioned, include those too."
    )

    if result.success:
        print(f"    Extracted ({result.llm_usage.total_tokens} tokens, "
              f"{result.execution.duration_seconds:.1f}s)")
        result.pprint()
        return result.data if isinstance(result.data, list) else [result.data]

    print(f"    Extraction failed: {result.error}")
    return []


def filter_jobs(listings: list[dict], criteria: dict) -> list[dict]:
    """
    Filter job listings against user-defined criteria.
    A listing matches if it contains at least one keyword and none of the
    exclusion terms (case-insensitive check across all string values).
    """
    keywords = [kw.lower() for kw in criteria.get("keywords", [])]
    excludes = [ex.lower() for ex in criteria.get("exclude", [])]

    matched: list[dict] = []
    for job in listings:
        if not isinstance(job, dict):
            continue
        blob = " ".join(str(v) for v in job.values()).lower()

        has_keyword = any(kw in blob for kw in keywords) if keywords else True
        has_exclude = any(ex in blob for ex in excludes)

        if has_keyword and not has_exclude:
            matched.append(job)

    return matched


def print_job_report(all_jobs: list[dict], filtered: list[dict]) -> None:
    """Print a formatted job search report."""
    print("\n" + "=" * 64)
    print("JOB SEARCH REPORT")
    print(f"Source: Hacker News Jobs")
    print(f"Keywords: {', '.join(FILTER_CRITERIA['keywords'])}")
    print(f"Exclude:  {', '.join(FILTER_CRITERIA['exclude'])}")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 64)

    print(f"\n  Total listings found: {len(all_jobs)}")
    print(f"  Matching criteria:    {len(filtered)}")

    if filtered:
        print("\n  --- Matching Jobs ---")
        for i, job in enumerate(filtered, start=1):
            title = job.get("title", "Untitled")
            company = job.get("company") or job.get("company_name") or "Unknown"
            location = job.get("location", "N/A")
            print(f"\n  {i}. {title}")
            print(f"     Company:  {company}")
            print(f"     Location: {location}")
    else:
        print("\n  No jobs matched the filter criteria.")

    print("\n" + "=" * 64)


async def main() -> None:
    """Run the job search workflow."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Job Search Workflow")
    print(f"Provider: {provider}  |  Model: {model}")
    print("=" * 64)

    all_jobs: list[dict] = []

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        if not await navigate_to_jobs(browser):
            print("Failed to reach jobs page -- aborting.")
            return

        all_jobs = await extract_job_listings(browser)

        # Screenshot for audit trail
        screenshot = await browser.screenshot()
        print(f"  Screenshot: {len(screenshot.get('data_base64', ''))} bytes")

        usage = browser.get_usage_summary()

    # Apply filters
    filtered = filter_jobs(all_jobs, FILTER_CRITERIA)
    print_job_report(all_jobs, filtered)

    # Persist results
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": HN_JOBS_URL,
        "criteria": FILTER_CRITERIA,
        "total_tokens": usage.get("total_tokens", 0),
        "all_listings": all_jobs,
        "filtered_listings": filtered,
    }
    filename = f"job_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved to: {filename}")
    print(f"Total LLM tokens: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
