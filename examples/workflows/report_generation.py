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
Multi-Source Report Generation
==============================

Collects data from Hacker News (top stories) and Wikipedia (current events),
then merges both data sets into a consolidated daily briefing report.
Demonstrates multi-site navigation, repeated extraction, data merging,
and usage tracking across a complex workflow.

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


HN_URL = "https://news.ycombinator.com"
WIKI_CURRENT_EVENTS_URL = "https://en.wikipedia.org/wiki/Portal:Current_events"


async def collect_hacker_news(browser: FlyBrowser) -> list[dict]:
    """Navigate to Hacker News and extract the top stories."""
    print("  [HN] Navigating to Hacker News...")
    await browser.goto(HN_URL)

    result = await browser.extract(
        "Extract the top 10 stories from the Hacker News front page. "
        "For each story include: rank, title, URL (if shown), points, "
        "number of comments, and the submitter username."
    )

    if result.success:
        print(f"  [HN] Extracted ({result.llm_usage.total_tokens} tokens, "
              f"{result.execution.duration_seconds:.1f}s)")
        result.pprint()
        return result.data if isinstance(result.data, list) else [result.data]

    print(f"  [HN] Extraction failed: {result.error}")
    return []


async def collect_wikipedia_current_events(browser: FlyBrowser) -> list[dict]:
    """Navigate to Wikipedia Current Events and extract headlines."""
    print("  [Wiki] Navigating to Wikipedia Current Events...")
    await browser.goto(WIKI_CURRENT_EVENTS_URL)

    result = await browser.extract(
        "Extract the most recent current events listed on this page. "
        "For each event include: date, headline summary, and the category "
        "(e.g. armed conflicts, politics, science, disasters, etc.)."
    )

    if result.success:
        print(f"  [Wiki] Extracted ({result.llm_usage.total_tokens} tokens, "
              f"{result.execution.duration_seconds:.1f}s)")
        result.pprint()
        return result.data if isinstance(result.data, list) else [result.data]

    print(f"  [Wiki] Extraction failed: {result.error}")
    return []


def build_briefing(hn_stories: list, wiki_events: list) -> str:
    """Merge the two data sources into a human-readable briefing."""
    lines: list[str] = [
        "=" * 64,
        "DAILY INTELLIGENCE BRIEFING",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 64,
        "",
        "SECTION 1 -- TECHNOLOGY (Hacker News Top Stories)",
        "-" * 48,
    ]

    if hn_stories:
        for i, story in enumerate(hn_stories[:10], start=1):
            if isinstance(story, dict):
                title = story.get("title", "Untitled")
                points = story.get("points", "?")
                comments = story.get("comments", "?")
                lines.append(f"  {i}. {title}  [{points} pts, {comments} comments]")
            else:
                lines.append(f"  {i}. {story}")
    else:
        lines.append("  (no data collected)")

    lines += [
        "",
        "SECTION 2 -- WORLD EVENTS (Wikipedia Current Events)",
        "-" * 48,
    ]

    if wiki_events:
        for event in wiki_events[:10]:
            if isinstance(event, dict):
                headline = event.get("headline", event.get("summary", str(event)))
                category = event.get("category", "General")
                lines.append(f"  [{category}]  {headline}")
            else:
                lines.append(f"  - {event}")
    else:
        lines.append("  (no data collected)")

    lines.append("\n" + "=" * 64)
    return "\n".join(lines)


async def main() -> None:
    """Run the multi-source report generation workflow."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Multi-Source Report Generation")
    print(f"Provider: {provider}  |  Model: {model}")
    print("=" * 64)

    hn_stories: list = []
    wiki_events: list = []

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        hn_stories = await collect_hacker_news(browser)
        wiki_events = await collect_wikipedia_current_events(browser)
        usage = browser.get_usage_summary()

    # Build and display the briefing
    briefing = build_briefing(hn_stories, wiki_events)
    print(briefing)

    # Save structured output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "total_tokens": usage.get("total_tokens", 0),
        "sources": {
            "hacker_news": {"url": HN_URL, "stories": hn_stories},
            "wikipedia": {"url": WIKI_CURRENT_EVENTS_URL, "events": wiki_events},
        },
    }
    filename = f"daily_briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nStructured data saved to: {filename}")

    # Final summary
    print(f"\nData collected: {len(hn_stories)} HN stories, {len(wiki_events)} world events")
    print(f"Total LLM tokens: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
