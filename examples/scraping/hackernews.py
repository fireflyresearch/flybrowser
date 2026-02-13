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
Advanced Hacker News Scraper with Pagination

Scrapes stories across multiple pages of Hacker News, extracting structured
fields (title, URL, score, author, comments count) for each story.  Pagination
is driven by clicking the "More" link at the bottom of each page, so extraction
and navigation happen in a single browser session.

Business value:
  - Competitive intelligence: monitor trending tech topics daily.
  - Content curation: feed extracted stories into a recommendation pipeline.
  - Analytics: track which domains and authors dominate HN over time.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export FLYBROWSER_LLM_PROVIDER="anthropic"     # optional
    export FLYBROWSER_LLM_MODEL="claude-sonnet-4-5-20250929"  # optional
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from flybrowser import FlyBrowser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")
MAX_PAGES = 3  # Number of HN pages to scrape (30 stories per page)
START_URL = "https://news.ycombinator.com"


async def scrape_hackernews(max_pages: int = MAX_PAGES) -> list[dict]:
    """
    Scrape Hacker News stories across multiple pages.

    For each page the scraper:
      1. Extracts all stories with structured fields.
      2. Clicks the "More" link to advance to the next page.

    Args:
        max_pages: How many pages to scrape (default 3 = ~90 stories).

    Returns:
        Aggregated list of story dicts.
    """
    all_stories: list[dict] = []
    cumulative_tokens = 0

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        await browser.goto(START_URL)
        print(f"Navigated to {START_URL}")

        for page_num in range(1, max_pages + 1):
            print(f"\n--- Page {page_num}/{max_pages} ---")

            # Extract stories from the current page
            result = await browser.extract(
                "Extract every story on this page. For each story return: "
                "rank (integer), title (string), url (the link the title "
                "points to), score (integer points), author (string username), "
                "and comments_count (integer number of comments, 0 if none)."
            )

            if result.success and result.data:
                stories = result.data if isinstance(result.data, list) else [result.data]

                # Tag each story with the page it came from
                for story in stories:
                    if isinstance(story, dict):
                        story["page"] = page_num

                all_stories.extend(stories)
                cumulative_tokens += result.llm_usage.total_tokens

                print(f"  Extracted {len(stories)} stories "
                      f"(tokens this page: {result.llm_usage.total_tokens:,}, "
                      f"duration: {result.execution.duration_seconds:.1f}s)")
            else:
                print(f"  Extraction failed: {result.error}")
                break

            # Navigate to the next page by clicking "More"
            if page_num < max_pages:
                nav = await browser.act(
                    "Click the 'More' link at the bottom of the story list"
                )
                if not nav.success:
                    print("  Could not navigate to next page. Stopping.")
                    break
                cumulative_tokens += nav.llm_usage.total_tokens

        # ----- Session-level summary -----
        usage = browser.get_usage_summary()

    return all_stories, usage, cumulative_tokens


def display_results(stories: list[dict], usage: dict, cumulative_tokens: int) -> None:
    """Pretty-print the scraping results and token usage."""
    print(f"\n{'=' * 70}")
    print(f"  HACKER NEWS SCRAPE RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total stories collected: {len(stories)}")
    print(f"  Cumulative tokens used:  {cumulative_tokens:,}")
    print(f"  Session total tokens:    {usage.get('total_tokens', 0):,}")
    print(f"  Session API calls:       {usage.get('calls_count', 0)}")
    print(f"  Estimated cost:          ${usage.get('cost_usd', 0):.4f}")

    # Display top stories by score
    scored = [s for s in stories if isinstance(s, dict) and isinstance(s.get("score"), (int, float))]
    scored.sort(key=lambda s: s.get("score", 0), reverse=True)

    print(f"\n  Top 10 stories by score:")
    print(f"  {'Rank':<6} {'Score':<8} {'Comments':<10} {'Title'}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*44}")

    for story in scored[:10]:
        rank = story.get("rank", "?")
        score = story.get("score", 0)
        comments = story.get("comments_count", 0)
        title = str(story.get("title", ""))[:44]
        print(f"  {rank:<6} {score:<8} {comments:<10} {title}")


async def main() -> None:
    """Entry point: scrape, display, and save results."""
    print(f"Hacker News Scraper | Provider: {PROVIDER} | Model: {MODEL}")
    print(f"Pages to scrape: {MAX_PAGES}")

    try:
        stories, usage, cumulative_tokens = await scrape_hackernews()
    except Exception as exc:
        print(f"\nFatal error during scraping: {exc}")
        return

    display_results(stories, usage, cumulative_tokens)

    # Persist to JSON
    output = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total_stories": len(stories),
        "stories": stories,
        "usage": usage,
    }

    filename = f"hackernews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
