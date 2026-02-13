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
fields (title, URL, score, author, comments count) for each story.

Performance strategy:
  - Uses direct URL navigation (goto) instead of agent-driven "click More"
    to avoid expensive multi-step LLM reasoning for simple pagination.
  - Asks the LLM to respond in JSON format for easier downstream parsing.
  - Handles both structured (JSON) and free-text LLM responses gracefully.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export FLYBROWSER_LLM_PROVIDER="anthropic"     # optional
    export FLYBROWSER_LLM_MODEL="claude-sonnet-4-5-20250929"  # optional
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone

from flybrowser import FlyBrowser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")
MAX_PAGES = 2  # Number of HN pages to scrape (30 stories per page)


def _parse_stories(raw: str, page_num: int) -> list[dict]:
    """Parse LLM response into a list of story dicts.

    The LLM may return:
      - A JSON array of objects (ideal)
      - A JSON array wrapped in markdown code fences
      - Free-text with story data embedded
    """
    if not raw:
        return []

    text = raw.strip()

    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    item["page"] = page_num
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: return the raw text as a single item
    return [{"raw_text": text, "page": page_num}]


async def scrape_hackernews(max_pages: int = MAX_PAGES) -> tuple:
    """Scrape Hacker News stories across multiple pages."""
    all_stories: list[dict] = []

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        for page_num in range(1, max_pages + 1):
            # Direct URL navigation — no LLM cost for pagination
            url = f"https://news.ycombinator.com/news?p={page_num}"
            await browser.goto(url)
            print(f"\n--- Page {page_num}/{max_pages} ({url}) ---")

            result = await browser.extract(
                "Extract every story visible on this page as a JSON array. "
                "Each object must have: rank (int), title (str), url (str), "
                "score (int), author (str), comments_count (int, 0 if none). "
                "Return ONLY the JSON array, no other text."
            )

            if result.success and result.data:
                stories = _parse_stories(str(result.data), page_num)
                all_stories.extend(stories)
                print(f"  Extracted {len(stories)} stories")
            else:
                print(f"  Extraction failed: {result.error}")
                break

        usage = browser.get_usage_summary()

    return all_stories, usage


def display_results(stories: list[dict], usage: dict) -> None:
    """Pretty-print the scraping results and token usage."""
    print(f"\n{'=' * 70}")
    print(f"  HACKER NEWS SCRAPE RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total stories collected: {len(stories)}")
    print(f"  Session total tokens:    {usage.get('total_tokens', 0):,}")
    print(f"  Estimated cost:          ${usage.get('cost_usd', 0):.4f}")
    print(f"  Model:                   {usage.get('model', 'N/A')}")

    # Display top stories by score
    scored = [s for s in stories if isinstance(s.get("score"), (int, float))]
    scored.sort(key=lambda s: s.get("score", 0), reverse=True)

    if scored:
        print(f"\n  Top 10 stories by score:")
        print(f"  {'Rank':<6} {'Score':<8} {'Comments':<10} {'Title'}")
        print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*44}")
        for story in scored[:10]:
            print(f"  {story.get('rank', '?'):<6} "
                  f"{story.get('score', 0):<8} "
                  f"{story.get('comments_count', 0):<10} "
                  f"{str(story.get('title', ''))[:44]}")
    else:
        print("\n  (LLM returned free-text — showing raw extractions)")
        for i, story in enumerate(stories[:5], 1):
            text = story.get("raw_text", str(story))[:120]
            print(f"  [{i}] {text}...")


async def main() -> None:
    """Entry point: scrape, display, and save results."""
    print(f"Hacker News Scraper | Provider: {PROVIDER} | Model: {MODEL}")
    print(f"Pages to scrape: {MAX_PAGES}")

    stories, usage = await scrape_hackernews()
    display_results(stories, usage)

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
