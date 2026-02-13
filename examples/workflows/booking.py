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
Restaurant / Venue Finder Workflow
===================================

Searches for restaurants on Yelp, extracts detailed listings, scores and
ranks them by user-defined criteria, and produces a recommendation report.
Demonstrates agent-driven navigation, act + extract chaining, multi-step
page interaction, and structured comparison logic.

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


SEARCH_URL = "https://www.yelp.com"
SEARCH_LOCATION = "San Francisco, CA"
SEARCH_CUISINE = "Italian"
MIN_RATING = 4.0


async def search_restaurants(browser: FlyBrowser) -> None:
    """Navigate to Yelp and perform a restaurant search."""
    print("  Navigating to Yelp...")
    await browser.goto(SEARCH_URL)

    instruction = (
        f"Search for '{SEARCH_CUISINE} restaurants' in '{SEARCH_LOCATION}'. "
        "Type the cuisine into the search field and the location into the "
        "location field, then submit the search."
    )
    act_result = await browser.act(instruction)

    if act_result.success:
        print(f"    Search submitted ({act_result.execution.duration_seconds:.1f}s)")
    else:
        print(f"    Search action failed: {act_result.error}")


async def extract_listings(browser: FlyBrowser) -> list[dict]:
    """Extract restaurant listings from the current search results page."""
    print("  Extracting restaurant listings...")
    result = await browser.extract(
        "Extract the first 5 restaurant listings shown on this page. "
        "For each, include: name, star rating (number), number of reviews, "
        "price range (e.g. $$), neighborhood, and a short description or "
        "cuisine tags if available."
    )

    if result.success:
        print(f"    Extracted ({result.llm_usage.total_tokens} tokens)")
        result.pprint()
        listings = result.data if isinstance(result.data, list) else [result.data]
        return listings

    print(f"    Extraction failed: {result.error}")
    return []


def rank_restaurants(listings: list[dict]) -> list[dict]:
    """Score and rank restaurants based on rating and review count."""
    scored: list[dict] = []
    for entry in listings:
        if not isinstance(entry, dict):
            continue
        try:
            rating = float(entry.get("star_rating") or entry.get("rating") or 0)
        except (ValueError, TypeError):
            rating = 0.0
        try:
            reviews = int(entry.get("number_of_reviews") or entry.get("reviews") or 0)
        except (ValueError, TypeError):
            reviews = 0

        # Simple weighted score: rating * 20 + log-scaled review count
        import math
        score = rating * 20 + math.log1p(reviews) * 5
        scored.append({**entry, "_score": round(score, 2), "_rating": rating, "_reviews": reviews})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def print_recommendation_report(ranked: list[dict]) -> None:
    """Print a formatted recommendation report."""
    print("\n" + "=" * 64)
    print("RESTAURANT RECOMMENDATION REPORT")
    print(f"Location: {SEARCH_LOCATION}  |  Cuisine: {SEARCH_CUISINE}")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 64)

    for i, r in enumerate(ranked, start=1):
        name = r.get("name", "Unknown")
        rating = r.get("_rating", "?")
        reviews = r.get("_reviews", "?")
        price = r.get("price_range") or r.get("price") or "N/A"
        area = r.get("neighborhood") or r.get("area") or "N/A"
        score = r.get("_score", 0)
        marker = " << TOP PICK" if i == 1 else ""
        print(f"\n  #{i}  {name}{marker}")
        print(f"      Rating: {rating}/5  |  Reviews: {reviews}  |  Price: {price}")
        print(f"      Area: {area}  |  Score: {score}")

    meets_min = [r for r in ranked if r.get("_rating", 0) >= MIN_RATING]
    print(f"\n  {len(meets_min)}/{len(ranked)} meet minimum rating of {MIN_RATING}")
    print("=" * 64)


async def main() -> None:
    """Run the restaurant finder workflow end to end."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Restaurant Finder Workflow")
    print(f"Provider: {provider}  |  Model: {model}")
    print("=" * 64)

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        await search_restaurants(browser)
        listings = await extract_listings(browser)

        # Observe the results page as a sanity check
        obs = await browser.observe("Are restaurant search results displayed on this page?")
        if obs.success:
            print(f"  Observation confirmed: {obs.data}")

        usage = browser.get_usage_summary()

    if listings:
        ranked = rank_restaurants(listings)
        print_recommendation_report(ranked)

        # Persist
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "search": {"location": SEARCH_LOCATION, "cuisine": SEARCH_CUISINE},
            "total_tokens": usage.get("total_tokens", 0),
            "restaurants": ranked,
        }
        filename = f"restaurant_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as fh:
            json.dump(output, fh, indent=2, default=str)
        print(f"\nData saved to: {filename}")
    else:
        print("\nNo listings extracted -- cannot generate report.")

    print(f"Total LLM tokens: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
