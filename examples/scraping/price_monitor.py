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
Price Monitor and Comparison Engine

Monitors book prices on books.toscrape.com and demonstrates three pricing
intelligence capabilities:

  1. Snapshot extraction - Capture current prices for a set of book categories.
  2. Cross-category comparison - Compare the cheapest book in each category.
  3. Price-tier classification - Bucket books into price tiers (budget,
     mid-range, premium) and identify outliers.

This script is designed to run repeatedly (e.g., via cron).  Each run appends
a timestamped snapshot to a local JSON file so you can track price changes
over time.

Business value:
  - Retail pricing strategy: understand competitor price distribution.
  - Procurement: identify the cheapest sources for bulk book purchases.
  - Market research: detect price increases or promotions automatically.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export FLYBROWSER_LLM_PROVIDER="anthropic"     # optional
    export FLYBROWSER_LLM_MODEL="claude-sonnet-4-5-20250929"  # optional
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flybrowser import FlyBrowser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")
HISTORY_FILE = Path("price_history.json")

BASE_URL = "https://books.toscrape.com"
CATEGORIES = [
    {"name": "Travel", "url": f"{BASE_URL}/catalogue/category/books/travel_2/index.html"},
    {"name": "Mystery", "url": f"{BASE_URL}/catalogue/category/books/mystery_3/index.html"},
    {"name": "Science Fiction", "url": f"{BASE_URL}/catalogue/category/books/science-fiction_16/index.html"},
]


def section(title: str) -> None:
    """Print a visible section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def load_history() -> list[dict]:
    """Load existing price history from disk."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as fh:
            return json.load(fh)
    return []


def save_history(history: list[dict]) -> None:
    """Append-safe write of price history."""
    with open(HISTORY_FILE, "w") as fh:
        json.dump(history, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Price extraction helpers
# ---------------------------------------------------------------------------
def parse_price(price_str: str) -> float:
    """Best-effort parse a price string like '51.77' or 'GBP 51.77' to float."""
    try:
        digits = "".join(c for c in str(price_str) if c.isdigit() or c == ".")
        return float(digits)
    except (ValueError, TypeError):
        return 0.0


def classify_tier(price: float) -> str:
    """Classify a price into budget / mid-range / premium."""
    if price < 20.0:
        return "budget"
    elif price < 40.0:
        return "mid-range"
    return "premium"


# ---------------------------------------------------------------------------
# Core scraping logic
# ---------------------------------------------------------------------------
async def snapshot_category(browser: FlyBrowser, category: dict) -> list[dict]:
    """
    Extract all book prices from a single category page.

    Returns list of dicts with: title, price, star_rating, in_stock.
    """
    await browser.goto(category["url"])

    result = await browser.extract(
        "Extract every book on this page. For each return: "
        "title (string), price (string with currency), "
        "star_rating (integer 1-5), in_stock (boolean)."
    )

    if not result.success:
        print(f"    Extraction failed for {category['name']}: {result.error}")
        return []

    books = result.data if isinstance(result.data, list) else [result.data]
    print(f"    {category['name']}: {len(books)} books "
          f"({result.llm_usage.total_tokens:,} tokens, "
          f"{result.execution.duration_seconds:.1f}s)")
    return books


async def run_price_monitor() -> dict:
    """
    Full monitoring run: snapshot every category, compare, and classify.

    Returns a snapshot dict suitable for appending to history.
    """
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "categories": {},
    }

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        # ---- 1. Snapshot each category ----
        section("1/3  Price Snapshot per Category")
        for cat in CATEGORIES:
            books = await snapshot_category(browser, cat)
            snapshot["categories"][cat["name"]] = books

        # ---- 2. Cross-category comparison ----
        section("2/3  Cross-Category Comparison (cheapest book per category)")
        comparison: list[dict] = []
        for cat_name, books in snapshot["categories"].items():
            valid = [b for b in books if isinstance(b, dict)]
            if not valid:
                continue
            cheapest = min(valid, key=lambda b: parse_price(b.get("price", "999")))
            cheapest_price = parse_price(cheapest.get("price", "0"))
            entry = {
                "category": cat_name,
                "title": cheapest.get("title", "Unknown"),
                "price": cheapest.get("price", "N/A"),
                "price_numeric": cheapest_price,
            }
            comparison.append(entry)

        comparison.sort(key=lambda e: e["price_numeric"])

        print(f"\n  {'Category':<20} {'Price':<10} {'Title'}")
        print(f"  {'-'*20} {'-'*10} {'-'*38}")
        for entry in comparison:
            title = str(entry["title"])[:38]
            print(f"  {entry['category']:<20} {entry['price']:<10} {title}")

        if comparison:
            best = comparison[0]
            print(f"\n  >>> Best deal: '{best['title']}' in {best['category']} "
                  f"at {best['price']}")

        snapshot["comparison"] = comparison

        # ---- 3. Price-tier classification ----
        section("3/3  Price Tier Analysis")
        tiers: dict[str, list] = {"budget": [], "mid-range": [], "premium": []}
        for cat_name, books in snapshot["categories"].items():
            for book in books:
                if not isinstance(book, dict):
                    continue
                price = parse_price(book.get("price", "0"))
                tier = classify_tier(price)
                tiers[tier].append({
                    "title": book.get("title", "Unknown"),
                    "price": price,
                    "category": cat_name,
                })

        for tier_name, tier_books in tiers.items():
            count = len(tier_books)
            if count > 0:
                avg = sum(b["price"] for b in tier_books) / count
                print(f"  {tier_name.upper():<12} {count:>3} books  "
                      f"(avg price: ${avg:.2f})")
            else:
                print(f"  {tier_name.upper():<12}   0 books")

        snapshot["tier_summary"] = {
            tier: len(books) for tier, books in tiers.items()
        }

        # ---- Session usage ----
        section("Session Usage")
        usage = browser.get_usage_summary()
        print(f"  Total tokens : {usage.get('total_tokens', 0):,}")
        print(f"  API calls    : {usage.get('calls_count', 0)}")
        print(f"  Est. cost    : ${usage.get('cost_usd', 0):.4f}")

        snapshot["usage"] = usage

    return snapshot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    """Run the price monitor and persist results."""

    print(f"Price Monitor | Provider: {PROVIDER} | Model: {MODEL}")
    print(f"Categories: {', '.join(c['name'] for c in CATEGORIES)}")

    try:
        snapshot = await run_price_monitor()
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        return

    # Append to history file
    history = load_history()
    history.append(snapshot)
    save_history(history)

    total_runs = len(history)
    print(f"\nSnapshot saved. History now contains {total_runs} run(s).")

    if total_runs > 1:
        prev = history[-2]
        print(f"  Previous run: {prev.get('timestamp', 'unknown')}")
        print(f"  Current run:  {snapshot['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
