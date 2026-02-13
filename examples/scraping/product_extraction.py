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
E-Commerce Product Extraction with Structured Schemas

Demonstrates three real-world extraction patterns against books.toscrape.com,
a freely available practice e-commerce site:

  1. Catalog extraction  - Pull every book on a category page with price,
                           rating, and availability.
  2. Product detail      - Navigate into an individual product page and
                           extract full metadata (description, UPC, tax, etc.).
  3. Category analysis   - Classify the catalog by price tier and star rating,
                           producing a summary useful for pricing analytics.

Business value:
  - Inventory monitoring: detect new products or stock-outs across competitors.
  - Pricing intelligence: feed structured product data into a BI dashboard.
  - Content enrichment: populate product databases from public catalogs.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export FLYBROWSER_LLM_PROVIDER="anthropic"     # optional
    export FLYBROWSER_LLM_MODEL="claude-sonnet-4-5-20250929"  # optional
"""

import asyncio
import json
import os

from flybrowser import FlyBrowser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

CATALOG_URL = "https://books.toscrape.com"
CATEGORY_URL = "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html"


def section(title: str) -> None:
    """Print a visible section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Catalog extraction - all books on a listing page
# ---------------------------------------------------------------------------
async def extract_catalog(browser: FlyBrowser, url: str) -> list[dict]:
    """
    Extract every book shown on a catalog/category page.

    Returns a list of dicts, each with: title, price, rating, availability.
    """
    await browser.goto(url)
    print(f"  Navigated to {url}")

    result = await browser.extract(
        "Extract every book shown on this page. For each book return: "
        "title (string), price (string including currency symbol), "
        "star_rating (integer 1-5), and in_stock (boolean)."
    )

    if not result.success:
        print(f"  Extraction failed: {result.error}")
        return []

    books = result.data if isinstance(result.data, list) else [result.data]
    print(f"  Extracted {len(books)} books "
          f"({result.llm_usage.total_tokens:,} tokens, "
          f"{result.execution.duration_seconds:.1f}s)")
    return books


# ---------------------------------------------------------------------------
# 2. Product detail extraction from an individual book page
# ---------------------------------------------------------------------------
async def extract_product_detail(browser: FlyBrowser, book_title: str) -> dict | None:
    """
    Click into a book and extract its full detail page.

    Returns a dict with: title, price, description, upc, availability,
    num_reviews, tax, and category.
    """
    # Navigate into the book
    nav = await browser.act(
        f"Click the link for the book titled '{book_title}'"
    )
    if not nav.success:
        print(f"  Could not navigate to '{book_title}': {nav.error}")
        return None

    result = await browser.extract(
        "Extract the full product information from this book detail page: "
        "title, price (with currency), star_rating (1-5), description "
        "(the paragraph below the title), UPC code, availability text, "
        "number_of_reviews, tax amount, and product category."
    )

    if not result.success:
        print(f"  Detail extraction failed: {result.error}")
        return None

    print(f"  Detail extracted ({result.llm_usage.total_tokens:,} tokens)")
    return result.data


# ---------------------------------------------------------------------------
# 3. Category analysis - classify books by price tier and rating
# ---------------------------------------------------------------------------
async def analyze_catalog(browser: FlyBrowser, url: str) -> dict | None:
    """
    Ask the agent to analyze the catalog and produce a pricing summary.

    Returns a structured summary with price tiers and rating distribution.
    """
    await browser.goto(url)

    result = await browser.extract(
        "Analyze all books on this page and produce a summary: "
        "1) total_books (int), "
        "2) price_range with min_price and max_price (strings), "
        "3) average_price (float), "
        "4) rating_distribution: how many books have 1, 2, 3, 4, and 5 stars, "
        "5) most_expensive_book title and price, "
        "6) cheapest_book title and price."
    )

    if not result.success:
        print(f"  Analysis failed: {result.error}")
        return None

    print(f"  Analysis complete ({result.llm_usage.total_tokens:,} tokens)")
    return result.data


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
async def main() -> None:
    """Run all three extraction patterns in a single browser session."""

    print(f"Product Extraction | Provider: {PROVIDER} | Model: {MODEL}")

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        # --- Pattern 1: Catalog listing extraction ---
        section("1/3  Catalog Extraction - Mystery Category")
        books = await extract_catalog(browser, CATEGORY_URL)

        if books:
            print(f"\n  {'Title':<45} {'Price':<10} {'Rating':<8} {'Stock'}")
            print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*6}")
            for book in books[:10]:
                if not isinstance(book, dict):
                    continue
                title = str(book.get("title", ""))[:43]
                price = str(book.get("price", "N/A"))[:10]
                rating = book.get("star_rating", "?")
                stock = "Yes" if book.get("in_stock") else "No"
                print(f"  {title:<45} {price:<10} {rating:<8} {stock}")
            if len(books) > 10:
                print(f"  ... and {len(books) - 10} more books")

        # --- Pattern 2: Product detail drill-down ---
        section("2/3  Product Detail Extraction")
        if books and isinstance(books[0], dict):
            first_title = books[0].get("title", "")
            if first_title:
                detail = await extract_product_detail(browser, first_title)
                if detail:
                    print("\n  Full product detail:")
                    print(f"  {json.dumps(detail, indent=4, default=str)}")

            # Navigate back to continue
            await browser.goto(CATEGORY_URL)

        # --- Pattern 3: Category-level analysis ---
        section("3/3  Category Analysis - Pricing Intelligence")
        analysis = await analyze_catalog(browser, CATEGORY_URL)
        if analysis:
            print("\n  Catalog analysis:")
            print(f"  {json.dumps(analysis, indent=4, default=str)}")

        # --- Session usage ---
        section("Session Usage")
        usage = browser.get_usage_summary()
        print(f"  Total tokens : {usage.get('total_tokens', 0):,}")
        print(f"  API calls    : {usage.get('calls_count', 0)}")
        print(f"  Est. cost    : ${usage.get('cost_usd', 0):.4f}")

    print("\nProduct extraction complete.")


if __name__ == "__main__":
    asyncio.run(main())
