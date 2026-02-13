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
Pagination Strategies on books.toscrape.com

Demonstrates three distinct approaches for scraping paginated content:

  1. Click-based pagination  - Use `act()` to click the "next" button and
                               scrape each subsequent page in the same session.
  2. Direct URL pagination   - Construct page URLs programmatically and
                               navigate to each with `goto()`.
  3. Accumulate-and-deduplicate - Combine results from multiple pages, remove
                               duplicates, and produce a clean dataset.

Target: books.toscrape.com (50 pages, 20 books per page, 1000 total books).

Business value:
  - Data pipeline ingestion: reliably paginate through any catalog.
  - Completeness assurance: accumulate + deduplicate guarantees no gaps.
  - Cost awareness: track token usage per page to estimate full-catalog cost.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export FLYBROWSER_LLM_PROVIDER="anthropic"     # optional
    export FLYBROWSER_LLM_MODEL="claude-sonnet-4-5-20250929"  # optional
"""

import asyncio
import os

from flybrowser import FlyBrowser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
MODEL = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")
BASE_URL = "https://books.toscrape.com/catalogue/page-{page}.html"
START_URL = "https://books.toscrape.com"
PAGES_TO_SCRAPE = 3  # Keep examples quick; increase for full catalog


def section(title: str) -> None:
    """Print a visible section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Strategy 1: Click-based "next" button pagination
# ---------------------------------------------------------------------------
async def paginate_by_clicking(max_pages: int = PAGES_TO_SCRAPE) -> list[dict]:
    """
    Scrape pages by clicking the 'next' button after each extraction.

    This mirrors how a human browses: extract, click next, repeat.
    """
    all_books: list[dict] = []
    total_tokens = 0

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        await browser.goto(START_URL)

        for page_num in range(1, max_pages + 1):
            print(f"\n  Page {page_num}/{max_pages}")

            result = await browser.extract(
                "Extract every book on this page. For each return: "
                "title (string), price (string with currency symbol), "
                "star_rating (integer 1-5), and in_stock (boolean)."
            )

            if result.success and result.data:
                books = result.data if isinstance(result.data, list) else [result.data]
                for book in books:
                    if isinstance(book, dict):
                        book["source_page"] = page_num
                all_books.extend(books)
                total_tokens += result.llm_usage.total_tokens
                print(f"    Extracted {len(books)} books "
                      f"(tokens: {result.llm_usage.total_tokens:,}, "
                      f"time: {result.execution.duration_seconds:.1f}s)")
            else:
                print(f"    Extraction failed: {result.error}")
                break

            # Click "next" to advance
            if page_num < max_pages:
                nav = await browser.act("Click the 'next' button to go to the next page")
                if not nav.success:
                    print("    No next button found - end of catalog.")
                    break
                total_tokens += nav.llm_usage.total_tokens

    print(f"\n  Click strategy totals: {len(all_books)} books, {total_tokens:,} tokens")
    return all_books


# ---------------------------------------------------------------------------
# Strategy 2: Direct URL construction
# ---------------------------------------------------------------------------
async def paginate_by_url(max_pages: int = PAGES_TO_SCRAPE) -> list[dict]:
    """
    Scrape pages by constructing URLs directly (page-1.html, page-2.html, ...).

    Faster than click-based because it skips the act() call, but requires
    knowledge of the URL pattern.
    """
    all_books: list[dict] = []
    total_tokens = 0

    async with FlyBrowser(
        llm_provider=PROVIDER,
        llm_model=MODEL,
        headless=True,
    ) as browser:

        for page_num in range(1, max_pages + 1):
            url = BASE_URL.format(page=page_num)
            print(f"\n  Page {page_num}/{max_pages}: {url}")

            await browser.goto(url)

            result = await browser.extract(
                "Extract every book on this page. For each return: "
                "title (string), price (string with currency symbol), "
                "star_rating (integer 1-5), and in_stock (boolean)."
            )

            if result.success and result.data:
                books = result.data if isinstance(result.data, list) else [result.data]
                for book in books:
                    if isinstance(book, dict):
                        book["source_page"] = page_num
                all_books.extend(books)
                total_tokens += result.llm_usage.total_tokens
                print(f"    Extracted {len(books)} books "
                      f"(tokens: {result.llm_usage.total_tokens:,}, "
                      f"time: {result.execution.duration_seconds:.1f}s)")
            else:
                print(f"    Extraction failed: {result.error}")
                # Empty page means we've passed the end
                break

    print(f"\n  URL strategy totals: {len(all_books)} books, {total_tokens:,} tokens")
    return all_books


# ---------------------------------------------------------------------------
# Strategy 3: Accumulate and deduplicate
# ---------------------------------------------------------------------------
def deduplicate(books: list[dict]) -> list[dict]:
    """
    Remove duplicate books based on title.

    In real pipelines, duplicates arise from overlapping extractions,
    retries, or pages that share featured items.
    """
    seen_titles: set[str] = set()
    unique: list[dict] = []

    for book in books:
        if not isinstance(book, dict):
            continue
        title = book.get("title", "").strip().lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append(book)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    """Run both pagination strategies, compare results, and deduplicate."""

    print(f"Pagination Demo | Provider: {PROVIDER} | Model: {MODEL}")
    print(f"Pages per strategy: {PAGES_TO_SCRAPE}")

    # --- Strategy 1: Click-based ---
    section("Strategy 1: Click-Based Pagination")
    click_books = await paginate_by_clicking()

    # --- Strategy 2: URL-based ---
    section("Strategy 2: Direct URL Pagination")
    url_books = await paginate_by_url()

    # --- Strategy 3: Merge and deduplicate ---
    section("Strategy 3: Accumulate and Deduplicate")
    combined = click_books + url_books
    unique_books = deduplicate(combined)

    print(f"  Books from click strategy:  {len(click_books)}")
    print(f"  Books from URL strategy:    {len(url_books)}")
    print(f"  Combined (raw):             {len(combined)}")
    print(f"  After deduplication:         {len(unique_books)}")
    print(f"  Duplicates removed:          {len(combined) - len(unique_books)}")

    # --- Display final dataset ---
    section("Final Dataset (first 15 books)")
    print(f"  {'#':<4} {'Title':<42} {'Price':<10} {'Rating':<8} {'Page'}")
    print(f"  {'-'*4} {'-'*42} {'-'*10} {'-'*8} {'-'*5}")

    for i, book in enumerate(unique_books[:15], 1):
        title = str(book.get("title", ""))[:40]
        price = str(book.get("price", "N/A"))[:10]
        rating = str(book.get("star_rating", "?"))
        page = str(book.get("source_page", "?"))
        print(f"  {i:<4} {title:<42} {price:<10} {rating:<8} {page}")

    if len(unique_books) > 15:
        print(f"  ... and {len(unique_books) - 15} more books")

    print(f"\nPagination demo complete. {len(unique_books)} unique books collected.")


if __name__ == "__main__":
    asyncio.run(main())
