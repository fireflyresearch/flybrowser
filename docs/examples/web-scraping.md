# Web Scraping Examples

This guide provides practical examples for extracting data from websites using FlyBrowser. From simple single-page extraction to complex multi-page scraping workflows.

## Single Page Extraction

### Basic Text Extraction

Extract text content from a page using natural language:

```python path=null start=null
"""
Example: Extract Hacker News Headlines

Demonstrates basic text extraction from a news site.
"""

import asyncio
from flybrowser import FlyBrowser

async def extract_headlines():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=True,
    ) as browser:
        await browser.goto("https://news.ycombinator.com")
        
        # Simple extraction
        result = await browser.extract(
            "Get the titles of the top 10 stories on the page"
        )
        
        if result.success:
            print("Top Headlines:")
            for i, title in enumerate(result.data, 1):
                print(f"  {i}. {title}")
        else:
            print(f"Extraction failed: {result.error}")

asyncio.run(extract_headlines())
```

**Expected Output:**
```
Top Headlines:
  1. Show HN: A new approach to neural networks
  2. The Future of Programming Languages
  3. Why I left Big Tech
  ...
```

### Structured Data Extraction

Extract data with a specific schema:

```python path=null start=null
"""
Example: Extract Product Information with Schema

Demonstrates structured extraction using JSON schema.
"""

import asyncio
import json
from flybrowser import FlyBrowser

# Define the expected data structure
PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "string"},
        "rating": {"type": "number"},
        "reviews_count": {"type": "integer"},
        "in_stock": {"type": "boolean"},
        "description": {"type": "string"}
    },
    "required": ["name", "price"]
}

async def extract_product_info():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        # Navigate to a product page
        await browser.goto("https://example-store.com/product/12345")
        
        # Extract with schema
        result = await browser.extract(
            "Extract the product information from this page",
            schema=PRODUCT_SCHEMA
        )
        
        if result.success:
            print("Product Information:")
            print(json.dumps(result.data, indent=2))
            
            # Access individual fields
            product = result.data
            print(f"\nProduct: {product.get('name')}")
            print(f"Price: {product.get('price')}")
            print(f"In Stock: {product.get('in_stock', 'Unknown')}")
        else:
            print(f"Failed: {result.error}")

asyncio.run(extract_product_info())
```

**Expected Output:**
```json
{
  "name": "Wireless Bluetooth Headphones",
  "price": "$79.99",
  "rating": 4.5,
  "reviews_count": 1247,
  "in_stock": true,
  "description": "Premium noise-canceling headphones with 30-hour battery life."
}
```

### Extracting Lists and Tables

Extract tabular or list data:

```python path=null start=null
"""
Example: Extract Table Data

Demonstrates extracting structured table data.
"""

import asyncio
import csv
from flybrowser import FlyBrowser

TABLE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "symbol": {"type": "string"},
            "price": {"type": "string"},
            "change": {"type": "string"},
            "volume": {"type": "string"}
        }
    }
}

async def extract_stock_data():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://finance.example.com/markets")
        
        # Extract table data
        result = await browser.extract(
            "Extract the stock market data table showing company name, "
            "ticker symbol, current price, daily change, and volume",
            schema=TABLE_SCHEMA
        )
        
        if result.success:
            stocks = result.data
            
            # Print as table
            print(f"{'Company':<30} {'Symbol':<10} {'Price':<10} {'Change':<10}")
            print("-" * 60)
            for stock in stocks:
                print(f"{stock.get('company', ''):<30} "
                      f"{stock.get('symbol', ''):<10} "
                      f"{stock.get('price', ''):<10} "
                      f"{stock.get('change', ''):<10}")
            
            # Optionally save to CSV
            with open('stocks.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['company', 'symbol', 'price', 'change', 'volume'])
                writer.writeheader()
                writer.writerows(stocks)
            
            print(f"\nSaved {len(stocks)} records to stocks.csv")

asyncio.run(extract_stock_data())
```

## Multi-Page Scraping

### Pagination Handling

Scrape data across multiple pages:

```python path=null start=null
"""
Example: Scrape Paginated Results

Demonstrates handling pagination to collect data across multiple pages.
"""

import asyncio
from flybrowser import FlyBrowser

async def scrape_paginated_results():
    all_items = []
    max_pages = 5
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        await browser.goto("https://example-store.com/products")
        
        for page_num in range(1, max_pages + 1):
            print(f"Scraping page {page_num}...")
            
            # Extract items from current page
            result = await browser.extract(
                "Extract all product names and prices from this page",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "string"}
                        }
                    }
                }
            )
            
            if result.success and result.data:
                all_items.extend(result.data)
                print(f"  Found {len(result.data)} items")
            else:
                print(f"  No items found or error: {result.error}")
                break
            
            # Check for and click next page
            if page_num < max_pages:
                nav_result = await browser.act("Click the 'Next' button or next page link")
                if not nav_result.success:
                    print("  No more pages available")
                    break
                
                # Wait for page to load
                await asyncio.sleep(1)
        
        print(f"\nTotal items collected: {len(all_items)}")
        return all_items

asyncio.run(scrape_paginated_results())
```

### Search and Extract Pattern

Search for items then extract details:

```python path=null start=null
"""
Example: Search and Extract

Demonstrates searching for content then extracting results.
"""

import asyncio
from flybrowser import FlyBrowser

async def search_and_extract(search_term: str):
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        # Navigate to search page
        await browser.goto("https://example-store.com")
        
        # Perform search using natural language
        await browser.act(f"Type '{search_term}' in the search box and press Enter")
        
        # Wait for results
        await asyncio.sleep(2)
        
        # Extract search results
        result = await browser.extract(
            "Get all search results with name, price, and rating",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "string"},
                        "rating": {"type": "number"},
                        "url": {"type": "string"}
                    }
                }
            }
        )
        
        if result.success:
            print(f"Found {len(result.data)} results for '{search_term}':")
            for item in result.data[:5]:  # Show top 5
                print(f"  - {item.get('name')}: {item.get('price')} ({item.get('rating')} stars)")
        
        return result.data if result.success else []

asyncio.run(search_and_extract("wireless headphones"))
```

### Category Navigation Scraper

Navigate through categories and collect data:

```python path=null start=null
"""
Example: Category-based Scraper

Navigates through category structure and collects products.
"""

import asyncio
from flybrowser import FlyBrowser

async def scrape_categories():
    results = {}
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example-store.com")
        
        # Get main categories
        categories_result = await browser.extract(
            "Get all main category names and their links from the navigation menu"
        )
        
        if not categories_result.success:
            print("Failed to get categories")
            return results
        
        categories = categories_result.data[:3]  # Limit to 3 for example
        
        for category in categories:
            category_name = category.get('name', 'Unknown')
            print(f"\nScraping category: {category_name}")
            
            # Navigate to category
            await browser.navigate(f"Click on the '{category_name}' category link")
            await asyncio.sleep(1)
            
            # Extract products
            products_result = await browser.extract(
                "Get all products with name and price"
            )
            
            if products_result.success:
                results[category_name] = products_result.data
                print(f"  Found {len(products_result.data)} products")
            
            # Return to main page
            await browser.goto("https://example-store.com")
        
        return results

asyncio.run(scrape_categories())
```

## Dynamic Content Scraping

### JavaScript-Rendered Content

Handle dynamically loaded content:

```python path=null start=null
"""
Example: Scraping Dynamic Content

Demonstrates handling JavaScript-rendered content with infinite scroll.
"""

import asyncio
from flybrowser import FlyBrowser

async def scrape_infinite_scroll():
    all_items = []
    scroll_count = 0
    max_scrolls = 10
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        speed_preset="thorough",  # More patient with dynamic content
    ) as browser:
        await browser.goto("https://example-feed.com")
        
        while scroll_count < max_scrolls:
            # Extract currently visible items
            result = await browser.extract(
                "Extract all post titles and authors visible on the page"
            )
            
            if result.success and result.data:
                # Track new items (avoid duplicates)
                new_items = [item for item in result.data 
                            if item not in all_items]
                all_items.extend(new_items)
                print(f"Scroll {scroll_count + 1}: Found {len(new_items)} new items")
            
            # Scroll to load more content
            await browser.act("Scroll down to load more content")
            await asyncio.sleep(2)  # Wait for content to load
            
            scroll_count += 1
            
            # Check if we've stopped loading new content
            if result.success and len(new_items) == 0:
                print("No new content loaded, stopping")
                break
        
        print(f"\nTotal unique items: {len(all_items)}")
        return all_items

asyncio.run(scrape_infinite_scroll())
```

### Handling AJAX Requests

Wait for AJAX content:

```python path=null start=null
"""
Example: AJAX Content Handling

Demonstrates waiting for and extracting AJAX-loaded content.
"""

import asyncio
from flybrowser import FlyBrowser

async def extract_ajax_content():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example-dashboard.com")
        
        # Trigger data load
        await browser.act("Click the 'Load Data' button")
        
        # Wait for loading indicator to disappear
        # The agent will intelligently wait for content
        await asyncio.sleep(2)
        
        # Extract the loaded data
        result = await browser.extract(
            "Extract the data from the table that was just loaded"
        )
        
        if result.success:
            print("Loaded data:")
            for row in result.data:
                print(f"  {row}")

asyncio.run(extract_ajax_content())
```

## Advanced Extraction Patterns

### Extracting with Vision

Use vision for visually complex pages:

```python path=null start=null
"""
Example: Vision-Based Extraction

Uses visual understanding for complex layouts.
"""

import asyncio
from flybrowser import FlyBrowser

async def extract_with_vision():
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4o",  # Vision-capable model
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example-infographic.com")
        
        # Enable vision for extraction
        result = await browser.extract(
            "Extract all statistics and data points shown in the infographic",
            use_vision=True
        )
        
        if result.success:
            print("Extracted data points:")
            for item in result.data:
                print(f"  - {item}")
        
        # Check usage (vision uses more tokens)
        usage = browser.get_usage_summary()
        print(f"\nTokens used: {usage['total_tokens']:,}")

asyncio.run(extract_with_vision())
```

### Conditional Extraction

Extract different data based on page content:

```python path=null start=null
"""
Example: Conditional Extraction

Adapts extraction strategy based on page type.
"""

import asyncio
from flybrowser import FlyBrowser

async def conditional_extract(url: str):
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto(url)
        
        # First, identify the page type
        page_info = await browser.extract(
            "What type of page is this? (product, article, listing, profile, other)"
        )
        
        page_type = page_info.data.lower() if page_info.success else "unknown"
        print(f"Detected page type: {page_type}")
        
        # Extract based on page type
        if "product" in page_type:
            result = await browser.extract(
                "Extract product name, price, description, and specifications",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "string"},
                        "description": {"type": "string"},
                        "specs": {"type": "object"}
                    }
                }
            )
        elif "article" in page_type:
            result = await browser.extract(
                "Extract article title, author, publish date, and main content",
                schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "date": {"type": "string"},
                        "content": {"type": "string"}
                    }
                }
            )
        elif "listing" in page_type:
            result = await browser.extract(
                "Extract all items with their names and prices",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "string"}
                        }
                    }
                }
            )
        else:
            result = await browser.extract(
                "Extract the main content and key information from this page"
            )
        
        return result.data if result.success else None

asyncio.run(conditional_extract("https://example.com/product/123"))
```

### Batch URL Scraping

Scrape multiple URLs efficiently:

```python path=null start=null
"""
Example: Batch URL Scraper

Efficiently scrapes data from a list of URLs.
"""

import asyncio
from flybrowser import FlyBrowser

async def scrape_urls(urls: list[str], extraction_query: str):
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Scraping: {url}")
            
            try:
                await browser.goto(url)
                result = await browser.extract(extraction_query)
                
                if result.success:
                    results.append({
                        "url": url,
                        "data": result.data,
                        "success": True
                    })
                else:
                    results.append({
                        "url": url,
                        "error": result.error,
                        "success": False
                    })
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "success": False
                })
            
            # Brief pause between requests
            await asyncio.sleep(0.5)
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(urls)} successful")
        
        return results

# Usage
urls = [
    "https://example.com/product/1",
    "https://example.com/product/2",
    "https://example.com/product/3",
]

asyncio.run(scrape_urls(urls, "Extract the product name and price"))
```

## Real-World Examples

### Hacker News Scraper

Complete example scraping Hacker News:

```python path=null start=null
"""
Example: Hacker News Scraper

A complete scraper for Hacker News front page stories.
"""

import asyncio
import json
from datetime import datetime
from flybrowser import FlyBrowser

STORY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "rank": {"type": "integer"},
            "title": {"type": "string"},
            "url": {"type": "string"},
            "points": {"type": "integer"},
            "author": {"type": "string"},
            "comments": {"type": "integer"},
            "time_ago": {"type": "string"}
        },
        "required": ["title"]
    }
}

async def scrape_hackernews(num_pages: int = 1):
    """Scrape Hacker News stories."""
    all_stories = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="normal",
    ) as browser:
        for page in range(1, num_pages + 1):
            # Navigate to page
            url = f"https://news.ycombinator.com/news?p={page}"
            await browser.goto(url)
            
            # Extract stories
            result = await browser.extract(
                "Extract all stories with their rank, title, URL, points, "
                "author, number of comments, and time posted",
                schema=STORY_SCHEMA,
                max_iterations=20
            )
            
            if result.success and result.data:
                # Add page info to each story
                for story in result.data:
                    story['page'] = page
                
                all_stories.extend(result.data)
                print(f"Page {page}: Extracted {len(result.data)} stories")
            else:
                print(f"Page {page}: Failed - {result.error}")
                break
        
        # Save to file
        output = {
            "scraped_at": datetime.now().isoformat(),
            "total_stories": len(all_stories),
            "stories": all_stories
        }
        
        with open("hackernews_stories.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved {len(all_stories)} stories to hackernews_stories.json")
        
        # Show usage stats
        usage = browser.get_usage_summary()
        print(f"Total tokens: {usage['total_tokens']:,}")
        print(f"Estimated cost: ${usage['cost_usd']:.4f}")
        
        return all_stories

asyncio.run(scrape_hackernews(num_pages=2))
```

### E-commerce Price Monitor

Monitor prices across multiple products:

```python path=null start=null
"""
Example: Price Monitor

Monitors product prices and detects changes.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from flybrowser import FlyBrowser

PRICE_HISTORY_FILE = "price_history.json"

def load_price_history() -> dict:
    """Load existing price history."""
    if Path(PRICE_HISTORY_FILE).exists():
        with open(PRICE_HISTORY_FILE) as f:
            return json.load(f)
    return {}

def save_price_history(history: dict):
    """Save price history."""
    with open(PRICE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

async def monitor_prices(products: list[dict]):
    """
    Monitor prices for a list of products.
    
    Args:
        products: List of {"name": str, "url": str}
    """
    history = load_price_history()
    alerts = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        for product in products:
            name = product["name"]
            url = product["url"]
            
            print(f"Checking: {name}")
            
            try:
                await browser.goto(url)
                
                # Extract current price
                result = await browser.extract(
                    "What is the current price of this product?",
                    schema={"type": "object", "properties": {"price": {"type": "string"}}}
                )
                
                if not result.success:
                    print(f"  Failed to get price: {result.error}")
                    continue
                
                current_price = result.data.get("price", "Unknown")
                timestamp = datetime.now().isoformat()
                
                # Initialize history for this product
                if name not in history:
                    history[name] = {
                        "url": url,
                        "prices": []
                    }
                
                # Get previous price
                prices = history[name]["prices"]
                previous_price = prices[-1]["price"] if prices else None
                
                # Record new price
                prices.append({
                    "price": current_price,
                    "timestamp": timestamp
                })
                
                # Check for price change
                if previous_price and previous_price != current_price:
                    alert = f"{name}: {previous_price} -> {current_price}"
                    alerts.append(alert)
                    print(f"  PRICE CHANGE: {alert}")
                else:
                    print(f"  Current price: {current_price}")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    # Save updated history
    save_price_history(history)
    
    # Report alerts
    if alerts:
        print("\n=== PRICE ALERTS ===")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\nNo price changes detected.")
    
    return alerts

# Usage
products = [
    {"name": "Headphones", "url": "https://example.com/product/headphones"},
    {"name": "Keyboard", "url": "https://example.com/product/keyboard"},
    {"name": "Monitor", "url": "https://example.com/product/monitor"},
]

asyncio.run(monitor_prices(products))
```

## Tips and Best Practices

### Extraction Query Tips

**Be Specific:**
```python path=null start=null
# Less effective
result = await browser.extract("Get the data")

# More effective
result = await browser.extract(
    "Extract the product name, price in USD, and star rating (1-5)"
)
```

**Use Schemas for Structured Data:**
```python path=null start=null
# Schema ensures consistent output format
result = await browser.extract(
    "Get product info",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"}
        },
        "required": ["name", "price"]
    }
)
```

**Handle Missing Data:**
```python path=null start=null
result = await browser.extract("Get all available product details")
data = result.data or {}
name = data.get("name", "Unknown")
price = data.get("price", "N/A")
```

### Performance Optimization

**Minimize Iterations:**
```python path=null start=null
# For simple extractions, reduce iterations
result = await browser.extract(
    "Get the page title",
    max_iterations=5  # Simple task needs fewer iterations
)
```

**Disable Vision When Not Needed:**
```python path=null start=null
# Text extraction doesn't need vision
result = await browser.extract(
    "Get the article text",
    use_vision=False  # Faster and cheaper
)
```

**Batch Similar Extractions:**
```python path=null start=null
# Instead of multiple extractions
name = await browser.extract("Get product name")
price = await browser.extract("Get product price")
rating = await browser.extract("Get product rating")

# Do one extraction
result = await browser.extract(
    "Get product name, price, and rating",
    schema={...}  # Schema for all fields
)
```

## Next Steps

- [UI Testing Examples](ui-testing.md) - Testing automation patterns
- [Workflow Automation](workflow-automation.md) - Complex task automation
- [Error Handling Guide](../guides/error-handling.md) - Robust error handling
