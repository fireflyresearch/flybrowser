# Data Extraction Guide

This guide covers techniques for extracting data from web pages using FlyBrowser. You will learn how to extract simple values, structured data, tables, and handle complex extraction scenarios.

## Basic Extraction

### Simple Text Extraction

The `extract()` method retrieves data using natural language queries:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com")
        
        # Extract simple text
        result = await browser.extract("get the page title")
        print(result.data)  # "Example Domain"
        
        # Extract specific content
        result = await browser.extract("what is the main heading?")
        print(result.data)

asyncio.run(main())
```

### Multiple Values

Extract multiple items at once:

```python
# Get a list of items
result = await browser.extract("get all the product names on this page")
for name in result.data:
    print(name)

# Get numbered items
result = await browser.extract("list the top 5 news headlines")
```

### Metadata About Extraction

Enable metadata to get execution details:

```python
result = await browser.extract(
    "get the price of the first product",
    return_metadata=True
)

print(f"Data: {result.data}")
print(f"Success: {result.success}")
print(f"Iterations: {result.execution.iterations}")
print(f"LLM calls: {result.llm_usage.calls_count}")
```

## Structured Data Extraction

### Using JSON Schema

For predictable data structures, provide a JSON Schema:

```python
# Define the expected structure
product_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "currency": {"type": "string"},
        "in_stock": {"type": "boolean"}
    },
    "required": ["name", "price"]
}

result = await browser.extract(
    "extract the product details",
    schema=product_schema
)

# Result matches the schema
print(result.data["name"])    # "MacBook Pro"
print(result.data["price"])   # 1999.99
print(result.data["in_stock"]) # True
```

### Array Schemas

Extract lists of structured items:

```python
products_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "price": {"type": "number"},
            "rating": {"type": "number"},
            "reviews_count": {"type": "integer"}
        },
        "required": ["title", "price"]
    }
}

result = await browser.extract(
    "extract all products with their prices and ratings",
    schema=products_schema
)

for product in result.data:
    print(f"{product['title']}: ${product['price']} ({product.get('rating', 'N/A')} stars)")
```

### Nested Structures

Handle complex nested data:

```python
article_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "bio": {"type": "string"}
            }
        },
        "content": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "published_date": {"type": "string"}
    }
}

result = await browser.extract(
    "extract the article with its author information and tags",
    schema=article_schema
)
```

## Table Extraction

### Extracting HTML Tables

FlyBrowser can extract tabular data:

```python
# Simple table extraction
result = await browser.extract("extract the pricing table")

# Returns a list of row dictionaries
for row in result.data:
    print(row)
```

### Structured Table Schema

For consistent table structure:

```python
pricing_table_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "plan_name": {"type": "string"},
            "monthly_price": {"type": "number"},
            "annual_price": {"type": "number"},
            "features": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
}

result = await browser.extract(
    "extract the pricing table with all plan details",
    schema=pricing_table_schema
)
```

### Complex Tables

For tables with merged cells or complex layouts:

```python
result = await browser.extract(
    "extract the comparison table, handling any merged headers",
    use_vision=True  # Vision helps with complex table layouts
)
```

## Using Vision for Extraction

### When to Use Vision

Enable vision for better extraction accuracy:

```python
# Extract data that requires visual understanding
result = await browser.extract(
    "what color is the Add to Cart button?",
    use_vision=True
)

# Extract from charts or graphs
result = await browser.extract(
    "what is the highest value shown in the bar chart?",
    use_vision=True
)

# Extract text from images
result = await browser.extract(
    "read the text in the promotional banner",
    use_vision=True
)
```

### Visual Layout Analysis

Vision helps understand spatial relationships:

```python
# Extract based on visual position
result = await browser.extract(
    "get the price displayed next to each product image",
    use_vision=True
)

# Understand complex layouts
result = await browser.extract(
    "extract the sidebar content separately from the main content",
    use_vision=True
)
```

## Observing Page Elements

### Using observe()

The `observe()` method finds elements without extracting their full content:

```python
# Find elements matching a description
result = await browser.observe("find all buttons on this page")

print(f"Found {len(result.data)} buttons")
for element in result.data:
    print(element)
```

### Getting Selectors

Request CSS selectors for found elements:

```python
result = await browser.observe(
    "find the search input field",
    return_selectors=True
)

# Use the selector directly if needed
selector = result.data[0]["selector"]
print(f"Selector: {selector}")  # e.g., "#search-input"
```

### Combining observe() with act()

Use observe to understand the page, then act:

```python
# First, discover what's available
elements = await browser.observe("find all filter options")

# Then interact based on findings
for element in elements.data:
    if "Price" in element.get("text", ""):
        await browser.act(f"click the {element['text']} filter")
        break
```

## Extraction Patterns

### E-commerce Product Data

```python
async def extract_product_data(browser, url):
    await browser.goto(url)
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "original_price": {"type": "number"},
            "discount_percent": {"type": "number"},
            "rating": {"type": "number"},
            "review_count": {"type": "integer"},
            "availability": {"type": "string"},
            "description": {"type": "string"},
            "specifications": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            },
            "images": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    result = await browser.extract(
        "extract complete product information including specs and images",
        schema=schema
    )
    
    return result.data
```

### News Article Extraction

```python
async def extract_article(browser, url):
    await browser.goto(url)
    
    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "subheadline": {"type": "string"},
            "author": {"type": "string"},
            "published_date": {"type": "string"},
            "updated_date": {"type": "string"},
            "content": {"type": "string"},
            "summary": {"type": "string"},
            "category": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    result = await browser.extract(
        "extract the full article content with metadata",
        schema=schema
    )
    
    return result.data
```

### Contact Information

```python
async def extract_contacts(browser, url):
    await browser.goto(url)
    
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "title": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "department": {"type": "string"}
            }
        }
    }
    
    result = await browser.extract(
        "extract all contact information from the team or contact page",
        schema=schema
    )
    
    return result.data
```

### Job Listings

```python
async def extract_jobs(browser, url):
    await browser.goto(url)
    
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "company": {"type": "string"},
                "location": {"type": "string"},
                "salary_range": {"type": "string"},
                "job_type": {"type": "string"},
                "posted_date": {"type": "string"},
                "description_summary": {"type": "string"}
            }
        }
    }
    
    result = await browser.extract(
        "extract all job listings with their details",
        schema=schema
    )
    
    return result.data
```

## Multi-Page Extraction

### Extracting Across Pagination

```python
async def extract_all_products(browser, start_url):
    await browser.goto(start_url)
    
    all_products = []
    page = 1
    
    while True:
        print(f"Extracting page {page}...")
        
        result = await browser.extract(
            "extract all products on this page",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "url": {"type": "string"}
                    }
                }
            }
        )
        
        if result.success and result.data:
            all_products.extend(result.data)
        
        # Try to navigate to next page
        next_result = await browser.act("click the Next page button")
        
        if not next_result.success:
            break  # No more pages
            
        page += 1
        await asyncio.sleep(1)  # Be polite
    
    return all_products
```

### Deep Extraction (Follow Links)

```python
async def extract_with_details(browser, listing_url):
    await browser.goto(listing_url)
    
    # Get summary list
    listings = await browser.extract(
        "extract all items with their detail page URLs",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"}
                }
            }
        }
    )
    
    detailed_items = []
    
    for item in listings.data:
        if item.get("url"):
            await browser.goto(item["url"])
            
            detail = await browser.extract(
                "extract full details of this item"
            )
            
            detailed_items.append({
                **item,
                "details": detail.data
            })
    
    return detailed_items
```

## Handling Extraction Challenges

### Dynamic Content

For content that loads after the initial page:

```python
async def extract_dynamic_content(browser, url):
    await browser.goto(url)
    
    # Wait for dynamic content
    await asyncio.sleep(2)
    
    # Or scroll to trigger lazy loading
    await browser.act("scroll to the bottom of the page")
    await asyncio.sleep(1)
    
    # Now extract
    result = await browser.extract("extract all loaded content")
    return result.data
```

### Extraction with Interaction

Sometimes you need to interact to reveal content:

```python
async def extract_hidden_content(browser, url):
    await browser.goto(url)
    
    # Expand all collapsed sections
    await browser.act("click all 'Show More' or 'Expand' buttons")
    await asyncio.sleep(1)
    
    # Now extract the revealed content
    result = await browser.extract("extract all content including expanded sections")
    return result.data
```

### Handling Missing Data

```python
result = await browser.extract(
    "extract product details",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "sku": {"type": "string"}  # Might not exist
        },
        "required": ["name", "price"]  # Only name and price required
    }
)

# Check for optional fields
data = result.data
if data.get("sku"):
    print(f"SKU: {data['sku']}")
else:
    print("No SKU available")
```

## Extraction Quality

### Improving Accuracy

Be specific in your queries:

```python
# Less specific (might return wrong content)
result = await browser.extract("get the price")

# More specific (better results)
result = await browser.extract(
    "get the current sale price of the main product, not the original price"
)
```

### Verifying Extraction Results

```python
result = await browser.extract(
    "extract the product price",
    schema={"type": "number"}
)

if result.success:
    price = result.data
    
    # Validate the extracted data
    if price <= 0:
        print("Warning: Invalid price extracted")
    elif price > 100000:
        print("Warning: Price seems unusually high")
    else:
        print(f"Verified price: ${price}")
else:
    print(f"Extraction failed: {result.error}")
```

### Handling Extraction Failures

```python
async def robust_extract(browser, query, schema, retries=3):
    for attempt in range(retries):
        result = await browser.extract(query, schema=schema)
        
        if result.success and result.data:
            return result
        
        print(f"Attempt {attempt + 1} failed, retrying...")
        
        # Try with vision on retry
        if attempt == 1:
            result = await browser.extract(
                query, 
                schema=schema, 
                use_vision=True
            )
            if result.success:
                return result
    
    return result  # Return last result even if failed
```

## Best Practices

### Use Schemas for Consistency

```python
# Define schemas upfront for reusable extractions
PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "currency": {"type": "string", "default": "USD"}
    },
    "required": ["name", "price"]
}

# Use consistently across extractions
result = await browser.extract("get product info", schema=PRODUCT_SCHEMA)
```

### Batch Extractions When Possible

```python
# Instead of multiple calls
name = await browser.extract("get product name")
price = await browser.extract("get product price")
desc = await browser.extract("get product description")

# Use a single call with schema
result = await browser.extract(
    "extract product name, price, and description",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "description": {"type": "string"}
        }
    }
)
```

### Monitor LLM Usage

```python
result = await browser.extract("get all products", return_metadata=True)

# Check costs
if result.llm_usage:
    print(f"Tokens used: {result.llm_usage.total_tokens}")
    print(f"Estimated cost: ${result.llm_usage.cost_usd:.4f}")

# After multiple extractions, check cumulative usage
summary = await browser.get_usage_summary()
print(f"Total cost this session: ${summary.total_cost_usd:.2f}")
```

## Next Steps

- [Form Automation Guide](form-automation.md) - Filling and submitting forms
- [Multi-Page Workflows](multi-page-workflows.md) - Complex extraction workflows
- [SDK Reference](../reference/sdk.md) - Complete extract() documentation
