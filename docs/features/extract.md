# extract() - Data Extraction

The `extract()` method retrieves data from web pages using natural language queries. It can extract text, numbers, structured data, and complex nested information.

## Basic Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com")
        
        # Simple text extraction
        result = await browser.extract("get the page title")
        print(result.data)
        
        # Specific data
        result = await browser.extract("what is the main heading?")
        print(result.data)

asyncio.run(main())
```

## Method Signature

```python
async def extract(
    self,
    query: str,
    use_vision: bool = False,
    schema: dict | None = None,
    return_metadata: bool = False,
) -> AgentRequestResponse
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Natural language description of what to extract |
| `use_vision` | `bool` | `False` | Include screenshot for visual context |
| `schema` | `dict` | `None` | JSON Schema to structure the extracted data |
| `return_metadata` | `bool` | `False` | Include detailed execution metadata |

### Returns

`AgentRequestResponse` with:
- `success` - Whether extraction succeeded
- `data` - The extracted data (type depends on schema)
- `error` - Error message if failed
- `operation` - "extract"
- `query` - The original query

## Simple Extraction

### Text Content

```python
# Get specific text
result = await browser.extract("get the page title")

# Get headings
result = await browser.extract("what is the main heading?")

# Get paragraph content
result = await browser.extract("get the first paragraph of the article")
```

### Numbers and Prices

```python
# Extract a price
result = await browser.extract("what is the product price?")

# Get numeric values
result = await browser.extract("how many items are in stock?")
result = await browser.extract("what is the rating score?")
```

### Lists

```python
# Get a list of items
result = await browser.extract("list all the product names")
for item in result.data:
    print(item)

# Get numbered items
result = await browser.extract("get the top 5 search results")
```

## Structured Extraction

### Using JSON Schema

Provide a schema to get consistently structured data:

```python
# Define expected structure
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

# Extract with schema
result = await browser.extract(
    "extract the product details",
    schema=product_schema
)

# Guaranteed structure
print(result.data["name"])
print(result.data["price"])
```

### Array Schemas

Extract lists of objects:

```python
products_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "price": {"type": "number"},
            "rating": {"type": "number"}
        },
        "required": ["title", "price"]
    }
}

result = await browser.extract(
    "extract all products",
    schema=products_schema
)

for product in result.data:
    print(f"{product['title']}: ${product['price']}")
```

### Nested Structures

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
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "published_date": {"type": "string"}
    }
}

result = await browser.extract(
    "extract article with author info",
    schema=article_schema
)
```

## Vision-Based Extraction

### When to Use Vision

Enable vision for:
- Data in images or graphics
- Complex visual layouts
- Position-dependent extraction
- Charts and diagrams

```python
# Extract from visual content
result = await browser.extract(
    "what text is shown in the hero image?",
    use_vision=True
)

# Chart data
result = await browser.extract(
    "what is the highest value in the bar chart?",
    use_vision=True
)

# Visual layout
result = await browser.extract(
    "get the sidebar content separately from the main area",
    use_vision=True
)
```

## Table Extraction

### Simple Tables

```python
result = await browser.extract("extract the data table")

for row in result.data:
    print(row)
```

### Structured Tables

```python
pricing_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "monthly": {"type": "number"},
            "annual": {"type": "number"},
            "features": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
}

result = await browser.extract(
    "extract the pricing comparison table",
    schema=pricing_schema
)
```

## Return Values

### Basic Response

```python
result = await browser.extract("get the price")

print(result.success)  # True/False
print(result.data)     # The extracted data
print(result.error)    # Error if failed
```

### With Metadata

```python
result = await browser.extract(
    "extract all products",
    return_metadata=True
)

# Execution details
print(f"Iterations: {result.execution.iterations}")
print(f"Duration: {result.execution.duration_seconds}s")

# LLM usage
print(f"Tokens: {result.llm_usage.total_tokens}")
print(f"Cost: ${result.llm_usage.cost_usd:.4f}")

# Pretty print full response
result.pprint()
```

## Error Handling

```python
result = await browser.extract("get the price")

if not result.success:
    print(f"Extraction failed: {result.error}")
else:
    # Validate extracted data
    if result.data is None:
        print("No data found")
    elif isinstance(result.data, dict) and "price" not in result.data:
        print("Price field missing from result")
    else:
        print(f"Price: {result.data}")
```

## Best Practices

### Be Specific

```python
# Less specific
result = await browser.extract("get the price")

# More specific
result = await browser.extract(
    "get the current sale price of the product, not the original price"
)
```

### Use Schemas for Consistency

```python
# Without schema - unpredictable format
result = await browser.extract("get product info")

# With schema - guaranteed format
result = await browser.extract(
    "get product info",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"}
        }
    }
)
```

### Batch Related Extractions

```python
# Instead of multiple calls
name = await browser.extract("get name")
price = await browser.extract("get price")
desc = await browser.extract("get description")

# Use one call with schema
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

### Handle Optional Fields

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "sku": {"type": "string"}  # Optional
    },
    "required": ["name", "price"]  # Only these are required
}

result = await browser.extract("get product details", schema=schema)

# Check for optional fields
if result.data.get("sku"):
    print(f"SKU: {result.data['sku']}")
```

## Operation Mode

The `extract()` method internally sets the operation mode to `SCRAPE`, which optimizes for:
- Content-focused exploration
- Balanced vision frequency
- Data-oriented prompts

## Schema Reference

### Supported Types

```python
# String
{"type": "string"}

# Number (integer or float)
{"type": "number"}

# Integer only
{"type": "integer"}

# Boolean
{"type": "boolean"}

# Array
{
    "type": "array",
    "items": {"type": "string"}
}

# Object
{
    "type": "object",
    "properties": {
        "field": {"type": "string"}
    }
}
```

### Constraints

```python
# Enum (allowed values)
{
    "type": "string",
    "enum": ["small", "medium", "large"]
}

# Default value
{
    "type": "string",
    "default": "unknown"
}

# Required fields
{
    "type": "object",
    "properties": {...},
    "required": ["field1", "field2"]
}
```

## Related Methods

- [act()](act.md) - Execute browser actions
- [observe()](observe.md) - Find elements
- [agent()](agent.md) - Complex multi-step extraction

## See Also

- [Data Extraction Guide](../guides/data-extraction.md) - Comprehensive examples
- [SDK Reference](../reference/sdk.md) - Complete API documentation
