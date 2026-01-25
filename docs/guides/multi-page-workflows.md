# Multi-Page Workflows Guide

This guide covers building automation workflows that span multiple pages. You will learn how to manage navigation, maintain state across pages, and orchestrate complex sequences.

## Basic Multi-Page Navigation

### Sequential Page Visits

```python
import asyncio
from flybrowser import FlyBrowser

async def visit_multiple_pages():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        # Visit first page
        await browser.goto("https://shop.example.com")
        
        # Navigate to products
        await browser.navigate("go to the products page")
        
        # Click a specific product
        await browser.act("click on the first product")
        
        # Add to cart
        await browser.act("click Add to Cart")
        
        # Go to cart
        await browser.navigate("go to the shopping cart")
        
        # Proceed to checkout
        await browser.act("click Checkout")

asyncio.run(visit_multiple_pages())
```

### Using agent() for Complex Flows

The `agent()` method handles multi-step workflows autonomously:

```python
result = await browser.agent(
    task="""
    1. Go to the products page
    2. Find the first laptop under $1000
    3. Add it to the cart
    4. Go to checkout
    5. Report the total price
    """,
    max_iterations=30,
    max_time_seconds=300
)

print(f"Task completed: {result.success}")
print(f"Result: {result.data}")
```

## State Management

### Tracking Visited Pages

```python
class WorkflowState:
    def __init__(self):
        self.visited_pages = []
        self.extracted_data = {}
        self.current_step = 0
        
async def workflow_with_state():
    state = WorkflowState()
    
    async with FlyBrowser(...) as browser:
        # Page 1: Search
        await browser.goto("https://example.com")
        state.visited_pages.append("home")
        
        await browser.act("search for 'laptop'")
        state.visited_pages.append("search_results")
        state.current_step += 1
        
        # Extract results
        results = await browser.extract("get all product names and prices")
        state.extracted_data["search_results"] = results.data
        
        # Continue to detail page
        await browser.act("click the first product")
        state.visited_pages.append("product_detail")
        state.current_step += 1
        
        # Extract details
        details = await browser.extract("get full product specifications")
        state.extracted_data["product_details"] = details.data
        
        return state
```

### Maintaining Context with agent()

Provide context to help the agent understand the workflow:

```python
result = await browser.agent(
    task="Complete the job application process",
    context={
        "applicant_name": "John Doe",
        "email": "john@example.com",
        "resume_path": "/path/to/resume.pdf",
        "cover_letter": "I am interested in this position..."
    },
    max_iterations=40
)
```

## Pagination Workflows

### Processing All Pages

```python
async def process_all_paginated_results(browser, url):
    await browser.goto(url)
    
    all_items = []
    page_number = 1
    
    while True:
        print(f"Processing page {page_number}...")
        
        # Extract current page items
        result = await browser.extract(
            "extract all items on this page",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "price": {"type": "number"},
                        "url": {"type": "string"}
                    }
                }
            }
        )
        
        if result.success and result.data:
            all_items.extend(result.data)
        
        # Check for next page
        has_next = await browser.extract("is there a next page button that is clickable?")
        
        if not has_next.data:
            break
        
        # Navigate to next page
        nav_result = await browser.act("click the Next page button")
        if not nav_result.success:
            break
            
        page_number += 1
        await asyncio.sleep(1)  # Rate limiting
    
    return all_items
```

### Parallel Page Processing

For independent pages, consider processing in parallel:

```python
async def process_urls_parallel(browser, urls, max_concurrent=3):
    """Process multiple URLs, returning to each one."""
    results = {}
    
    for i, url in enumerate(urls):
        await browser.goto(url)
        
        result = await browser.extract("extract the main content")
        results[url] = result.data
        
        # Progress update
        print(f"Processed {i + 1}/{len(urls)}: {url}")
    
    return results
```

## Detail Page Crawling

### Master-Detail Pattern

```python
async def crawl_with_details(browser, listing_url):
    """Get listing, then visit each detail page."""
    await browser.goto(listing_url)
    
    # Get all items with their URLs
    items = await browser.extract(
        "get all items with their detail page links",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"}
                }
            }
        }
    )
    
    detailed_items = []
    
    for item in items.data:
        if not item.get("url"):
            continue
            
        # Visit detail page
        await browser.goto(item["url"])
        
        # Extract full details
        details = await browser.extract(
            "extract all product details",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "description": {"type": "string"},
                    "specifications": {"type": "object"},
                    "images": {"type": "array", "items": {"type": "string"}}
                }
            }
        )
        
        detailed_items.append({
            **item,
            "details": details.data
        })
        
        # Rate limiting
        await asyncio.sleep(0.5)
    
    return detailed_items
```

### Breadcrumb Navigation

```python
async def navigate_via_breadcrumbs(browser):
    """Navigate using breadcrumb trail."""
    await browser.goto("https://shop.example.com/category/subcategory/product")
    
    # Extract breadcrumb path
    breadcrumbs = await browser.extract(
        "get the breadcrumb navigation path",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"}
                }
            }
        }
    )
    
    # Navigate up to parent category
    await browser.act("click the category link in the breadcrumbs")
```

## Form Workflows Across Pages

### Multi-Step Registration

```python
async def complete_registration_flow(browser, user_data):
    """Complete a multi-page registration process."""
    
    await browser.goto("https://example.com/register")
    
    # Step 1: Account Info
    await browser.act(f"type '{user_data['email']}' in the email field")
    await browser.act(f"type '{user_data['password']}' in the password field")
    await browser.act("click Continue or Next")
    
    # Wait for page transition
    await asyncio.sleep(1)
    
    # Step 2: Personal Info
    await browser.act(f"type '{user_data['first_name']}' in the first name field")
    await browser.act(f"type '{user_data['last_name']}' in the last name field")
    await browser.act(f"type '{user_data['phone']}' in the phone field")
    await browser.act("click Continue or Next")
    
    await asyncio.sleep(1)
    
    # Step 3: Preferences
    await browser.act("select my communication preferences")
    await browser.act("check the terms and conditions checkbox")
    await browser.act("click Submit or Complete Registration")
    
    # Verify success
    await asyncio.sleep(2)
    result = await browser.extract("was registration successful?")
    
    return result.data
```

### E-commerce Checkout Flow

```python
async def complete_checkout(browser, order_data):
    """Complete full checkout process."""
    
    # Cart page
    await browser.goto("https://shop.example.com/cart")
    
    cart_total = await browser.extract("what is the cart total?")
    print(f"Cart total: {cart_total.data}")
    
    await browser.act("click Proceed to Checkout")
    await asyncio.sleep(1)
    
    # Shipping page
    await browser.act(f"type '{order_data['shipping']['name']}' in the name field")
    await browser.act(f"type '{order_data['shipping']['address']}' in the address field")
    await browser.act(f"type '{order_data['shipping']['city']}' in the city field")
    await browser.act(f"select '{order_data['shipping']['state']}' from the state dropdown")
    await browser.act(f"type '{order_data['shipping']['zip']}' in the zip field")
    await browser.act("click Continue to Payment")
    await asyncio.sleep(1)
    
    # Payment page
    await browser.act(f"type '{order_data['payment']['card']}' in the card number field")
    await browser.act(f"type '{order_data['payment']['expiry']}' in the expiration field")
    await browser.act(f"type '{order_data['payment']['cvv']}' in the security code field")
    await browser.act("click Review Order")
    await asyncio.sleep(1)
    
    # Review page
    order_summary = await browser.extract(
        "get the complete order summary",
        schema={
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "subtotal": {"type": "number"},
                "shipping": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"}
            }
        }
    )
    
    print(f"Order summary: {order_summary.data}")
    
    # Place order
    await browser.act("click Place Order")
    await asyncio.sleep(3)
    
    # Confirmation page
    confirmation = await browser.extract("get the order confirmation number")
    
    return {
        "success": True,
        "confirmation": confirmation.data,
        "summary": order_summary.data
    }
```

## Site Exploration

### Sitemap Crawling

```python
async def explore_site(browser, start_url, max_pages=50):
    """Explore a site systematically."""
    visited = set()
    to_visit = [start_url]
    site_data = []
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
        
        await browser.goto(url)
        visited.add(url)
        
        # Extract page data
        page_data = await browser.extract(
            "extract the page title, main content summary, and all internal links",
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "links": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        )
        
        site_data.append({
            "url": url,
            **page_data.data
        })
        
        # Add new links to visit
        for link in page_data.data.get("links", []):
            if link not in visited and link.startswith(start_url):
                to_visit.append(link)
        
        print(f"Visited {len(visited)}/{max_pages}: {url}")
    
    return site_data
```

### Category Navigation

```python
async def explore_categories(browser, shop_url):
    """Explore all product categories."""
    await browser.goto(shop_url)
    
    # Get all categories
    categories = await browser.extract(
        "get all product category links from the navigation",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"}
                }
            }
        }
    )
    
    category_data = []
    
    for category in categories.data:
        await browser.goto(category["url"])
        
        # Get products in this category
        products = await browser.extract(
            "get all products in this category",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"}
                    }
                }
            }
        )
        
        category_data.append({
            "category": category["name"],
            "products": products.data,
            "count": len(products.data)
        })
    
    return category_data
```

## Error Recovery in Workflows

### Checkpoint-Based Recovery

```python
class WorkflowCheckpoint:
    def __init__(self):
        self.checkpoints = {}
    
    def save(self, name, data):
        self.checkpoints[name] = data
    
    def get(self, name):
        return self.checkpoints.get(name)
    
    def has(self, name):
        return name in self.checkpoints

async def workflow_with_checkpoints(browser):
    checkpoint = WorkflowCheckpoint()
    
    try:
        # Step 1
        if not checkpoint.has("step1"):
            await browser.goto("https://example.com/start")
            data = await browser.extract("get initial data")
            checkpoint.save("step1", data.data)
        
        # Step 2
        if not checkpoint.has("step2"):
            await browser.act("click Continue")
            await browser.act("fill in the form...")
            checkpoint.save("step2", {"completed": True})
        
        # Step 3
        if not checkpoint.has("step3"):
            await browser.act("submit the form")
            result = await browser.extract("get confirmation")
            checkpoint.save("step3", result.data)
        
        return checkpoint.checkpoints
        
    except Exception as e:
        print(f"Error at checkpoint: {e}")
        print(f"Saved checkpoints: {list(checkpoint.checkpoints.keys())}")
        raise
```

### Retry with Backoff

```python
async def navigate_with_retry(browser, instruction, max_retries=3):
    """Navigate with exponential backoff on failure."""
    for attempt in range(max_retries):
        result = await browser.act(instruction)
        
        if result.success:
            return result
        
        wait_time = 2 ** attempt  # 1, 2, 4 seconds
        print(f"Attempt {attempt + 1} failed, waiting {wait_time}s...")
        await asyncio.sleep(wait_time)
    
    raise Exception(f"Failed after {max_retries} attempts: {instruction}")
```

## Best Practices

### Use Meaningful Waits

```python
async def workflow_with_waits(browser):
    await browser.goto("https://example.com")
    
    # Wait after navigation
    await browser.act("click Products")
    await asyncio.sleep(1)  # Wait for page load
    
    # Wait after AJAX actions
    await browser.act("filter by category Electronics")
    await asyncio.sleep(0.5)  # Wait for filter to apply
    
    # Extract after everything settles
    result = await browser.extract("get filtered products")
```

### Validate Each Step

```python
async def validated_workflow(browser):
    await browser.goto("https://example.com/checkout")
    
    # Fill and validate shipping
    await browser.act("fill shipping address...")
    
    # Verify the form is valid
    errors = await browser.extract("are there any form errors?")
    if errors.data:
        raise Exception(f"Shipping form errors: {errors.data}")
    
    await browser.act("click Continue")
    
    # Verify we moved to next step
    current_step = await browser.extract("what is the current checkout step?")
    if "payment" not in str(current_step.data).lower():
        raise Exception("Failed to advance to payment step")
```

### Log Progress

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workflow")

async def logged_workflow(browser, urls):
    for i, url in enumerate(urls):
        logger.info(f"Processing {i + 1}/{len(urls)}: {url}")
        
        await browser.goto(url)
        
        result = await browser.extract("get page data")
        
        if result.success:
            logger.info(f"Successfully extracted data from {url}")
        else:
            logger.error(f"Failed to extract from {url}: {result.error}")
```

## Next Steps

- [Authentication Guide](authentication.md) - Handle logins across pages
- [Error Handling Guide](error-handling.md) - Robust error recovery
- [SDK Reference](../reference/sdk.md) - Complete API documentation
