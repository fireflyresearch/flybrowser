# Workflow Automation Examples

This guide provides practical examples for automating complex, multi-step workflows using FlyBrowser. From business process automation to scheduled monitoring tasks.

## Introduction to Workflow Automation

FlyBrowser's `agent()` method excels at complex workflows that require:

- Multi-step decision making
- Dynamic adaptation to page state
- Context retention across steps
- Handling unexpected obstacles

The agent uses the ReAct (Reasoning + Acting) framework to intelligently complete tasks.

## Business Process Automation

### Invoice Processing Workflow

Automate invoice download and organization:

```python path=null start=null
"""
Example: Invoice Processing Workflow

Logs into a vendor portal, downloads invoices, and organizes them.
"""

import asyncio
from pathlib import Path
from datetime import datetime
from flybrowser import FlyBrowser

async def process_invoices(vendor_url: str, username: str, password: str):
    """Download and process invoices from vendor portal."""
    
    downloads_dir = Path(f"invoices/{datetime.now().strftime('%Y-%m')}")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="verbose",
    ) as browser:
        # Step 1: Login to vendor portal
        print("Step 1: Logging into vendor portal")
        await browser.goto(vendor_url)
        
        # Store credentials securely
        user_id = browser.store_credential("vendor_user", username, "email")
        pwd_id = browser.store_credential("vendor_pwd", password, "password")
        
        await browser.secure_fill("#username", user_id)
        await browser.secure_fill("#password", pwd_id)
        await browser.act("Click the Login button")
        
        await asyncio.sleep(2)
        
        # Verify login success
        login_check = await browser.extract("Am I logged in? Look for dashboard or account info")
        print(f"  Login result: {login_check.data}")
        
        # Step 2: Navigate to invoices section
        print("\nStep 2: Navigating to invoices")
        await browser.agent(
            "Find and navigate to the invoices or billing section of this portal"
        )
        
        # Step 3: Find unpaid/new invoices
        print("\nStep 3: Finding pending invoices")
        invoices = await browser.extract(
            "List all pending or unpaid invoices with their invoice number, "
            "date, amount, and due date",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "invoice_number": {"type": "string"},
                        "date": {"type": "string"},
                        "amount": {"type": "string"},
                        "due_date": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            }
        )
        
        if not invoices.success:
            print("  Failed to extract invoices")
            return []
        
        print(f"  Found {len(invoices.data)} invoices")
        
        # Step 4: Download each invoice
        print("\nStep 4: Downloading invoices")
        downloaded = []
        
        for invoice in invoices.data:
            inv_num = invoice.get("invoice_number", "unknown")
            print(f"  Downloading: {inv_num}")
            
            # Use agent to handle the download process
            result = await browser.agent(
                f"Download invoice {inv_num}. Click on the invoice, "
                f"then find and click the download or PDF button"
            )
            
            if result.success:
                downloaded.append(invoice)
                print(f"    Downloaded successfully")
            else:
                print(f"    Failed: {result.error}")
        
        # Step 5: Logout
        print("\nStep 5: Logging out")
        await browser.agent("Log out of the portal safely")
        
        return {
            "total_found": len(invoices.data),
            "downloaded": len(downloaded),
            "invoices": downloaded
        }

asyncio.run(process_invoices(
    "https://vendor-portal.example.com",
    "user@company.com",
    "secure-password"
))
```

### Report Generation Workflow

Automate report generation and export:

```python path=null start=null
"""
Example: Automated Report Generation

Generates a report by collecting data from multiple pages.
"""

import asyncio
import json
from datetime import datetime
from flybrowser import FlyBrowser

async def generate_sales_report():
    """Generate comprehensive sales report from dashboard."""
    
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "sections": {}
    }
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        # Login to dashboard
        await browser.goto("https://sales-dashboard.example.com/login")
        await browser.act("Type 'analyst@company.com' in email field")
        await browser.act("Type 'password123' in password field")
        await browser.act("Click Login")
        await asyncio.sleep(2)
        
        # Section 1: Executive Summary
        print("Collecting: Executive Summary")
        await browser.goto("https://sales-dashboard.example.com/summary")
        
        summary = await browser.extract(
            "Extract the key metrics: total revenue, total orders, "
            "average order value, and growth percentage"
        )
        report_data["sections"]["executive_summary"] = summary.data
        
        # Section 2: Top Products
        print("Collecting: Top Products")
        await browser.goto("https://sales-dashboard.example.com/products")
        
        products = await browser.extract(
            "Get the top 10 products by revenue with name, units sold, and revenue"
        )
        report_data["sections"]["top_products"] = products.data
        
        # Section 3: Regional Performance
        print("Collecting: Regional Performance")
        await browser.goto("https://sales-dashboard.example.com/regions")
        
        regions = await browser.extract(
            "Extract performance data for each region including "
            "region name, revenue, growth rate, and top product"
        )
        report_data["sections"]["regional_performance"] = regions.data
        
        # Section 4: Customer Insights
        print("Collecting: Customer Insights")
        await browser.goto("https://sales-dashboard.example.com/customers")
        
        customers = await browser.extract(
            "Get customer metrics: new customers, returning customers, "
            "churn rate, and customer lifetime value"
        )
        report_data["sections"]["customer_insights"] = customers.data
        
        # Section 5: Take screenshots of key charts
        print("Capturing: Visual Charts")
        await browser.goto("https://sales-dashboard.example.com/charts")
        
        charts_screenshot = await browser.screenshot(full_page=True)
        report_data["sections"]["charts_screenshot_id"] = charts_screenshot.get("screenshot_id")
        
        # Save report
        report_filename = f"sales_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_filename, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nReport saved to: {report_filename}")
        
        return report_data

asyncio.run(generate_sales_report())
```

## E-commerce Automation

### Complete Purchase Workflow

Automate a complete purchase process:

```python path=null start=null
"""
Example: Automated Purchase Workflow

Completes a purchase based on specified criteria.
"""

import asyncio
from flybrowser import FlyBrowser

async def automated_purchase(
    product_search: str,
    max_price: float,
    shipping_info: dict,
):
    """
    Automatically find and purchase a product.
    
    Args:
        product_search: Product to search for
        max_price: Maximum acceptable price
        shipping_info: Dictionary with shipping details
    """
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="verbose",
    ) as browser:
        await browser.goto("https://example-store.com")
        
        # Step 1: Search for product
        print(f"Searching for: {product_search}")
        await browser.act(f"Search for '{product_search}'")
        await asyncio.sleep(2)
        
        # Step 2: Filter by price
        print(f"Filtering by max price: ${max_price}")
        await browser.agent(
            f"Filter the search results to show only items under ${max_price}. "
            f"Use the price filter or sort by price low to high."
        )
        
        # Step 3: Find best option
        print("Finding best option")
        products = await browser.extract(
            f"Get the first 5 products that are under ${max_price} "
            f"with their name, price, rating, and review count",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "rating": {"type": "number"},
                        "reviews": {"type": "integer"}
                    }
                }
            }
        )
        
        if not products.success or not products.data:
            print("No products found within budget")
            return None
        
        # Select best rated product within budget
        best_product = max(
            [p for p in products.data if p.get("price", 999) <= max_price],
            key=lambda x: (x.get("rating", 0), x.get("reviews", 0)),
            default=None
        )
        
        if not best_product:
            print("No suitable product found")
            return None
        
        print(f"Selected: {best_product['name']} at ${best_product.get('price')}")
        
        # Step 4: Add to cart
        await browser.agent(
            f"Click on the product '{best_product['name']}' and add it to cart"
        )
        
        # Step 5: Proceed to checkout
        print("Proceeding to checkout")
        await browser.agent("Go to cart and proceed to checkout")
        
        # Step 6: Fill shipping information
        print("Filling shipping information")
        await browser.agent(
            f"Fill the shipping form with: "
            f"Name: {shipping_info['name']}, "
            f"Address: {shipping_info['address']}, "
            f"City: {shipping_info['city']}, "
            f"State: {shipping_info['state']}, "
            f"ZIP: {shipping_info['zip']}, "
            f"Phone: {shipping_info['phone']}"
        )
        
        # Step 7: Continue to payment page (but don't complete)
        print("Navigating to payment")
        await browser.act("Click Continue to proceed to payment options")
        await asyncio.sleep(1)
        
        # Verify order summary
        order_summary = await browser.extract(
            "What is the order total including shipping and tax?"
        )
        
        print(f"\nOrder Summary: {order_summary.data}")
        print("\n*** Payment step skipped - would complete purchase here ***")
        
        return {
            "product": best_product,
            "order_summary": order_summary.data,
            "status": "ready_for_payment"
        }

# Usage
asyncio.run(automated_purchase(
    product_search="wireless earbuds",
    max_price=50.00,
    shipping_info={
        "name": "John Doe",
        "address": "123 Main Street",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94105",
        "phone": "555-123-4567"
    }
))
```

### Price Comparison Workflow

Compare prices across multiple sites:

```python path=null start=null
"""
Example: Multi-Site Price Comparison

Compares prices for a product across multiple e-commerce sites.
"""

import asyncio
from flybrowser import FlyBrowser

SITES = [
    {"name": "Store A", "url": "https://store-a.example.com"},
    {"name": "Store B", "url": "https://store-b.example.com"},
    {"name": "Store C", "url": "https://store-c.example.com"},
]

async def compare_prices(product_name: str):
    """Compare prices across multiple stores."""
    
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        for site in SITES:
            print(f"\nSearching {site['name']}...")
            
            try:
                await browser.goto(site["url"])
                
                # Search for product
                await browser.act(f"Search for '{product_name}'")
                await asyncio.sleep(2)
                
                # Get best match
                product = await browser.extract(
                    f"Find the product most similar to '{product_name}' "
                    f"and get its exact name, price, availability, and rating",
                    schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "string"},
                            "available": {"type": "boolean"},
                            "rating": {"type": "number"},
                            "shipping": {"type": "string"}
                        }
                    }
                )
                
                if product.success and product.data:
                    results.append({
                        "store": site["name"],
                        "url": site["url"],
                        **product.data
                    })
                    print(f"  Found: {product.data.get('name')} at {product.data.get('price')}")
                else:
                    print(f"  Product not found")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    # Sort by price (extracting numeric value)
    def extract_price(item):
        price_str = item.get("price", "$99999")
        try:
            return float(''.join(c for c in price_str if c.isdigit() or c == '.'))
        except:
            return 99999
    
    results.sort(key=extract_price)
    
    # Display comparison
    print("\n" + "=" * 60)
    print("PRICE COMPARISON RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        availability = "In Stock" if result.get("available") else "Out of Stock"
        rating = result.get("rating", "N/A")
        print(f"\n{i}. {result['store']}")
        print(f"   Product: {result.get('name')}")
        print(f"   Price: {result.get('price')}")
        print(f"   Rating: {rating}")
        print(f"   Status: {availability}")
        print(f"   Shipping: {result.get('shipping', 'Unknown')}")
    
    if results:
        best = results[0]
        print(f"\n*** BEST DEAL: {best['store']} at {best.get('price')} ***")
    
    return results

asyncio.run(compare_prices("Sony WH-1000XM5 Headphones"))
```

## Monitoring and Alerts

### Website Monitoring Workflow

Monitor website health and content:

```python path=null start=null
"""
Example: Website Health Monitor

Periodically checks website health and sends alerts.
"""

import asyncio
from datetime import datetime
from flybrowser import FlyBrowser

# Monitoring configuration
CHECKS = [
    {
        "name": "Homepage Load",
        "url": "https://example.com",
        "type": "load_time",
        "threshold_seconds": 5.0
    },
    {
        "name": "Login Page",
        "url": "https://example.com/login",
        "type": "element_present",
        "element": "login form"
    },
    {
        "name": "API Status",
        "url": "https://example.com/status",
        "type": "content_check",
        "expected": "operational"
    },
]

async def run_health_check():
    """Run all health checks."""
    
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        for check in CHECKS:
            print(f"Running check: {check['name']}")
            
            start_time = datetime.now()
            
            try:
                await browser.goto(check["url"])
                load_time = (datetime.now() - start_time).total_seconds()
                
                if check["type"] == "load_time":
                    passed = load_time <= check["threshold_seconds"]
                    message = f"Load time: {load_time:.2f}s (threshold: {check['threshold_seconds']}s)"
                    
                elif check["type"] == "element_present":
                    element = await browser.observe(f"find the {check['element']}")
                    passed = element.success and len(element.data) > 0
                    message = f"Element '{check['element']}': {'found' if passed else 'not found'}"
                    
                elif check["type"] == "content_check":
                    content = await browser.extract("What is the status shown on this page?")
                    passed = check["expected"].lower() in str(content.data).lower()
                    message = f"Content check: expected '{check['expected']}', found '{content.data}'"
                
                else:
                    passed = True
                    message = "Unknown check type"
                
                results.append({
                    "name": check["name"],
                    "url": check["url"],
                    "passed": passed,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                })
                
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {message}")
                
            except Exception as e:
                results.append({
                    "name": check["name"],
                    "url": check["url"],
                    "passed": False,
                    "message": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                print(f"  [FAIL] Error: {e}")
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== Health Check Summary: {passed}/{len(results)} passed ===")
    
    # Alert on failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        send_alert(failures)
    
    return results

def send_alert(failures: list):
    """Send alert for failed checks (implement your notification method)."""
    print("\n!!! ALERT: Health check failures detected !!!")
    for f in failures:
        print(f"  - {f['name']}: {f['message']}")
    # Add your notification logic: email, Slack, PagerDuty, etc.

# Run monitoring check
asyncio.run(run_health_check())
```

### Content Change Detection

Monitor pages for content changes:

```python path=null start=null
"""
Example: Content Change Detector

Monitors specific pages for content changes.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from datetime import datetime
from flybrowser import FlyBrowser

MONITOR_FILE = "content_monitor_state.json"

def load_state() -> dict:
    """Load previous state."""
    if Path(MONITOR_FILE).exists():
        with open(MONITOR_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    """Save current state."""
    with open(MONITOR_FILE, "w") as f:
        json.dump(state, f, indent=2)

async def monitor_content(pages: list[dict]):
    """
    Monitor pages for content changes.
    
    Args:
        pages: List of {"name": str, "url": str, "selector": str}
    """
    
    state = load_state()
    changes = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        for page in pages:
            name = page["name"]
            url = page["url"]
            selector = page.get("selector", "main content")
            
            print(f"Checking: {name}")
            
            try:
                await browser.goto(url)
                
                # Extract the specific content to monitor
                content = await browser.extract(
                    f"Extract the {selector} from this page. "
                    f"Return the text content only."
                )
                
                if not content.success:
                    print(f"  Failed to extract content")
                    continue
                
                # Hash the content for comparison
                current_hash = hashlib.md5(
                    str(content.data).encode()
                ).hexdigest()
                
                previous_hash = state.get(url, {}).get("hash")
                
                if previous_hash is None:
                    # First time monitoring
                    print(f"  Initial capture (no previous state)")
                    
                elif current_hash != previous_hash:
                    # Content changed!
                    print(f"  CHANGE DETECTED!")
                    changes.append({
                        "name": name,
                        "url": url,
                        "old_content": state[url].get("content", "")[:200],
                        "new_content": str(content.data)[:200],
                        "detected_at": datetime.now().isoformat()
                    })
                    
                else:
                    print(f"  No changes")
                
                # Update state
                state[url] = {
                    "name": name,
                    "hash": current_hash,
                    "content": str(content.data)[:1000],
                    "last_checked": datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"  Error: {e}")
    
    save_state(state)
    
    # Report changes
    if changes:
        print("\n=== CONTENT CHANGES DETECTED ===")
        for change in changes:
            print(f"\n{change['name']} ({change['url']})")
            print(f"  Previous: {change['old_content'][:100]}...")
            print(f"  Current: {change['new_content'][:100]}...")
    else:
        print("\nNo content changes detected.")
    
    return changes

# Monitor configuration
pages_to_monitor = [
    {
        "name": "Pricing Page",
        "url": "https://example.com/pricing",
        "selector": "pricing table"
    },
    {
        "name": "Product Features",
        "url": "https://example.com/features",
        "selector": "feature list"
    },
    {
        "name": "Terms of Service",
        "url": "https://example.com/terms",
        "selector": "terms content"
    },
]

asyncio.run(monitor_content(pages_to_monitor))
```

## Data Entry Automation

### Form Data Entry Workflow

Automate repetitive data entry tasks:

```python path=null start=null
"""
Example: Bulk Data Entry Automation

Enters data from a source into a web application.
"""

import asyncio
import csv
from flybrowser import FlyBrowser

async def bulk_data_entry(data_file: str, target_url: str):
    """
    Enter data from CSV file into web application.
    
    Args:
        data_file: Path to CSV file with data
        target_url: URL of the data entry form
    """
    
    # Load data
    with open(data_file) as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    print(f"Loaded {len(records)} records to enter")
    
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto(target_url)
        
        for i, record in enumerate(records, 1):
            print(f"\nProcessing record {i}/{len(records)}")
            
            try:
                # Navigate to new entry form
                await browser.act("Click the 'New Entry' or 'Add Record' button")
                await asyncio.sleep(1)
                
                # Fill the form with record data
                for field, value in record.items():
                    if value:  # Skip empty fields
                        await browser.act(
                            f"Type '{value}' in the {field} field"
                        )
                
                # Submit the form
                await browser.act("Click the Save or Submit button")
                await asyncio.sleep(1)
                
                # Verify submission
                result = await browser.extract(
                    "Was the record saved successfully? Look for success message or error"
                )
                
                success = "success" in str(result.data).lower() or \
                         "saved" in str(result.data).lower()
                
                results.append({
                    "record": i,
                    "data": record,
                    "success": success,
                    "message": result.data
                })
                
                if success:
                    print(f"  Success: {result.data}")
                else:
                    print(f"  Failed: {result.data}")
                
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "record": i,
                    "data": record,
                    "success": False,
                    "message": str(e)
                })
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n=== Data Entry Complete: {successful}/{len(records)} successful ===")
    
    return results

# Example CSV structure expected:
# first_name,last_name,email,phone,department
# John,Doe,john@example.com,555-1234,Sales
# Jane,Smith,jane@example.com,555-5678,Marketing

asyncio.run(bulk_data_entry("employees.csv", "https://hr-system.example.com/employees"))
```

## Complex Multi-System Workflows

### Cross-Platform Data Sync

Sync data between two systems:

```python path=null start=null
"""
Example: Cross-Platform Data Synchronization

Extracts data from one system and enters it into another.
"""

import asyncio
from flybrowser import FlyBrowser

async def sync_systems(source_url: str, target_url: str, source_creds: dict, target_creds: dict):
    """
    Sync data from source system to target system.
    """
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        # === PHASE 1: Extract from source ===
        print("=== PHASE 1: Extracting from source system ===")
        
        await browser.goto(source_url)
        
        # Login to source
        user_id = browser.store_credential("src_user", source_creds["username"], "email")
        pwd_id = browser.store_credential("src_pwd", source_creds["password"], "password")
        
        await browser.secure_fill("#username", user_id)
        await browser.secure_fill("#password", pwd_id)
        await browser.act("Click Login")
        await asyncio.sleep(2)
        
        # Navigate to data export
        await browser.agent("Navigate to the reports or export section")
        
        # Extract data
        source_data = await browser.extract(
            "Extract all records from the table or list view with all visible columns",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True
                }
            }
        )
        
        if not source_data.success:
            print("Failed to extract source data")
            return {"success": False, "error": "Source extraction failed"}
        
        print(f"Extracted {len(source_data.data)} records from source")
        
        # Logout from source
        await browser.agent("Log out of the current system")
        
        # === PHASE 2: Import to target ===
        print("\n=== PHASE 2: Importing to target system ===")
        
        await browser.goto(target_url)
        
        # Login to target
        tgt_user_id = browser.store_credential("tgt_user", target_creds["username"], "email")
        tgt_pwd_id = browser.store_credential("tgt_pwd", target_creds["password"], "password")
        
        await browser.secure_fill("#username", tgt_user_id)
        await browser.secure_fill("#password", tgt_pwd_id)
        await browser.act("Click Login")
        await asyncio.sleep(2)
        
        # Import each record
        imported = 0
        errors = []
        
        for i, record in enumerate(source_data.data, 1):
            print(f"Importing record {i}/{len(source_data.data)}")
            
            try:
                # Use agent for intelligent form filling
                result = await browser.agent(
                    f"Create a new record with this data: {record}. "
                    f"Click 'New' or 'Add', fill in the appropriate fields, "
                    f"and save the record."
                )
                
                if result.success:
                    imported += 1
                else:
                    errors.append({"record": i, "error": result.error})
                    
            except Exception as e:
                errors.append({"record": i, "error": str(e)})
        
        # Logout from target
        await browser.agent("Log out of the current system")
        
        print(f"\n=== SYNC COMPLETE ===")
        print(f"  Source records: {len(source_data.data)}")
        print(f"  Successfully imported: {imported}")
        print(f"  Errors: {len(errors)}")
        
        return {
            "success": imported == len(source_data.data),
            "total": len(source_data.data),
            "imported": imported,
            "errors": errors
        }

asyncio.run(sync_systems(
    source_url="https://old-system.example.com",
    target_url="https://new-system.example.com",
    source_creds={"username": "admin", "password": "oldpass"},
    target_creds={"username": "admin", "password": "newpass"}
))
```

## Scheduled Workflows

### Running Workflows on Schedule

Set up scheduled execution:

```python path=null start=null
"""
Example: Scheduled Workflow Execution

Runs workflows on a defined schedule.
"""

import asyncio
import schedule
import time
from datetime import datetime
from flybrowser import FlyBrowser

async def daily_report_task():
    """Task that runs daily."""
    print(f"\n[{datetime.now()}] Running daily report task")
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        await browser.goto("https://dashboard.example.com")
        
        # Login and extract daily metrics
        await browser.act("Login with saved credentials")
        await asyncio.sleep(2)
        
        metrics = await browser.extract(
            "Get today's key metrics: total sales, new users, and active sessions"
        )
        
        print(f"Daily Metrics: {metrics.data}")
        
        # Save or send report
        save_daily_report(metrics.data)

async def hourly_price_check():
    """Task that runs hourly."""
    print(f"\n[{datetime.now()}] Running hourly price check")
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="minimal",
    ) as browser:
        await browser.goto("https://competitor.example.com/products")
        
        prices = await browser.extract(
            "Get the current prices for the first 5 products"
        )
        
        print(f"Current Prices: {prices.data}")
        
        # Check for significant changes
        check_price_alerts(prices.data)

def save_daily_report(data):
    """Save daily report data."""
    filename = f"reports/daily_{datetime.now().strftime('%Y%m%d')}.json"
    print(f"  Saving to: {filename}")

def check_price_alerts(prices):
    """Check if prices trigger any alerts."""
    print("  Checking price thresholds...")

def run_async_task(coro):
    """Helper to run async task from sync scheduler."""
    asyncio.run(coro())

# Schedule configuration
schedule.every().day.at("09:00").do(run_async_task, daily_report_task)
schedule.every().hour.do(run_async_task, hourly_price_check)

print("Scheduler started. Press Ctrl+C to exit.")
print("Scheduled tasks:")
print("  - Daily report: 09:00")
print("  - Hourly price check: every hour")

# Keep scheduler running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

## Error Recovery Patterns

### Robust Workflow with Retry Logic

Handle errors gracefully in workflows:

```python path=null start=null
"""
Example: Robust Workflow with Error Recovery

Demonstrates error handling and retry patterns.
"""

import asyncio
from flybrowser import FlyBrowser

MAX_RETRIES = 3
RETRY_DELAY = 5

async def robust_workflow(task_description: str):
    """
    Execute a workflow with automatic retry and error recovery.
    """
    
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\nAttempt {attempt}/{MAX_RETRIES}")
        
        try:
            async with FlyBrowser(
                llm_provider="openai",
                api_key="sk-...",
                log_verbosity="normal",
            ) as browser:
                # Execute the main task
                result = await browser.agent(
                    task_description,
                    max_iterations=50,
                    max_time_seconds=300
                )
                
                if result.success:
                    print("Task completed successfully!")
                    return {
                        "success": True,
                        "data": result.data,
                        "attempts": attempt
                    }
                else:
                    print(f"Task failed: {result.error}")
                    last_error = result.error
                    
                    # Analyze failure and potentially adjust approach
                    if "timeout" in str(result.error).lower():
                        print("  Timeout detected - will retry with longer timeout")
                    elif "not found" in str(result.error).lower():
                        print("  Element not found - will retry")
                    else:
                        print(f"  Unknown error: {result.error}")
                        
        except Exception as e:
            print(f"Exception occurred: {e}")
            last_error = str(e)
        
        if attempt < MAX_RETRIES:
            print(f"Waiting {RETRY_DELAY}s before retry...")
            await asyncio.sleep(RETRY_DELAY)
    
    print(f"\nAll {MAX_RETRIES} attempts failed")
    return {
        "success": False,
        "error": last_error,
        "attempts": MAX_RETRIES
    }

# Usage
asyncio.run(robust_workflow(
    "Go to example.com, find the contact form, and submit an inquiry "
    "about product pricing"
))
```

## Best Practices

### Workflow Organization

```python path=null start=null
# Structure workflows as reusable functions
class WorkflowBase:
    """Base class for workflows."""
    
    def __init__(self, browser_config: dict):
        self.browser_config = browser_config
    
    async def execute(self):
        """Override in subclass."""
        raise NotImplementedError
    
    async def setup(self, browser):
        """Workflow setup steps."""
        pass
    
    async def teardown(self, browser):
        """Workflow cleanup steps."""
        pass
    
    async def run(self):
        """Execute the full workflow."""
        async with FlyBrowser(**self.browser_config) as browser:
            await self.setup(browser)
            try:
                result = await self.execute(browser)
                return result
            finally:
                await self.teardown(browser)
```

### Logging and Monitoring

```python path=null start=null
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"workflows_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("workflow")

async def logged_workflow():
    logger.info("Starting workflow")
    try:
        # ... workflow steps ...
        logger.info("Workflow completed successfully")
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise
```

### Configuration Management

```python path=null start=null
import os
from dataclasses import dataclass

@dataclass
class WorkflowConfig:
    """Workflow configuration from environment."""
    llm_provider: str = os.getenv("FLYBROWSER_LLM_PROVIDER", "openai")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    headless: bool = os.getenv("FLYBROWSER_HEADLESS", "true").lower() == "true"
    log_level: str = os.getenv("FLYBROWSER_LOG_LEVEL", "normal")
    
    def to_browser_config(self) -> dict:
        return {
            "llm_provider": self.llm_provider,
            "api_key": self.api_key,
            "headless": self.headless,
            "log_verbosity": self.log_level,
        }
```

## Next Steps

- [Web Scraping Examples](web-scraping.md) - Data extraction patterns
- [UI Testing Examples](ui-testing.md) - Testing automation
- [Error Handling Guide](../guides/error-handling.md) - Robust error handling
- [Performance Guide](../advanced/performance.md) - Optimizing workflows
