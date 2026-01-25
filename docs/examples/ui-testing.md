# UI Testing Examples

This guide provides practical examples for automated UI testing using FlyBrowser. From form validation to end-to-end user flow testing.

## Why FlyBrowser for UI Testing?

FlyBrowser offers unique advantages for UI testing:

1. **Natural Language Tests** - Write tests that describe user intent, not implementation details
2. **Self-Healing** - Tests adapt to minor UI changes without breaking
3. **Visual Verification** - Use vision to verify visual appearance, not just DOM state
4. **Intelligent Waiting** - Automatically waits for dynamic content, no flaky sleep statements

## Basic Form Testing

### Form Input Validation

Test form validation behavior:

```python path=null start=null
"""
Example: Form Validation Testing

Tests input validation on a registration form.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_form_validation():
    """Test that form validation works correctly."""
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=True,
    ) as browser:
        await browser.goto("https://example.com/register")
        
        # Test 1: Empty form submission
        print("Test 1: Empty form submission")
        await browser.act("Click the Submit button without filling any fields")
        
        # Check for validation errors
        errors = await browser.extract("What error messages are shown on the form?")
        
        if errors.success and errors.data:
            print(f"  PASS: Validation errors shown: {errors.data}")
            results.append(("Empty form validation", True))
        else:
            print("  FAIL: No validation errors shown")
            results.append(("Empty form validation", False))
        
        # Test 2: Invalid email format
        print("\nTest 2: Invalid email format")
        await browser.act("Type 'invalid-email' in the email field")
        await browser.act("Click outside the email field to trigger validation")
        
        email_error = await browser.extract(
            "Is there an error message about the email format?"
        )
        
        if email_error.success and "invalid" in str(email_error.data).lower():
            print("  PASS: Email format error shown")
            results.append(("Invalid email validation", True))
        else:
            print("  FAIL: No email format error")
            results.append(("Invalid email validation", False))
        
        # Test 3: Password requirements
        print("\nTest 3: Password requirements")
        await browser.act("Clear the email field and type 'test@example.com'")
        await browser.act("Type '123' in the password field")
        await browser.act("Click outside the password field")
        
        pwd_error = await browser.extract(
            "What does the password requirement error say?"
        )
        
        if pwd_error.success and pwd_error.data:
            print(f"  PASS: Password error shown: {pwd_error.data}")
            results.append(("Password requirements", True))
        else:
            print("  FAIL: No password requirement error")
            results.append(("Password requirements", False))
        
        # Test 4: Successful validation
        print("\nTest 4: Valid form submission")
        await browser.act("Clear all fields")
        await browser.act("Type 'test@example.com' in the email field")
        await browser.act("Type 'SecurePassword123!' in the password field")
        await browser.act("Type 'SecurePassword123!' in the confirm password field")
        await browser.act("Check the 'I agree to terms' checkbox")
        
        # Verify no errors are shown
        final_errors = await browser.extract(
            "Are there any validation errors currently visible on the form?"
        )
        
        if not final_errors.data or "no" in str(final_errors.data).lower():
            print("  PASS: Form validates successfully")
            results.append(("Valid form passes", True))
        else:
            print(f"  FAIL: Unexpected errors: {final_errors.data}")
            results.append(("Valid form passes", False))
    
    # Summary
    passed = sum(1 for _, p in results if p)
    print(f"\n=== Results: {passed}/{len(results)} tests passed ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    return results

asyncio.run(test_form_validation())
```

### Form Submission Test

Test complete form submission:

```python path=null start=null
"""
Example: Form Submission Testing

Tests a complete form submission workflow.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_form_submission():
    """Test successful form submission."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com/contact")
        
        # Fill out the contact form
        await browser.act("Type 'John Doe' in the name field")
        await browser.act("Type 'john@example.com' in the email field")
        await browser.act("Select 'Sales Inquiry' from the subject dropdown")
        await browser.act("Type 'I would like to learn more about your enterprise plan.' in the message field")
        
        # Take screenshot before submission
        before_screenshot = await browser.screenshot()
        
        # Submit the form
        await browser.act("Click the Submit button")
        
        # Wait for response
        await asyncio.sleep(2)
        
        # Verify submission
        result = await browser.extract(
            "Is there a success message? What does it say?"
        )
        
        if result.success:
            print(f"Submission result: {result.data}")
            
            # Additional verification
            current_url = await browser.extract("What is the current page URL or title?")
            print(f"Current page: {current_url.data}")
        
        # Take screenshot after submission
        after_screenshot = await browser.screenshot()
        
        return {
            "success": "success" in str(result.data).lower() or "thank" in str(result.data).lower(),
            "message": result.data,
            "before_screenshot": before_screenshot,
            "after_screenshot": after_screenshot
        }

asyncio.run(test_form_submission())
```

## Navigation Testing

### Link Navigation Test

Verify navigation links work correctly:

```python path=null start=null
"""
Example: Navigation Link Testing

Tests that all main navigation links work correctly.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_navigation_links():
    """Test that navigation links lead to correct pages."""
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        base_url = "https://example.com"
        await browser.goto(base_url)
        
        # Get all navigation links
        nav_links = await browser.extract(
            "Get all main navigation links with their text and expected destinations"
        )
        
        if not nav_links.success:
            print("Failed to extract navigation links")
            return []
        
        print(f"Found {len(nav_links.data)} navigation links")
        
        for link in nav_links.data:
            link_text = link.get("text", "Unknown")
            print(f"\nTesting: {link_text}")
            
            # Navigate using the link
            await browser.goto(base_url)  # Reset to home
            await browser.act(f"Click the '{link_text}' navigation link")
            
            await asyncio.sleep(1)
            
            # Verify the page loaded correctly
            page_info = await browser.extract(
                "What is the current page title and main heading?"
            )
            
            # Check if page matches expected destination
            expected = link_text.lower()
            actual = str(page_info.data).lower()
            
            passed = expected in actual or link_text.lower() in actual
            results.append({
                "link": link_text,
                "page_info": page_info.data,
                "passed": passed
            })
            
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] Page: {page_info.data}")
    
    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n=== Navigation Test Results: {passed_count}/{len(results)} passed ===")
    
    return results

asyncio.run(test_navigation_links())
```

### Breadcrumb Navigation Test

Test breadcrumb navigation:

```python path=null start=null
"""
Example: Breadcrumb Navigation Test

Tests that breadcrumb navigation works correctly.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_breadcrumbs():
    """Test breadcrumb navigation functionality."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        # Navigate to a deep page
        await browser.goto("https://example-store.com/electronics/phones/iphone-15")
        
        # Get breadcrumb trail
        breadcrumbs = await browser.extract(
            "What are the breadcrumb links shown on this page?"
        )
        
        print(f"Breadcrumb trail: {breadcrumbs.data}")
        
        # Test each breadcrumb link
        if breadcrumbs.success and isinstance(breadcrumbs.data, list):
            for i, crumb in enumerate(breadcrumbs.data[:-1]):  # Skip current page
                print(f"\nTesting breadcrumb: {crumb}")
                
                await browser.act(f"Click the '{crumb}' breadcrumb link")
                await asyncio.sleep(1)
                
                # Verify navigation
                current = await browser.extract("What page am I on now?")
                print(f"  Navigated to: {current.data}")
                
                # Go back to original page for next test
                await browser.goto("https://example-store.com/electronics/phones/iphone-15")

asyncio.run(test_breadcrumbs())
```

## Authentication Testing

### Login Flow Test

Test complete login workflow:

```python path=null start=null
"""
Example: Login Flow Testing

Tests the complete login workflow including error handling.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_login_flow():
    """Test login functionality."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com/login")
        
        # Test 1: Invalid credentials
        print("Test 1: Invalid credentials")
        await browser.act("Type 'wrong@example.com' in the email field")
        await browser.act("Type 'wrongpassword' in the password field")
        await browser.act("Click the Login button")
        
        await asyncio.sleep(2)
        
        error = await browser.extract("Is there a login error message?")
        
        if error.success and error.data:
            print(f"  PASS: Error shown: {error.data}")
        else:
            print("  FAIL: No error for invalid credentials")
        
        # Test 2: Empty password
        print("\nTest 2: Empty password")
        await browser.goto("https://example.com/login")
        await browser.act("Type 'test@example.com' in the email field")
        await browser.act("Leave the password field empty and click Login")
        
        pwd_error = await browser.extract("What validation error is shown?")
        print(f"  Validation: {pwd_error.data}")
        
        # Test 3: Valid login (use secure credential storage)
        print("\nTest 3: Valid login")
        await browser.goto("https://example.com/login")
        
        # Store credentials securely
        email_id = browser.store_credential("test_email", "test@example.com", "email")
        pwd_id = browser.store_credential("test_password", "securepassword123", "password")
        
        # Fill form with secure credentials
        await browser.act("Clear the email field")
        await browser.secure_fill("#email", email_id)
        await browser.secure_fill("#password", pwd_id)
        await browser.act("Click the Login button")
        
        await asyncio.sleep(3)
        
        # Verify successful login
        logged_in = await browser.extract(
            "Am I logged in? Is there a user profile or dashboard visible?"
        )
        
        if logged_in.success:
            print(f"  Login result: {logged_in.data}")
        
        return logged_in.data

asyncio.run(test_login_flow())
```

### Session Persistence Test

Test that sessions persist correctly:

```python path=null start=null
"""
Example: Session Persistence Test

Tests that user session persists across page navigations.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_session_persistence():
    """Test that login session persists."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        # Login first
        await browser.goto("https://example.com/login")
        await browser.act("Type 'test@example.com' in the email field")
        await browser.act("Type 'password123' in the password field")
        await browser.act("Click the Login button")
        
        await asyncio.sleep(2)
        
        # Navigate to different pages and check session
        pages_to_test = [
            ("https://example.com/products", "Products page"),
            ("https://example.com/account", "Account page"),
            ("https://example.com/orders", "Orders page"),
        ]
        
        results = []
        
        for url, page_name in pages_to_test:
            await browser.goto(url)
            await asyncio.sleep(1)
            
            # Check if still logged in
            session_check = await browser.extract(
                "Is the user still logged in? Look for user name, logout button, or login prompt"
            )
            
            logged_in = "logout" in str(session_check.data).lower() or \
                       "user" in str(session_check.data).lower()
            
            results.append((page_name, logged_in))
            
            status = "PASS" if logged_in else "FAIL"
            print(f"[{status}] {page_name}: Session {'active' if logged_in else 'lost'}")
        
        # Final check after navigation
        await browser.goto("https://example.com/account/settings")
        final_check = await browser.extract("Can I access account settings?")
        
        print(f"\nFinal session check: {final_check.data}")
        
        return results

asyncio.run(test_session_persistence())
```

## Visual Testing

### Screenshot Comparison

Capture screenshots for visual comparison:

```python path=null start=null
"""
Example: Visual Screenshot Testing

Captures screenshots at various states for visual comparison.
"""

import asyncio
import base64
from pathlib import Path
from datetime import datetime
from flybrowser import FlyBrowser

async def capture_visual_states(url: str, test_name: str):
    """Capture screenshots at different viewport sizes and states."""
    
    output_dir = Path(f"screenshots/{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto(url)
        
        # Capture initial state
        screenshot = await browser.screenshot(full_page=True)
        save_screenshot(screenshot, output_dir / "initial.png")
        print("Captured: initial state")
        
        # Capture after scrolling
        await browser.act("Scroll down to the middle of the page")
        screenshot = await browser.screenshot()
        save_screenshot(screenshot, output_dir / "middle_viewport.png")
        print("Captured: middle viewport")
        
        # Capture footer area
        await browser.act("Scroll to the bottom of the page")
        screenshot = await browser.screenshot()
        save_screenshot(screenshot, output_dir / "footer.png")
        print("Captured: footer")
        
        # Capture with modal open (if applicable)
        try:
            await browser.act("Click a button that opens a modal or popup")
            await asyncio.sleep(1)
            screenshot = await browser.screenshot()
            save_screenshot(screenshot, output_dir / "modal_open.png")
            print("Captured: modal state")
        except:
            print("Skipped: modal state (no modal found)")
        
        print(f"\nScreenshots saved to: {output_dir}")
        return str(output_dir)

def save_screenshot(screenshot_data: dict, path: Path):
    """Save screenshot from base64 data."""
    image_data = base64.b64decode(screenshot_data["data_base64"])
    with open(path, "wb") as f:
        f.write(image_data)

asyncio.run(capture_visual_states("https://example.com", "homepage"))
```

### Visual Element Verification

Verify visual elements are present:

```python path=null start=null
"""
Example: Visual Element Verification

Uses vision to verify visual elements are displayed correctly.
"""

import asyncio
from flybrowser import FlyBrowser

async def verify_visual_elements():
    """Verify key visual elements using vision."""
    
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4o",  # Vision model
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com")
        
        # List of visual elements to verify
        elements_to_check = [
            "company logo in the header",
            "navigation menu",
            "hero image or banner",
            "call-to-action button",
            "footer with social media links",
        ]
        
        results = []
        
        for element in elements_to_check:
            # Use vision to check if element is visible
            check = await browser.extract(
                f"Is the {element} visible and properly displayed on the page? "
                "Describe what you see.",
                use_vision=True
            )
            
            visible = check.success and check.data and "yes" in str(check.data).lower()
            
            results.append({
                "element": element,
                "visible": visible,
                "description": check.data
            })
            
            status = "PASS" if visible else "FAIL"
            print(f"[{status}] {element}")
            if check.data:
                print(f"       {check.data[:100]}...")
        
        # Summary
        visible_count = sum(1 for r in results if r["visible"])
        print(f"\n=== Visual Check: {visible_count}/{len(results)} elements verified ===")
        
        return results

asyncio.run(verify_visual_elements())
```

## Responsive Testing

### Multi-Viewport Testing

Test across different viewport sizes:

```python path=null start=null
"""
Example: Responsive Design Testing

Tests the page across multiple viewport sizes.
"""

import asyncio
from flybrowser import FlyBrowser

# Common viewport sizes
VIEWPORTS = [
    {"name": "Mobile", "width": 375, "height": 667},
    {"name": "Tablet", "width": 768, "height": 1024},
    {"name": "Desktop", "width": 1920, "height": 1080},
]

async def test_responsive_design(url: str):
    """Test responsive behavior across viewport sizes."""
    
    results = []
    
    for viewport in VIEWPORTS:
        print(f"\n=== Testing {viewport['name']} ({viewport['width']}x{viewport['height']}) ===")
        
        async with FlyBrowser(
            llm_provider="openai",
            llm_model="gpt-4o",
            api_key="sk-...",
            headless=True,
        ) as browser:
            await browser.goto(url)
            
            # Set viewport (through JavaScript as FlyBrowser uses Playwright)
            page = browser.browser_manager.page
            await page.set_viewport_size({
                "width": viewport["width"],
                "height": viewport["height"]
            })
            
            await asyncio.sleep(1)  # Wait for reflow
            
            # Take screenshot
            screenshot = await browser.screenshot()
            
            # Check layout with vision
            layout_check = await browser.extract(
                f"Analyze the current layout. Is it properly adapted for a "
                f"{viewport['name'].lower()} viewport? Check for: "
                f"1) Content not overflowing "
                f"2) Text is readable "
                f"3) Navigation is accessible "
                f"4) Images are properly sized",
                use_vision=True
            )
            
            # Check specific responsive elements
            nav_check = await browser.extract(
                "Is the navigation menu visible? Is it a hamburger menu or full menu?",
                use_vision=True
            )
            
            results.append({
                "viewport": viewport["name"],
                "layout": layout_check.data,
                "navigation": nav_check.data,
                "screenshot_id": screenshot.get("screenshot_id")
            })
            
            print(f"  Layout: {str(layout_check.data)[:100]}...")
            print(f"  Navigation: {nav_check.data}")
    
    return results

asyncio.run(test_responsive_design("https://example.com"))
```

## Interactive Component Testing

### Dropdown and Select Testing

Test dropdown and select components:

```python path=null start=null
"""
Example: Dropdown Component Testing

Tests dropdown and select component behavior.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_dropdown_components():
    """Test various dropdown interactions."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com/form-with-dropdowns")
        
        # Test 1: Basic dropdown selection
        print("Test 1: Basic dropdown selection")
        await browser.act("Click on the Country dropdown")
        
        # Verify dropdown opened
        options = await browser.extract("What options are shown in the dropdown?")
        print(f"  Available options: {options.data}")
        
        # Select an option
        await browser.act("Select 'United States' from the dropdown")
        
        # Verify selection
        selected = await browser.extract("What country is currently selected?")
        
        passed = "united states" in str(selected.data).lower()
        print(f"  {'PASS' if passed else 'FAIL'}: Selected = {selected.data}")
        
        # Test 2: Cascading dropdowns
        print("\nTest 2: Cascading dropdowns (State depends on Country)")
        
        # Check if state dropdown is now populated
        await browser.act("Click on the State dropdown")
        states = await browser.extract("What states are shown?")
        
        has_us_states = any(
            state in str(states.data).lower() 
            for state in ["california", "texas", "new york", "florida"]
        )
        print(f"  {'PASS' if has_us_states else 'FAIL'}: US states loaded")
        
        # Test 3: Search/filter dropdown
        print("\nTest 3: Searchable dropdown")
        await browser.act("Select 'California' from the state dropdown")
        
        # If there's a searchable dropdown
        await browser.act("Click on the City dropdown and type 'San' to filter")
        filtered = await browser.extract("What cities are shown after filtering?")
        print(f"  Filtered results: {filtered.data}")

asyncio.run(test_dropdown_components())
```

### Modal and Dialog Testing

Test modal dialogs:

```python path=null start=null
"""
Example: Modal Dialog Testing

Tests modal dialog behavior and interactions.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_modal_dialogs():
    """Test modal dialog functionality."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com/page-with-modals")
        
        # Test 1: Open modal
        print("Test 1: Opening modal")
        await browser.act("Click the button that opens a modal dialog")
        await asyncio.sleep(0.5)
        
        # Verify modal is open
        modal_visible = await browser.extract(
            "Is there a modal dialog visible? What is its title?"
        )
        print(f"  Modal: {modal_visible.data}")
        
        # Test 2: Modal content interaction
        print("\nTest 2: Interacting with modal content")
        await browser.act("Fill out any form fields inside the modal")
        
        # Test 3: Close modal with X button
        print("\nTest 3: Close with X button")
        await browser.act("Click the X button to close the modal")
        await asyncio.sleep(0.5)
        
        closed = await browser.extract("Is the modal still visible?")
        print(f"  Modal closed: {'no' in str(closed.data).lower()}")
        
        # Test 4: Close with backdrop click
        print("\nTest 4: Close by clicking backdrop")
        await browser.act("Open the modal again")
        await asyncio.sleep(0.5)
        await browser.act("Click outside the modal on the dark backdrop")
        await asyncio.sleep(0.5)
        
        closed_backdrop = await browser.extract("Is the modal still visible?")
        print(f"  Backdrop close works: {'no' in str(closed_backdrop.data).lower()}")
        
        # Test 5: Close with Escape key
        print("\nTest 5: Close with Escape key")
        await browser.act("Open the modal again")
        await asyncio.sleep(0.5)
        await browser.act("Press the Escape key")
        await asyncio.sleep(0.5)
        
        closed_escape = await browser.extract("Is the modal still visible?")
        print(f"  Escape close works: {'no' in str(closed_escape.data).lower()}")

asyncio.run(test_modal_dialogs())
```

## End-to-End Test Examples

### E-commerce Checkout Flow

Complete checkout flow test:

```python path=null start=null
"""
Example: E-commerce Checkout Flow Test

Tests the complete checkout process from product selection to order confirmation.
"""

import asyncio
from flybrowser import FlyBrowser

async def test_checkout_flow():
    """Test complete e-commerce checkout flow."""
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        log_verbosity="verbose",
    ) as browser:
        await browser.goto("https://example-store.com")
        
        # Step 1: Search for product
        print("Step 1: Search for product")
        await browser.act("Search for 'wireless headphones'")
        await asyncio.sleep(2)
        
        results = await browser.extract("How many search results are shown?")
        print(f"  Search results: {results.data}")
        
        # Step 2: Select product
        print("\nStep 2: Select product")
        await browser.act("Click on the first product in the results")
        await asyncio.sleep(1)
        
        product = await browser.extract("What is the product name and price?")
        print(f"  Product: {product.data}")
        
        # Step 3: Add to cart
        print("\nStep 3: Add to cart")
        await browser.act("Click the 'Add to Cart' button")
        await asyncio.sleep(1)
        
        cart_confirmation = await browser.extract(
            "Is there a confirmation that the item was added to cart?"
        )
        print(f"  Cart confirmation: {cart_confirmation.data}")
        
        # Step 4: Go to cart
        print("\nStep 4: View cart")
        await browser.act("Click on the shopping cart icon or 'View Cart' button")
        await asyncio.sleep(1)
        
        cart_contents = await browser.extract(
            "What items are in the cart? What is the total?"
        )
        print(f"  Cart: {cart_contents.data}")
        
        # Step 5: Proceed to checkout
        print("\nStep 5: Proceed to checkout")
        await browser.act("Click 'Proceed to Checkout' or 'Checkout' button")
        await asyncio.sleep(2)
        
        # Step 6: Fill shipping information
        print("\nStep 6: Fill shipping info")
        await browser.agent(
            "Fill out the shipping form with: "
            "Name: John Doe, "
            "Address: 123 Test Street, "
            "City: San Francisco, "
            "State: California, "
            "ZIP: 94105, "
            "Phone: 555-123-4567"
        )
        
        # Step 7: Continue to payment
        print("\nStep 7: Continue to payment")
        await browser.act("Click Continue or Next to proceed to payment")
        await asyncio.sleep(1)
        
        # Verify we're on payment page
        payment_page = await browser.extract(
            "Am I on the payment page? What payment options are available?"
        )
        print(f"  Payment options: {payment_page.data}")
        
        # Step 8: Verify order summary
        print("\nStep 8: Verify order summary")
        order_summary = await browser.extract(
            "What is shown in the order summary? Include items, shipping, and total."
        )
        print(f"  Order summary: {order_summary.data}")
        
        print("\n=== Checkout Flow Test Complete ===")
        print("Note: Payment step skipped for test purposes")
        
        return {
            "product": product.data,
            "cart": cart_contents.data,
            "order_summary": order_summary.data
        }

asyncio.run(test_checkout_flow())
```

### User Registration Flow

Test user registration:

```python path=null start=null
"""
Example: User Registration Flow Test

Tests the complete user registration process.
"""

import asyncio
import random
import string
from flybrowser import FlyBrowser

def generate_test_email():
    """Generate unique test email."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=8))
    return f"test_{suffix}@example-test.com"

async def test_registration_flow():
    """Test complete registration flow."""
    
    test_email = generate_test_email()
    test_password = "TestPassword123!"
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com/register")
        
        # Step 1: Fill registration form
        print("Step 1: Fill registration form")
        await browser.act(f"Type 'John' in the first name field")
        await browser.act(f"Type 'Doe' in the last name field")
        await browser.act(f"Type '{test_email}' in the email field")
        await browser.act(f"Type '{test_password}' in the password field")
        await browser.act(f"Type '{test_password}' in the confirm password field")
        
        # Step 2: Accept terms
        print("\nStep 2: Accept terms and conditions")
        await browser.act("Check the 'I agree to Terms and Conditions' checkbox")
        
        # Step 3: Complete CAPTCHA if present
        print("\nStep 3: Handle CAPTCHA (if present)")
        captcha = await browser.extract("Is there a CAPTCHA on this page?")
        if "yes" in str(captcha.data).lower():
            print("  CAPTCHA detected - manual intervention may be required")
        
        # Step 4: Submit registration
        print("\nStep 4: Submit registration")
        await browser.act("Click the Register or Sign Up button")
        await asyncio.sleep(3)
        
        # Step 5: Verify success
        print("\nStep 5: Verify registration")
        result = await browser.extract(
            "Was the registration successful? Look for success message, "
            "email verification prompt, or redirect to dashboard"
        )
        
        print(f"  Result: {result.data}")
        
        # Check current page
        current_page = await browser.extract("What page am I on now?")
        print(f"  Current page: {current_page.data}")
        
        success = any(word in str(result.data).lower() for word in 
                     ["success", "welcome", "verify", "email sent", "dashboard"])
        
        return {
            "email": test_email,
            "success": success,
            "result": result.data
        }

asyncio.run(test_registration_flow())
```

## Test Framework Integration

### pytest Integration

Integrate FlyBrowser with pytest:

```python path=null start=null
"""
Example: pytest Integration

Shows how to integrate FlyBrowser tests with pytest.

Run with: pytest test_example.py -v
"""

import pytest
import asyncio
from flybrowser import FlyBrowser

# Fixture for browser instance
@pytest.fixture
async def browser():
    """Create browser instance for tests."""
    browser = FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=True,
        log_verbosity="minimal",
    )
    await browser.start()
    yield browser
    await browser.stop()

# Test functions
@pytest.mark.asyncio
async def test_homepage_loads(browser):
    """Test that homepage loads correctly."""
    await browser.goto("https://example.com")
    
    result = await browser.extract("What is the page title?")
    
    assert result.success
    assert result.data is not None
    assert "example" in str(result.data).lower()

@pytest.mark.asyncio
async def test_navigation_menu_visible(browser):
    """Test that navigation menu is visible."""
    await browser.goto("https://example.com")
    
    result = await browser.observe("find the main navigation menu")
    
    assert result.success
    assert len(result.data) > 0

@pytest.mark.asyncio
async def test_search_functionality(browser):
    """Test that search works."""
    await browser.goto("https://example.com")
    
    await browser.act("Type 'test query' in the search box and press Enter")
    await asyncio.sleep(2)
    
    result = await browser.extract("Are search results shown?")
    
    assert result.success
    assert "results" in str(result.data).lower() or "found" in str(result.data).lower()

@pytest.mark.asyncio
async def test_contact_form_validation(browser):
    """Test contact form validation."""
    await browser.goto("https://example.com/contact")
    
    # Submit empty form
    await browser.act("Click the Submit button without filling any fields")
    
    result = await browser.extract("Are validation errors shown?")
    
    assert result.success
    assert any(word in str(result.data).lower() for word in ["error", "required", "invalid"])
```

### Running Tests with Custom Configuration

```python path=null start=null
"""
Example: Configurable Test Suite

Demonstrates running tests with different configurations.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from flybrowser import FlyBrowser

@dataclass
class TestConfig:
    """Test configuration."""
    base_url: str
    llm_provider: str = "openai"
    api_key: Optional[str] = None
    headless: bool = True
    timeout: float = 30.0
    
# Test configurations for different environments
CONFIGS = {
    "staging": TestConfig(
        base_url="https://staging.example.com",
        headless=True,
    ),
    "production": TestConfig(
        base_url="https://example.com",
        headless=True,
    ),
    "local": TestConfig(
        base_url="http://localhost:3000",
        headless=False,  # Show browser for local debugging
    ),
}

async def run_test_suite(env: str = "staging"):
    """Run test suite for specified environment."""
    config = CONFIGS.get(env)
    if not config:
        raise ValueError(f"Unknown environment: {env}")
    
    print(f"Running tests against: {config.base_url}")
    
    results = []
    
    async with FlyBrowser(
        llm_provider=config.llm_provider,
        api_key=config.api_key,
        headless=config.headless,
        timeout=config.timeout,
    ) as browser:
        # Run tests
        tests = [
            ("Homepage loads", test_homepage(browser, config.base_url)),
            ("Navigation works", test_navigation(browser, config.base_url)),
            ("Search works", test_search(browser, config.base_url)),
        ]
        
        for test_name, test_coro in tests:
            try:
                result = await test_coro
                results.append((test_name, result, None))
                print(f"[PASS] {test_name}")
            except Exception as e:
                results.append((test_name, False, str(e)))
                print(f"[FAIL] {test_name}: {e}")
    
    # Summary
    passed = sum(1 for _, r, _ in results if r)
    print(f"\n=== {passed}/{len(results)} tests passed ===")
    
    return results

async def test_homepage(browser, base_url):
    await browser.goto(base_url)
    result = await browser.extract("Is the homepage loaded correctly?")
    return result.success

async def test_navigation(browser, base_url):
    await browser.goto(base_url)
    await browser.act("Click on the About link")
    result = await browser.extract("Am I on the About page?")
    return "about" in str(result.data).lower()

async def test_search(browser, base_url):
    await browser.goto(base_url)
    await browser.act("Search for 'test'")
    await asyncio.sleep(2)
    result = await browser.extract("Are search results displayed?")
    return result.success and result.data

# Run for staging environment
asyncio.run(run_test_suite("staging"))
```

## Best Practices

### Test Organization

```python path=null start=null
# Organize tests by feature
tests/
  test_auth.py       # Authentication tests
  test_forms.py      # Form tests
  test_navigation.py # Navigation tests
  test_checkout.py   # E-commerce tests
  conftest.py        # Shared fixtures
```

### Assertion Helpers

```python path=null start=null
async def assert_element_visible(browser, description: str):
    """Assert an element is visible."""
    result = await browser.observe(f"find {description}")
    assert result.success and len(result.data) > 0, \
        f"Element not found: {description}"

async def assert_text_present(browser, text: str):
    """Assert text is present on page."""
    result = await browser.extract(f"Is the text '{text}' visible on the page?")
    assert result.success and "yes" in str(result.data).lower(), \
        f"Text not found: {text}"

async def assert_navigation_to(browser, expected_page: str):
    """Assert navigation to expected page."""
    result = await browser.extract("What page am I on?")
    assert expected_page.lower() in str(result.data).lower(), \
        f"Expected {expected_page}, got {result.data}"
```

## Next Steps

- [Workflow Automation](workflow-automation.md) - Complex automation patterns
- [Web Scraping Examples](web-scraping.md) - Data extraction patterns
- [Error Handling Guide](../guides/error-handling.md) - Robust test error handling
