# Copyright 2026 Firefly Software Solutions Inc.
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
Example: Form Validation Testing

Navigates to httpbin.org/forms/post, fills form fields using act(),
verifies form interactions work, and validates submission results.

Prerequisites:
    pip install flybrowser
    export ANTHROPIC_API_KEY="sk-ant-..."

Environment Variables:
    ANTHROPIC_API_KEY          - Required. Your LLM provider API key.
    FLYBROWSER_LLM_PROVIDER    - Optional. Defaults to "anthropic".
    FLYBROWSER_LLM_MODEL       - Optional. Defaults to "claude-sonnet-4-5-20250929".
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from flybrowser import FlyBrowser

FORM_URL = "https://httpbin.org/forms/post"


@dataclass
class FormTestResult:
    """Outcome of a single form-interaction test step."""

    step_name: str
    passed: bool = False
    detail: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FormTestSuite:
    """Collects all form test results."""

    results: list[FormTestResult] = field(default_factory=list)

    def add(self, result: FormTestResult) -> None:
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        msg = result.detail or result.error or ""
        print(f"    [{status}] {result.step_name}: {msg[:100]}")

    def print_summary(self) -> None:
        passed = sum(1 for r in self.results if r.passed)
        print("\n" + "=" * 65)
        print("  FORM VALIDATION TEST RESULTS")
        print("=" * 65)
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            print(f"    [{tag}] {r.step_name}")
            if r.error:
                print(f"           Error: {r.error}")
        print("-" * 65)
        total = len(self.results)
        print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {total - passed}")
        print("=" * 65)


async def test_identify_form_fields(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Extract and verify all visible form fields on the page."""
    result = FormTestResult(step_name="Identify form fields")
    try:
        await browser.goto(FORM_URL)
        extraction = await browser.extract(
            "List all form fields visible on this page, including their labels and types "
            "(text input, textarea, select, radio, checkbox)."
        )
        if extraction.success and extraction.data:
            result.passed = True
            result.detail = str(extraction.data)[:200]
        else:
            result.error = extraction.error or "No form fields detected"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_fill_customer_name(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Fill in the customer name field."""
    result = FormTestResult(step_name="Fill customer name")
    try:
        act_result = await browser.act(
            "Type 'Jane Doe' into the 'Customer name' or 'custname' text input field."
        )
        if act_result.success:
            result.passed = True
            result.detail = "Customer name filled with 'Jane Doe'"
        else:
            result.error = act_result.error or "act() returned success=False"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_fill_telephone(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Fill the telephone field."""
    result = FormTestResult(step_name="Fill telephone")
    try:
        act_result = await browser.act(
            "Type '555-123-4567' into the telephone or 'custtel' field."
        )
        if act_result.success:
            result.passed = True
            result.detail = "Telephone filled with '555-123-4567'"
        else:
            result.error = act_result.error or "act() returned success=False"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_fill_email(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Fill the email field."""
    result = FormTestResult(step_name="Fill email address")
    try:
        act_result = await browser.act(
            "Type 'jane.doe@example.com' into the email or 'custemail' field."
        )
        if act_result.success:
            result.passed = True
            result.detail = "Email filled with 'jane.doe@example.com'"
        else:
            result.error = act_result.error or "act() returned success=False"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_select_pizza_size(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Select a pizza size radio button."""
    result = FormTestResult(step_name="Select pizza size (Large)")
    try:
        act_result = await browser.act(
            "Select the 'Large' pizza size radio button."
        )
        if act_result.success:
            result.passed = True
            result.detail = "Selected Large pizza size"
        else:
            result.error = act_result.error or "Could not select pizza size"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_select_toppings(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Check multiple topping checkboxes."""
    result = FormTestResult(step_name="Select toppings (bacon, cheese)")
    try:
        act_result = await browser.act(
            "Check the 'Bacon' and 'Extra Cheese' topping checkboxes. "
            "If 'Extra Cheese' is not available, check 'Cheese' instead."
        )
        if act_result.success:
            result.passed = True
            result.detail = "Toppings selected"
        else:
            result.error = act_result.error or "Could not select toppings"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_fill_delivery_instructions(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Fill the delivery instructions textarea."""
    result = FormTestResult(step_name="Fill delivery instructions")
    try:
        act_result = await browser.act(
            "Type 'Please ring the doorbell twice. Leave at front door if no answer.' "
            "into the delivery instructions or 'comments' textarea."
        )
        if act_result.success:
            result.passed = True
            result.detail = "Delivery instructions filled"
        else:
            result.error = act_result.error or "Could not fill textarea"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_verify_filled_form(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Use extract() to confirm the form is filled correctly before submission."""
    result = FormTestResult(step_name="Verify form contents before submit")
    try:
        extraction = await browser.extract(
            "Read all current values in the form fields. "
            "What is the customer name, telephone, email, pizza size, "
            "selected toppings, and delivery instructions?"
        )
        if extraction.success and extraction.data:
            data_str = str(extraction.data).lower()
            checks_passed = 0
            for keyword in ["jane", "555", "example.com", "large"]:
                if keyword in data_str:
                    checks_passed += 1
            result.passed = checks_passed >= 2
            result.detail = f"Verified {checks_passed}/4 fields. Data: {str(extraction.data)[:150]}"
        else:
            result.error = extraction.error or "Could not read form contents"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def test_submit_form(browser: FlyBrowser, suite: FormTestSuite) -> None:
    """Submit the form and verify the response page."""
    result = FormTestResult(step_name="Submit form and verify response")
    try:
        act_result = await browser.act(
            "Click the 'Submit Order' or submit button to submit the form."
        )
        if not act_result.success:
            result.error = act_result.error or "Failed to click submit"
            suite.add(result)
            return

        await asyncio.sleep(2)

        # httpbin.org/post returns JSON with the submitted data
        extraction = await browser.extract(
            "What does the response page show? Look for the submitted form data "
            "including custname, custemail, custtel values."
        )
        if extraction.success and extraction.data:
            response_str = str(extraction.data).lower()
            if "jane" in response_str or "custname" in response_str:
                result.passed = True
                result.detail = f"Form submitted. Response: {str(extraction.data)[:150]}"
            else:
                result.error = f"Response did not contain expected data: {str(extraction.data)[:150]}"
        else:
            result.error = extraction.error or "Could not read response page"
    except Exception as exc:
        result.error = str(exc)
    suite.add(result)


async def main() -> None:
    """Run the full form validation test suite against httpbin.org."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser Form Validation Testing")
    print(f"  Target: {FORM_URL}")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 65)

    suite = FormTestSuite()

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Step 1: Identify available fields
        print("\n[Step 1] Identify form fields")
        await test_identify_form_fields(browser, suite)

        # Step 2: Fill each field individually
        print("\n[Step 2] Fill form fields")
        await test_fill_customer_name(browser, suite)
        await test_fill_telephone(browser, suite)
        await test_fill_email(browser, suite)
        await test_select_pizza_size(browser, suite)
        await test_select_toppings(browser, suite)
        await test_fill_delivery_instructions(browser, suite)

        # Step 3: Verify filled form
        print("\n[Step 3] Verify form state")
        await test_verify_filled_form(browser, suite)

        # Step 4: Submit and validate response
        print("\n[Step 4] Submit form")
        await test_submit_form(browser, suite)

        # Print usage
        usage = browser.get_usage_summary()
        print(f"\n  LLM Usage: {usage}")

    suite.print_summary()

    if suite.results and all(not r.passed for r in suite.results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
