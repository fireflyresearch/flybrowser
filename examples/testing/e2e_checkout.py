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
Example: End-to-End Checkout Testing

Performs a complete e-commerce user journey on books.toscrape.com:
browse the catalogue, select a product, add it to the basket, and
verify basket contents. All interactions use the FlyBrowser SDK
(act, extract, observe) with AgentRequestResponse objects.

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

STORE_URL = "https://books.toscrape.com"


@dataclass
class CheckoutStep:
    """Result of one step in the checkout flow."""

    name: str
    passed: bool = False
    detail: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CheckoutReport:
    """Aggregated checkout test report."""

    steps: list[CheckoutStep] = field(default_factory=list)

    def record(self, step: CheckoutStep) -> None:
        self.steps.append(step)
        tag = "PASS" if step.passed else "FAIL"
        msg = step.detail or step.error or ""
        print(f"    [{tag}] {step.name}: {msg[:120]}")

    def print_summary(self) -> None:
        passed = sum(1 for s in self.steps if s.passed)
        total = len(self.steps)
        print("\n" + "=" * 65)
        print("  E2E CHECKOUT TEST REPORT")
        print("=" * 65)
        for s in self.steps:
            tag = "PASS" if s.passed else "FAIL"
            print(f"    [{tag}] {s.name}")
            if s.detail:
                print(f"           {s.detail[:120]}")
            if s.error:
                print(f"           Error: {s.error[:120]}")
        print("-" * 65)
        print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {total - passed}")
        verdict = "ALL STEPS PASSED" if passed == total else f"{total - passed} STEP(S) FAILED"
        print(f"  Verdict: {verdict}")
        print("=" * 65)


async def step_load_homepage(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Navigate to the bookstore homepage and verify it loaded."""
    step = CheckoutStep(name="Load homepage")
    try:
        await browser.goto(STORE_URL)
        result = await browser.extract(
            "What is the site name or main heading? Is this a bookstore?"
        )
        if result.success and result.data:
            data_str = str(result.data).lower()
            if "book" in data_str or "scrape" in data_str:
                step.passed = True
                step.detail = str(result.data)[:150]
            else:
                step.error = f"Unexpected content: {result.data}"
        else:
            step.error = result.error or "extract() returned no data"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_browse_catalogue(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Browse the catalogue and verify book listings are visible."""
    step = CheckoutStep(name="Browse catalogue")
    try:
        result = await browser.extract(
            "List the first 3 book titles visible on this page with their prices."
        )
        if result.success and result.data:
            step.passed = True
            step.detail = str(result.data)[:200]
        else:
            step.error = result.error or "Could not extract book listings"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_observe_categories(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Use observe() to find category navigation links."""
    step = CheckoutStep(name="Observe category sidebar")
    try:
        result = await browser.observe(
            "What book categories or genres are listed in the left sidebar?"
        )
        if result.success and result.data:
            step.passed = True
            step.detail = str(result.data)[:200]
        else:
            step.error = result.error or "observe() returned no data"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_select_book(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Click on a specific book to view its detail page."""
    step = CheckoutStep(name="Select a book")
    try:
        act_result = await browser.act(
            "Click on the first book title or image link to open its product page."
        )
        if not act_result.success:
            step.error = act_result.error or "Could not click on a book"
            report.record(step)
            return False

        await asyncio.sleep(1)

        detail = await browser.extract(
            "What is the book title, price, availability, and description on this page?"
        )
        if detail.success and detail.data:
            step.passed = True
            step.detail = str(detail.data)[:200]
        else:
            step.error = detail.error or "Could not read product detail page"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_add_to_basket(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Add the currently viewed book to the basket."""
    step = CheckoutStep(name="Add to basket")
    try:
        act_result = await browser.act(
            "Click the 'Add to basket' button on the product page."
        )
        if not act_result.success:
            step.error = act_result.error or "Could not click 'Add to basket'"
            report.record(step)
            return False

        await asyncio.sleep(1)

        # Verify the basket was updated (banner or basket count)
        check = await browser.extract(
            "Is there a success message or a basket indicator showing items were added? "
            "What does the basket count show?"
        )
        if check.success and check.data:
            step.passed = True
            step.detail = str(check.data)[:150]
        else:
            step.error = check.error or "Could not confirm basket update"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_add_second_book(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Go back to catalogue and add a second book to the basket."""
    step = CheckoutStep(name="Add second book")
    try:
        # Navigate back to catalogue
        await browser.goto(STORE_URL)
        await asyncio.sleep(1)

        # Click on a different book (second one)
        act_result = await browser.act(
            "Click on the second book title or image link to view its product page."
        )
        if not act_result.success:
            step.error = act_result.error or "Could not select second book"
            report.record(step)
            return False

        await asyncio.sleep(1)

        # Add to basket
        add_result = await browser.act("Click the 'Add to basket' button.")
        if add_result.success:
            step.passed = True
            step.detail = "Second book added to basket"
        else:
            step.error = add_result.error or "Could not add second book"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_view_basket(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Navigate to the basket page and verify its contents."""
    step = CheckoutStep(name="View basket contents")
    try:
        act_result = await browser.act(
            "Click on the basket link or 'View basket' button to see the basket page."
        )
        if not act_result.success:
            step.error = act_result.error or "Could not navigate to basket"
            report.record(step)
            return False

        await asyncio.sleep(1)

        basket = await browser.extract(
            "What items are in the basket? List each book title, quantity, and price. "
            "Also provide the total price if visible."
        )
        if basket.success and basket.data:
            step.passed = True
            step.detail = str(basket.data)[:200]
            # Pretty-print the result
            print("\n    --- Basket Contents (pprint) ---")
            basket.pprint()
            print("    --- End pprint ---")
        else:
            step.error = basket.error or "Could not read basket contents"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def step_verify_basket_total(browser: FlyBrowser, report: CheckoutReport) -> bool:
    """Verify the basket shows a reasonable total."""
    step = CheckoutStep(name="Verify basket total")
    try:
        result = await browser.extract(
            "What is the basket total or order total displayed on this page? "
            "Return just the numeric total with currency symbol."
        )
        if result.success and result.data:
            total_str = str(result.data)
            # Check that some currency indicator is present
            if any(c in total_str for c in ["$", "£", "EUR", "."]):
                step.passed = True
                step.detail = f"Basket total: {total_str}"
            else:
                step.error = f"No currency found in total: {total_str}"
        else:
            step.error = result.error or "Could not extract basket total"
    except Exception as exc:
        step.error = str(exc)
    report.record(step)
    return step.passed


async def main() -> None:
    """Run the full end-to-end checkout test on books.toscrape.com."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 65)
    print("  FlyBrowser E2E Checkout Test")
    print(f"  Store: {STORE_URL}")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 65)

    report = CheckoutReport()

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        # Run each step; stop early only on critical failures
        print("\n[Phase 1] Load and Browse")
        if not await step_load_homepage(browser, report):
            print("  CRITICAL: Homepage failed to load. Aborting.")
            report.print_summary()
            sys.exit(1)

        await step_browse_catalogue(browser, report)
        await step_observe_categories(browser, report)

        print("\n[Phase 2] Product Selection")
        await step_select_book(browser, report)
        await step_add_to_basket(browser, report)

        print("\n[Phase 3] Add Second Book")
        await step_add_second_book(browser, report)

        print("\n[Phase 4] Basket Verification")
        await step_view_basket(browser, report)
        await step_verify_basket_total(browser, report)

        # Usage summary
        usage = browser.get_usage_summary()
        print(f"\n  LLM Usage: {usage}")

    report.print_summary()

    if report.steps and any(not s.passed for s in report.steps):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
