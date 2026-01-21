#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
PII-Safe Automation Example

This example demonstrates FlyBrowser's PII protection features:
- Secure credential storage
- Placeholder-based masking for LLM safety
- Secure form filling
- PII masking in logs and prompts

IMPORTANT: Credentials are NEVER sent to LLM providers!

Run with: python examples/06_pii_safe_automation.py
"""

import asyncio
import os

from flybrowser import FlyBrowser


async def main():
    """Demonstrate PII-safe automation."""
    print("=" * 60)
    print("FlyBrowser PII-Safe Automation Example")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        pii_masking_enabled=True,  # Enable PII masking (default)
    ) as browser:
        # Store credentials securely
        print("\n[1] Storing credentials securely...")
        email_id = browser.store_credential("email", "user@example.com", pii_type="email")
        password_id = browser.store_credential("password", "SuperSecret123!", pii_type="password")
        print(f"    Email credential ID: {email_id}")
        print(f"    Password credential ID: {password_id}")

        # Demonstrate PII masking
        print("\n[2] Demonstrating PII masking...")
        original_text = "Login with user@example.com and password SuperSecret123!"
        masked_text = browser.mask_pii(original_text)
        print(f"    Original: {original_text}")
        print(f"    Masked:   {masked_text}")
        print("    (Notice: credentials are replaced with placeholders)")

        # Navigate to a login page (demo)
        print("\n[3] Navigating to a demo page...")
        await browser.goto("https://www.python.org")

        # In a real scenario, you would use secure_fill like this:
        print("\n[4] Secure form filling demonstration...")
        print("    In a real login scenario, you would use:")
        print("    await browser.secure_fill('#email-input', email_id)")
        print("    await browser.secure_fill('#password-input', password_id)")
        print("")
        print("    The LLM sees: 'Fill {{CREDENTIAL:email}} into email field'")
        print("    The browser fills: 'user@example.com'")
        print("    The LLM NEVER sees the actual credential value!")

        # Demonstrate how actions work with PII
        print("\n[5] PII-safe action execution...")
        print("    When you run: await browser.act('type my email into the form')")
        print("    1. Your instruction is masked before sending to LLM")
        print("    2. LLM plans actions using placeholders")
        print("    3. Placeholders are resolved just before browser execution")
        print("    4. Actual values never appear in logs or LLM prompts")

    print("\n" + "=" * 60)
    print("PII-Safe Automation example completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Store credentials with store_credential()")
    print("- Use secure_fill() for form fields")
    print("- LLM sees {{CREDENTIAL:name}} placeholders, not real values")
    print("- All logs and prompts are automatically masked")


if __name__ == "__main__":
    asyncio.run(main())

