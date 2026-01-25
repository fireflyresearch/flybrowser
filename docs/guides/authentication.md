# Authentication Guide

This guide covers handling authentication and sessions in FlyBrowser. You will learn how to log into websites, manage sessions, handle different authentication methods, and secure credential handling.

## Basic Login

### Simple Login Flow

```python
import asyncio
from flybrowser import FlyBrowser

async def login_to_site():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com/login")
        
        # Fill credentials
        await browser.act("type 'myusername' in the username field")
        await browser.act("type 'mypassword' in the password field")
        
        # Submit
        await browser.act("click the Login button")
        
        # Verify login succeeded
        await asyncio.sleep(2)
        result = await browser.extract("am I logged in? Look for username or logout link")
        
        if result.data:
            print("Login successful!")
        else:
            print("Login may have failed")

asyncio.run(login_to_site())
```

### Using Environment Variables

Never hardcode credentials. Use environment variables:

```python
import os

async def secure_login(browser):
    username = os.environ.get("SITE_USERNAME")
    password = os.environ.get("SITE_PASSWORD")
    
    if not username or not password:
        raise ValueError("Missing credentials in environment variables")
    
    await browser.goto("https://example.com/login")
    await browser.act(f"type '{username}' in the username field")
    await browser.act(f"type '{password}' in the password field")
    await browser.act("click Login")
```

## Secure Credential Handling

### Using store_credential

FlyBrowser provides secure credential storage that prevents sensitive data from appearing in logs:

```python
async with FlyBrowser(...) as browser:
    # Store credential securely (never logged)
    await browser.store_credential(
        name="bank_password",
        value="my-secret-password",
        pii_type="password"
    )
    
    await browser.goto("https://bank.example.com/login")
    
    # Type username normally
    await browser.act("type 'myusername' in the username field")
    
    # Use secure_fill for password (uses stored credential)
    await browser.secure_fill("#password", "bank_password")
    
    await browser.act("click Sign In")
```

### PII Types

When storing credentials, specify the PII type for proper masking:

```python
# Password
await browser.store_credential("user_pass", password, pii_type="password")

# Credit card
await browser.store_credential("card_num", card_number, pii_type="credit_card")

# Social Security Number
await browser.store_credential("ssn", ssn_value, pii_type="ssn")

# Generic sensitive data
await browser.store_credential("api_key", key_value, pii_type="sensitive")
```

## Session Management

### Maintaining Session Across Operations

Once logged in, the browser session persists:

```python
async def work_with_session(browser):
    # Login
    await browser.goto("https://example.com/login")
    await browser.act("type credentials and login...")
    
    # Session is maintained for subsequent operations
    await browser.goto("https://example.com/dashboard")
    dashboard_data = await browser.extract("get dashboard info")
    
    await browser.goto("https://example.com/profile")
    profile_data = await browser.extract("get profile info")
    
    await browser.goto("https://example.com/settings")
    settings_data = await browser.extract("get settings")
    
    # All these work because we're in the same session
    return {
        "dashboard": dashboard_data.data,
        "profile": profile_data.data,
        "settings": settings_data.data
    }
```

### Checking Session Status

Verify you're still logged in before critical operations:

```python
async def verify_session(browser):
    """Check if session is still valid."""
    result = await browser.extract(
        "am I logged in? Check for logout button, username display, or login prompt"
    )
    
    # Parse the response
    response_lower = str(result.data).lower()
    
    if any(word in response_lower for word in ["logged in", "logout", "sign out"]):
        return True
    elif any(word in response_lower for word in ["login", "sign in", "not logged"]):
        return False
    
    return None  # Uncertain

async def ensure_logged_in(browser, login_func):
    """Ensure user is logged in, re-authenticate if needed."""
    if not await verify_session(browser):
        print("Session expired, re-authenticating...")
        await login_func(browser)
```

## Two-Factor Authentication

### TOTP (Authenticator Apps)

For time-based OTP codes:

```python
import pyotp

async def login_with_2fa(browser, username, password, totp_secret):
    await browser.goto("https://example.com/login")
    
    # First factor
    await browser.act(f"type '{username}' in the username field")
    await browser.act(f"type '{password}' in the password field")
    await browser.act("click Login")
    
    await asyncio.sleep(2)
    
    # Check if 2FA is required
    needs_2fa = await browser.extract("is there a 2FA or verification code prompt?")
    
    if needs_2fa.data:
        # Generate TOTP code
        totp = pyotp.TOTP(totp_secret)
        code = totp.now()
        
        await browser.act(f"type '{code}' in the verification code field")
        await browser.act("click Verify or Submit")
```

### SMS/Email Codes

For codes sent via SMS or email, you may need manual intervention:

```python
async def login_with_sms_2fa(browser, username, password):
    await browser.goto("https://example.com/login")
    
    await browser.act(f"type '{username}' in the username field")
    await browser.act(f"type '{password}' in the password field")
    await browser.act("click Login")
    
    await asyncio.sleep(2)
    
    # Check if code was sent
    result = await browser.extract("was a verification code sent? To phone or email?")
    
    if result.data:
        # In automated scenarios, you'd need another way to retrieve the code
        # For interactive use, prompt the user
        code = input("Enter the verification code sent to your device: ")
        
        await browser.act(f"type '{code}' in the verification code field")
        await browser.act("click Verify")
```

## OAuth and Social Login

### Google Login

```python
async def google_oauth_login(browser, google_email, google_password):
    await browser.goto("https://example.com/login")
    
    # Click the Google login button
    await browser.act("click 'Sign in with Google' or 'Continue with Google'")
    
    await asyncio.sleep(2)
    
    # Google's login page
    await browser.act(f"type '{google_email}' in the email field")
    await browser.act("click Next")
    
    await asyncio.sleep(2)
    
    await browser.act(f"type '{google_password}' in the password field")
    await browser.act("click Next")
    
    await asyncio.sleep(3)
    
    # May need to approve permissions
    result = await browser.extract("is there a permissions approval screen?")
    if result.data:
        await browser.act("click Allow or Approve")
```

### Handling OAuth Popups

Some OAuth flows open popup windows:

```python
async def handle_oauth_popup(browser):
    await browser.goto("https://example.com/login")
    
    # This might open a popup
    await browser.act("click 'Sign in with GitHub'")
    
    # The agent handles the popup/redirect automatically
    await asyncio.sleep(3)
    
    # Check if we ended up logged in
    result = await browser.extract("am I now logged into the main site?")
    return result.data
```

## Handling Login Challenges

### Captchas

FlyBrowser can detect captchas but cannot solve them automatically:

```python
async def login_with_captcha_detection(browser, username, password):
    await browser.goto("https://example.com/login")
    
    await browser.act(f"type '{username}' in the username field")
    await browser.act(f"type '{password}' in the password field")
    
    # Check for captcha before submitting
    captcha = await browser.extract(
        "is there a captcha on this page? Look for reCAPTCHA, hCaptcha, or similar"
    )
    
    if captcha.data and "yes" in str(captcha.data).lower():
        print("CAPTCHA detected!")
        print("Please solve the captcha manually or use a captcha service")
        # You'd need manual intervention or a third-party service here
        input("Press Enter after solving the captcha...")
    
    await browser.act("click Login")
```

### Rate Limiting

Handle rate limiting gracefully:

```python
async def login_with_rate_limit_handling(browser, username, password, max_retries=3):
    for attempt in range(max_retries):
        await browser.goto("https://example.com/login")
        
        await browser.act(f"type '{username}' in username")
        await browser.act(f"type '{password}' in password")
        await browser.act("click Login")
        
        await asyncio.sleep(2)
        
        # Check for rate limit message
        result = await browser.extract(
            "is there an error message about too many attempts or rate limiting?"
        )
        
        if result.data and "rate" in str(result.data).lower():
            wait_time = 60 * (attempt + 1)  # Increasing backoff
            print(f"Rate limited, waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            continue
        
        # Check if logged in
        logged_in = await browser.extract("am I logged in?")
        if logged_in.data:
            return True
    
    return False
```

### Account Lockouts

```python
async def safe_login_attempt(browser, username, password):
    """Login with lockout awareness."""
    await browser.goto("https://example.com/login")
    
    # Check if account is locked before attempting
    await browser.act(f"type '{username}' in the username field")
    
    # Some sites show lockout status after entering username
    lockout_check = await browser.extract(
        "is there a message about account being locked or disabled?"
    )
    
    if lockout_check.data and "lock" in str(lockout_check.data).lower():
        return {"success": False, "reason": "Account locked"}
    
    await browser.act(f"type '{password}' in the password field")
    await browser.act("click Login")
    
    await asyncio.sleep(2)
    
    # Check result
    result = await browser.extract(
        "what is the login result? Success, wrong password, or locked?"
    )
    
    return result.data
```

## Password Management

### Password Reset Flows

```python
async def initiate_password_reset(browser, email):
    await browser.goto("https://example.com/login")
    
    # Find and click forgot password
    await browser.act("click 'Forgot Password' or 'Reset Password'")
    
    await asyncio.sleep(1)
    
    # Enter email
    await browser.act(f"type '{email}' in the email field")
    await browser.act("click Send or Submit")
    
    # Verify email was sent
    result = await browser.extract("was a password reset email sent?")
    return result.data
```

### Remember Me / Stay Signed In

```python
async def login_with_remember(browser, username, password, remember=True):
    await browser.goto("https://example.com/login")
    
    await browser.act(f"type '{username}' in the username field")
    await browser.act(f"type '{password}' in the password field")
    
    if remember:
        await browser.act("check the 'Remember me' or 'Stay signed in' checkbox")
    
    await browser.act("click Login")
```

## Enterprise Authentication

### SAML/SSO

```python
async def enterprise_sso_login(browser, corporate_email, password):
    """Login via corporate SSO."""
    await browser.goto("https://app.example.com/login")
    
    # Enter corporate email to trigger SSO redirect
    await browser.act(f"type '{corporate_email}' in the email field")
    await browser.act("click Continue or Next")
    
    await asyncio.sleep(3)
    
    # Should redirect to corporate identity provider
    current_url = await browser.extract("what is the current page URL or title?")
    
    if "okta" in str(current_url.data).lower() or "azure" in str(current_url.data).lower():
        # Corporate IdP page
        await browser.act(f"type '{password}' in the password field")
        await browser.act("click Sign In")
        
        # May need MFA
        await asyncio.sleep(2)
        # Handle MFA if needed...
```

## Best Practices

### Credential Security Checklist

```python
import os

def get_credentials():
    """Securely retrieve credentials."""
    username = os.environ.get("APP_USERNAME")
    password = os.environ.get("APP_PASSWORD")
    
    if not username or not password:
        raise ValueError(
            "Credentials not found. Set APP_USERNAME and APP_PASSWORD "
            "environment variables."
        )
    
    return username, password

async def secure_workflow():
    username, password = get_credentials()
    
    async with FlyBrowser(pii_masking_enabled=True, ...) as browser:
        await browser.store_credential("pass", password, pii_type="password")
        
        await browser.goto("https://example.com/login")
        await browser.act(f"type '{username}' in the username field")
        await browser.secure_fill("#password", "pass")
        await browser.act("click Login")
```

### Session Timeout Handling

```python
async def session_aware_operation(browser, operation_func):
    """Wrapper that handles session timeouts."""
    try:
        return await operation_func(browser)
    except Exception as e:
        if "session" in str(e).lower() or "unauthorized" in str(e).lower():
            print("Session expired, re-authenticating...")
            await perform_login(browser)
            return await operation_func(browser)
        raise
```

### Logout Properly

```python
async def proper_logout(browser):
    """Properly log out to clean up session."""
    try:
        await browser.act("click the user menu or profile icon")
        await asyncio.sleep(0.5)
        await browser.act("click Logout or Sign Out")
        
        # Verify logged out
        await asyncio.sleep(1)
        result = await browser.extract("am I logged out? Is there a login button?")
        return result.data
    except Exception as e:
        print(f"Logout error (non-critical): {e}")
```

## Next Steps

- [Error Handling Guide](error-handling.md) - Handle authentication errors
- [Form Automation Guide](form-automation.md) - Complex form patterns
- [SDK Reference](../reference/sdk.md) - Security features documentation
