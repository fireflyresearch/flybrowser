# PII Masking and Secure Credentials

FlyBrowser provides comprehensive Personally Identifiable Information (PII) handling to ensure sensitive data is never exposed to LLM providers, logs, or external systems.

## Overview

The PII system ensures:

- Credentials are NEVER sent to LLM providers
- Sensitive data is encrypted at rest
- Automatic masking in logs and debugging output
- Placeholder-based system for LLM instructions
- Secure form filling without value exposure

## Core Concepts

### The Placeholder System

FlyBrowser uses a placeholder-based approach for handling credentials:

1. User stores credentials with names (e.g., "email", "password")
2. Instructions containing sensitive values are converted to placeholders
3. LLM sees `{{CREDENTIAL:password}}` instead of actual values
4. Just before browser execution, placeholders are resolved to real values
5. The actual value is never exposed to the LLM

```python
# Example flow:
# 1. User instruction: "Login with user@example.com and password secret123"
# 2. For LLM: "Login with {{CREDENTIAL:email}} and password {{CREDENTIAL:password}}"
# 3. LLM plans actions using placeholders
# 4. For browser: placeholders resolved to actual values
```

## Storing Credentials

### Using the SDK

```python
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    # Store a credential
    credential_id = browser.store_credential(
        name="login_password",
        value="my_secret_password",
        pii_type="password"
    )
    
    # Later use for secure form filling
    await browser.secure_fill("#password", credential_id)
```

### Method Signature

```python
def store_credential(
    name: str,
    value: str,
    pii_type: str = "password"
) -> str
```

**Parameters:**

- `name` (str) - Name/identifier for the credential (used in placeholders)
- `value` (str) - The sensitive value to store
- `pii_type` (str) - Type of PII. Options:
  - `"password"` - Passwords and secrets
  - `"username"` - Usernames and login IDs
  - `"email"` - Email addresses
  - `"phone"` - Phone numbers
  - `"ssn"` - Social Security Numbers
  - `"credit_card"` - Credit card numbers
  - `"cvv"` - Card verification values
  - `"address"` - Physical addresses
  - `"name"` - Personal names
  - `"date_of_birth"` - Birth dates
  - `"api_key"` - API keys
  - `"token"` - Authentication tokens
  - `"custom"` - Custom sensitive data

**Returns:**

String credential ID for later retrieval.

## Secure Form Filling

### Using secure_fill()

Fill form fields without exposing credential values:

```python
# Store credentials
email_id = browser.store_credential("email", "user@example.com", "email")
password_id = browser.store_credential("password", "secret123", "password")

# Navigate to login page
await browser.goto("https://example.com/login")

# Securely fill form fields
await browser.secure_fill("#email", email_id)
await browser.secure_fill("#password", password_id)

# Submit the form
await browser.act("click the login button")
```

### Method Signature

```python
async def secure_fill(
    selector: str,
    credential_id: str,
    clear_first: bool = True
) -> bool
```

**Parameters:**

- `selector` (str) - CSS selector for the input field
- `credential_id` (str) - ID of the stored credential
- `clear_first` (bool, default: True) - Clear the field before filling

**Returns:**

Boolean indicating success.

## PII Masking in Text

### Using mask_pii()

Mask sensitive information in text:

```python
text = "My email is user@example.com and SSN is 123-45-6789"
masked = browser.mask_pii(text)
print(masked)
# Output: "My email is ****@****.*** and SSN is ********"
```

### Automatic PII Detection

The masker automatically detects:

- **Email addresses** - `user@example.com` → `****@****.***`
- **Phone numbers** - `(555) 123-4567` → `(***) ***-****`
- **Social Security Numbers** - `123-45-6789` → `********`
- **Credit card numbers** - `4111-1111-1111-1111` → `************1111`
- **API keys** - `sk-abc123...` → `********`
- **JWT tokens** - `eyJ...` → `********`

### Format Preservation

Masking preserves format hints when possible:

```python
# Email format preserved
"user@example.com" → "****@****.***"

# Phone format preserved
"555-123-4567" → "***-***-****"

# Credit card shows last 4 digits
"4111111111111111" → "************1111"
```

## Configuration

### PIIConfig Options

Configure PII handling when creating FlyBrowser:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    pii_masking_enabled=True,  # Enable/disable PII masking globally
)
```

### Advanced Configuration (Embedded Mode)

For embedded mode, you can configure the PIIHandler directly:

```python
from flybrowser.security.pii_handler import PIIHandler, PIIConfig

config = PIIConfig(
    enabled=True,
    mask_in_logs=True,
    mask_in_llm_prompts=True,
    encryption_enabled=True,
    mask_character="*",
    mask_length=8,
    preserve_format=True,
    auto_detect_pii=True,
    use_placeholders_for_llm=True,
    placeholder_prefix="{{CREDENTIAL:",
    placeholder_suffix="}}",
    credential_timeout=0,  # 0 = no timeout
    sensitive_field_patterns=[
        r"password", r"passwd", r"pwd", r"secret", r"token",
        r"api[_-]?key", r"credit[_-]?card", r"cvv", r"ssn",
    ]
)

handler = PIIHandler(config)
```

## Working with LLM Instructions

### Creating Secure Instructions

When you need to create instructions containing sensitive values:

```python
# Direct approach - store and use placeholders manually
email_id = browser.store_credential("email", "user@example.com", "email")
password_id = browser.store_credential("password", "secret123", "password")

# Use in act() - the system handles placeholder resolution
await browser.act("type {{CREDENTIAL:email}} in the email field")
await browser.act("type {{CREDENTIAL:password}} in the password field")
```

### How Placeholder Resolution Works

1. **Storage**: `store_credential("email", "user@example.com")` → returns credential ID
2. **Placeholder**: `{{CREDENTIAL:email}}` is used in LLM instructions
3. **LLM Processing**: LLM sees only the placeholder, never the actual value
4. **Resolution**: Just before browser action, placeholder is resolved
5. **Execution**: Browser fills the actual value

## Security Best Practices

### Never Log Credentials

```python
# Bad - value visible in logs
print(f"Logging in with password: {password}")

# Good - use masked version
print(f"Logging in with credential: {credential_id}")
```

### Use Secure Fill for All Sensitive Fields

```python
# Store all sensitive data as credentials
username_id = browser.store_credential("username", username, "username")
password_id = browser.store_credential("password", password, "password")
otp_id = browser.store_credential("otp", otp_code, "token")

# Use secure_fill instead of act() for credentials
await browser.secure_fill("#username", username_id)
await browser.secure_fill("#password", password_id)
await browser.secure_fill("#otp", otp_id)
```

### Credential Timeout

For extra security, set credential timeout:

```python
from flybrowser.security.pii_handler import PIIConfig

config = PIIConfig(
    credential_timeout=300.0  # Credentials expire after 5 minutes
)
```

### Clean Up Credentials

```python
# Delete specific credential
browser.pii_handler.delete_credential(credential_id)

# Clean up expired credentials
browser.pii_handler.cleanup_expired_credentials()

# Clear all credentials
browser.pii_handler.clear_all_credentials()
```

## PII Types Reference

| Type | Description | Auto-Detection Pattern |
|------|-------------|----------------------|
| `password` | Passwords, secrets | Field names containing "password", "pwd", "secret" |
| `username` | Login usernames | Field names containing "user", "login" |
| `email` | Email addresses | `name@domain.tld` pattern |
| `phone` | Phone numbers | US/international phone formats |
| `ssn` | Social Security Numbers | `XXX-XX-XXXX` pattern |
| `credit_card` | Credit card numbers | 16 digits with optional separators |
| `cvv` | Card verification values | Field names containing "cvv", "cvc" |
| `address` | Physical addresses | Field names containing "address" |
| `name` | Personal names | Field names containing "name" |
| `date_of_birth` | Birth dates | Field names containing "dob", "birth" |
| `api_key` | API keys | `sk-`, `pk-`, `api_` prefixes |
| `token` | Auth tokens | JWT format and similar |

## Complete Example

```python
import asyncio
from flybrowser import FlyBrowser

async def secure_login_example():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        pii_masking_enabled=True,
    ) as browser:
        # Store credentials securely
        email_id = browser.store_credential("email", "user@example.com", "email")
        password_id = browser.store_credential("password", "MySecureP@ss!", "password")
        
        # Navigate to login
        await browser.goto("https://example.com/login")
        
        # Fill form securely - values never exposed to LLM
        await browser.secure_fill("#email", email_id)
        await browser.secure_fill("#password", password_id)
        
        # Submit
        await browser.act("click the Sign In button")
        
        # Verify login success
        result = await browser.extract("Am I logged in? What is the username shown?")
        
        # The extracted result is automatically masked if it contains PII
        print(result)

asyncio.run(secure_login_example())
```

## Related Features

- [Authentication Guide](../guides/authentication.md) - Complete authentication patterns
- [Form Automation Guide](../guides/form-automation.md) - Form handling techniques

## See Also

- [SDK Reference](../reference/sdk.md) - Complete API documentation
