# Security Architecture

FlyBrowser provides role-based access control (RBAC) with JWT authentication for the API server. The security layer wraps fireflyframework-genai's `RBACManager` and maintains backward compatibility with API key authentication.

## Overview

The authentication system supports two modes:

1. **JWT tokens** (recommended) -- Role-based tokens with configurable expiry
2. **API keys** (legacy) -- Simple key validation for backward compatibility

Both modes can be used simultaneously. JWT tokens are checked first; if no Bearer token is present, the system falls back to API key validation.

## Roles and Permissions

FlyBrowser defines three roles with progressively broader permissions:

### admin

Full access to all operations.

| Permission | Description |
|-----------|-------------|
| `*` | Wildcard -- all current and future permissions |

### operator

Can create and use sessions, perform automations, and view recordings.

| Permission | Description |
|-----------|-------------|
| `sessions.create` | Create new browser sessions |
| `sessions.delete` | Close/delete sessions |
| `sessions.list` | List active sessions |
| `sessions.get` | Get session details |
| `sessions.navigate` | Navigate within a session |
| `sessions.extract` | Extract data from a session |
| `sessions.act` | Perform actions in a session |
| `sessions.agent` | Run agent tasks |
| `sessions.screenshot` | Take screenshots |
| `sessions.observe` | Observe page elements |
| `sessions.stream` | Start/stop live streams |
| `recordings.list` | List recordings |
| `recordings.download` | Download recordings |

### viewer

Read-only access to sessions and recordings.

| Permission | Description |
|-----------|-------------|
| `sessions.list` | List active sessions |
| `sessions.get` | Get session details |
| `recordings.list` | List recordings |
| `recordings.download` | Download recordings |

## JWT Authentication

### Creating Tokens

Use the `RBACAuthManager` to create signed JWT tokens:

```python
from flybrowser.service.auth import RBACAuthManager

mgr = RBACAuthManager(jwt_secret="your-production-secret")

# Create an operator token
token = mgr.create_token(
    user_id="alice",
    roles=["operator"],
)

# Create an admin token with custom claims
admin_token = mgr.create_token(
    user_id="admin",
    roles=["admin"],
    custom_claims={"department": "engineering"},
)
```

### Token Structure

JWT tokens contain the following claims:

```json
{
  "user_id": "alice",
  "roles": ["operator"],
  "iat": 1707811200,
  "exp": 1707897600,
  "department": "engineering"
}
```

| Claim | Description |
|-------|-------------|
| `user_id` | Unique user identifier |
| `roles` | List of assigned roles |
| `iat` | Issued-at timestamp |
| `exp` | Expiration timestamp (default: 24 hours) |
| (custom) | Any additional claims passed at creation |

### Using Tokens in API Requests

Pass the JWT token in the `Authorization` header:

```bash
# Using Bearer token
curl -X POST http://localhost:8000/sessions \
  -H "Authorization: Bearer eyJhbGci..." \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "openai"}'
```

### Validating Tokens

```python
# Validate a token and get claims
claims = mgr.validate_token(token)
if claims:
    print(f"User: {claims['user_id']}")
    print(f"Roles: {claims['roles']}")
else:
    print("Invalid or expired token")
```

Token validation checks:
- Signature integrity (using the JWT secret)
- Expiration time
- Required claims presence

### Checking Permissions

```python
# Check if a role has a specific permission
mgr.has_permission("operator", "sessions.create")  # True
mgr.has_permission("viewer", "sessions.create")     # False
mgr.has_permission("admin", "anything.at.all")      # True (wildcard)
```

## API Key Authentication (Backward Compatibility)

For services that already use API key authentication, FlyBrowser supports the `X-API-Key` header:

```bash
curl -X GET http://localhost:8000/sessions \
  -H "X-API-Key: flybrowser_dev_abc123..."
```

### API Key Management

```python
from flybrowser.service.auth import APIKeyManager

key_mgr = APIKeyManager()

# Create a key with expiration and rate limit
key = key_mgr.create_key(
    name="CI Pipeline",
    expires_in_days=90,
    rate_limit=1000,
)
print(f"Key: {key.key}")

# Validate a key
validated = key_mgr.validate_key("flybrowser_...")
if validated:
    print(f"Key name: {validated.name}")

# Revoke a key
key_mgr.revoke_key("flybrowser_...")

# List all keys
for k in key_mgr.list_keys():
    print(f"  {k.name}: {'active' if k.enabled else 'revoked'}")
```

### API Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `key` | str | The key string (prefix: `flybrowser_`) |
| `name` | str | Human-readable name |
| `created_at` | datetime | Creation timestamp |
| `expires_at` | datetime or None | Expiration timestamp |
| `rate_limit` | int or None | Requests per hour limit |
| `enabled` | bool | Whether the key is active |

## Configuration

### Setup Wizard

The easiest way to configure security is through the setup wizard:

```bash
flybrowser setup security
```

This generates a JWT secret and an initial admin token. See [Setup Wizard](../getting-started/setup-wizard.md) for details.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FLYBROWSER_JWT_SECRET` | JWT signing secret (required for production) |
| `FLYBROWSER_RBAC_ENABLED` | Enable/disable RBAC (`true`/`false`) |
| `FLYBROWSER_ADMIN_TOKEN` | Pre-configured admin token |

### Programmatic Configuration

```python
from flybrowser.service.auth import RBACAuthManager

mgr = RBACAuthManager(
    jwt_secret="your-production-secret",  # Required in production
    token_expiry_hours=24,                 # Token lifetime
)
```

If no `jwt_secret` is provided, a random secret is generated at startup. This is suitable for development but means tokens will be invalidated on every restart.

## Authentication Flow

```
Client Request
  |
  +-- Has "Authorization: Bearer <token>" header?
  |     |
  |     YES --> RBACAuthManager.validate_token(token)
  |              |
  |              +-- Valid? --> Extract roles, check permissions --> Allow/Deny
  |              +-- Invalid? --> 401 Unauthorized
  |
  +-- Has "X-API-Key: <key>" header?
  |     |
  |     YES --> APIKeyManager.validate_key(key)
  |              |
  |              +-- Valid? --> Allow (all permissions)
  |              +-- Invalid? --> 401 Unauthorized
  |
  +-- No auth headers?
        |
        --> 401 Unauthorized
```

## Best Practices

### 1. Use a Strong JWT Secret in Production

```bash
# Generate a strong secret
python -c "import secrets; print(secrets.token_urlsafe(64))"

# Set in environment
export FLYBROWSER_JWT_SECRET="your-generated-secret"
```

### 2. Use Short-Lived Tokens

Create tokens with appropriate lifetimes:

```python
# Short-lived token for a CI pipeline
mgr = RBACAuthManager(jwt_secret="...", token_expiry_hours=1)
token = mgr.create_token(user_id="ci-bot", roles=["operator"])
```

### 3. Apply Least Privilege

Assign the most restrictive role that meets the user's needs:

- **Monitoring dashboards** -- `viewer` role
- **Automation pipelines** -- `operator` role
- **Administration** -- `admin` role

### 4. Rotate Secrets Periodically

Change the JWT secret periodically. Existing tokens will be invalidated, so coordinate with token holders.

### 5. Secure API Keys

- Never commit API keys to version control
- Use environment variables or secret managers
- Set expiration dates on all keys
- Revoke keys immediately when compromised

## See Also

- [Setup Wizard](../getting-started/setup-wizard.md) -- Interactive security configuration
- [Framework Integration](framework-integration.md) -- How RBAC integrates with the framework
- [REST API Reference](../reference/rest-api.md) -- API authentication requirements
- [Configuration Reference](../reference/configuration.md) -- All security-related settings
