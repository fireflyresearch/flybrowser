# Validation System

FlyBrowser implements validation for security, data integrity, and reliable operation. This document explains the validation architecture.

## Overview

The validation system provides:

- Browser scope validation for security
- URL safety validation
- Tool parameter validation
- Pydantic-based configuration validation

## Browser Scope Validator

The `BrowserScopeValidator` ensures tasks stay within browser automation scope:

```python
from flybrowser.agents.scope_validator import BrowserScopeValidator

validator = BrowserScopeValidator()

# Valid browser task
is_valid, error = validator.validate_task("Navigate to example.com and click login")
# is_valid = True

# Invalid - filesystem operation
is_valid, error = validator.validate_task("Delete all files in /tmp")
# is_valid = False

# URL validation
is_valid, error = validator.validate_url("https://example.com")
# is_valid = True

is_valid, error = validator.validate_url("file:///etc/passwd")
# is_valid = False
```

### Prohibited Operations

- Filesystem operations (delete, read, write files)
- System operations (execute command, run script, sudo)
- Network operations outside browser (curl, wget, ssh)
- Database operations (SQL queries)
- Security-sensitive actions (code injection, exploits)

### URL Validation

Only `http` and `https` schemes are allowed. Prohibited schemes include `file`, `javascript`, `data`, `ftp`, and others.

## Configuration Validation

FlyBrowser uses Pydantic for configuration validation:

```python
from pydantic import BaseModel, Field

class BrowserAgentConfig:
    model: str = "openai:gpt-4o"
    max_iterations: int = 50
    max_time: int = 1800
    budget_limit_usd: float = 5.0
```

## Validation Flow

```
User Request
     |
     v
Browser Scope Validation  (Is task safe for browser automation?)
     |
     v
URL Validation             (Are URLs safe?)
     |
     v
Configuration Validation   (Are parameters valid?)
     |
     v
Tool Execution             (FireflyAgent handles tool parameter validation)
     |
     v
Result
```

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Tools System](tools.md) - Tool parameter validation
- [Error Handling Guide](../guides/error-handling.md) - Handling validation errors
