# Validation System

FlyBrowser implements comprehensive validation for security, data integrity, and reliable LLM responses. This document explains the validation architecture.

## Overview

The validation system provides:

- Browser scope validation for security
- URL safety validation
- JSON schema validation for structured outputs
- Automatic repair for malformed LLM responses
- Tool parameter validation
- Pydantic-based configuration validation

## Validation Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Validation System                           │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Browser Scope │  │  Structured LLM │  │      Tool       │  │
│  │    Validator    │  │    Wrapper      │  │   Validation    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ URL Validation  │  │ JSON Schema     │  │  Parameter      │  │
│  │ Task Validation │  │ Validation      │  │  Validation     │  │
│  │ Security Checks │  │ Auto Repair     │  │  Type Checking  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Browser Scope Validator

The `BrowserScopeValidator` ensures tasks stay within browser automation scope:

```python
class BrowserScopeValidator:
    """Validates that tasks are appropriate for browser automation."""
    
    def validate_task(
        self,
        task: str,
        skip_browser_keyword_check: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a task is appropriate for browser automation.
        
        Args:
            task: The task description to validate
            skip_browser_keyword_check: Skip browser keyword check for SDK methods
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """Validate that a URL is safe for browser automation."""
        ...
```

### Prohibited Operations

The validator blocks operations outside browser scope:

**Filesystem Operations:**
- delete file, read file, write to file
- /etc/, /tmp/, ~/.ssh
- c:\, d:\

**System Operations:**
- execute command, run script
- bash, powershell, cmd.exe
- kill process, sudo, chmod

**Network Operations (Outside Browser):**
- send email, smtp
- curl, wget, ftp
- ssh connection

**Database Operations:**
- sql query, insert into, delete from
- mysql, postgres, mongodb

**Security-Sensitive:**
- steal data, inject code
- sql injection, exploit

### Browser Keywords

Valid browser automation tasks include:

**Navigation:**
- navigate, visit, go to, open, browse

**Interaction:**
- click, type, fill, submit, select
- hover, scroll, drag, drop

**Extraction:**
- extract, scrape, get data, find
- read text, parse, collect

**Web Elements:**
- button, link, form, input
- menu, dropdown, checkbox

### URL Validation

```python
# Allowed schemes
ALLOWED_URL_SCHEMES = ["http", "https"]

# Prohibited schemes
PROHIBITED_URL_SCHEMES = [
    "file",        # Local file access
    "ftp",         # FTP protocol
    "javascript",  # Script injection
    "data",        # Data URLs
    "vbscript",    # VBScript
    "about",       # Browser internal
    "blob",        # Blob URLs
    "filesystem",  # Filesystem API
]
```

### Usage Examples

```python
from flybrowser.agents.scope_validator import BrowserScopeValidator

validator = BrowserScopeValidator()

# Valid browser task
is_valid, error = validator.validate_task("Navigate to example.com and click login")
# is_valid = True

# Invalid - filesystem operation
is_valid, error = validator.validate_task("Delete all files in /tmp")
# is_valid = False
# error = "Task contains prohibited operation: 'delete file'..."

# SDK method call - skip browser keyword check
is_valid, error = validator.validate_task(
    "Get the titles and scores",
    skip_browser_keyword_check=True,
)
# is_valid = True (security checks still apply)

# URL validation
is_valid, error = validator.validate_url("https://example.com")
# is_valid = True

is_valid, error = validator.validate_url("file:///etc/passwd")
# is_valid = False
```

## Structured LLM Wrapper

The `StructuredLLMWrapper` ensures reliable JSON responses from LLMs:

```python
class StructuredLLMWrapper:
    """Wrapper for reliable structured LLM responses."""
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        max_repair_attempts: int = 2,
        repair_temperature: float = 0.1,
    ):
        """
        Initialize the wrapper.
        
        Args:
            llm_provider: LLM provider to wrap
            max_repair_attempts: Maximum repair attempts
            repair_temperature: Temperature for repairs (low = deterministic)
        """
        ...
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        custom_validator: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON response with validation and repair."""
        ...
    
    async def generate_structured_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, List[bytes]],
        schema: Dict[str, Any],
        ...
    ) -> Dict[str, Any]:
        """Generate structured JSON with vision and validation."""
        ...
```

### JSON Schema Validation

The wrapper validates responses against JSON schemas:

```python
def validate_json_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    path: str = "",
) -> tuple[bool, List[str]]:
    """
    Validate data against a JSON schema.
    
    Checks:
    - Required fields
    - Types (string, number, integer, boolean, array, object, null)
    - Nested objects
    - Array items
    - oneOf schemas
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
```

Example schema validation:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
    },
    "required": ["name"],
}

data = {"name": "John", "age": 30, "tags": ["developer"]}
is_valid, errors = validate_json_schema(data, schema)
# is_valid = True, errors = []

data = {"age": "thirty"}  # Missing required, wrong type
is_valid, errors = validate_json_schema(data, schema)
# is_valid = False
# errors = ["Missing required field 'name'", "'age' must be integer, got str"]
```

### Automatic Repair

When validation fails, the wrapper attempts to repair the response:

```python
# Repair flow
1. Generate initial response
2. Validate against schema
3. If invalid:
   a. Build repair prompt with errors
   b. Ask LLM to fix the response
   c. Validate repaired response
   d. Repeat up to max_repair_attempts
4. Return validated response or raise ValueError
```

Repair prompt includes:
- Validation errors
- Malformed output
- Required schema
- Original context

### Usage Example

```python
from flybrowser.agents.structured_llm import StructuredLLMWrapper

wrapper = StructuredLLMWrapper(
    llm_provider=provider,
    max_repair_attempts=2,
    repair_temperature=0.1,
)

schema = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "selector": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "selector"],
}

# Generate structured response
result = await wrapper.generate_structured(
    prompt="Analyze this page and determine the next action",
    schema=schema,
    system_prompt="You are a browser automation agent",
)

# result is guaranteed to match schema
print(result)  # {"action": "click", "selector": "#submit", "confidence": 0.95}
```

## Tool Parameter Validation

Tools validate their parameters against JSON schemas:

```python
class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against the tool's JSON schema.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = self.metadata.to_json_schema()
        # Validate required parameters
        for required in schema.get("required", []):
            if required not in params:
                return False, f"Missing required parameter: {required}"
        # Validate types
        for name, value in params.items():
            if name in schema["properties"]:
                expected_type = schema["properties"][name].get("type")
                # Type checking...
        return True, None
```

### ToolParameter Definition

```python
@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    
    name: str                           # Parameter name
    type: str                           # JSON type
    description: str = ""               # Description
    required: bool = False              # Whether required
    default: Any = None                 # Default value
    enum: Optional[List[Any]] = None   # Allowed values
    items_type: Optional[str] = None   # For array types
```

### Tool Metadata Schema

Tools generate JSON schemas from their metadata:

```python
# Tool metadata
metadata = ToolMetadata(
    name="click",
    description="Click on an element",
    parameters=[
        ToolParameter(
            name="selector",
            type="string",
            description="CSS selector",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in ms",
            default=5000,
        ),
    ],
    ...
)

# Generated JSON schema
schema = metadata.to_json_schema()
# {
#     "type": "object",
#     "properties": {
#         "selector": {
#             "type": "string",
#             "description": "CSS selector"
#         },
#         "timeout": {
#             "type": "number",
#             "description": "Timeout in ms",
#             "default": 5000
#         }
#     },
#     "required": ["selector"]
# }
```

## Configuration Validation

FlyBrowser uses Pydantic for configuration validation:

```python
from pydantic import BaseModel, Field, field_validator

class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    provider_type: LLMProviderType
    model: str
    api_key: Optional[str] = None
    timeout: float = Field(default=60.0, ge=1.0, le=300.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v, info):
        """Validate base URL for local providers."""
        # Custom validation logic
        ...
```

### Field Constraints

Pydantic validates field constraints:

```python
class RetryConfig(BaseModel):
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter: bool = Field(default=True)

# Valid
config = RetryConfig(max_retries=5, initial_delay=2.0)

# Invalid - raises ValidationError
config = RetryConfig(max_retries=100)  # ge=0, le=10 violated
```

## Validation Flow

```
User Request
     │
     ▼
┌─────────────────┐
│ Browser Scope   │  Is task within browser automation scope?
│   Validation    │  Are URLs safe?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Configuration   │  Are parameters valid?
│   Validation    │  (Pydantic)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tool Parameter  │  Do action parameters match schema?
│   Validation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Response    │  Does response match expected schema?
│   Validation    │  Auto-repair if needed?
└────────┬────────┘
         │
         ▼
   Validated Result
```

## Custom Validators

Create custom validators for specialized validation:

```python
# Custom validator function
def validate_extraction_schema(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Custom validation for extraction results."""
    errors = []
    
    if "items" not in data:
        errors.append("Missing 'items' field")
        return False, errors
    
    items = data["items"]
    if not isinstance(items, list):
        errors.append("'items' must be a list")
        return False, errors
    
    for i, item in enumerate(items):
        if "value" not in item:
            errors.append(f"Item {i} missing 'value' field")
    
    return len(errors) == 0, errors

# Use custom validator
result = await wrapper.generate_structured(
    prompt="Extract product prices",
    schema=schema,
    custom_validator=validate_extraction_schema,
)
```

## Error Handling

Validation errors are surfaced clearly:

```python
# Browser scope validation error
try:
    await browser.agent("Delete all files in /tmp")
except ValueError as e:
    print(e)
    # "Task contains prohibited operation: 'delete file'. 
    #  FlyBrowser only supports browser automation..."

# URL validation error
try:
    await browser.goto("file:///etc/passwd")
except ValueError as e:
    print(e)
    # "URL scheme 'file' is not allowed. Use http:// or https://"

# Schema validation (after repair attempts exhausted)
try:
    result = await wrapper.generate_structured(prompt, schema)
except ValueError as e:
    print(e)
    # "Failed to generate valid response after 2 repair attempts.
    #  Errors: ['Missing required field: action']"
```

## Best Practices

1. **Always validate user input** - Use BrowserScopeValidator for task descriptions
2. **Define strict schemas** - Use required fields and type constraints
3. **Handle validation errors gracefully** - Provide actionable error messages
4. **Use custom validators** - For domain-specific validation logic
5. **Set appropriate repair attempts** - Balance reliability vs latency

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Tools System](tools.md) - Tool parameter validation
- [Error Handling Guide](../guides/error-handling.md) - Handling validation errors
