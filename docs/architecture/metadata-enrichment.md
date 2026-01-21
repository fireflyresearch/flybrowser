# Metadata Enrichment Architecture

## Overview

All FlyBrowser agents now use a centralized, professional metadata enrichment system to ensure consistent, accurate metadata in all responses.

## Problem

LLMs (especially local models like Llama2) may:
- Return hardcoded example URLs (e.g., `https://example.com/query`)
- Extract HTML attributes instead of visible content
- Populate metadata incorrectly or incompletely

## Solution

### 1. Base-Level Implementation

Added `enrich_response_metadata()` method to `BaseAgent` that all agents inherit:

```python path=/Users/ancongui/Development/flybrowser/flybrowser/agents/base_agent.py start=199
async def enrich_response_metadata(
    self,
    result: Dict[str, Any],
    query_or_instruction: str,
    include_page_context: bool = True
) -> Dict[str, Any]:
    """
    Enrich agent response with accurate metadata.
    
    This ensures all responses have:
    - Correct source_url (actual page URL, not hardcoded examples)
    - Accurate query/instruction
    - Timestamp
    - Page title
    """
```

**Features:**
- Detects and fixes hardcoded example URLs
- Adds missing metadata fields
- Includes timestamp and page title
- Works across all agents consistently

### 2. Improved Schema Definitions

Schemas now include explicit descriptions:

```python
schema = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "description": "Extract actual visible content, not HTML attributes"
        },
        "metadata": {
            "type": "object",
            "description": "Metadata (will be auto-populated)",
            "properties": {
                "source_url": {
                    "type": "string",
                    "description": f"The source page URL: {actual_url}"
                }
            }
        }
    }
}
```

### 3. Enhanced System Prompts

Prompts now explicitly instruct LLMs:

```
GUIDELINES:
- Extract actual visible content from the page (text, headings, links, data)
- DO NOT extract HTML attributes, element IDs, CSS classes, or technical metadata
- DO NOT extract URL parameters, query strings, or technical page data
- Focus on human-readable information that answers the query
- Metadata fields (source_url, extraction_query) will be auto-populated
```

## Usage Across Agents

### ExtractionAgent

```python
# After validation
result = await self.validator.validate_and_fix(response.content, schema)

# Enrich metadata
result = await self.enrich_response_metadata(
    result,
    query_or_instruction=query,
    include_page_context=True
)
```

### Other Agents

All agents should follow the same pattern:

```python
# ActionAgent, NavigationAgent, etc.
result = await self.enrich_response_metadata(
    result,
    query_or_instruction=instruction,
    include_page_context=True
)
```

## Benefits

1. **Consistency**: All agents use the same metadata enrichment
2. **Accuracy**: Guaranteed correct URLs, timestamps, and context
3. **Maintainability**: Single source of truth in BaseAgent
4. **Extensibility**: Easy to add new metadata fields globally
5. **Debugging**: Proper metadata aids in troubleshooting

## Example Output

**Before (Incorrect):**
```json
{
  "data": {
    "cshid": "false",
    "eid": "1",
    "glmm": "1"
  },
  "metadata": {
    "source_url": "https://example.com/query",
    "extraction_query": "What is the main heading?"
  }
}
```

**After (Correct):**
```json
{
  "data": {
    "heading": "Google",
    "description": "Search engine homepage"
  },
  "metadata": {
    "source_url": "https://google.com",
    "extraction_query": "What is the main heading?",
    "timestamp": "2026-01-21T12:25:00Z",
    "page_title": "Google"
  }
}
```

## Testing

Test the enrichment:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(llm_provider="ollama", llm_model="llama2")
await browser.start()
await browser.goto("https://google.com")

# Extraction
data = await browser.extract("What is the main heading?")
assert data['data']['metadata']['source_url'] == "https://google.com"
assert "example.com" not in data['data']['metadata']['source_url']

# Verify actual content extraction
assert "cshid" not in str(data['data']['data'])  # No HTML attributes
assert "heading" in str(data['data']['data']).lower()  # Actual content

await browser.stop()
```

## Future Enhancements

Potential additions to metadata:
- User agent information
- Session ID
- Agent type/version
- Execution environment (embedded/standalone/cluster)
- Performance metrics (LLM tokens, latency)
