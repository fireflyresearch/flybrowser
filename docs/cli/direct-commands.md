# Direct Commands

FlyBrowser provides one-shot CLI commands that map directly to SDK operations. These commands let you perform a single browser action from the command line without entering the REPL or writing Python code.

## Quick Reference

```bash
flybrowser goto <url>                    # Navigate to a URL
flybrowser extract <query>               # Extract data from the page
flybrowser act <instruction>             # Perform a browser action
flybrowser screenshot                    # Capture the current page
flybrowser agent <task>                  # Run an autonomous agent task
```

## Auto-Ephemeral Sessions

When you run a direct command without specifying `--session`, FlyBrowser automatically:

1. Creates an ephemeral browser session on the server
2. Executes the command
3. Closes the session

This makes direct commands ideal for quick, one-off tasks. To reuse a session across multiple commands, pass `--session <id>` (see [Session Management](session-management.md) to create persistent sessions).

## Commands

### goto

Navigate the browser to a specific URL.

```bash
flybrowser goto <url> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--session` | | (ephemeral) | Existing session ID to reuse |
| `--wait-for` | | | CSS selector to wait for after navigation |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Examples:**

```bash
# Navigate to a URL
$ flybrowser goto https://news.ycombinator.com

Navigation Result
  URL:    https://news.ycombinator.com
  Title:  Hacker News
  Status: OK

# Navigate with a wait condition
$ flybrowser goto https://example.com --wait-for "#content"

# Navigate using an existing session
$ flybrowser goto https://example.com --session sess_abc123
```

### extract

Extract data from the current page using a natural language query.

```bash
flybrowser extract <query> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--session` | | (ephemeral) | Existing session ID to reuse |
| `--schema` | | | Path to a JSON schema file for structured extraction |
| `--format` | `-f` | json | Output format: `json`, `csv`, or `table` |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Examples:**

```bash
# Simple extraction (JSON output)
$ flybrowser extract "get the top 5 story titles"
{
  "data": [
    "Show HN: FlyBrowser - AI-Powered Browser Automation",
    "Rust 2026 Edition Released",
    ...
  ]
}

# Structured extraction with a schema file
$ flybrowser extract "get product listings" --schema products.json --format table

name                    | price  | rating
---------------------------------------------
Widget Pro              | 29.99  | 4.5
Gadget Ultra            | 49.99  | 4.8

# CSV output for piping
$ flybrowser extract "get all links" --format csv > links.csv
```

**Schema file example** (`products.json`):

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "price": {"type": "number"},
      "rating": {"type": "number"}
    },
    "required": ["name", "price"]
  }
}
```

### act

Perform a browser action described in natural language.

```bash
flybrowser act <instruction> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--session` | | (ephemeral) | Existing session ID to reuse |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Examples:**

```bash
# Click a button
$ flybrowser act "click the Sign In button"

Action Result
  Instruction: click the Sign In button
  Success:     True
  Details:     Clicked element matching 'Sign In' button

# Fill a form
$ flybrowser act "type 'hello@example.com' in the email field"

# Scroll the page
$ flybrowser act "scroll down to the footer"
```

### screenshot

Capture a screenshot of the current browser page.

```bash
flybrowser screenshot [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--session` | | (ephemeral) | Existing session ID to reuse |
| `--output` | `-o` | screenshot.png | Output file path |
| `--full-page` | | False | Capture the entire scrollable page |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Examples:**

```bash
# Save to default file
$ flybrowser screenshot
Screenshot saved to screenshot.png

# Custom output path and full page
$ flybrowser screenshot --output ~/captures/page.png --full-page
Screenshot saved to /Users/you/captures/page.png

# Screenshot of an existing session
$ flybrowser screenshot --session sess_abc123 -o session_state.png
```

### agent

Run an autonomous agent that navigates and interacts with pages to complete a multi-step task.

```bash
flybrowser agent <task> [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--session` | | (ephemeral) | Existing session ID to reuse |
| `--max-iterations` | | 50 | Maximum number of reasoning iterations |
| `--stream` | | False | Stream progress to stdout in real time |
| `--endpoint` | `-e` | http://localhost:8000 | FlyBrowser server endpoint |

**Examples:**

```bash
# Run a complex task
$ flybrowser agent "Go to Hacker News, find the top story, and extract the article text"
{
  "success": true,
  "result": "The article discusses...",
  "iterations": 8,
  "cost_usd": 0.042
}

# Limit iterations for a simpler task
$ flybrowser agent "Search Google for 'python web automation' and get the first result title" \
    --max-iterations 10

# Stream agent progress
$ flybrowser agent "Fill out the contact form on example.com" --stream
```

## Output Formats

The `extract` command supports three output formats:

### JSON (default)

```bash
$ flybrowser extract "get product names and prices" --format json
{
  "data": [
    {"name": "Widget", "price": 29.99},
    {"name": "Gadget", "price": 49.99}
  ]
}
```

### CSV

```bash
$ flybrowser extract "get product names and prices" --format csv
name,price
Widget,29.99
Gadget,49.99
```

### Table

```bash
$ flybrowser extract "get product names and prices" --format table
name   | price
--------------
Widget | 29.99
Gadget | 49.99
```

## Combining with Session Commands

Direct commands work well with persistent sessions for multi-step workflows:

```bash
# Create a persistent session
flybrowser session create -p openai
# Session ID: sess_abc123

# Use direct commands against it
flybrowser goto https://shop.example.com --session sess_abc123
flybrowser act "search for 'laptop'" --session sess_abc123
flybrowser extract "get the top 3 results with prices" --session sess_abc123 --format csv
flybrowser screenshot --session sess_abc123 -o results.png

# Clean up
flybrowser session close sess_abc123
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FLYBROWSER_ENDPOINT` | Default server endpoint for all commands |
| `FLYBROWSER_LLM_PROVIDER` | Default LLM provider for ephemeral sessions |
| `FLYBROWSER_LLM_MODEL` | Default LLM model for ephemeral sessions |

## See Also

- [Session Management](session-management.md) -- Manage persistent sessions
- [Pipelines](pipelines.md) -- Run multi-step workflows from YAML
- [CLI Reference](../reference/cli.md) -- Full CLI command reference
- [SDK Reference](../reference/sdk.md) -- Python SDK equivalents
