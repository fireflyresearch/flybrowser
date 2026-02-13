# Pipelines

FlyBrowser pipelines let you define and execute multi-step browser workflows from YAML files or inline command strings. This is useful for repeatable automation tasks, CI/CD integration, and batch processing.

## Quick Reference

```bash
# Run a YAML workflow
flybrowser run workflow.yaml

# Run inline commands
flybrowser run --inline "goto https://example.com && extract 'get the title'"
```

## YAML Workflow Format

A workflow file defines sessions and a sequence of steps:

```yaml
name: my-workflow
sessions:
  main:
    provider: openai
    model: gpt-4o
    headless: true
steps:
  - name: navigate
    session: main
    action: goto
    url: https://example.com
  - name: extract-data
    session: main
    action: extract
    query: "Get all product names and prices"
  - name: take-screenshot
    session: main
    action: screenshot
    full_page: true
```

### Workflow Structure

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Workflow name (defaults to the filename) |
| `sessions` | No | Named session configurations |
| `steps` | Yes | Ordered list of steps to execute |

### Session Configuration

Each entry under `sessions` defines a browser session:

```yaml
sessions:
  main:
    provider: openai         # LLM provider (openai, anthropic, gemini, ollama)
    model: gpt-4o            # Model name
    headless: true           # Run without visible browser
  research:
    provider: anthropic
    model: claude-sonnet-4-5-20250929
    headless: true
```

If a step references a session name that is not defined under `sessions`, a default session (OpenAI, headless) is created automatically.

### Step Actions

Each step must have an `action` field. Available actions:

| Action | Required Fields | Optional Fields | Description |
|--------|----------------|-----------------|-------------|
| `goto` | `url` | | Navigate to a URL |
| `extract` | `query` | | Extract data using natural language |
| `act` | `instruction` | | Perform a browser action |
| `screenshot` | | `full_page` | Capture the current page |
| `agent` | `task` | `max_iterations` | Run an autonomous agent task |

### Step Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Human-readable step name (for logging) |
| `session` | No | Session name (defaults to `"default"`) |
| `action` | Yes | One of: goto, extract, act, screenshot, agent |

## Examples

### Web Scraping Workflow

```yaml
name: scrape-hn
sessions:
  browser:
    provider: openai
    model: gpt-4o
    headless: true
steps:
  - name: open-hn
    session: browser
    action: goto
    url: https://news.ycombinator.com

  - name: get-stories
    session: browser
    action: extract
    query: "Get the top 10 stories with title, score, and comment count"

  - name: capture-page
    session: browser
    action: screenshot
    full_page: false
```

Run it:

```bash
$ flybrowser run scrape-hn.yaml

Pipeline
  Workflow: scrape-hn
  Steps:   3

Pipeline Complete
  Steps Completed: 3
  Status:          Success
```

### Form Automation Workflow

```yaml
name: submit-form
sessions:
  main:
    provider: openai
    headless: false
steps:
  - name: open-form
    session: main
    action: goto
    url: https://example.com/contact

  - name: fill-name
    session: main
    action: act
    instruction: "Type 'Jane Doe' in the name field"

  - name: fill-email
    session: main
    action: act
    instruction: "Type 'jane@example.com' in the email field"

  - name: fill-message
    session: main
    action: act
    instruction: "Type 'I would like a demo' in the message field"

  - name: submit
    session: main
    action: act
    instruction: "Click the Submit button"

  - name: verify
    session: main
    action: extract
    query: "Check if there is a success confirmation message"
```

### Multi-Session Workflow

Use multiple sessions to work across different sites simultaneously:

```yaml
name: cross-site-comparison
sessions:
  site-a:
    provider: openai
    model: gpt-4o
    headless: true
  site-b:
    provider: openai
    model: gpt-4o
    headless: true
steps:
  - name: open-site-a
    session: site-a
    action: goto
    url: https://shop-a.example.com/laptops

  - name: open-site-b
    session: site-b
    action: goto
    url: https://shop-b.example.com/laptops

  - name: prices-a
    session: site-a
    action: extract
    query: "Get all laptop names and prices"

  - name: prices-b
    session: site-b
    action: extract
    query: "Get all laptop names and prices"
```

### Agent-Powered Workflow

Use the `agent` action for steps that require multi-step reasoning:

```yaml
name: research-task
sessions:
  researcher:
    provider: anthropic
    model: claude-sonnet-4-5-20250929
steps:
  - name: research
    session: researcher
    action: agent
    task: "Go to Wikipedia, search for 'browser automation', and extract the first paragraph of the article"
    max_iterations: 20

  - name: screenshot
    session: researcher
    action: screenshot
```

## Inline Commands

For quick, ad-hoc workflows, use inline commands instead of YAML files. Commands are separated by `&&`:

```bash
flybrowser run --inline "goto https://example.com && extract 'get the title'"
```

### Inline Command Syntax

Each command follows the format `<action> <arguments>`:

```
goto <url>
extract '<query>'
act '<instruction>'
screenshot
agent '<task>'
```

Quoted arguments are handled correctly:

```bash
# Single quotes
flybrowser run --inline "goto https://example.com && extract 'get all links'"

# Double quotes (escaped in shell)
flybrowser run --inline "goto https://example.com && act \"click the login button\""
```

### Inline Examples

```bash
# Navigate and extract
flybrowser run --inline "goto https://news.ycombinator.com && extract 'top 5 stories'"

# Multi-step interaction
flybrowser run --inline "goto https://google.com && act 'type python in the search box' && act 'click search' && extract 'first 3 results'"

# Navigate and screenshot
flybrowser run --inline "goto https://example.com && screenshot"
```

## Pipeline Execution

When you run a pipeline, FlyBrowser:

1. **Parses** the YAML file or inline commands into a workflow definition
2. **Creates sessions** on demand as steps reference them
3. **Executes steps** sequentially in the order defined
4. **Collects results** from each step
5. **Cleans up** all sessions when the pipeline completes (or fails)

### Error Handling

If a step fails, the pipeline stops and reports the error:

```
Error: Pipeline failed after 2 steps: Connection refused
```

All sessions are cleaned up even when the pipeline fails.

### Result Structure

Internally, the pipeline returns a result dictionary:

```json
{
  "success": true,
  "steps_completed": 3,
  "results": [
    {
      "step": "navigate",
      "action": "goto",
      "result": {"url": "https://example.com", "title": "Example"}
    },
    {
      "step": "extract-data",
      "action": "extract",
      "result": {"data": ["Item 1", "Item 2"]}
    },
    {
      "step": "take-screenshot",
      "action": "screenshot",
      "result": {"screenshot": "base64..."}
    }
  ]
}
```

## Environment Variables

Pipelines respect the standard FlyBrowser environment variables:

| Variable | Description |
|----------|-------------|
| `FLYBROWSER_ENDPOINT` | Server endpoint for pipeline execution |
| `FLYBROWSER_LLM_PROVIDER` | Default provider for sessions without explicit config |
| `FLYBROWSER_LLM_MODEL` | Default model for sessions without explicit config |

## CI/CD Integration

Pipelines are designed for automated environments. Example GitHub Actions usage:

```yaml
# .github/workflows/automation.yml
name: Browser Automation
on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9 AM
jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install flybrowser && playwright install chromium
      - run: flybrowser serve &
      - run: sleep 5
      - run: flybrowser run workflows/weekly-report.yaml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          FLYBROWSER_ENDPOINT: http://localhost:8000
```

## See Also

- [Session Management](session-management.md) -- Manage long-lived sessions
- [Direct Commands](direct-commands.md) -- One-shot commands
- [CLI Reference](../reference/cli.md) -- Full CLI command reference
- [Workflow Automation Examples](../examples/workflow-automation.md) -- More workflow examples
