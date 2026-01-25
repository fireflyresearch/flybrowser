# search() - Web Search

The `search()` method performs web searches using multiple providers with intelligent ranking. It supports both API-based search (fast, reliable) and browser-based search (human-like navigation).

## Basic Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        search_provider="serper",  # Optional: defaults to auto
    ) as browser:
        # Simple search
        results = await browser.search("Python web scraping tutorials")
        
        # Access results
        for item in results.data["results"]:
            print(f"{item['title']}: {item['url']}")

asyncio.run(main())
```

## Method Signature

```python
async def search(
    self,
    query: str,
    search_type: str = "auto",
    max_results: int = 10,
    ranking: str = "auto",
    return_metadata: bool = True,
) -> AgentRequestResponse | dict
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query or natural language instruction |
| `search_type` | `str` | `"auto"` | Type of search: web, images, news, videos, places, shopping |
| `max_results` | `int` | `10` | Maximum results to return (1-50) |
| `ranking` | `str` | `"auto"` | Ranking preference: auto, balanced, relevance, freshness, authority |
| `return_metadata` | `bool` | `True` | Include execution metadata in response |

### Returns

`AgentRequestResponse` with:
- `success` - Whether the search completed successfully
- `data` - Search results (see Response Format below)
- `error` - Error message if failed
- `operation` - "search"
- `query` - The original search query

When `return_metadata=True`, also includes:
- `execution` - ExecutionInfo with duration and details
- `llm_usage` - LLMUsageInfo with token counts and cost

## Search Types

```python
# Web search (default)
results = await browser.search("Python tutorials", search_type="web")

# Image search
images = await browser.search("sunset photography", search_type="images")

# News search
news = await browser.search("AI developments", search_type="news")

# Video search
videos = await browser.search("Python tutorial", search_type="videos")

# Local/Places search
places = await browser.search("coffee shops near me", search_type="places")

# Shopping search
products = await browser.search("wireless headphones", search_type="shopping")

# Auto-detect from query
results = await browser.search("Find the latest news about Python 4.0")
# Automatically detects this is a news search
```

## Ranking Options

```python
# Balanced ranking (default)
results = await browser.search("Python tutorials", ranking="balanced")

# Prioritize relevance (keyword matching)
results = await browser.search("async await Python", ranking="relevance")

# Prioritize freshness (recent results)
results = await browser.search("AI news", ranking="freshness")

# Prioritize authority (trusted sources)
results = await browser.search("Python security best practices", ranking="authority")

# Auto-select based on query
results = await browser.search("breaking news technology")
# Automatically selects freshness ranking for news queries
```

## Response Format

```python
results = await browser.search("Python tutorials")

# Response structure
{
    "query": "Python tutorials",
    "search_type": "web",
    "results": [
        {
            "title": "Python Tutorial - W3Schools",
            "url": "https://www.w3schools.com/python/",
            "snippet": "Learn Python programming with our easy tutorial...",
            "relevance_score": 0.95,
            "domain": "w3schools.com",
        },
        # ... more results
    ],
    "result_count": 10,
    "provider_used": "serper",
    "answer_box": {
        "title": "Python Tutorial",
        "answer": "...",
        "url": "..."
    },  # Optional, from featured snippets
    "knowledge_graph": {...},  # Optional, entity information
    "related_searches": ["python tutorial for beginners", "..."],
}
```

## Search Providers

FlyBrowser supports multiple search providers:

### Serper.dev (Recommended)

```python
# Configure at initialization
browser = FlyBrowser(
    llm_provider="openai",
    search_provider="serper",
    search_api_key="your-serper-api-key",
)

# Or via environment variable
# export SERPER_API_KEY=your-serper-api-key
```

**Features:**
- Fast response times (~100-300ms)
- Affordable pricing
- Rich result metadata (answer boxes, knowledge graphs)
- Supports all search types

### Google Custom Search

```python
browser = FlyBrowser(
    llm_provider="openai",
    search_provider="google",
)

# Requires environment variables:
# export GOOGLE_CUSTOM_SEARCH_API_KEY=your-api-key
# export GOOGLE_CUSTOM_SEARCH_CX=your-search-engine-id
```

### Bing Web Search

```python
browser = FlyBrowser(
    llm_provider="openai",
    search_provider="bing",
)

# Requires environment variable:
# export BING_SEARCH_API_KEY=your-api-key
```

## Runtime Configuration

Change search settings after initialization:

```python
async with FlyBrowser(llm_provider="openai") as browser:
    # Switch provider
    browser.configure_search(
        provider="serper",
        api_key="your-api-key"
    )
    
    # Adjust ranking weights
    browser.configure_search(
        ranking_weights={
            "bm25": 0.25,        # Keyword relevance
            "freshness": 0.45,   # Prioritize recent
            "domain_authority": 0.15,
            "position": 0.15,    # Original ranking
        }
    )
    
    # Enable/disable ranking
    browser.configure_search(enable_ranking=False)
    
    # Set cache TTL
    browser.configure_search(cache_ttl_seconds=600)
```

## configure_search() Method

```python
def configure_search(
    self,
    provider: str | None = None,
    api_key: str | None = None,
    enable_ranking: bool | None = None,
    ranking_weights: dict[str, float] | None = None,
    cache_ttl_seconds: int | None = None,
) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `None` | Search provider: serper, google, bing, auto |
| `api_key` | `str` | `None` | API key for the provider |
| `enable_ranking` | `bool` | `None` | Enable intelligent result ranking |
| `ranking_weights` | `dict` | `None` | Custom ranking weights |
| `cache_ttl_seconds` | `int` | `None` | Result cache TTL in seconds |

## Intelligent Ranking

FlyBrowser uses a multi-factor ranking system:

### BM25 Ranker
Scores results based on keyword relevance using the Okapi BM25 algorithm.

### Freshness Ranker
Boosts recent content. Configurable time windows:
- Very recent (< 24 hours): 1.0
- Recent (< 7 days): 0.8
- Semi-recent (< 30 days): 0.6
- Older: 0.4

### Domain Authority Ranker
Scores results based on source trustworthiness. Built-in authority scores for common domains (Wikipedia, GitHub, official documentation, etc.).

### Composite Ranker
Combines all rankers with configurable weights:
```python
{
    "bm25": 0.35,           # Keyword relevance
    "freshness": 0.20,      # Recency
    "domain_authority": 0.15,  # Source quality
    "position": 0.30,       # Original search engine ranking
}
```

## ReAct Integration

Search is fully integrated with the ReAct framework. The agent automatically uses search when appropriate:

```python
# The agent will search as needed
result = await browser.agent(
    "Find the current price of Bitcoin and compare it to Ethereum"
)
# Agent automatically:
# 1. Detects search intent
# 2. Performs appropriate searches
# 3. Navigates to sources if needed
# 4. Extracts and compares data
```

### Intent Detection

When using `agent()` or `act()`, FlyBrowser uses LLM-powered intent detection to determine when to search:

```python
# These will trigger automatic search
await browser.act("Search for Python tutorials")
await browser.act("Find the latest news about AI")
await browser.act("Look up the weather in Tokyo")

# Search happens transparently
result = await browser.agent(
    "Research the top 3 JavaScript frameworks and summarize their pros and cons"
)
```

## Browser-Based Fallback

If API search is unavailable, FlyBrowser falls back to browser-based search:

```python
# Uses human-like navigation when no API configured
async with FlyBrowser(llm_provider="openai") as browser:
    # Without search_api_key, uses browser navigation
    result = await browser.search("Python tutorials")
    # Opens google.com, types query, extracts results
```

Browser-based search:
- Works without API keys
- Human-like behavior (useful for sites that block APIs)
- Slower than API search (~2-5 seconds vs ~200ms)
- Supports Google, Bing, DuckDuckGo

## Examples

### Research Task

```python
async def research_topic(topic: str):
    async with FlyBrowser(
        llm_provider="openai",
        search_provider="serper",
    ) as browser:
        # Search for information
        results = await browser.search(
            f"latest research on {topic}",
            search_type="news",
            ranking="freshness",
            max_results=20,
        )
        
        # Process results
        for item in results.data["results"][:5]:
            print(f"- {item['title']}")
            print(f"  {item['snippet']}\n")
```

### Multi-Source Search

```python
async def comprehensive_search(query: str):
    async with FlyBrowser(llm_provider="openai") as browser:
        # Search across different types
        web_results = await browser.search(query, search_type="web")
        news_results = await browser.search(query, search_type="news")
        
        return {
            "web": web_results.data["results"],
            "news": news_results.data["results"],
        }
```

### Price Comparison

```python
async def find_best_price(product: str):
    async with FlyBrowser(
        llm_provider="openai",
        search_provider="serper",
    ) as browser:
        # Shopping search
        results = await browser.search(
            product,
            search_type="shopping",
            max_results=20,
        )
        
        # Sort by price (if available)
        return sorted(
            results.data["results"],
            key=lambda x: x.get("price", float("inf"))
        )
```

## Error Handling

```python
results = await browser.search("my query")

if not results.success:
    print(f"Search failed: {results.error}")
    
    # Common errors
    error = results.error.lower()
    
    if "api key" in error:
        # Missing or invalid API key
        browser.configure_search(api_key="new-key")
    elif "rate limit" in error:
        # Rate limited, wait and retry
        await asyncio.sleep(1)
    elif "timeout" in error:
        # Network timeout
        pass
```

## Best Practices

### Use Appropriate Search Types

```python
# Explicit types for better results
await browser.search("Python tutorials", search_type="web")
await browser.search("AI news today", search_type="news")
await browser.search("sunset wallpaper", search_type="images")
```

### Customize Ranking for Use Case

```python
# For news/current events
browser.configure_search(ranking_weights={"freshness": 0.5})

# For technical documentation
browser.configure_search(ranking_weights={"domain_authority": 0.4})

# For broad research
browser.configure_search(ranking_weights={"bm25": 0.5})
```

### Cache Results

```python
# Increase cache TTL for stable queries
browser.configure_search(cache_ttl_seconds=3600)  # 1 hour

# Disable cache for real-time data
browser.configure_search(cache_ttl_seconds=0)
```

## Related Methods

- [agent()](agent.md) - Autonomous multi-step tasks with search
- [act()](act.md) - Action execution (can trigger search)
- [extract()](extract.md) - Extract data from search results
- [navigate()](navigation.md) - Navigate to search result URLs

## See Also

- [Configuration Reference](../reference/configuration.md) - Search configuration options
- [SDK Reference](../reference/sdk.md) - Full API documentation
- [Architecture: Tools](../architecture/tools.md) - How search tools work
