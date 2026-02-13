"""Search ToolKit wrapping an existing search coordinator.

Provides a single ``search`` tool that delegates to whichever search
backend has been configured, all built on the fireflyframework-genai
ToolKit pattern.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fireflyframework_genai.tools.decorators import firefly_tool
from fireflyframework_genai.tools.toolkit import ToolKit


def create_search_toolkit(search_coordinator: Optional[Any]) -> ToolKit:
    """Create a search toolkit that wraps *search_coordinator*.

    The returned :class:`ToolKit` contains one tool:

    * **search** -- perform a web search via the configured coordinator

    If *search_coordinator* is ``None``, the tool returns a polite
    message indicating that no search provider is available.
    """

    @firefly_tool(
        name="search",
        description=(
            "Search the web for information. Supports different search types "
            "(auto, web, news, images) and configurable result count."
        ),
        auto_register=False,
    )
    async def search(
        query: str, search_type: str = "auto", max_results: int = 10
    ) -> str:
        if search_coordinator is None:
            return "No search provider configured."

        results = await search_coordinator.search(
            query, search_type=search_type, max_results=max_results
        )

        if not results or not results.get("results"):
            return f"No results found for: {query}"

        items = results["results"]
        total = results.get("total", len(items))

        parts = [f"Search results for '{query}' ({total} total):"]
        for i, item in enumerate(items, 1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            parts.append(f"\n{i}. {title}\n   URL: {url}\n   {snippet}")

        return "\n".join(parts)

    return ToolKit(
        "search",
        [search],
        description="Tools for web search via configured search provider.",
    )
