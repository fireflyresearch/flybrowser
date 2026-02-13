"""Navigation ToolKit for browser page navigation.

Provides tools for URL navigation, browser history traversal,
and page refresh, all built on the fireflyframework-genai ToolKit
pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fireflyframework_genai.tools.decorators import firefly_tool
from fireflyframework_genai.tools.toolkit import ToolKit

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


def create_navigation_toolkit(page: PageController) -> ToolKit:
    """Create a navigation toolkit bound to the given *page*.

    The returned :class:`ToolKit` contains four tools:

    * **navigate** -- go to a URL
    * **go_back** -- browser back button
    * **go_forward** -- browser forward button
    * **refresh** -- reload the current page

    Each tool is a closure over *page* so it can drive the browser
    without requiring dependency injection at call time.
    """

    @firefly_tool(
        name="navigate",
        description=(
            "Navigate to a URL. Accepts an optional wait_until parameter "
            "(default: 'domcontentloaded')."
        ),
        auto_register=False,
    )
    async def navigate(url: str, wait_until: str = "domcontentloaded") -> str:
        await page.goto(url, wait_until=wait_until)
        state = await page.get_page_state()
        return f"Navigated to {state.get('url', url)}. Title: {state.get('title', '')}"

    @firefly_tool(
        name="go_back",
        description="Navigate back to the previous page in browser history.",
        auto_register=False,
    )
    async def go_back() -> str:
        await page.page.go_back()
        state = await page.get_page_state()
        return f"Navigated back to {state.get('url', '')}. Title: {state.get('title', '')}"

    @firefly_tool(
        name="go_forward",
        description="Navigate forward to the next page in browser history.",
        auto_register=False,
    )
    async def go_forward() -> str:
        await page.page.go_forward()
        state = await page.get_page_state()
        return f"Navigated forward to {state.get('url', '')}. Title: {state.get('title', '')}"

    @firefly_tool(
        name="refresh",
        description="Refresh (reload) the current page.",
        auto_register=False,
    )
    async def refresh() -> str:
        await page.page.reload()
        state = await page.get_page_state()
        return f"Refreshed page {state.get('url', '')}. Title: {state.get('title', '')}"

    return ToolKit(
        "navigation",
        [navigate, go_back, go_forward, refresh],
        description="Tools for browser page navigation.",
    )
