"""Browser automation ToolKits built on fireflyframework-genai."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional

from fireflyframework_genai.tools.toolkit import ToolKit

from flybrowser.agents.toolkits.navigation import create_navigation_toolkit
from flybrowser.agents.toolkits.interaction import create_interaction_toolkit
from flybrowser.agents.toolkits.extraction import create_extraction_toolkit
from flybrowser.agents.toolkits.system import create_system_toolkit
from flybrowser.agents.toolkits.search import create_search_toolkit
from flybrowser.agents.toolkits.captcha import create_captcha_toolkit

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


def create_all_toolkits(
    page: PageController,
    search_coordinator: Optional[Any] = None,
    captcha_solver: Optional[Any] = None,
    user_input_callback: Optional[Any] = None,
) -> List[ToolKit]:
    """Create all 6 browser ToolKits."""
    return [
        create_navigation_toolkit(page),
        create_interaction_toolkit(page),
        create_extraction_toolkit(page),
        create_system_toolkit(user_input_callback=user_input_callback),
        create_search_toolkit(search_coordinator),
        create_captcha_toolkit(page, captcha_solver),
    ]

__all__ = [
    "create_all_toolkits",
    "create_navigation_toolkit", "create_interaction_toolkit",
    "create_extraction_toolkit", "create_system_toolkit",
    "create_search_toolkit", "create_captcha_toolkit",
]
