"""CAPTCHA ToolKit for detecting and solving CAPTCHAs.

Provides tools for CAPTCHA detection, solving, and waiting for
resolution, all built on the fireflyframework-genai ToolKit pattern.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from fireflyframework_genai.tools.decorators import firefly_tool
from fireflyframework_genai.tools.toolkit import ToolKit

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


def create_captcha_toolkit(
    page: PageController, captcha_solver: Optional[Any] = None
) -> ToolKit:
    """Create a CAPTCHA toolkit bound to the given *page*.

    The returned :class:`ToolKit` contains three tools:

    * **detect_captcha** -- detect if a CAPTCHA is present on the page
    * **solve_captcha** -- attempt to solve a detected CAPTCHA
    * **wait_captcha_resolved** -- wait for a CAPTCHA to be resolved

    If *captcha_solver* is ``None``, each tool returns a message
    indicating that no solver is configured.
    """

    @firefly_tool(
        name="detect_captcha",
        description="Detect if a CAPTCHA is present on the current page.",
        auto_register=False,
    )
    async def detect_captcha() -> str:
        if captcha_solver is None:
            return "No CAPTCHA solver configured."
        result = await captcha_solver.detect()
        return json.dumps(result, default=str)

    @firefly_tool(
        name="solve_captcha",
        description=(
            "Attempt to solve a CAPTCHA on the current page. Set captcha_type "
            "to specify the type (default: 'auto' for automatic detection)."
        ),
        auto_register=False,
    )
    async def solve_captcha(captcha_type: str = "auto") -> str:
        if captcha_solver is None:
            return "No CAPTCHA solver configured."
        result = await captcha_solver.solve(captcha_type=captcha_type)
        return json.dumps(result, default=str)

    @firefly_tool(
        name="wait_captcha_resolved",
        description=(
            "Wait for a CAPTCHA to be resolved, with a configurable timeout "
            "in seconds (default: 60)."
        ),
        auto_register=False,
    )
    async def wait_captcha_resolved(timeout: int = 60) -> str:
        if captcha_solver is None:
            return "No CAPTCHA solver configured."
        result = await captcha_solver.wait_resolved(timeout=timeout)
        if result:
            return "CAPTCHA resolved successfully."
        return "CAPTCHA resolution timed out."

    return ToolKit(
        "captcha",
        [detect_captcha, solve_captcha, wait_captcha_resolved],
        description="Tools for detecting and solving CAPTCHAs.",
    )
