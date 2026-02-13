# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ObstacleDetectionMiddleware — detect popups/captchas before agent actions."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

_DETECT_OBSTACLES_JS = """() => {
    const hasCookieBanner = !!(
        document.querySelector('[class*="cookie"]') ||
        document.querySelector('[id*="cookie"]') ||
        document.querySelector('[class*="consent"]') ||
        document.querySelector('[id*="consent"]')
    );
    const hasPopup = !!(
        document.querySelector('[class*="modal"].show') ||
        document.querySelector('[role="dialog"][aria-modal="true"]')
    );
    return { hasCookieBanner, hasPopup };
}"""


class ObstacleDetectionMiddleware:
    def __init__(self, page) -> None:
        self._page = page

    async def before_run(self, ctx: Any) -> None:
        try:
            result = await self._page.page.evaluate(_DETECT_OBSTACLES_JS)
            if result.get("hasCookieBanner"):
                ctx.metadata["obstacle_detected"] = "cookie_banner"
            elif result.get("hasPopup"):
                ctx.metadata["obstacle_detected"] = "popup"
        except Exception as e:
            logger.debug(f"Obstacle detection failed: {e}")

    async def after_run(self, ctx: Any, result: Any) -> None:
        pass
