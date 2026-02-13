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

"""ScreenshotOnErrorMiddleware — capture screenshot when agent errors."""
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ScreenshotOnErrorMiddleware:
    def __init__(self, page) -> None:
        self._page = page

    async def before_run(self, ctx: Any) -> None:
        pass

    async def after_run(self, ctx: Any, result: Any) -> Any:
        if getattr(result, "error", None):
            try:
                img_bytes = await self._page.screenshot()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                ctx.metadata["error_screenshot_b64"] = b64[:100]
            except Exception as e:
                logger.debug(f"Error screenshot failed: {e}")
        return result
