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

"""Shared fixtures for toolkit tests.

These mock the browser layer (PageController, Playwright Page) so toolkit
tests run without a real browser.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock


@pytest.fixture
def mock_playwright_page():
    """Mock Playwright Page object with common methods."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")

    # Navigation
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()
    page.reload = AsyncMock()

    # Content
    page.content = AsyncMock(return_value="<html><body>Hello</body></html>")
    page.evaluate = AsyncMock(return_value=None)
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")

    # Interaction
    page.keyboard = MagicMock()
    page.keyboard.press = AsyncMock()
    page.mouse = MagicMock()
    page.mouse.move = AsyncMock()

    # Locator
    mock_locator = AsyncMock()
    mock_locator.count = AsyncMock(return_value=1)
    mock_locator.first = mock_locator
    mock_locator.click = AsyncMock()
    mock_locator.dblclick = AsyncMock()
    mock_locator.fill = AsyncMock()
    mock_locator.clear = AsyncMock()
    mock_locator.hover = AsyncMock()
    mock_locator.focus = AsyncMock()
    mock_locator.check = AsyncMock()
    mock_locator.uncheck = AsyncMock()
    mock_locator.is_checked = AsyncMock(return_value=False)
    mock_locator.select_option = AsyncMock(return_value=["opt1"])
    mock_locator.set_input_files = AsyncMock()
    mock_locator.drag_to = AsyncMock()
    mock_locator.text_content = AsyncMock(return_value="Button Text")
    mock_locator.input_value = AsyncMock(return_value="")
    mock_locator.get_attribute = AsyncMock(return_value="https://example.com")
    mock_locator.wait_for = AsyncMock()
    mock_locator.bounding_box = AsyncMock(
        return_value={"x": 100, "y": 200, "width": 50, "height": 30}
    )
    mock_locator.evaluate = AsyncMock(return_value={"tag": "button", "id": "btn1"})

    page.locator = MagicMock(return_value=mock_locator)
    page._mock_locator = mock_locator  # expose for assertions

    return page


@pytest.fixture
def mock_page_controller(mock_playwright_page):
    """Mock PageController wrapping a mock Playwright page."""
    pc = MagicMock()
    pc.page = mock_playwright_page
    pc.goto = AsyncMock()
    pc.navigate = AsyncMock()
    pc.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")
    pc.get_page_state = AsyncMock(
        return_value={"url": "https://example.com", "title": "Example Page"}
    )
    pc.get_rich_state = AsyncMock(
        return_value={
            "url": "https://example.com",
            "title": "Example Page",
            "viewport": {"width": 1280, "height": 720},
            "scrollPosition": {"x": 0, "y": 0},
            "links": [],
            "buttons": [],
            "forms": [],
            "inputs": [],
            "hiddenLinks": [],
            "contentSections": [],
        }
    )
    pc.click_and_track = AsyncMock(
        return_value={"success": True, "navigated": False}
    )
    pc.type_and_track = AsyncMock(
        return_value={"success": True, "navigated": False}
    )
    pc.hover_and_track = AsyncMock(return_value={"success": True})
    pc.focus_and_track = AsyncMock(return_value={"success": True})
    pc.press_key_and_track = AsyncMock(return_value={"success": True})
    pc.scroll_page = AsyncMock()
    pc.get_html = AsyncMock(return_value="<html><body>Hello</body></html>")
    pc.get_title = AsyncMock(return_value="Example Page")
    pc.get_url = AsyncMock(return_value="https://example.com")
    pc.wait_for_selector = AsyncMock()
    return pc
