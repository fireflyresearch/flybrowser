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

"""Tests for ExtractionToolKit."""

import json

import pytest

from flybrowser.agents.toolkits.extraction import create_extraction_toolkit


class TestExtractionToolKit:
    def test_toolkit_has_three_tools(self, mock_page_controller):
        toolkit = create_extraction_toolkit(mock_page_controller)
        assert len(toolkit.tools) == 3

    def test_tool_names(self, mock_page_controller):
        toolkit = create_extraction_toolkit(mock_page_controller)
        names = {t.name for t in toolkit.tools}
        assert names == {"extract_text", "screenshot", "get_page_state"}

    @pytest.mark.asyncio
    async def test_extract_text_full_page(self, mock_page_controller):
        mock_page_controller.page.evaluate.return_value = {
            "title": "Example Page",
            "url": "https://example.com",
            "headings": ["Welcome"],
            "navLinks": [{"text": "Home", "href": "/"}],
            "mainContent": "Hello world",
            "visibleText": "Hello world full",
        }
        toolkit = create_extraction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "extract_text")
        result = await tool.execute()
        mock_page_controller.page.evaluate.assert_called_once()
        assert "Example Page" in result
        assert "https://example.com" in result

    @pytest.mark.asyncio
    async def test_extract_text_with_selector(self, mock_page_controller):
        toolkit = create_extraction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "extract_text")
        result = await tool.execute(selector="#content")
        mock_page_controller.page.locator.assert_called_with("#content")
        assert "Button Text" in result

    @pytest.mark.asyncio
    async def test_screenshot(self, mock_page_controller):
        toolkit = create_extraction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "screenshot")
        result = await tool.execute()
        mock_page_controller.screenshot.assert_called_once_with(full_page=False)
        assert "Screenshot captured" in result

    @pytest.mark.asyncio
    async def test_get_page_state(self, mock_page_controller):
        toolkit = create_extraction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "get_page_state")
        result = await tool.execute()
        mock_page_controller.get_rich_state.assert_called_once()
        # Result should be valid JSON-like summary
        assert "example.com" in result
