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

"""Tests for InteractionToolKit."""

import json

import pytest

from flybrowser.agents.toolkits.interaction import create_interaction_toolkit


EXPECTED_TOOL_NAMES = {
    "click",
    "type_text",
    "scroll",
    "hover",
    "press_key",
    "fill",
    "select_option",
    "check_checkbox",
    "focus",
    "wait_for_selector",
    "double_click",
    "right_click",
    "drag_and_drop",
    "upload_file",
    "evaluate_javascript",
    "get_attribute",
    "clear_input",
}


class TestInteractionToolKit:
    def test_toolkit_has_seventeen_tools(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        assert len(toolkit.tools) == 17

    def test_toolkit_name(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        assert toolkit.name == "interaction"

    def test_tool_names(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        names = {t.name for t in toolkit.tools}
        assert names == EXPECTED_TOOL_NAMES

    @pytest.mark.asyncio
    async def test_click_calls_click_and_track(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "click")
        result = await tool.execute(selector="#btn")
        mock_page_controller.click_and_track.assert_called_once_with("#btn")
        assert "Clicked element: #btn" in result

    @pytest.mark.asyncio
    async def test_type_text_calls_type_and_track(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "type_text")
        result = await tool.execute(selector="#input", text="hello")
        mock_page_controller.type_and_track.assert_called_once_with(
            "#input", "hello", clear_first=True, press_enter=False
        )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_type_text_with_options(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "type_text")
        await tool.execute(
            selector="#input", text="hello", clear_first=False, press_enter=True
        )
        mock_page_controller.type_and_track.assert_called_once_with(
            "#input", "hello", clear_first=False, press_enter=True
        )

    @pytest.mark.asyncio
    async def test_scroll_calls_page_evaluate(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "scroll")
        result = await tool.execute(direction="down", amount=300)
        mock_page_controller.page.evaluate.assert_called_once()
        assert "down" in result.lower()

    @pytest.mark.asyncio
    async def test_scroll_top(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "scroll")
        result = await tool.execute(direction="top")
        mock_page_controller.page.evaluate.assert_called_once()
        assert "top" in result.lower()

    @pytest.mark.asyncio
    async def test_hover_calls_locator_hover(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "hover")
        result = await tool.execute(selector="#link")
        mock_page_controller.page.locator.assert_called_with("#link")
        mock_page_controller.page._mock_locator.hover.assert_called_once()
        assert "#link" in result

    @pytest.mark.asyncio
    async def test_press_key_calls_keyboard_press(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "press_key")
        result = await tool.execute(key="Enter")
        mock_page_controller.page.keyboard.press.assert_called_once_with("Enter")
        assert "Enter" in result

    @pytest.mark.asyncio
    async def test_fill_calls_locator_fill(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "fill")
        result = await tool.execute(selector="#name", value="Alice")
        mock_page_controller.page.locator.assert_called_with("#name")
        mock_page_controller.page._mock_locator.fill.assert_called_once_with("Alice")
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_select_option_calls_locator_select_option(
        self, mock_page_controller
    ):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "select_option")
        result = await tool.execute(selector="select#color", option="Red")
        mock_page_controller.page.locator.assert_called_with("select#color")
        mock_page_controller.page._mock_locator.select_option.assert_called_once_with(
            label="Red"
        )
        assert "Red" in result

    @pytest.mark.asyncio
    async def test_select_option_fallback_to_value(self, mock_page_controller):
        """When select_option(label=...) fails, fall back to value=."""
        mock_page_controller.page._mock_locator.select_option.side_effect = [
            Exception("Label not found"),
            ["opt1"],
        ]
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "select_option")
        result = await tool.execute(selector="select#color", option="red")
        assert mock_page_controller.page._mock_locator.select_option.call_count == 2
        assert "red" in result

    @pytest.mark.asyncio
    async def test_check_checkbox_calls_locator_check(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "check_checkbox")
        result = await tool.execute(selector="#agree")
        mock_page_controller.page.locator.assert_called_with("#agree")
        mock_page_controller.page._mock_locator.check.assert_called_once()
        assert "Checked" in result

    @pytest.mark.asyncio
    async def test_check_checkbox_uncheck(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "check_checkbox")
        result = await tool.execute(selector="#agree", checked=False)
        mock_page_controller.page._mock_locator.uncheck.assert_called_once()
        assert "Unchecked" in result

    @pytest.mark.asyncio
    async def test_focus_calls_locator_focus(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "focus")
        result = await tool.execute(selector="#input")
        mock_page_controller.page.locator.assert_called_with("#input")
        mock_page_controller.page._mock_locator.focus.assert_called_once()
        assert "#input" in result

    @pytest.mark.asyncio
    async def test_wait_for_selector_calls_locator_wait_for(
        self, mock_page_controller
    ):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "wait_for_selector")
        result = await tool.execute(selector="#modal", state="visible", timeout=5000)
        mock_page_controller.page.locator.assert_called_with("#modal")
        mock_page_controller.page._mock_locator.wait_for.assert_called_once_with(
            state="visible", timeout=5000
        )
        assert "#modal" in result

    @pytest.mark.asyncio
    async def test_double_click_calls_locator_dblclick(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "double_click")
        result = await tool.execute(selector="#item")
        mock_page_controller.page.locator.assert_called_with("#item")
        mock_page_controller.page._mock_locator.dblclick.assert_called_once()
        assert "#item" in result

    @pytest.mark.asyncio
    async def test_right_click_calls_locator_click_right(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "right_click")
        result = await tool.execute(selector="#item")
        mock_page_controller.page.locator.assert_called_with("#item")
        mock_page_controller.page._mock_locator.click.assert_called_once_with(
            button="right"
        )
        assert "#item" in result

    @pytest.mark.asyncio
    async def test_drag_and_drop(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "drag_and_drop")
        result = await tool.execute(source="#src", target="#tgt")
        mock_page_controller.page._mock_locator.drag_to.assert_called_once()
        assert "#src" in result
        assert "#tgt" in result

    @pytest.mark.asyncio
    async def test_upload_file(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "upload_file")
        result = await tool.execute(selector="#file", file_path="/tmp/test.txt")
        mock_page_controller.page.locator.assert_called_with("#file")
        mock_page_controller.page._mock_locator.set_input_files.assert_called_once_with(
            "/tmp/test.txt"
        )
        assert "/tmp/test.txt" in result

    @pytest.mark.asyncio
    async def test_evaluate_javascript_calls_page_evaluate(
        self, mock_page_controller
    ):
        mock_page_controller.page.evaluate.return_value = {"key": "value"}
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "evaluate_javascript")
        result = await tool.execute(script="document.title")
        mock_page_controller.page.evaluate.assert_called_once_with("document.title")
        assert "JavaScript result:" in result
        parsed = json.loads(result.split("JavaScript result: ")[1])
        assert parsed == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_attribute(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "get_attribute")
        result = await tool.execute(selector="a#link", attribute="href")
        mock_page_controller.page.locator.assert_called_with("a#link")
        mock_page_controller.page._mock_locator.get_attribute.assert_called_once_with(
            "href"
        )
        assert "href" in result

    @pytest.mark.asyncio
    async def test_clear_input_calls_locator_clear(self, mock_page_controller):
        toolkit = create_interaction_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "clear_input")
        result = await tool.execute(selector="#input")
        mock_page_controller.page.locator.assert_called_with("#input")
        mock_page_controller.page._mock_locator.clear.assert_called_once()
        assert "#input" in result
