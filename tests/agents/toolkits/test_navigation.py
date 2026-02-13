"""Tests for NavigationToolKit."""
import pytest
from flybrowser.agents.toolkits.navigation import create_navigation_toolkit


class TestNavigationToolKit:
    def test_toolkit_has_four_tools(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        assert len(toolkit.tools) == 4

    def test_toolkit_name(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        assert toolkit.name == "navigation"

    def test_tool_names(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        names = {t.name for t in toolkit.tools}
        assert names == {"navigate", "go_back", "go_forward", "refresh"}

    @pytest.mark.asyncio
    async def test_navigate_calls_page_goto(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        nav_tool = next(t for t in toolkit.tools if t.name == "navigate")
        result = await nav_tool.execute(url="https://example.com")
        mock_page_controller.goto.assert_called_once()
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_navigate_with_wait_until(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        nav_tool = next(t for t in toolkit.tools if t.name == "navigate")
        await nav_tool.execute(url="https://test.com", wait_until="networkidle")
        mock_page_controller.goto.assert_called_once_with("https://test.com", wait_until="networkidle")

    @pytest.mark.asyncio
    async def test_go_back(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "go_back")
        result = await tool.execute()
        mock_page_controller.page.go_back.assert_called_once()
        assert "back" in result.lower()

    @pytest.mark.asyncio
    async def test_go_forward(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "go_forward")
        result = await tool.execute()
        mock_page_controller.page.go_forward.assert_called_once()
        assert "forward" in result.lower()

    @pytest.mark.asyncio
    async def test_refresh(self, mock_page_controller):
        toolkit = create_navigation_toolkit(mock_page_controller)
        tool = next(t for t in toolkit.tools if t.name == "refresh")
        result = await tool.execute()
        mock_page_controller.page.reload.assert_called_once()
        assert "refresh" in result.lower()
