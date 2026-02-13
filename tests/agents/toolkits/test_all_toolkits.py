"""Tests for the all_toolkits factory."""
import pytest
from flybrowser.agents.toolkits import create_all_toolkits


class TestAllToolKits:
    def test_creates_six_toolkits(self, mock_page_controller):
        toolkits = create_all_toolkits(page=mock_page_controller)
        assert len(toolkits) == 6

    def test_toolkit_names(self, mock_page_controller):
        toolkits = create_all_toolkits(page=mock_page_controller)
        names = {tk.name for tk in toolkits}
        assert names == {"navigation", "interaction", "extraction", "system", "search", "captcha"}

    def test_total_tool_count(self, mock_page_controller):
        toolkits = create_all_toolkits(page=mock_page_controller)
        total = sum(len(tk.tools) for tk in toolkits)
        assert total == 32  # 4+17+3+4+1+3
