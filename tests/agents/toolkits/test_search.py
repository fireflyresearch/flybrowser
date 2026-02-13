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

"""Tests for SearchToolKit."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from flybrowser.agents.toolkits.search import create_search_toolkit


@pytest.fixture
def mock_search_coordinator():
    coord = MagicMock()
    coord.search = AsyncMock(return_value={
        "results": [{"title": "Result 1", "url": "https://r1.com", "snippet": "First"}],
        "total": 1,
    })
    return coord


class TestSearchToolKit:
    def test_toolkit_has_one_tool(self, mock_search_coordinator):
        toolkit = create_search_toolkit(mock_search_coordinator)
        assert len(toolkit.tools) == 1

    @pytest.mark.asyncio
    async def test_search_calls_coordinator(self, mock_search_coordinator):
        toolkit = create_search_toolkit(mock_search_coordinator)
        tool = next(t for t in toolkit.tools if t.name == "search")
        result = await tool.execute(query="test query")
        mock_search_coordinator.search.assert_called_once_with(
            "test query", search_type="auto", max_results=10
        )
        assert "Result 1" in result

    @pytest.mark.asyncio
    async def test_search_with_no_coordinator(self):
        toolkit = create_search_toolkit(None)
        tool = next(t for t in toolkit.tools if t.name == "search")
        result = await tool.execute(query="test query")
        assert "No search provider configured" in result
