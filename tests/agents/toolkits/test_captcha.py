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

"""Tests for CaptchaToolKit."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from flybrowser.agents.toolkits.captcha import create_captcha_toolkit


@pytest.fixture
def mock_captcha_solver():
    solver = MagicMock()
    solver.detect = AsyncMock(return_value={"detected": True, "type": "recaptcha_v2"})
    solver.solve = AsyncMock(return_value={"solved": True})
    solver.wait_resolved = AsyncMock(return_value=True)
    return solver


class TestCaptchaToolKit:
    def test_toolkit_has_three_tools(self, mock_page_controller):
        toolkit = create_captcha_toolkit(mock_page_controller)
        assert len(toolkit.tools) == 3

    def test_tool_names(self, mock_page_controller):
        toolkit = create_captcha_toolkit(mock_page_controller)
        names = {t.name for t in toolkit.tools}
        assert names == {"detect_captcha", "solve_captcha", "wait_captcha_resolved"}

    @pytest.mark.asyncio
    async def test_detect_captcha(self, mock_page_controller, mock_captcha_solver):
        toolkit = create_captcha_toolkit(mock_page_controller, captcha_solver=mock_captcha_solver)
        tool = next(t for t in toolkit.tools if t.name == "detect_captcha")
        result = await tool.execute()
        mock_captcha_solver.detect.assert_called_once()
        assert "recaptcha_v2" in result

    @pytest.mark.asyncio
    async def test_solve_captcha(self, mock_page_controller, mock_captcha_solver):
        toolkit = create_captcha_toolkit(mock_page_controller, captcha_solver=mock_captcha_solver)
        tool = next(t for t in toolkit.tools if t.name == "solve_captcha")
        result = await tool.execute()
        mock_captcha_solver.solve.assert_called_once_with(captcha_type="auto")
        assert "solved" in result.lower() or "Solved" in result
