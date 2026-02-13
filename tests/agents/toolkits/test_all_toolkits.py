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
