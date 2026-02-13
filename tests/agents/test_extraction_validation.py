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

"""Tests for extraction validation using OutputReviewer."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig


class ProductInfo(BaseModel):
    """Sample Pydantic schema for testing."""

    name: str
    price: float
    currency: str


class TestExtractAcceptsPydanticSchema:
    """Verify the extract method signature accepts Pydantic models and max_retries."""

    def test_extract_accepts_pydantic_schema(self, mock_page_controller):
        """The extract method should accept a Pydantic model as the schema parameter."""
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        sig = inspect.signature(agent.extract)
        params = sig.parameters

        assert "schema" in params, "extract must have a 'schema' parameter"
        assert "max_retries" in params, "extract must have a 'max_retries' parameter"
        assert params["max_retries"].default == 3, "max_retries default should be 3"


class TestExtractWithSchemaUsesReviewer:
    """Verify that when a schema is provided, OutputReviewer is used."""

    @pytest.mark.asyncio
    async def test_extract_with_schema_uses_reviewer(self, mock_page_controller):
        """When schema is provided, extract should use OutputReviewer.review()."""
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )

        mock_review_result = MagicMock()
        mock_review_result.output = ProductInfo(
            name="Widget", price=9.99, currency="USD"
        )
        mock_review_result.attempts = 1
        mock_review_result.retry_history = []

        mock_reviewer_instance = AsyncMock()
        mock_reviewer_instance.review = AsyncMock(return_value=mock_review_result)

        with patch(
            "flybrowser.agents.browser_agent.OutputReviewer",
            return_value=mock_reviewer_instance,
        ) as mock_reviewer_cls:
            result = await agent.extract(
                query="Get product info",
                schema=ProductInfo,
                max_retries=2,
            )

            # OutputReviewer should have been instantiated with the schema and max_retries
            mock_reviewer_cls.assert_called_once_with(
                output_type=ProductInfo, max_retries=2
            )

            # review() should have been called with the agent and the prompt
            mock_reviewer_instance.review.assert_awaited_once()
            call_args = mock_reviewer_instance.review.call_args
            assert call_args[0][0] is agent._agent  # first arg is the agent
            assert "Get product info" in call_args[0][1]  # second arg is the prompt

        # The result should be formatted through _format_result
        # Since mock_review_result.output is a Pydantic model (not dict),
        # _format_result wraps it in the standard dict
        assert result["success"] is True
        assert result["task"] == "Get product info"

    @pytest.mark.asyncio
    async def test_extract_with_schema_default_retries(self, mock_page_controller):
        """When schema is provided without explicit max_retries, default of 3 is used."""
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )

        mock_review_result = MagicMock()
        mock_review_result.output = ProductInfo(
            name="Gadget", price=19.99, currency="EUR"
        )

        mock_reviewer_instance = AsyncMock()
        mock_reviewer_instance.review = AsyncMock(return_value=mock_review_result)

        with patch(
            "flybrowser.agents.browser_agent.OutputReviewer",
            return_value=mock_reviewer_instance,
        ) as mock_reviewer_cls:
            await agent.extract(query="Get product info", schema=ProductInfo)

            # Default max_retries should be 3
            mock_reviewer_cls.assert_called_once_with(
                output_type=ProductInfo, max_retries=3
            )


class TestExtractWithoutSchemaUsesDirectRun:
    """Verify that when no schema is provided, the agent runs directly (no reviewer)."""

    @pytest.mark.asyncio
    async def test_extract_without_schema_uses_direct_run(self, mock_page_controller):
        """Without a schema, extract should call agent.run() directly, not OutputReviewer."""
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )

        # Mock the underlying agent's run method
        agent._agent.run = AsyncMock(return_value="extracted text data")

        with patch(
            "flybrowser.agents.browser_agent.OutputReviewer"
        ) as mock_reviewer_cls:
            result = await agent.extract(query="Get page title")

            # OutputReviewer should NOT have been instantiated
            mock_reviewer_cls.assert_not_called()

        # agent.run should have been called directly
        agent._agent.run.assert_awaited_once()
        call_args = agent._agent.run.call_args
        assert "Get page title" in call_args[0][0]

        # Result should be the standard format
        assert result["success"] is True
        assert result["result"] == "extracted text data"
        assert result["task"] == "Get page title"
