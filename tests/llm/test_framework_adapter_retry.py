"""Unit tests for FrameworkLLMAdapter rate limit retry integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.llm.framework_adapter import FrameworkLLMAdapter


@pytest.mark.asyncio
class TestFrameworkAdapterRetry:
    """Verify FrameworkLLMAdapter delegates to _execute_with_rate_limit_retry."""

    async def test_generate_uses_retry_wrapper(self):
        """generate() should call _execute_with_rate_limit_retry."""
        adapter = FrameworkLLMAdapter("openai:gpt-4o")

        mock_result = MagicMock()
        mock_result.output = "hello"
        mock_result.usage = None
        mock_result._usage = None

        with patch.object(
            adapter,
            "_execute_with_rate_limit_retry",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_retry:
            response = await adapter.generate("test prompt")

        mock_retry.assert_called_once()
        assert response.content == "hello"

    async def test_generate_with_vision_uses_retry_wrapper(self):
        """generate_with_vision() should call _execute_with_rate_limit_retry."""
        adapter = FrameworkLLMAdapter("openai:gpt-4o")

        mock_result = MagicMock()
        mock_result.output = "I see an image"
        mock_result.usage = None
        mock_result._usage = None

        with patch.object(
            adapter,
            "_execute_with_rate_limit_retry",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_retry:
            response = await adapter.generate_with_vision(
                "describe this", images=[b"\x89PNG"]
            )

        mock_retry.assert_called_once()
        assert response.content == "I see an image"

    async def test_adapter_429_retries_via_base(self):
        """Verify that a 429 from pydantic-ai triggers retry in the base class."""
        adapter = FrameworkLLMAdapter("openai:gpt-4o")

        call_count = 0
        mock_success = MagicMock()
        mock_success.output = "success"
        mock_success.usage = None
        mock_success._usage = None

        class FakeRateLimit(Exception):
            status_code = 429

        async def fake_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeRateLimit("rate limit exceeded")
            return mock_success

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._execute_with_rate_limit_retry(
                fake_api_call, max_retries=3, base_delay=0.01
            )

        assert result is mock_success
        assert call_count == 3
