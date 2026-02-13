# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for the three-layer rate limit retry architecture.

Layer 1: SDK built-in (not tested here — handled by anthropic/openai SDKs)
Layer 2: FrameworkLLMAdapter._execute_with_rate_limit_retry()
Layer 3: FireflyAgent.run() retry loop driven by RetryMiddleware
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fireflyframework_genai.agents.builtin_middleware import RetryMiddleware
from fireflyframework_genai.agents.middleware import MiddlewareContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRateLimitError(Exception):
    """Simulates an LLM provider 429 error."""

    def __init__(self, msg: str = "rate limit exceeded", status_code: int = 429):
        super().__init__(msg)
        self.status_code = status_code


class FakeRunResult:
    """Minimal stand-in for a pydantic-ai RunResult."""

    def __init__(self, output: str = "ok"):
        self.output = output
        self.data = output  # older pydantic-ai compat

    def usage(self):
        return MagicMock(total_tokens=10, request_tokens=5, response_tokens=5)


# ===========================================================================
# 1. RetryMiddleware unit tests
# ===========================================================================


class TestRetryMiddleware:
    """Verify RetryMiddleware stores config in MiddlewareContext.metadata."""

    @pytest.mark.asyncio
    async def test_before_run_stores_config(self):
        mw = RetryMiddleware(max_retries=5, base_delay=1.5, max_delay=90.0, backoff_multiplier=3.0)
        ctx = MiddlewareContext(agent_name="test", prompt="hello")

        await mw.before_run(ctx)

        cfg = ctx.metadata["_retry_config"]
        assert cfg["max_retries"] == 5
        assert cfg["base_delay"] == 1.5
        assert cfg["max_delay"] == 90.0
        assert cfg["backoff_multiplier"] == 3.0

    @pytest.mark.asyncio
    async def test_after_run_passthrough(self):
        mw = RetryMiddleware()
        ctx = MiddlewareContext(agent_name="test", prompt="hello")
        sentinel = object()

        result = await mw.after_run(ctx, sentinel)
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_defaults(self):
        mw = RetryMiddleware()
        ctx = MiddlewareContext(agent_name="test", prompt="hello")

        await mw.before_run(ctx)

        cfg = ctx.metadata["_retry_config"]
        assert cfg["max_retries"] == 3
        assert cfg["base_delay"] == 2.0
        assert cfg["max_delay"] == 120.0
        assert cfg["backoff_multiplier"] == 2.0


# ===========================================================================
# 2. _is_rate_limit_error detection tests
# ===========================================================================


class TestIsRateLimitError:
    """Verify FireflyAgent._is_rate_limit_error detects 429s from any provider."""

    @staticmethod
    def _check(exc: Exception) -> bool:
        from fireflyframework_genai.agents.base import FireflyAgent

        return FireflyAgent._is_rate_limit_error(exc)

    def test_framework_rate_limit_error(self):
        from fireflyframework_genai.exceptions import RateLimitError

        assert self._check(RateLimitError("too many requests"))

    def test_status_code_429(self):
        assert self._check(FakeRateLimitError("overloaded", status_code=429))

    def test_status_code_non_429(self):
        exc = FakeRateLimitError("server error", status_code=500)
        # Message doesn't contain "rate limit" and status isn't 429
        assert not self._check(exc)

    def test_string_pattern_rate_limit(self):
        assert self._check(Exception("Error: rate limit exceeded"))

    def test_string_pattern_429(self):
        assert self._check(Exception("HTTP 429 Too Many Requests"))

    def test_unrelated_error(self):
        assert not self._check(ValueError("invalid input"))

    def test_generic_timeout_not_matched(self):
        assert not self._check(TimeoutError("connection timed out"))


# ===========================================================================
# 3. FireflyAgent.run() retry integration
# ===========================================================================


class TestFireflyAgentRunRetry:
    """Verify the retry loop in FireflyAgent.run()."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        """Agent.run() retries when rate-limited and RetryMiddleware is configured."""
        from fireflyframework_genai.agents.base import FireflyAgent

        agent = FireflyAgent(
            "test-retry",
            model="openai:gpt-4o",
            middleware=[RetryMiddleware(max_retries=2, base_delay=0.01, max_delay=0.05)],
            auto_register=False,
        )

        # Make the inner pydantic-ai agent.run() fail twice then succeed
        call_count = 0

        async def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise FakeRateLimitError("rate limit exceeded")
            return FakeRunResult("success")

        agent._agent = MagicMock()
        agent._agent.run = fake_run

        result = await agent.run("test prompt")

        assert call_count == 3  # 2 failures + 1 success
        assert result.output == "success"

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self):
        """Agent.run() raises when all retries are exhausted."""
        from fireflyframework_genai.agents.base import FireflyAgent

        agent = FireflyAgent(
            "test-exhaust",
            model="openai:gpt-4o",
            middleware=[RetryMiddleware(max_retries=2, base_delay=0.01, max_delay=0.05)],
            auto_register=False,
        )

        async def always_fail(*args, **kwargs):
            raise FakeRateLimitError("rate limit exceeded")

        agent._agent = MagicMock()
        agent._agent.run = always_fail

        with pytest.raises(FakeRateLimitError, match="rate limit"):
            await agent.run("test prompt")

    @pytest.mark.asyncio
    async def test_no_retry_when_config_retries_zero(self):
        """With rate_limit_max_retries=0, rate limit errors propagate immediately."""
        from fireflyframework_genai.agents.base import FireflyAgent

        agent = FireflyAgent(
            "test-no-retry",
            model="openai:gpt-4o",
            middleware=[],
            default_middleware=False,
            auto_register=False,
        )

        call_count = 0

        async def fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise FakeRateLimitError("rate limit exceeded")

        agent._agent = MagicMock()
        agent._agent.run = fail_once

        with patch("fireflyframework_genai.agents.base.get_config") as mock_cfg:
            cfg = mock_cfg.return_value
            cfg.rate_limit_max_retries = 0
            cfg.rate_limit_base_delay = 1.0
            cfg.rate_limit_max_delay = 60.0

            with pytest.raises(FakeRateLimitError):
                await agent.run("test prompt")

        assert call_count == 1  # No retry when max_retries=0

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_not_retried(self):
        """Non-rate-limit errors are not retried even with RetryMiddleware."""
        from fireflyframework_genai.agents.base import FireflyAgent

        agent = FireflyAgent(
            "test-no-retry-other",
            model="openai:gpt-4o",
            middleware=[RetryMiddleware(max_retries=3, base_delay=0.01)],
            auto_register=False,
        )

        call_count = 0

        async def fail_with_value_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        agent._agent = MagicMock()
        agent._agent.run = fail_with_value_error

        with pytest.raises(ValueError, match="bad input"):
            await agent.run("test prompt")

        assert call_count == 1


# ===========================================================================
# 4. FrameworkLLMAdapter retry integration
# ===========================================================================


class TestFrameworkLLMAdapterRetry:
    """Verify FrameworkLLMAdapter.generate() retries on 429."""

    @pytest.mark.asyncio
    async def test_generate_retries_on_rate_limit(self):
        from flybrowser.llm.framework_adapter import FrameworkLLMAdapter

        adapter = FrameworkLLMAdapter("openai:gpt-4o", api_key="test-key")

        call_count = 0

        async def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise FakeRateLimitError("rate limit exceeded")
            return FakeRunResult("hello")

        with patch("flybrowser.llm.framework_adapter.PydanticAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = fake_run
            MockAgent.return_value = mock_agent_instance

            # Override retry delays for fast test
            response = await adapter._execute_with_rate_limit_retry(
                lambda: mock_agent_instance.run("test"),
                max_retries=2,
                base_delay=0.01,
                max_delay=0.05,
            )

        assert call_count == 2
        assert response.output == "hello"


# ===========================================================================
# 5. BrowserAgentConfig has retry fields
# ===========================================================================


class TestBrowserAgentConfig:
    """Verify BrowserAgentConfig exposes retry settings."""

    def test_default_retry_config(self):
        from flybrowser.agents.browser_agent import BrowserAgentConfig

        config = BrowserAgentConfig()
        assert config.max_retries == 3
        assert config.retry_base_delay == 2.0

    def test_custom_retry_config(self):
        from flybrowser.agents.browser_agent import BrowserAgentConfig

        config = BrowserAgentConfig(max_retries=5, retry_base_delay=1.0)
        assert config.max_retries == 5
        assert config.retry_base_delay == 1.0
