"""Tests for SDK integration with BrowserAgent."""

import pytest
from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig


class TestSDKCreatesBrowserAgent:
    def test_config_from_params(self):
        config = BrowserAgentConfig(
            model="anthropic:claude-3-5-sonnet-latest", max_iterations=30
        )
        assert config.model == "anthropic:claude-3-5-sonnet-latest"
        assert config.max_iterations == 30

    def test_agent_created_with_mock(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        assert agent is not None
        toolkit_names = {tk.name for tk in agent._toolkits}
        assert "navigation" in toolkit_names
        assert "interaction" in toolkit_names
        assert "extraction" in toolkit_names

    def test_agent_with_custom_config(self, mock_page_controller):
        config = BrowserAgentConfig(
            model="openai:gpt-4o-mini",
            max_iterations=25,
            max_time=900,
            session_id="test-session-123",
        )
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=config
        )
        assert agent._config.model == "openai:gpt-4o-mini"
        assert agent._config.max_iterations == 25
        assert agent._config.max_time == 900
        assert agent._config.session_id == "test-session-123"

    def test_agent_memory_accessible(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        memory = agent.memory
        assert memory is not None
        # Memory should start empty
        assert memory.format_for_prompt() == "No browser memory recorded yet."
