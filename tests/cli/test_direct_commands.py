# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlyBrowser direct SDK-like CLI commands."""

import argparse
import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_run_async_sequence(return_values: List[Any]):
    """Create a mock for _run_async that returns values from a sequence.

    Each successive call to _run_async returns the next value from the list.
    Coroutines passed in are properly closed to avoid RuntimeWarning.
    """
    call_index = {"i": 0}

    def side_effect(coro):
        # Close the coroutine to prevent RuntimeWarning
        coro.close()
        idx = call_index["i"]
        call_index["i"] += 1
        if idx < len(return_values):
            return return_values[idx]
        return None

    return MagicMock(side_effect=side_effect)


class TestDirectGoto:
    """Tests for the goto command."""

    def test_goto_url(self, capsys):
        """Test navigating to a URL via the goto command."""
        from flybrowser.cli.direct import cmd_goto

        args = argparse.Namespace(
            url="https://example.com",
            session=None,
            wait_for=None,
            endpoint="http://localhost:8000",
        )

        mock_client = AsyncMock()
        mock_nav_result = {
            "success": True,
            "url": "https://example.com",
            "title": "Example Domain",
        }

        # Sequence: _get_or_create_session, navigate, close_session, stop
        mock_run = _mock_run_async_sequence([
            ("sess-abc123", mock_client),
            mock_nav_result,
            None,
            None,
        ])

        with patch("flybrowser.cli.direct._run_async", mock_run):
            result = cmd_goto(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "example.com" in captured.out.lower()


class TestDirectExtract:
    """Tests for the extract command."""

    def test_extract_query(self, capsys):
        """Test extracting data from a page via the extract command."""
        from flybrowser.cli.direct import cmd_extract

        args = argparse.Namespace(
            query="Get all product names",
            session=None,
            schema=None,
            format="json",
            endpoint="http://localhost:8000",
        )

        mock_client = AsyncMock()
        mock_extract_result = {
            "success": True,
            "data": ["Product A", "Product B", "Product C"],
        }

        # Sequence: _get_or_create_session, extract, close_session, stop
        mock_run = _mock_run_async_sequence([
            ("sess-abc123", mock_client),
            mock_extract_result,
            None,
            None,
        ])

        with patch("flybrowser.cli.direct._run_async", mock_run):
            result = cmd_extract(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "Product A" in captured.out


class TestDirectAct:
    """Tests for the act command."""

    def test_act_instruction(self, capsys):
        """Test performing an action via the act command."""
        from flybrowser.cli.direct import cmd_act

        args = argparse.Namespace(
            instruction="Click the login button",
            session=None,
            endpoint="http://localhost:8000",
        )

        mock_client = AsyncMock()
        mock_act_result = {
            "success": True,
            "action": "click",
            "element": "login button",
        }

        # Sequence: _get_or_create_session, action, close_session, stop
        mock_run = _mock_run_async_sequence([
            ("sess-abc123", mock_client),
            mock_act_result,
            None,
            None,
        ])

        with patch("flybrowser.cli.direct._run_async", mock_run):
            result = cmd_act(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "click" in captured.out.lower()


class TestDirectScreenshot:
    """Tests for the screenshot command."""

    def test_screenshot(self, capsys, tmp_path):
        """Test taking a screenshot via the screenshot command."""
        from flybrowser.cli.direct import cmd_screenshot

        output_file = str(tmp_path / "screenshot.png")

        args = argparse.Namespace(
            session=None,
            output=output_file,
            full_page=False,
            endpoint="http://localhost:8000",
        )

        import base64

        mock_client = AsyncMock()
        mock_screenshot_result = {
            "success": True,
            "screenshot": base64.b64encode(b"PNG_FAKE_DATA").decode(),
            "format": "png",
        }

        # Sequence: _get_or_create_session, screenshot, close_session, stop
        mock_run = _mock_run_async_sequence([
            ("sess-abc123", mock_client),
            mock_screenshot_result,
            None,
            None,
        ])

        with patch("flybrowser.cli.direct._run_async", mock_run):
            result = cmd_screenshot(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "screenshot" in captured.out.lower()


class TestDirectAgent:
    """Tests for the agent command."""

    def test_agent_task(self, capsys):
        """Test running an agent task via the agent command."""
        from flybrowser.cli.direct import cmd_agent

        args = argparse.Namespace(
            task="Find the cheapest flight from NYC to LA",
            session=None,
            max_iterations=50,
            stream=False,
            endpoint="http://localhost:8000",
        )

        mock_client = AsyncMock()
        mock_agent_result = {
            "success": True,
            "result_data": {"cheapest_flight": "$199"},
            "iterations": 12,
            "duration_seconds": 45.2,
        }

        # Sequence: _get_or_create_session, agent, close_session, stop
        mock_run = _mock_run_async_sequence([
            ("sess-abc123", mock_client),
            mock_agent_result,
            None,
            None,
        ])

        with patch("flybrowser.cli.direct._run_async", mock_run):
            result = cmd_agent(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "$199" in captured.out


class TestAddDirectSubparsers:
    """Tests for subparser registration."""

    def test_registers_all_subcommands(self):
        """Test that add_direct_subparsers registers goto, extract, act, screenshot, agent."""
        from flybrowser.cli.direct import add_direct_subparsers

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        add_direct_subparsers(subparsers)

        # Verify that the subcommands are registered by parsing known commands
        for cmd in ["goto", "extract", "act", "screenshot", "agent"]:
            # Each command should be parseable without error
            # We need to pass required args
            if cmd == "goto":
                args = parser.parse_args([cmd, "https://example.com"])
                assert args.url == "https://example.com"
            elif cmd == "extract":
                args = parser.parse_args([cmd, "Get the title"])
                assert args.query == "Get the title"
            elif cmd == "act":
                args = parser.parse_args([cmd, "Click button"])
                assert args.instruction == "Click button"
            elif cmd == "screenshot":
                args = parser.parse_args([cmd])
                assert hasattr(args, "output")
            elif cmd == "agent":
                args = parser.parse_args([cmd, "Do something"])
                assert args.task == "Do something"
