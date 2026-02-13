# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlyBrowser pipeline execution CLI commands."""

import argparse
import json
import textwrap
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


class TestWorkflowParsing:
    """Tests for workflow parsing from YAML files and inline commands."""

    def test_parse_yaml_workflow(self, tmp_path):
        """Test parsing a YAML workflow file into a workflow dict."""
        from flybrowser.cli.pipeline import parse_workflow

        workflow_yaml = textwrap.dedent("""\
            name: my-workflow
            sessions:
              main:
                provider: openai
                model: gpt-4o
                headless: true
            steps:
              - name: navigate
                session: main
                action: goto
                url: https://example.com
              - name: extract
                session: main
                action: extract
                query: "Get all prices"
        """)

        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(workflow_yaml)

        workflow = parse_workflow(path=str(workflow_file))

        assert workflow["name"] == "my-workflow"
        assert "sessions" in workflow
        assert "main" in workflow["sessions"]
        assert workflow["sessions"]["main"]["provider"] == "openai"
        assert workflow["sessions"]["main"]["model"] == "gpt-4o"
        assert workflow["sessions"]["main"]["headless"] is True
        assert len(workflow["steps"]) == 2
        assert workflow["steps"][0]["action"] == "goto"
        assert workflow["steps"][0]["url"] == "https://example.com"
        assert workflow["steps"][1]["action"] == "extract"
        assert workflow["steps"][1]["query"] == "Get all prices"

    def test_parse_inline_commands(self):
        """Test parsing inline commands into a workflow dict."""
        from flybrowser.cli.pipeline import parse_workflow

        inline = "goto https://example.com && extract 'get prices'"

        workflow = parse_workflow(inline_commands=inline)

        assert "steps" in workflow
        assert len(workflow["steps"]) == 2

        # First step: goto
        assert workflow["steps"][0]["action"] == "goto"
        assert workflow["steps"][0]["url"] == "https://example.com"

        # Second step: extract
        assert workflow["steps"][1]["action"] == "extract"
        assert workflow["steps"][1]["query"] == "get prices"

    def test_parse_yaml_workflow_file_not_found(self):
        """Test that parsing a nonexistent YAML file raises FileNotFoundError."""
        from flybrowser.cli.pipeline import parse_workflow

        with pytest.raises(FileNotFoundError):
            parse_workflow(path="/nonexistent/workflow.yaml")

    def test_parse_no_input_raises_error(self):
        """Test that calling parse_workflow with no arguments raises ValueError."""
        from flybrowser.cli.pipeline import parse_workflow

        with pytest.raises(ValueError):
            parse_workflow()

    def test_parse_inline_single_command(self):
        """Test parsing a single inline command (no &&)."""
        from flybrowser.cli.pipeline import parse_workflow

        workflow = parse_workflow(inline_commands="goto https://example.com")

        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["action"] == "goto"
        assert workflow["steps"][0]["url"] == "https://example.com"

    def test_parse_inline_act_command(self):
        """Test parsing an inline act command."""
        from flybrowser.cli.pipeline import parse_workflow

        workflow = parse_workflow(inline_commands="act 'click the login button'")

        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["action"] == "act"
        assert workflow["steps"][0]["instruction"] == "click the login button"

    def test_parse_inline_screenshot_command(self):
        """Test parsing an inline screenshot command."""
        from flybrowser.cli.pipeline import parse_workflow

        workflow = parse_workflow(inline_commands="screenshot")

        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["action"] == "screenshot"


class TestPipelineRun:
    """Tests for pipeline run command handler."""

    def test_run_workflow_file(self, tmp_path):
        """Test running a workflow from a YAML file (mock execute_workflow)."""
        from flybrowser.cli.pipeline import cmd_run

        workflow_yaml = textwrap.dedent("""\
            name: test-workflow
            sessions:
              main:
                provider: openai
                model: gpt-4o
                headless: true
            steps:
              - name: navigate
                session: main
                action: goto
                url: https://example.com
        """)

        workflow_file = tmp_path / "test-workflow.yaml"
        workflow_file.write_text(workflow_yaml)

        args = argparse.Namespace(
            workflow=str(workflow_file),
            inline=None,
        )

        with patch(
            "flybrowser.cli.pipeline.execute_workflow", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = {"success": True, "steps_completed": 1}

            result = cmd_run(args)

            assert result == 0
            mock_execute.assert_called_once()
            # Verify the workflow dict passed to execute_workflow
            call_args = mock_execute.call_args
            workflow = call_args[0][0]
            assert workflow["name"] == "test-workflow"
            assert len(workflow["steps"]) == 1

    def test_run_inline_commands(self):
        """Test running inline commands via cmd_run."""
        from flybrowser.cli.pipeline import cmd_run

        args = argparse.Namespace(
            workflow=None,
            inline="goto https://example.com && extract 'get prices'",
        )

        with patch(
            "flybrowser.cli.pipeline.execute_workflow", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = {"success": True, "steps_completed": 2}

            result = cmd_run(args)

            assert result == 0
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            workflow = call_args[0][0]
            assert len(workflow["steps"]) == 2

    def test_run_no_input_returns_error(self, capsys):
        """Test that running without a workflow file or inline commands fails."""
        from flybrowser.cli.pipeline import cmd_run

        args = argparse.Namespace(
            workflow=None,
            inline=None,
        )

        result = cmd_run(args)

        assert result == 1

    def test_run_handles_execute_error(self, tmp_path, capsys):
        """Test that cmd_run handles execution errors gracefully."""
        from flybrowser.cli.pipeline import cmd_run

        workflow_yaml = textwrap.dedent("""\
            name: failing-workflow
            steps:
              - name: navigate
                action: goto
                url: https://example.com
        """)

        workflow_file = tmp_path / "fail.yaml"
        workflow_file.write_text(workflow_yaml)

        args = argparse.Namespace(
            workflow=str(workflow_file),
            inline=None,
        )

        with patch(
            "flybrowser.cli.pipeline.execute_workflow", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = Exception("Connection refused")

            result = cmd_run(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "error" in captured.err.lower()


class TestAddPipelineSubparser:
    """Tests for pipeline subparser registration."""

    def test_registers_run_subcommand(self):
        """Test that add_pipeline_subparser registers the 'run' subcommand."""
        from flybrowser.cli.pipeline import add_pipeline_subparser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        add_pipeline_subparser(subparsers)

        # Test parsing with a workflow file argument
        args = parser.parse_args(["run", "workflow.yaml"])
        assert args.workflow == "workflow.yaml"
        assert args.command == "run"

    def test_registers_inline_option(self):
        """Test that the --inline option is registered."""
        from flybrowser.cli.pipeline import add_pipeline_subparser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        add_pipeline_subparser(subparsers)

        args = parser.parse_args(["run", "--inline", "goto https://example.com"])
        assert args.inline == "goto https://example.com"
        assert args.workflow is None
