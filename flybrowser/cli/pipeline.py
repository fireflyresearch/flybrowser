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

"""
FlyBrowser Pipeline Execution CLI.

Provides the ability to run multi-step browser workflows from YAML files
or inline command strings.

Usage:
    flybrowser run <workflow.yaml>
    flybrowser run --inline "goto https://example.com && extract 'get prices'"

YAML workflow format:
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

Inline format:
    "goto https://example.com && extract 'get prices'"
    Commands are split on '&&' and each is parsed as a step.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from flybrowser.cli.output import CLIOutput

# CLI output instance
_cli_output = CLIOutput()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    return asyncio.run(coro)


def _parse_inline_command(command_str: str) -> Dict[str, Any]:
    """Parse a single inline command string into a step dict.

    Supported command forms:
        goto <url>
        extract '<query>'
        act '<instruction>'
        screenshot

    Args:
        command_str: A single command string (e.g. "goto https://example.com").

    Returns:
        A step dict with at least an 'action' key.
    """
    command_str = command_str.strip()
    if not command_str:
        return {"action": "noop"}

    # Use shlex to properly handle quoted arguments
    try:
        parts = shlex.split(command_str)
    except ValueError:
        # Fallback to simple split if shlex fails (e.g. unmatched quotes)
        parts = command_str.split()

    action = parts[0].lower()
    step: Dict[str, Any] = {"action": action}

    if action == "goto" and len(parts) > 1:
        step["url"] = parts[1]
    elif action == "extract" and len(parts) > 1:
        step["query"] = " ".join(parts[1:])
    elif action == "act" and len(parts) > 1:
        step["instruction"] = " ".join(parts[1:])
    elif action == "screenshot":
        pass  # No additional args required
    elif action == "agent" and len(parts) > 1:
        step["task"] = " ".join(parts[1:])

    return step


def parse_workflow(
    path: Optional[str] = None,
    inline_commands: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a YAML workflow file or inline commands into a workflow dict.

    Exactly one of ``path`` or ``inline_commands`` must be provided.

    Args:
        path: Path to a YAML workflow file.
        inline_commands: Inline commands string separated by '&&'.

    Returns:
        A workflow dict with 'steps' (and optionally 'name', 'sessions').

    Raises:
        ValueError: If neither ``path`` nor ``inline_commands`` is given.
        FileNotFoundError: If the YAML file does not exist.
    """
    if path is None and inline_commands is None:
        raise ValueError(
            "Either a workflow file path or inline commands must be provided."
        )

    if path is not None:
        workflow_path = Path(path)
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        # Ensure essential keys exist
        if "steps" not in workflow:
            workflow["steps"] = []
        if "name" not in workflow:
            workflow["name"] = workflow_path.stem

        return workflow

    # Parse inline commands
    raw_commands = inline_commands.split("&&")
    steps: List[Dict[str, Any]] = []

    for i, cmd in enumerate(raw_commands):
        step = _parse_inline_command(cmd)
        if step.get("action") != "noop":
            step.setdefault("name", f"step-{i + 1}")
            steps.append(step)

    return {
        "name": "inline-workflow",
        "steps": steps,
    }


async def execute_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Execute workflow steps sequentially.

    For each step, creates or reuses sessions as defined in the workflow,
    then dispatches the action to the appropriate handler.

    Args:
        workflow: A workflow dict as returned by :func:`parse_workflow`.

    Returns:
        A result dict with 'success' and 'steps_completed' keys.
    """
    from flybrowser.client import FlyBrowserClient

    sessions_config = workflow.get("sessions", {})
    steps = workflow.get("steps", [])
    results: List[Dict[str, Any]] = []

    # Track active sessions: session_name -> (session_id, client)
    active_sessions: Dict[str, tuple] = {}

    endpoint = os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000")

    try:
        for step in steps:
            session_name = step.get("session", "default")
            action = step.get("action", "")

            # Ensure we have a session
            if session_name not in active_sessions:
                session_cfg = sessions_config.get(session_name, {})
                client = FlyBrowserClient(endpoint)
                await client.start()

                create_result = await client.create_session(
                    llm_provider=session_cfg.get("provider", "openai"),
                    llm_model=session_cfg.get("model"),
                    headless=session_cfg.get("headless", True),
                )
                session_id = create_result.get("session_id", "")
                active_sessions[session_name] = (session_id, client)

            session_id, client = active_sessions[session_name]

            # Dispatch action
            step_result: Dict[str, Any] = {}

            if action == "goto":
                step_result = await client.navigate(session_id, step.get("url", ""))
            elif action == "extract":
                step_result = await client.extract(
                    session_id, step.get("query", "")
                )
            elif action == "act":
                step_result = await client.action(
                    session_id, step.get("instruction", "")
                )
            elif action == "screenshot":
                step_result = await client.screenshot(
                    session_id,
                    full_page=step.get("full_page", False),
                )
            elif action == "agent":
                step_result = await client.agent(
                    session_id,
                    task=step.get("task", ""),
                    max_iterations=step.get("max_iterations", 50),
                )
            else:
                step_result = {"warning": f"Unknown action: {action}"}

            results.append(
                {
                    "step": step.get("name", action),
                    "action": action,
                    "result": step_result,
                }
            )

        return {
            "success": True,
            "steps_completed": len(results),
            "results": results,
        }

    except Exception as exc:
        return {
            "success": False,
            "steps_completed": len(results),
            "results": results,
            "error": str(exc),
        }

    finally:
        # Clean up all sessions
        for session_name, (session_id, client) in active_sessions.items():
            try:
                await client.close_session(session_id)
                await client.stop()
            except Exception:
                pass


def cmd_run(args: argparse.Namespace) -> int:
    """Handle 'flybrowser run' command.

    Parses a workflow from a YAML file or inline commands and executes it.

    Args:
        args: Parsed command-line arguments. Expected attributes:
            - workflow: Path to a YAML workflow file (or None).
            - inline: Inline command string (or None).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        workflow = parse_workflow(
            path=args.workflow,
            inline_commands=args.inline,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _cli_output.print_summary(
        "Pipeline",
        {
            "Workflow": workflow.get("name", "(unnamed)"),
            "Steps": len(workflow.get("steps", [])),
        },
    )

    try:
        result = _run_async(execute_workflow(workflow))
    except Exception as exc:
        print(f"Error: Pipeline execution failed: {exc}", file=sys.stderr)
        return 1

    if result.get("success"):
        _cli_output.print_summary(
            "Pipeline Complete",
            {
                "Steps Completed": result.get("steps_completed", 0),
                "Status": "Success",
            },
        )
        return 0
    else:
        print(
            f"Error: Pipeline failed after {result.get('steps_completed', 0)} steps: "
            f"{result.get('error', 'unknown error')}",
            file=sys.stderr,
        )
        return 1


def add_pipeline_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'run' subcommand on the CLI parser.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    run_parser = subparsers.add_parser(
        "run",
        help="Run a multi-step browser workflow",
        description=(
            "Execute a browser automation workflow from a YAML file "
            "or inline commands."
        ),
    )
    run_parser.add_argument(
        "workflow",
        nargs="?",
        default=None,
        help="Path to a YAML workflow file",
    )
    run_parser.add_argument(
        "--inline", "-i",
        default=None,
        help=(
            "Inline commands separated by '&&' "
            "(e.g. \"goto https://example.com && extract 'get prices'\")"
        ),
    )
    run_parser.set_defaults(func=cmd_run)
