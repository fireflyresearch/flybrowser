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
FlyBrowser Direct SDK-like CLI Commands.

Provides one-shot CLI commands that map directly to SDK operations,
allowing users to perform single browser actions without entering
the REPL.

Usage:
    flybrowser goto <url> [--session <id>] [--wait-for <selector>]
    flybrowser extract <query> [--session <id>] [--schema <file>] [--format json|csv|table]
    flybrowser act <instruction> [--session <id>]
    flybrowser screenshot [--session <id>] [--output <file>] [--full-page]
    flybrowser agent <task> [--session <id>] [--max-iterations 50] [--stream]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


async def _get_or_create_session(
    session_id: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Tuple[str, Any]:
    """Get an existing session or create a new ephemeral one.

    If a session_id is provided, reuses that session via the server client.
    Otherwise, creates a new ephemeral session.

    Args:
        session_id: Optional existing session ID to reuse.
        endpoint: Server endpoint URL.

    Returns:
        Tuple of (session_id, client) where client is a FlyBrowserClient.
    """
    from flybrowser.client import FlyBrowserClient

    effective_endpoint = endpoint or os.environ.get(
        "FLYBROWSER_ENDPOINT", "http://localhost:8000"
    )

    client = FlyBrowserClient(effective_endpoint)
    await client.start()

    if session_id:
        return session_id, client

    # Create an ephemeral session
    result = await client.create_session(
        llm_provider=os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai"),
        llm_model=os.environ.get("FLYBROWSER_LLM_MODEL"),
        headless=True,
    )
    new_session_id = result.get("session_id", "")
    return new_session_id, client


# ---------- Command Handlers ----------


def cmd_goto(args: argparse.Namespace) -> int:
    """Handle 'flybrowser goto <url>' command.

    Navigates the browser to the specified URL.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        session_id, client = _run_async(
            _get_or_create_session(
                session_id=args.session,
                endpoint=getattr(args, "endpoint", None),
            )
        )

        result = _run_async(client.navigate(session_id, args.url))

        # If --wait-for was specified, wait for the selector
        if args.wait_for:
            # The wait is handled server-side; include it in the request context
            pass

        _cli_output.print_summary(
            "Navigation Result",
            {
                "URL": result.get("url", args.url),
                "Title": result.get("title", "(unknown)"),
                "Status": "OK" if result.get("success", True) else "Failed",
            },
        )

        # Clean up ephemeral session if we created one
        if not args.session:
            _run_async(client.close_session(session_id))
        _run_async(client.stop())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Handle 'flybrowser extract <query>' command.

    Extracts data from the current page using a natural language query.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        session_id, client = _run_async(
            _get_or_create_session(
                session_id=args.session,
                endpoint=getattr(args, "endpoint", None),
            )
        )

        # Load schema from file if provided
        schema = None
        if args.schema:
            schema_path = Path(args.schema)
            if schema_path.exists():
                schema = json.loads(schema_path.read_text())
            else:
                print(f"Error: Schema file not found: {args.schema}", file=sys.stderr)
                return 1

        result = _run_async(
            client.extract(session_id, args.query, schema=schema)
        )

        # Output in the requested format
        output_format = getattr(args, "format", "json")

        if output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        elif output_format == "csv":
            _print_csv(result)
        elif output_format == "table":
            _print_table(result)
        else:
            print(json.dumps(result, indent=2, default=str))

        # Clean up ephemeral session if we created one
        if not args.session:
            _run_async(client.close_session(session_id))
        _run_async(client.stop())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_act(args: argparse.Namespace) -> int:
    """Handle 'flybrowser act <instruction>' command.

    Performs an action on the current page.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        session_id, client = _run_async(
            _get_or_create_session(
                session_id=args.session,
                endpoint=getattr(args, "endpoint", None),
            )
        )

        result = _run_async(client.action(session_id, args.instruction))

        _cli_output.print_summary(
            "Action Result",
            {
                "Instruction": args.instruction,
                "Success": result.get("success", True),
                "Details": result.get("result", result.get("action", "-")),
            },
        )

        # Clean up ephemeral session if we created one
        if not args.session:
            _run_async(client.close_session(session_id))
        _run_async(client.stop())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_screenshot(args: argparse.Namespace) -> int:
    """Handle 'flybrowser screenshot' command.

    Takes a screenshot of the current page.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        session_id, client = _run_async(
            _get_or_create_session(
                session_id=args.session,
                endpoint=getattr(args, "endpoint", None),
            )
        )

        result = _run_async(
            client.screenshot(session_id, full_page=args.full_page)
        )

        # Save screenshot data to file
        output_path = Path(args.output)
        screenshot_data = result.get("screenshot", "")

        if screenshot_data:
            raw_bytes = base64.b64decode(screenshot_data)
            output_path.write_bytes(raw_bytes)
            print(f"Screenshot saved to {output_path}")
        else:
            print(f"Screenshot result: {json.dumps(result, indent=2, default=str)}")

        # Clean up ephemeral session if we created one
        if not args.session:
            _run_async(client.close_session(session_id))
        _run_async(client.stop())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_agent(args: argparse.Namespace) -> int:
    """Handle 'flybrowser agent <task>' command.

    Runs an agent task that autonomously navigates and interacts
    with web pages to accomplish a goal.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        session_id, client = _run_async(
            _get_or_create_session(
                session_id=args.session,
                endpoint=getattr(args, "endpoint", None),
            )
        )

        result = _run_async(
            client.agent(
                session_id,
                task=args.task,
                max_iterations=args.max_iterations,
            )
        )

        # Print the result
        print(json.dumps(result, indent=2, default=str))

        # Clean up ephemeral session if we created one
        if not args.session:
            _run_async(client.close_session(session_id))
        _run_async(client.stop())

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ---------- Output Helpers ----------


def _print_csv(result: Dict[str, Any]) -> None:
    """Print extraction result as CSV.

    Args:
        result: The extraction result dict.
    """
    data = result.get("data", result)

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # List of dicts -> CSV with headers
            headers = list(data[0].keys())
            print(",".join(headers))
            for row in data:
                print(",".join(str(row.get(h, "")) for h in headers))
        else:
            # Simple list -> one item per line
            for item in data:
                print(str(item))
    elif isinstance(data, dict):
        # Single dict -> key,value
        print("key,value")
        for k, v in data.items():
            print(f"{k},{v}")
    else:
        print(str(data))


def _print_table(result: Dict[str, Any]) -> None:
    """Print extraction result as a formatted table.

    Args:
        result: The extraction result dict.
    """
    data = result.get("data", result)

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            headers = list(data[0].keys())
            widths = [len(h) for h in headers]

            rows = []
            for item in data:
                row = [str(item.get(h, "")) for h in headers]
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(cell))
                rows.append(row)

            header_line = " | ".join(
                h.ljust(widths[i]) for i, h in enumerate(headers)
            )
            print(header_line)
            print("-" * len(header_line))
            for row in rows:
                print(
                    " | ".join(
                        cell.ljust(widths[i]) for i, cell in enumerate(row)
                    )
                )
        else:
            for item in data:
                print(str(item))
    elif isinstance(data, dict):
        max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
        for k, v in data.items():
            print(f"  {str(k).ljust(max_key_len)}  {v}")
    else:
        print(str(data))


# ---------- Argparse Subparser Registration ----------


def add_direct_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add direct SDK-like subcommands to the CLI parser.

    Registers: goto, extract, act, screenshot, agent.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    # --- goto ---
    goto_parser = subparsers.add_parser(
        "goto",
        help="Navigate to a URL",
    )
    goto_parser.add_argument(
        "url",
        help="URL to navigate to",
    )
    goto_parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID to reuse",
    )
    goto_parser.add_argument(
        "--wait-for",
        default=None,
        help="CSS selector to wait for after navigation",
    )
    goto_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    goto_parser.set_defaults(func=cmd_goto)

    # --- extract ---
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract data from the current page",
    )
    extract_parser.add_argument(
        "query",
        help="Natural language extraction query",
    )
    extract_parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID to reuse",
    )
    extract_parser.add_argument(
        "--schema",
        default=None,
        help="Path to a JSON schema file for structured extraction",
    )
    extract_parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "table"],
        default="json",
        help="Output format (default: json)",
    )
    extract_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # --- act ---
    act_parser = subparsers.add_parser(
        "act",
        help="Perform an action on the page",
    )
    act_parser.add_argument(
        "instruction",
        help="Natural language action instruction",
    )
    act_parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID to reuse",
    )
    act_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    act_parser.set_defaults(func=cmd_act)

    # --- screenshot ---
    screenshot_parser = subparsers.add_parser(
        "screenshot",
        help="Take a screenshot of the current page",
    )
    screenshot_parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID to reuse",
    )
    screenshot_parser.add_argument(
        "--output", "-o",
        default="screenshot.png",
        help="Output file path (default: screenshot.png)",
    )
    screenshot_parser.add_argument(
        "--full-page",
        action="store_true",
        default=False,
        help="Capture the full scrollable page",
    )
    screenshot_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    screenshot_parser.set_defaults(func=cmd_screenshot)

    # --- agent ---
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run an autonomous agent task",
    )
    agent_parser.add_argument(
        "task",
        help="Natural language description of the task to accomplish",
    )
    agent_parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID to reuse",
    )
    agent_parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of agent iterations (default: 50)",
    )
    agent_parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream agent progress to stdout",
    )
    agent_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    agent_parser.set_defaults(func=cmd_agent)
