#!/usr/bin/env python3
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
FlyBrowser Unified CLI.

The main entry point for all FlyBrowser command-line functionality.

Usage:
    flybrowser                    # Launch interactive REPL
    flybrowser repl               # Launch interactive REPL (explicit)
    flybrowser setup [COMMAND]    # Installation/configuration wizard
    flybrowser serve [OPTIONS]    # Start the FlyBrowser service
    flybrowser cluster [COMMAND]  # Cluster management
    flybrowser admin [COMMAND]    # Administrative commands
    flybrowser doctor             # Diagnose installation issues
    flybrowser version            # Show version information
    flybrowser --help             # Show help

Examples:
    # Quick start - launch interactive REPL
    flybrowser

    # Configure FlyBrowser interactively
    flybrowser setup configure

    # Start the service
    flybrowser serve --port 8000

    # Check installation health
    flybrowser doctor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def get_version() -> str:
    """Get the FlyBrowser version."""
    try:
        import flybrowser
        return getattr(flybrowser, "__version__", "unknown")
    except ImportError:
        return "unknown"


def get_banner() -> str:
    """Get the FlyBrowser banner from banner.txt or fallback.

    Returns:
        The banner string
    """
    # Try to load from banner.txt
    banner_paths = [
        Path(__file__).parent.parent / "banner.txt",  # flybrowser/banner.txt
        Path(__file__).parent.parent.parent / "flybrowser" / "banner.txt",
    ]

    for banner_path in banner_paths:
        if banner_path.exists():
            try:
                return banner_path.read_text()
            except Exception:
                pass

    # Fallback banner (matches banner.txt)
    return r"""  _____.__         ___.
_/ ____\  | ___.__.\\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/"""


def print_banner() -> None:
    """Print the FlyBrowser banner."""
    print()
    print(get_banner())
    print()
    print(f"  Browser Automation Powered by LLM Agents")
    print(f"  Version: {get_version()}")
    print()


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    version = get_version()
    
    if args.json:
        import json
        import platform
        info = {
            "flybrowser": version,
            "python": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
        }
        print(json.dumps(info, indent=2))
    else:
        print(f"FlyBrowser {version}")
    
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run installation diagnostics."""
    from flybrowser.cli.setup import verify_installation
    
    print_banner()
    print("Running FlyBrowser diagnostics...\n")
    
    checks = []
    
    # Check 1: Python version
    import platform
    py_version = platform.python_version()
    major, minor = map(int, py_version.split(".")[:2])
    if major >= 3 and minor >= 9:
        print(f"[OK] Python {py_version}")
        checks.append(True)
    else:
        print(f"[FAIL] Python {py_version} (3.9+ required)")
        checks.append(False)
    
    # Check 2: FlyBrowser import
    try:
        import flybrowser
        version = getattr(flybrowser, "__version__", "unknown")
        print(f"[OK] FlyBrowser {version} installed")
        checks.append(True)
    except ImportError as e:
        print(f"[FAIL] FlyBrowser not installed: {e}")
        checks.append(False)
    
    # Check 3: Playwright
    try:
        from playwright.sync_api import sync_playwright
        print("[OK] Playwright installed")
        checks.append(True)
    except ImportError:
        print("[FAIL] Playwright not installed")
        print("      Run: pip install playwright && playwright install")
        checks.append(False)
    
    # Check 4: Browser availability
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("[OK] Chromium browser available")
            checks.append(True)
        else:
            print("[WARN] Chromium may need installation")
            print("       Run: playwright install chromium")
            checks.append(True)  # Not a hard failure
    except Exception:
        print("[WARN] Could not verify browser installation")
        checks.append(True)  # Not a hard failure
    
    # Check 5: LLM providers
    print("\n--- LLM Provider Status ---")
    
    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        print("[OK] OpenAI API key configured")
    else:
        print("[INFO] OpenAI API key not set (optional)")
    
    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[OK] Anthropic API key configured")
    else:
        print("[INFO] Anthropic API key not set (optional)")
    
    # Ollama
    try:
        import subprocess
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("[OK] Ollama running at localhost:11434")
        else:
            print("[INFO] Ollama not running (optional)")
    except Exception:
        print("[INFO] Ollama not detected (optional)")
    
    # Check 6: Config files
    print("\n--- Configuration ---")
    config_paths = [
        Path.cwd() / ".env",
        Path.home() / ".flybrowser" / "config",
        Path.home() / ".config" / "flybrowser" / "config",
    ]
    
    found_config = False
    for config_path in config_paths:
        if config_path.exists():
            print(f"[OK] Config found: {config_path}")
            found_config = True
            break
    
    if not found_config:
        print("[INFO] No config file found (using defaults)")
        print("       Run 'flybrowser setup configure' to create one")
    
    # Summary
    print("\n" + "=" * 50)
    if all(checks):
        print("[SUCCESS] All checks passed!")
        print("\nYou're ready to use FlyBrowser. Try:")
        print("  flybrowser repl         # Interactive mode")
        print("  flybrowser serve        # Start service")
        return 0
    else:
        print("[FAIL] Some checks failed. Please fix the issues above.")
        return 1


def cmd_repl(args: argparse.Namespace) -> int:
    """Launch the interactive REPL."""
    try:
        from flybrowser.cli.repl import FlyBrowserREPL
        
        print_banner()
        
        repl = FlyBrowserREPL(
            llm_provider=args.provider,
            llm_model=args.model,
            headless=args.headless,
            api_key=args.api_key or os.environ.get(f"{args.provider.upper()}_API_KEY"),
        )
        repl.run()
        return 0
    except ImportError as e:
        print(f"Error: REPL dependencies not available: {e}")
        print("Install with: pip install prompt_toolkit")
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0


def cmd_setup(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to setup CLI."""
    from flybrowser.cli import setup
    
    # Reconstruct argv for the setup module
    sys.argv = ["flybrowser-setup"] + remaining
    setup.main()
    return 0


def cmd_serve(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to serve CLI."""
    from flybrowser.cli import serve
    
    # Reconstruct argv for the serve module
    sys.argv = ["flybrowser-serve"] + remaining
    serve.main()
    return 0


def cmd_cluster(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to cluster CLI."""
    from flybrowser.cli import cluster
    
    # Reconstruct argv for the cluster module
    sys.argv = ["flybrowser-cluster"] + remaining
    sys.exit(cluster.main())


def cmd_admin(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to admin CLI."""
    from flybrowser.cli import admin
    
    # Reconstruct argv for the admin module
    sys.argv = ["flybrowser-admin"] + remaining
    sys.exit(admin.main())


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="flybrowser",
        description="FlyBrowser - Browser Automation Powered by LLM Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)      Launch interactive REPL (default)
  repl        Launch interactive REPL
  setup       Installation and configuration wizard
  serve       Start the FlyBrowser API service
  cluster     Cluster management commands
  admin       Administrative commands
  doctor      Diagnose installation issues
  version     Show version information

Examples:
  flybrowser                          # Start REPL
  flybrowser setup configure          # Interactive setup
  flybrowser serve --port 8000        # Start server
  flybrowser cluster status           # Check cluster
  flybrowser doctor                   # Run diagnostics

Documentation: https://flybrowser.dev/docs
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    version_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )
    version_parser.set_defaults(func=cmd_version)
    
    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose installation issues",
    )
    doctor_parser.set_defaults(func=cmd_doctor)
    
    # repl command
    repl_parser = subparsers.add_parser(
        "repl",
        help="Launch interactive REPL",
    )
    repl_parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai"),
        help="LLM provider (default: openai)",
    )
    repl_parser.add_argument(
        "--model", "-m",
        default=os.environ.get("FLYBROWSER_LLM_MODEL"),
        help="LLM model (provider default if not specified)",
    )
    repl_parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)",
    )
    repl_parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run browser with visible UI",
    )
    repl_parser.add_argument(
        "--api-key",
        help="LLM API key (uses env var if not provided)",
    )
    repl_parser.set_defaults(func=cmd_repl)
    
    # setup command (pass-through)
    setup_parser = subparsers.add_parser(
        "setup",
        help="Installation and configuration wizard",
        add_help=False,
    )
    
    # serve command (pass-through)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FlyBrowser API service",
        add_help=False,
    )
    
    # cluster command (pass-through)
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster management commands",
        add_help=False,
    )
    
    # admin command (pass-through)
    admin_parser = subparsers.add_parser(
        "admin",
        help="Administrative commands",
        add_help=False,
    )
    
    return parser


def main() -> None:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    
    # Parse only the known args to allow pass-through for subcommands
    args, remaining = parser.parse_known_args()
    
    # Handle commands that need pass-through
    if args.command == "setup":
        sys.exit(cmd_setup(args, remaining))
    elif args.command == "serve":
        sys.exit(cmd_serve(args, remaining))
    elif args.command == "cluster":
        sys.exit(cmd_cluster(args, remaining))
    elif args.command == "admin":
        sys.exit(cmd_admin(args, remaining))
    
    # Handle other commands normally
    if args.command is None:
        # Default to REPL if no command given
        # First check if we're in interactive mode
        if sys.stdin.isatty():
            # Set default REPL args
            args.provider = os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai")
            args.model = os.environ.get("FLYBROWSER_LLM_MODEL")
            args.headless = True
            args.api_key = None
            sys.exit(cmd_repl(args))
        else:
            # Not interactive, show help
            parser.print_help()
            sys.exit(0)
    elif hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
