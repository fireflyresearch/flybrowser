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
FlyBrowser Interactive REPL.

An interactive Read-Eval-Print Loop for browser automation.

Features:
- Tab completion for commands
- Persistent command history
- Session management (browser stays open between commands)
- Natural language commands for browser automation
- Built-in help system

Commands:
    goto <url>              Navigate to a URL
    extract <query>         Extract data using natural language
    act <action>            Perform an action (click, type, etc.)
    screenshot [filename]   Take a screenshot
    html                    Show current page HTML
    url                     Show current URL
    title                   Show current page title
    wait <condition>        Wait for a condition
    session                 Show session info
    reset                   Reset the browser session
    config                  Show/edit configuration
    help [command]          Show help
    exit/quit               Exit the REPL

Usage:
    from flybrowser.cli.repl import FlyBrowserREPL
    
    repl = FlyBrowserREPL(llm_provider="openai")
    repl.run()
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Try to import prompt_toolkit for enhanced REPL experience
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    # Create stub classes when prompt_toolkit is not available
    Completer = object
    Completion = None
    WordCompleter = None
    PromptSession = None
    AutoSuggestFromHistory = None
    FileHistory = None
    Style = None


class FlyBrowserCompleter(Completer):  # type: ignore[misc]
    """Custom completer for FlyBrowser REPL commands."""
    
    COMMANDS = [
        "goto", "extract", "act", "screenshot", "html", "url", "title",
        "wait", "session", "reset", "config", "help", "exit", "quit",
        "history", "clear", "status", "headless", "provider", "model",
    ]
    
    HELP_TOPICS = COMMANDS + ["commands", "examples", "shortcuts"]
    
    def get_completions(self, document, complete_event):
        """Generate completions for the current input."""
        text = document.text_before_cursor
        words = text.split()
        
        if not words:
            # Show all commands
            for cmd in self.COMMANDS:
                yield Completion(cmd, start_position=0)
            return
        
        current_word = words[-1] if text and not text.endswith(" ") else ""
        
        if len(words) == 1 and not text.endswith(" "):
            # Completing first word (command)
            for cmd in self.COMMANDS:
                if cmd.startswith(current_word):
                    yield Completion(cmd, start_position=-len(current_word))
        
        elif words[0] == "help":
            # Complete help topics
            for topic in self.HELP_TOPICS:
                if topic.startswith(current_word):
                    yield Completion(topic, start_position=-len(current_word))
        
        elif words[0] == "goto":
            # Suggest common URLs
            urls = ["https://", "http://", "https://google.com", "https://example.com"]
            for url in urls:
                if url.startswith(current_word):
                    yield Completion(url, start_position=-len(current_word))
        
        elif words[0] == "config":
            # Config subcommands
            subcommands = ["show", "set", "reset"]
            for sub in subcommands:
                if sub.startswith(current_word):
                    yield Completion(sub, start_position=-len(current_word))


class FlyBrowserREPL:
    """Interactive REPL for FlyBrowser automation."""
    
    PROMPT_STYLE = Style.from_dict({
        "prompt": "#00aa00 bold",
        "command": "#0000aa",
    }) if PROMPT_TOOLKIT_AVAILABLE else None
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        headless: bool = True,
        api_key: Optional[str] = None,
    ):
        """Initialize the REPL.
        
        Args:
            llm_provider: LLM provider name (openai, anthropic, ollama, gemini)
            llm_model: Model name (uses provider default if not specified)
            headless: Run browser in headless mode
            api_key: API key for LLM provider
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.headless = headless
        self.api_key = api_key
        
        self.browser = None
        self.session_started = None
        self.command_history: List[Tuple[datetime, str, Any]] = []
        
        # Setup history file
        self.history_dir = Path.home() / ".flybrowser"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "repl_history"
        
        # Setup prompt session
        if PROMPT_TOOLKIT_AVAILABLE:
            self.session = PromptSession(
                history=FileHistory(str(self.history_file)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=FlyBrowserCompleter(),
                style=self.PROMPT_STYLE,
            )
        else:
            self.session = None
        
        # Command handlers
        self.commands: Dict[str, Callable] = {
            "goto": self._cmd_goto,
            "extract": self._cmd_extract,
            "act": self._cmd_act,
            "screenshot": self._cmd_screenshot,
            "html": self._cmd_html,
            "url": self._cmd_url,
            "title": self._cmd_title,
            "wait": self._cmd_wait,
            "session": self._cmd_session,
            "reset": self._cmd_reset,
            "config": self._cmd_config,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "history": self._cmd_history,
            "clear": self._cmd_clear,
            "status": self._cmd_status,
            "headless": self._cmd_headless,
            "provider": self._cmd_provider,
            "model": self._cmd_model,
        }
    
    def _print_error(self, msg: str) -> None:
        """Print an error message."""
        print(f"\033[91m[ERROR]\033[0m {msg}")
    
    def _print_success(self, msg: str) -> None:
        """Print a success message."""
        print(f"\033[92m[OK]\033[0m {msg}")
    
    def _print_info(self, msg: str) -> None:
        """Print an info message."""
        print(f"\033[94m[INFO]\033[0m {msg}")
    
    def _print_warning(self, msg: str) -> None:
        """Print a warning message."""
        print(f"\033[93m[WARN]\033[0m {msg}")
    
    def _ensure_browser(self) -> bool:
        """Ensure browser is initialized."""
        if self.browser is not None:
            return True
        
        self._print_info("Starting browser session...")
        
        try:
            from flybrowser import FlyBrowser
            
            # Determine API key
            api_key = self.api_key
            if not api_key:
                env_var = f"{self.llm_provider.upper()}_API_KEY"
                api_key = os.environ.get(env_var)
            
            # For Ollama, we don't need an API key
            if self.llm_provider == "ollama":
                api_key = None
            
            self.browser = FlyBrowser(
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                headless=self.headless,
                api_key=api_key,
            )
            
            # Start the browser
            asyncio.get_event_loop().run_until_complete(
                self.browser.__aenter__()
            )
            
            self.session_started = datetime.now()
            self._print_success(f"Browser session started (headless={self.headless})")
            return True
            
        except Exception as e:
            self._print_error(f"Failed to start browser: {e}")
            if "API key" in str(e) or "api_key" in str(e).lower():
                self._print_info(f"Set your API key: export {self.llm_provider.upper()}_API_KEY=your-key")
            return False
    
    def _close_browser(self) -> None:
        """Close the browser session."""
        if self.browser is not None:
            try:
                asyncio.get_event_loop().run_until_complete(
                    self.browser.__aexit__(None, None, None)
                )
            except Exception:
                pass
            self.browser = None
            self.session_started = None
    
    async def _run_async(self, coro):
        """Run an async coroutine."""
        return await coro
    
    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return asyncio.get_event_loop().run_until_complete(coro)
    
    # Command implementations
    
    def _cmd_goto(self, args: str) -> None:
        """Navigate to a URL."""
        if not args:
            self._print_error("Usage: goto <url>")
            return
        
        if not self._ensure_browser():
            return
        
        url = args.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        try:
            self._run(self.browser.goto(url))
            self._print_success(f"Navigated to {url}")
        except Exception as e:
            self._print_error(f"Navigation failed: {e}")
    
    def _cmd_extract(self, args: str) -> None:
        """Extract data using natural language query."""
        if not args:
            self._print_error("Usage: extract <query>")
            self._print_info("Example: extract What is the main heading?")
            return
        
        if not self._ensure_browser():
            return
        
        try:
            result = self._run(self.browser.extract(args))
            print("\n--- Extracted Data ---")
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
            print("---")
        except Exception as e:
            self._print_error(f"Extraction failed: {e}")
    
    def _cmd_act(self, args: str) -> None:
        """Perform an action using natural language."""
        if not args:
            self._print_error("Usage: act <action>")
            self._print_info("Example: act click the login button")
            return
        
        if not self._ensure_browser():
            return
        
        try:
            result = self._run(self.browser.act(args))
            self._print_success(f"Action completed")
            if result:
                print(f"Result: {result}")
        except Exception as e:
            self._print_error(f"Action failed: {e}")
    
    def _cmd_screenshot(self, args: str) -> None:
        """Take a screenshot."""
        if not self._ensure_browser():
            return
        
        filename = args.strip() if args else f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            page = self.browser._page
            if page:
                self._run(page.screenshot(path=filename))
                self._print_success(f"Screenshot saved to {filename}")
            else:
                self._print_error("No page available")
        except Exception as e:
            self._print_error(f"Screenshot failed: {e}")
    
    def _cmd_html(self, args: str) -> None:
        """Show current page HTML."""
        if not self._ensure_browser():
            return
        
        try:
            page = self.browser._page
            if page:
                content = self._run(page.content())
                # Truncate if too long
                if len(content) > 2000:
                    print(content[:2000])
                    print(f"\n... (truncated, {len(content)} total chars)")
                else:
                    print(content)
            else:
                self._print_error("No page available")
        except Exception as e:
            self._print_error(f"Failed to get HTML: {e}")
    
    def _cmd_url(self, args: str) -> None:
        """Show current URL."""
        if not self._ensure_browser():
            return
        
        try:
            page = self.browser._page
            if page:
                print(page.url)
            else:
                self._print_error("No page available")
        except Exception as e:
            self._print_error(f"Failed to get URL: {e}")
    
    def _cmd_title(self, args: str) -> None:
        """Show current page title."""
        if not self._ensure_browser():
            return
        
        try:
            page = self.browser._page
            if page:
                title = self._run(page.title())
                print(title)
            else:
                self._print_error("No page available")
        except Exception as e:
            self._print_error(f"Failed to get title: {e}")
    
    def _cmd_wait(self, args: str) -> None:
        """Wait for a condition."""
        if not args:
            self._print_error("Usage: wait <condition>")
            self._print_info("Example: wait for the page to load")
            return
        
        if not self._ensure_browser():
            return
        
        try:
            result = self._run(self.browser.monitor(args))
            self._print_success("Condition met")
            if result:
                print(f"Result: {result}")
        except Exception as e:
            self._print_error(f"Wait failed: {e}")
    
    def _cmd_session(self, args: str) -> None:
        """Show session information."""
        print("\n--- Session Info ---")
        print(f"Provider: {self.llm_provider}")
        print(f"Model: {self.llm_model or '(default)'}")
        print(f"Headless: {self.headless}")
        print(f"Browser active: {self.browser is not None}")
        if self.session_started:
            duration = datetime.now() - self.session_started
            print(f"Session duration: {duration}")
        print(f"Commands executed: {len(self.command_history)}")
        print("---")
    
    def _cmd_reset(self, args: str) -> None:
        """Reset the browser session."""
        self._print_info("Resetting browser session...")
        self._close_browser()
        self._print_success("Session reset. Browser will restart on next command.")
    
    def _cmd_config(self, args: str) -> None:
        """Show or modify configuration."""
        parts = args.split(maxsplit=1) if args else []
        
        if not parts or parts[0] == "show":
            print("\n--- Configuration ---")
            print(f"llm_provider: {self.llm_provider}")
            print(f"llm_model: {self.llm_model or '(default)'}")
            print(f"headless: {self.headless}")
            print(f"history_file: {self.history_file}")
            print("---")
        
        elif parts[0] == "set" and len(parts) > 1:
            # Parse key=value
            match = re.match(r"(\w+)=(.+)", parts[1])
            if match:
                key, value = match.groups()
                if key == "headless":
                    self.headless = value.lower() in ("true", "1", "yes")
                    self._print_success(f"Set headless={self.headless}")
                    self._print_info("Run 'reset' to apply to browser")
                elif key == "provider":
                    self.llm_provider = value
                    self._print_success(f"Set provider={self.llm_provider}")
                    self._print_info("Run 'reset' to apply to browser")
                elif key == "model":
                    self.llm_model = value
                    self._print_success(f"Set model={self.llm_model}")
                    self._print_info("Run 'reset' to apply to browser")
                else:
                    self._print_error(f"Unknown config key: {key}")
            else:
                self._print_error("Usage: config set key=value")
        
        else:
            self._print_error("Usage: config [show|set key=value]")
    
    def _cmd_help(self, args: str) -> None:
        """Show help information."""
        topic = args.strip() if args else None
        
        if not topic or topic == "commands":
            print("""
FlyBrowser REPL Commands
========================

Navigation:
  goto <url>              Navigate to a URL
  
Data Extraction:
  extract <query>         Extract data using natural language
  html                    Show current page HTML
  url                     Show current URL
  title                   Show current page title
  
Actions:
  act <action>            Perform an action (click, type, etc.)
  wait <condition>        Wait for a condition
  screenshot [filename]   Take a screenshot
  
Session Management:
  session                 Show session info
  reset                   Reset the browser session
  status                  Show browser status
  
Configuration:
  config [show|set]       Show/modify configuration
  headless [on|off]       Toggle headless mode
  provider <name>         Change LLM provider
  model <name>            Change LLM model
  
Utility:
  history                 Show command history
  clear                   Clear the screen
  help [topic]            Show help
  exit/quit               Exit the REPL

Type 'help <command>' for detailed help on a specific command.
Type 'help examples' for usage examples.
""")
        
        elif topic == "examples":
            print("""
FlyBrowser REPL Examples
========================

Basic navigation:
  goto google.com
  goto https://news.ycombinator.com
  
Data extraction:
  extract What is the main headline?
  extract List all the article titles
  extract Find the price of the first product
  
Performing actions:
  act click the search button
  act type "flybrowser" into the search box
  act scroll down
  act click on the first result
  
Waiting for conditions:
  wait for the page to finish loading
  wait until the login form appears
  wait for "Success" to appear on the page
  
Screenshots:
  screenshot
  screenshot login_page.png
  
Changing settings:
  config set headless=false
  provider ollama
  model gpt-4
  reset
""")
        
        elif topic in self.commands:
            # Show help for specific command
            cmd_help = {
                "goto": "Navigate to a URL.\nUsage: goto <url>\nExample: goto https://example.com",
                "extract": "Extract data from the page using natural language.\nUsage: extract <query>\nExample: extract What is the main heading?",
                "act": "Perform an action on the page.\nUsage: act <action>\nExample: act click the submit button",
                "screenshot": "Take a screenshot.\nUsage: screenshot [filename]\nExample: screenshot page.png",
                "wait": "Wait for a condition.\nUsage: wait <condition>\nExample: wait for the loading spinner to disappear",
                "reset": "Reset the browser session.\nUsage: reset",
                "config": "Show or modify configuration.\nUsage: config [show|set key=value]",
            }
            print(cmd_help.get(topic, f"No detailed help available for '{topic}'"))
        
        else:
            self._print_error(f"Unknown help topic: {topic}")
            self._print_info("Try 'help' or 'help commands'")
    
    def _cmd_exit(self, args: str) -> None:
        """Exit the REPL."""
        raise EOFError()
    
    def _cmd_history(self, args: str) -> None:
        """Show command history."""
        if not self.command_history:
            print("No commands in history.")
            return
        
        print("\n--- Command History ---")
        for i, (timestamp, cmd, result) in enumerate(self.command_history[-20:], 1):
            time_str = timestamp.strftime("%H:%M:%S")
            print(f"{i:3}. [{time_str}] {cmd}")
        print("---")
    
    def _cmd_clear(self, args: str) -> None:
        """Clear the screen."""
        os.system("clear" if os.name != "nt" else "cls")
    
    def _cmd_status(self, args: str) -> None:
        """Show browser status."""
        if self.browser is None:
            print("Browser: Not started")
        else:
            print("Browser: Running")
            try:
                page = self.browser._page
                if page:
                    print(f"  URL: {page.url}")
                    title = self._run(page.title())
                    print(f"  Title: {title[:50]}..." if len(title) > 50 else f"  Title: {title}")
            except Exception:
                print("  (unable to get page info)")
    
    def _cmd_headless(self, args: str) -> None:
        """Toggle or set headless mode."""
        if not args:
            print(f"Headless mode: {self.headless}")
            print("Usage: headless [on|off]")
            return
        
        if args.lower() in ("on", "true", "1", "yes"):
            self.headless = True
        elif args.lower() in ("off", "false", "0", "no"):
            self.headless = False
        else:
            self._print_error(f"Invalid value: {args}")
            return
        
        self._print_success(f"Headless mode: {self.headless}")
        if self.browser:
            self._print_info("Run 'reset' to apply changes")
    
    def _cmd_provider(self, args: str) -> None:
        """Change LLM provider."""
        if not args:
            print(f"Current provider: {self.llm_provider}")
            print("Available: openai, anthropic, ollama, gemini")
            return
        
        providers = ["openai", "anthropic", "ollama", "gemini"]
        if args.lower() not in providers:
            self._print_error(f"Unknown provider: {args}")
            print(f"Available: {', '.join(providers)}")
            return
        
        self.llm_provider = args.lower()
        self._print_success(f"Provider set to: {self.llm_provider}")
        if self.browser:
            self._print_info("Run 'reset' to apply changes")
    
    def _cmd_model(self, args: str) -> None:
        """Change LLM model."""
        if not args:
            print(f"Current model: {self.llm_model or '(provider default)'}")
            return
        
        self.llm_model = args
        self._print_success(f"Model set to: {self.llm_model}")
        if self.browser:
            self._print_info("Run 'reset' to apply changes")
    
    def _parse_command(self, line: str) -> Tuple[str, str]:
        """Parse a command line into command and arguments."""
        line = line.strip()
        if not line:
            return "", ""
        
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        return cmd, args
    
    def _execute(self, line: str) -> None:
        """Execute a command line."""
        cmd, args = self._parse_command(line)
        
        if not cmd:
            return
        
        if cmd in self.commands:
            try:
                self.command_history.append((datetime.now(), line, None))
                self.commands[cmd](args)
            except Exception as e:
                self._print_error(f"Command failed: {e}")
                if os.environ.get("FLYBROWSER_DEBUG"):
                    traceback.print_exc()
        else:
            self._print_error(f"Unknown command: {cmd}")
            self._print_info("Type 'help' for available commands")
    
    def run(self) -> None:
        """Run the REPL main loop."""
        print("Welcome to the FlyBrowser Interactive Shell!")
        print("Type 'help' for available commands, 'exit' to quit.\n")
        
        # Ensure we have an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            while True:
                try:
                    # Get input
                    if self.session:
                        line = self.session.prompt("flybrowser> ")
                    else:
                        line = input("flybrowser> ")
                    
                    self._execute(line)
                    
                except KeyboardInterrupt:
                    print("\n(Use 'exit' or Ctrl+D to quit)")
                    continue
                except EOFError:
                    print("\nGoodbye!")
                    break
        
        finally:
            self._close_browser()


def main() -> None:
    """Main entry point for running the REPL directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlyBrowser Interactive REPL")
    parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai"),
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("FLYBROWSER_LLM_MODEL"),
        help="LLM model",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser with visible UI",
    )
    
    args = parser.parse_args()
    
    repl = FlyBrowserREPL(
        llm_provider=args.provider,
        llm_model=args.model,
        headless=not args.no_headless,
    )
    repl.run()


if __name__ == "__main__":
    main()
