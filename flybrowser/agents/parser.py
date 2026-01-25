# Copyright 2026 Firefly Software Solutions Inc.
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
ReAct LLM Response Parser.

This module provides robust parsing of LLM responses into structured
Thought-Action pairs for the ReAct framework. It supports multiple
response formats and includes comprehensive error recovery.

Supported formats:
- JSON with thought/action structure
- XML-like tagged format (<thought>, <action>)
- Natural language with markers
- Markdown code blocks wrapping any of the above

Example JSON format:
    {
        "thought": "I need to navigate to the page",
        "action": {
            "tool": "navigate",
            "parameters": {"url": "https://example.com"}
        }
    }

Example XML format:
    <thought>I need to navigate to the page</thought>
    <action tool="navigate">
        <param name="url">https://example.com</param>
    </action>
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .types import Thought, Action, ReasoningStrategy

if TYPE_CHECKING:
    from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ParseFormat(Enum):
    """Detected or expected format of LLM response."""
    JSON = "json"
    XML = "xml"
    NATURAL = "natural"
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    """
    Result of parsing an LLM response.
    
    Attributes:
        success: Whether parsing was successful
        thought: Extracted thought (reasoning)
        action: Extracted action to execute
        format_detected: Format that was detected/used
        confidence: Confidence score for the parse (0.0-1.0)
        raw_content: Original content that was parsed
        error: Error message if parsing failed
        warnings: Non-fatal warnings during parsing
    """
    success: bool
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    format_detected: ParseFormat = ParseFormat.UNKNOWN
    confidence: float = 0.0
    raw_content: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def failure(cls, error: str, raw_content: str = "") -> "ParseResult":
        """Create a failed parse result."""
        return cls(
            success=False,
            error=error,
            raw_content=raw_content,
            confidence=0.0,
        )
    
    @classmethod
    def partial(
        cls,
        thought: Optional[Thought] = None,
        action: Optional[Action] = None,
        format_detected: ParseFormat = ParseFormat.UNKNOWN,
        raw_content: str = "",
        warnings: Optional[List[str]] = None,
    ) -> "ParseResult":
        """Create a partial parse result (some fields missing)."""
        return cls(
            success=thought is not None or action is not None,
            thought=thought,
            action=action,
            format_detected=format_detected,
            confidence=0.5 if thought and action else 0.25,
            raw_content=raw_content,
            warnings=warnings or [],
        )


class ReActParser:
    """
    Parser for ReAct-style LLM responses.
    
    Handles multiple response formats and provides robust error recovery.
    Can optionally validate actions against a tool registry.
    
    Attributes:
        tool_registry: Optional registry for validating tool names
        strict_mode: If True, fail on any validation error
        
    Example:
        >>> parser = ReActParser(tool_registry=registry)
        >>> result = parser.parse(llm_response_content)
        >>> if result.success:
        ...     print(f"Thought: {result.thought.content}")
        ...     print(f"Action: {result.action.tool_name}")
    """
    
    def __init__(
        self,
        tool_registry: Optional["ToolRegistry"] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the parser.
        
        Args:
            tool_registry: Optional tool registry for validation
            strict_mode: Whether to fail on validation errors
        """
        self.tool_registry = tool_registry
        self.strict_mode = strict_mode
    
    def parse(self, content: str) -> ParseResult:
        """
        Parse LLM response content into Thought-Action pair.
        
        Tries multiple parsing strategies in order of preference:
        1. JSON format
        2. XML-like tagged format
        3. Natural language extraction
        
        Args:
            content: Raw LLM response content
            
        Returns:
            ParseResult with extracted thought/action or error
        """
        if not content or not content.strip():
            return ParseResult.failure("Empty response content", content)
        
        # Clean content (remove markdown code blocks if present)
        cleaned = self._extract_from_markdown(content)
        
        # Try parsing strategies in order
        strategies = [
            (self._parse_json, ParseFormat.JSON),
            (self._parse_xml, ParseFormat.XML),
            (self._parse_natural, ParseFormat.NATURAL),
        ]
        
        last_error = ""
        for parse_fn, format_type in strategies:
            try:
                result = parse_fn(cleaned)
                if result.success:
                    result.format_detected = format_type
                    result.raw_content = content
                    
                    # Validate action if registry available
                    if result.action and self.tool_registry is not None:
                        validation = self._validate_action(result.action)
                        if validation:
                            result.warnings.append(validation)
                            if self.strict_mode:
                                return ParseResult.failure(validation, content)
                    
                    return result
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Parse strategy {format_type.value} failed: {e}")
                continue
        
        # All strategies failed
        logger.warning(f"All parsing strategies failed for content:\n{content[:500]}")
        return ParseResult.failure(
            f"Failed to parse response: {last_error}",
            content,
        )

    def _extract_from_markdown(self, content: str) -> str:
        """
        Extract content from markdown code blocks.

        Handles both ```json and plain ``` code blocks.

        Args:
            content: Content potentially wrapped in markdown

        Returns:
            Extracted content without markdown wrapper
        """
        # Pattern for ```json or ```xml or plain ``` blocks
        code_block_pattern = r"```(?:json|xml|text)?\s*\n?(.*?)\n?```"
        match = re.search(code_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return content.strip()

    def _parse_json(self, content: str) -> ParseResult:
        """
        Parse JSON format response.

        Expected format:
            {
                "thought": "reasoning text",
                "action": {
                    "tool": "tool_name",
                    "parameters": {...}
                }
            }

        Args:
            content: Content to parse

        Returns:
            ParseResult with extracted thought/action
        """
        # Try to find JSON object in content
        json_content = self._find_json_object(content)
        if not json_content:
            return ParseResult.failure("No JSON object found")

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            return ParseResult.failure(f"Invalid JSON: {e}")

        if not isinstance(data, dict):
            return ParseResult.failure("JSON is not an object")

        # Extract thought (handle multiple formats)
        thought = None
        
        # Check for tree_of_thoughts format
        if "tree_of_thoughts" in data:
            tot = data["tree_of_thoughts"]
            if isinstance(tot, dict):
                # Extract reasoning from selected branch
                selected_id = tot.get("selected_branch")
                rationale = tot.get("selection_rationale", "")
                
                # Build thought from branches and selection
                branches_summary = ""
                if "branches" in tot and isinstance(tot["branches"], list):
                    selected_branch = next(
                        (b for b in tot["branches"] if b.get("branch_id") == selected_id),
                        None
                    )
                    if selected_branch:
                        branches_summary = (
                            f"Selected Branch {selected_id}: {selected_branch.get('strategy', 'unknown')} - "
                            f"{selected_branch.get('reasoning', 'no reasoning')}"
                        )
                
                thought_content = f"{branches_summary}\nSelection Rationale: {rationale}"
                thought = Thought(
                    content=thought_content,
                    confidence=tot.get("confidence", 0.8),
                    strategy=ReasoningStrategy.TREE_OF_THOUGHT,
                )
        
        # Fall back to standard thought fields if not tree_of_thoughts
        if not thought:
            thought_content = data.get("thought") or data.get("thinking") or data.get("reasoning")
            if thought_content:
                thought = Thought(
                    content=str(thought_content),
                    confidence=data.get("confidence", 0.8),
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                )

        # Extract action
        action = None
        action_data = data.get("action")
        if action_data and isinstance(action_data, dict):
            tool_name = action_data.get("tool") or action_data.get("name") or action_data.get("tool_name")
            if tool_name:
                action = Action(
                    tool_name=str(tool_name),
                    parameters=action_data.get("parameters", action_data.get("params", {})),
                    requires_approval=action_data.get("requires_approval", False),
                )
            else:
                # Handle shorthand format: {"action": {"tool_name": {params}}} or {"tool_name": {params}}
                # e.g., {"action": {"click": {"selector": "button"}}} or {"action": {"extract_text": {"selector": "body"}}}
                for key, value in action_data.items():
                    if key not in ("tool", "name", "tool_name", "parameters", "params", "requires_approval"):
                        # This key might be the tool name itself
                        if isinstance(value, dict):
                            action = Action(
                                tool_name=str(key),
                                parameters=value,
                                requires_approval=False,
                            )
                            logger.debug(f"[Parser] Detected shorthand action format: {key}({value})")
                            break
        
        # Also check for top-level shorthand format (tool name directly in data)
        # e.g., {"thought": "...", "click": {"selector": "button"}}
        if not action:
            known_tools = {
                "click", "navigate", "type_text", "scroll", "screenshot", 
                "extract_text", "get_page_state", "wait", "go_back", "go_forward",
                "refresh", "hover", "select_option", "press_key", "evaluate_js",
                "fill_form", "extract_table", "extract_links"
            }
            for key, value in data.items():
                if key in known_tools and isinstance(value, dict):
                    action = Action(
                        tool_name=str(key),
                        parameters=value,
                        requires_approval=False,
                    )
                    logger.debug(f"[Parser] Detected top-level shorthand action: {key}({value})")
                    break
        
        # Auto-generate action from Tree of Thoughts if missing
        if not action and "tree_of_thoughts" in data:
            action = self._generate_action_from_tot(data["tree_of_thoughts"])
            if action:
                logger.info(f"[ToT] Auto-generated action from selected branch: {action.tool_name}")

        if not thought and not action:
            return ParseResult.failure("No thought or action found in JSON")

        return ParseResult(
            success=True,
            thought=thought,
            action=action,
            format_detected=ParseFormat.JSON,
            confidence=0.95 if thought and action else 0.7,
        )

    def _find_json_object(self, content: str) -> Optional[str]:
        """
        Find and extract JSON object from content.
        
        Handles truncated JSON by attempting to auto-complete it.

        Args:
            content: Content to search

        Returns:
            Extracted JSON string or None
        """
        # Find the first { and last }
        start = content.find("{")
        end = content.rfind("}")

        if start == -1:
            return None
        
        # If no closing brace found, JSON might be truncated
        if end == -1 or end <= start:
            logger.warning("[Parser] JSON appears truncated, attempting auto-completion")
            # Try to salvage truncated JSON by closing it properly
            json_str = content[start:]
            # Count unclosed braces and brackets
            open_braces = json_str.count("{") - json_str.count("}")
            open_brackets = json_str.count("[") - json_str.count("]")
            open_quotes = json_str.count('"') % 2  # Odd number means unclosed string
            
            # Close any unclosed strings
            if open_quotes:
                json_str += '"'
            
            # Close any unclosed arrays/objects
            json_str += "]" * open_brackets
            json_str += "}" * open_braces
            
            logger.debug(f"[Parser] Auto-completed JSON: ...{json_str[-100:]}")
            return json_str

        return content[start:end + 1]

    def _parse_xml(self, content: str) -> ParseResult:
        """
        Parse XML-like tagged format response.

        Expected format:
            <thought>I need to navigate to the page</thought>
            <action tool="navigate">
                <param name="url">https://example.com</param>
            </action>

        Args:
            content: Content to parse

        Returns:
            ParseResult with extracted thought/action
        """
        thought = None
        action = None

        # Extract thought from <thought>...</thought> tags
        thought_match = re.search(
            r"<thought[^>]*>(.*?)</thought>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = Thought(
                content=thought_match.group(1).strip(),
                confidence=0.8,
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            )

        # Extract action from <action>...</action> tags
        action_match = re.search(
            r'<action\s+(?:tool|name)=["\']([^"\']+)["\'][^>]*>(.*?)</action>',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            tool_name = action_match.group(1)
            action_body = action_match.group(2)

            # Extract parameters from <param> tags
            params: Dict[str, Any] = {}
            param_matches = re.finditer(
                r'<param\s+name=["\']([^"\']+)["\'][^>]*>([^<]*)</param>',
                action_body,
                re.IGNORECASE,
            )
            for param_match in param_matches:
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                # Try to parse as JSON for complex values
                try:
                    params[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    params[param_name] = param_value

            action = Action(
                tool_name=tool_name,
                parameters=params,
            )

        # Also try self-closing action format: <action tool="name" param1="val1" />
        if not action:
            action_match = re.search(
                r'<action\s+(?:tool|name)=["\']([^"\']+)["\']([^/>]*)/?>',
                content,
                re.IGNORECASE,
            )
            if action_match:
                tool_name = action_match.group(1)
                attrs_str = action_match.group(2)

                # Extract attributes as parameters
                params = {}
                attr_matches = re.finditer(
                    r'(\w+)=["\']([^"\']*)["\']',
                    attrs_str,
                )
                for attr_match in attr_matches:
                    attr_name = attr_match.group(1)
                    if attr_name.lower() not in ("tool", "name"):
                        params[attr_name] = attr_match.group(2)

                action = Action(
                    tool_name=tool_name,
                    parameters=params,
                )

        if not thought and not action:
            return ParseResult.failure("No thought or action tags found")

        return ParseResult(
            success=True,
            thought=thought,
            action=action,
            format_detected=ParseFormat.XML,
            confidence=0.9 if thought and action else 0.6,
        )

    def _parse_natural(self, content: str) -> ParseResult:
        """
        Parse natural language format response.

        Expected format:
            Thought: I need to navigate to the page
            Action: navigate
            Parameters:
              url: https://example.com

        Also handles:
            - "I think..." / "Let me..." style thoughts
            - "I will use [tool]" / "I'll call [tool]" action patterns

        Args:
            content: Content to parse

        Returns:
            ParseResult with extracted thought/action
        """
        thought = None
        action = None

        # Pattern 1: Explicit "Thought:" label
        thought_match = re.search(
            r"(?:^|\n)\s*(?:Thought|Thinking|Reasoning)\s*:\s*(.+?)(?=\n\s*(?:Action|$))",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if thought_match:
            thought = Thought(
                content=thought_match.group(1).strip(),
                confidence=0.7,
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            )

        # Pattern 2: Natural thought patterns ("I think...", "Let me...")
        if not thought:
            natural_thought_match = re.search(
                r"(?:^|\n)\s*((?:I think|I need to|I should|I will|I'll|Let me|First,? I)[^.!?]*[.!?])",
                content,
                re.IGNORECASE,
            )
            if natural_thought_match:
                thought = Thought(
                    content=natural_thought_match.group(1).strip(),
                    confidence=0.5,
                    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                )

        # Pattern 1: Explicit "Action:" label with parameters (handles both "Parameters:" and "Action Input:" formats)
        action_match = re.search(
            r"(?:^|\n)\s*Action\s*:\s*(\w+)\s*(?:\n\s*(?:Action\s*Input|Parameters?)\s*:?\s*(.+?))?(?=\n\s*(?:Thought|Action|Observation|$)|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if action_match:
            tool_name = action_match.group(1).strip()
            params_str = action_match.group(2) or ""
            
            # Try to parse as JSON first (common for Action Input format)
            params = {}
            if params_str.strip():
                try:
                    params = json.loads(params_str.strip())
                except (json.JSONDecodeError, ValueError):
                    # Fall back to natural language parsing
                    params = self._parse_natural_params(params_str)

            action = Action(
                tool_name=tool_name,
                parameters=params,
            )

        # Pattern 2: "I will use [tool]" or "I'll call [tool]" patterns
        if not action:
            use_pattern_match = re.search(
                r"(?:I will|I'll|Let me|Going to)\s+(?:use|call|execute|run|invoke)\s+(?:the\s+)?['\"]?(\w+)['\"]?",
                content,
                re.IGNORECASE,
            )
            if use_pattern_match:
                tool_name = use_pattern_match.group(1)
                # Try to find parameters mentioned nearby
                params = self._extract_params_from_context(content, tool_name)

                action = Action(
                    tool_name=tool_name,
                    parameters=params,
                )

        if not thought and not action:
            return ParseResult.failure("No thought or action patterns found in natural language")

        return ParseResult(
            success=True,
            thought=thought,
            action=action,
            format_detected=ParseFormat.NATURAL,
            confidence=0.6 if thought and action else 0.4,
        )

    def _parse_natural_params(self, params_str: str) -> Dict[str, Any]:
        """
        Parse parameters from natural language format.

        Handles formats like:
            url: https://example.com
            selector: #button
            - url: https://example.com
            url = "https://example.com"

        Args:
            params_str: Parameter string to parse

        Returns:
            Dictionary of parameter name to value
        """
        params: Dict[str, Any] = {}

        if not params_str.strip():
            return params

        # Try key: value format (YAML-like)
        kv_matches = re.finditer(
            r"(?:^|\n)\s*[-*]?\s*(\w+)\s*[:=]\s*(.+?)(?=\n\s*[-*]?\s*\w+\s*[:=]|$)",
            params_str,
            re.DOTALL,
        )
        for match in kv_matches:
            key = match.group(1).strip()
            value = match.group(2).strip().strip("\"'")
            # Try to parse as JSON for complex values
            try:
                params[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[key] = value

        return params

    def _extract_params_from_context(self, content: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameters from surrounding context when action pattern is found.

        Args:
            content: Full content to search
            tool_name: Name of the tool for context

        Returns:
            Dictionary of extracted parameters
        """
        params: Dict[str, Any] = {}

        # Look for common parameter patterns near the tool mention
        # URL pattern
        url_match = re.search(r'(?:url|link|page)\s*[:=]?\s*["\']?(https?://[^\s"\']+)', content, re.IGNORECASE)
        if url_match:
            params["url"] = url_match.group(1)

        # Selector pattern
        selector_match = re.search(r'(?:selector|element|css)\s*[:=]?\s*["\']?([#.][^\s"\']+)', content, re.IGNORECASE)
        if selector_match:
            params["selector"] = selector_match.group(1)

        # Text/query pattern
        text_match = re.search(r'(?:text|query|search|input)\s*[:=]?\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
        if text_match:
            params["text"] = text_match.group(1)

        return params

    def _validate_action(self, action: Action) -> Optional[str]:
        """
        Validate action against tool registry.

        Checks that the action references a known tool and optionally
        validates parameters against the tool's schema.

        Args:
            action: Action to validate

        Returns:
            None if valid, error message string if invalid
        """
        if self.tool_registry is None:
            return None  # No registry to validate against

        # Get tool for parameter validation (also serves as existence check).
        tool = self.tool_registry.get_tool(action.tool_name)
        if not tool:
            # ToolRegistry.list_tools() returns List[str], but keep this robust
            # to any iterable.
            available_tools = list(self.tool_registry.list_tools())
            return f"Unknown tool: '{action.tool_name}'. Available tools: {available_tools}"

        # Validate required parameters
        schema = tool.get_parameters_schema()
        required_params = schema.get("required", [])
        properties = schema.get("properties", {})

        missing_params = []
        for param in required_params:
            if param not in action.parameters:
                missing_params.append(param)

        if missing_params:
            return f"Missing required parameters for '{action.tool_name}': {missing_params}"

        # Validate parameter types (basic type checking)
        type_errors = []
        for param_name, param_value in action.parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type and not self._check_type(param_value, expected_type):
                    type_errors.append(
                        f"Parameter '{param_name}' expected type '{expected_type}', "
                        f"got '{type(param_value).__name__}'"
                    )

        if type_errors:
            return f"Type errors in '{action.tool_name}': {'; '.join(type_errors)}"

        return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value matches the expected JSON schema type.

        Args:
            value: The value to check
            expected_type: JSON schema type string

        Returns:
            True if type matches, False otherwise
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            # Unknown type, assume valid
            return True

        return isinstance(value, expected_python_type)
    
    def _generate_action_from_tot(self, tot_data: Dict[str, Any]) -> Optional[Action]:
        """
        Generate an Action from Tree of Thoughts data when action is missing.
        
        This handles cases where LLM provides branches and selection but forgets
        to include the action object. We auto-generate it from the selected branch.
        
        Args:
            tot_data: Tree of Thoughts dictionary with branches and selection
            
        Returns:
            Action object or None if cannot be generated
        """
        if not isinstance(tot_data, dict):
            return None
        
        branches = tot_data.get("branches", [])
        if not branches or not isinstance(branches, list):
            return None
        
        # Get selected branch ID
        selected_id = tot_data.get("selected_branch")
        
        # If no selection, auto-select highest scored branch
        if selected_id is None:
            logger.warning("[ToT] No selected_branch specified, auto-selecting highest score")
            highest_branch = max(branches, key=lambda b: b.get("overall_score", 0))
            selected_id = highest_branch.get("branch_id")
            logger.info(f"[ToT] Auto-selected branch {selected_id} with score {highest_branch.get('overall_score')}")
        
        # Find the selected branch
        selected_branch = next(
            (b for b in branches if b.get("branch_id") == selected_id),
            None
        )
        
        if not selected_branch:
            logger.warning(f"[ToT] Selected branch {selected_id} not found in branches")
            # Fallback to first branch
            selected_branch = branches[0]
            logger.info(f"[ToT] Falling back to first branch: {selected_branch.get('branch_id')}")
        
        # Extract tool name from tools_needed
        tools_needed = selected_branch.get("tools_needed", [])
        if not tools_needed or not isinstance(tools_needed, list):
            logger.warning("[ToT] Selected branch has no tools_needed")
            return None
        
        # Use first tool from the list
        tool_name = tools_needed[0]
        
        # Try to extract parameters from reasoning and strategy
        parameters = self._extract_parameters_from_branch(selected_branch, tool_name)
        
        logger.info(f"[ToT] Generated action: {tool_name}({parameters}) from branch '{selected_branch.get('strategy')}'")
        
        return Action(
            tool_name=tool_name,
            parameters=parameters,
            requires_approval=False,
        )
    
    def _extract_parameters_from_branch(self, branch: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Extract action parameters from a ToT branch's reasoning and strategy.
        
        Args:
            branch: The selected branch dictionary
            tool_name: Name of the tool to execute
            
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        
        # Combine strategy and reasoning text for extraction
        text = f"{branch.get('strategy', '')} {branch.get('reasoning', '')}"
        
        # Extract common parameters based on tool
        if tool_name == "navigate":
            # Look for URLs
            url_match = re.search(r'https?://[^\s"\'<>]+', text)
            if url_match:
                params["url"] = url_match.group(0)
        
        elif tool_name == "click":
            # Look for selectors or element descriptions
            # Try XPath first
            xpath_match = re.search(r'//[a-zA-Z]+\[.*?\]', text)
            if xpath_match:
                params["selector"] = xpath_match.group(0)
            else:
                # Try CSS selector
                css_match = re.search(r'[#.][-\w]+', text)
                if css_match:
                    params["selector"] = css_match.group(0)
                else:
                    # Try text-based (button with text)
                    text_match = re.search(r'(?:button|link|element).*?["\']([^"\']+)["\']', text, re.IGNORECASE)
                    if text_match:
                        params["selector"] = f"button:has-text('{text_match.group(1)}')"
        
        elif tool_name == "type_text":
            # Look for selector and text
            selector_match = re.search(r'(?:selector|element|input)[^a-zA-Z]+([#.][\w-]+|\w+)', text, re.IGNORECASE)
            if selector_match:
                params["selector"] = selector_match.group(1)
            
            text_match = re.search(r'(?:text|type|enter)[^a-zA-Z]+["\']([^"\']+)["\']', text, re.IGNORECASE)
            if text_match:
                params["text"] = text_match.group(1)
        
        elif tool_name == "scroll":
            # Look for direction
            if re.search(r'\bdown\b', text, re.IGNORECASE):
                params["direction"] = "down"
            elif re.search(r'\bup\b', text, re.IGNORECASE):
                params["direction"] = "up"
            elif re.search(r'\bleft\b', text, re.IGNORECASE):
                params["direction"] = "left"
            elif re.search(r'\bright\b', text, re.IGNORECASE):
                params["direction"] = "right"
        
        elif tool_name == "extract_text":
            # Look for selector
            selector_match = re.search(r'(?:selector|element)[^a-zA-Z]+([#.][\w-]+)', text, re.IGNORECASE)
            if selector_match:
                params["selector"] = selector_match.group(1)
        
        # If no parameters extracted and tool typically needs them, log warning
        if not params and tool_name not in {"get_page_state", "screenshot", "refresh", "go_back", "go_forward"}:
            logger.warning(f"[ToT] Could not extract parameters for {tool_name} from: {text[:200]}")
        
        return params

