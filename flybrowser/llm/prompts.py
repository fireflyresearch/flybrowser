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
Prompt templates for LLM interactions.

This module provides backward-compatible prompt constants while also
integrating with the centralized prompt management system in flybrowser.prompts.

For new code, prefer using the PromptManager:
    >>> from flybrowser.prompts import PromptManager
    >>> manager = PromptManager()
    >>> prompts = manager.get_prompt("action_planning", instruction="...", url="...")

The constants below are maintained for backward compatibility.
"""

from typing import Any, Dict, Optional

# Lazy-loaded prompt manager for centralized prompts
_prompt_manager = None


def get_prompt_manager():
    """Get or create the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        from flybrowser.prompts import PromptManager
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(
    name: str,
    version: Optional[str] = None,
    **variables: Any,
) -> Dict[str, str]:
    """
    Get a rendered prompt from the centralized prompt system.

    Args:
        name: Template name (e.g., "action_planning", "data_extraction")
        version: Optional template version
        **variables: Variables to render the template

    Returns:
        Dictionary with 'system' and 'user' prompts

    Example:
        >>> prompts = get_prompt(
        ...     "action_planning",
        ...     instruction="Click the login button",
        ...     url="https://example.com",
        ...     title="Example",
        ...     visible_elements="[...]"
        ... )
        >>> print(prompts["system"])
        >>> print(prompts["user"])
    """
    manager = get_prompt_manager()
    return manager.get_prompt(name, version=version, **variables)


# ============================================================================
# BACKWARD COMPATIBLE CONSTANTS
# These are maintained for existing code that imports them directly.
# New code should use get_prompt() or PromptManager instead.
# ============================================================================

# System prompts
ELEMENT_DETECTION_SYSTEM = """You are an expert web automation assistant. Your task is to identify elements on a web page based on natural language descriptions.

You will be provided with:
1. A screenshot of the web page
2. The page's HTML structure
3. A description of the element to find

Analyze the page and identify the best selector for the described element. Consider:
- Element visibility and interactability
- Semantic meaning and context
- Reliability of the selector

CRITICAL: You MUST respond with ONLY a valid JSON object. Do NOT include:
- Explanations or reasoning outside the JSON
- Markdown code blocks (```json)
- Any text before or after the JSON object

Respond with ONLY this JSON structure:
{
  "selector": "CSS selector or XPath string",
  "selector_type": "css" or "xpath",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}

Start your response with { and end with } - nothing else.
"""

EXTRACTION_SYSTEM = """You are a data extraction specialist. Extract specific information from web pages as JSON.

You will be provided with:
1. A screenshot of the web page (optional)
2. The page's HTML content
3. The current page URL and title
4. A query describing what data to extract

EXTRACTION RULES:
✓ Extract ACTUAL CONTENT from the page (visible text, numbers, headings)
✓ Return REAL VALUES you see on the page
✓ Structure as clear, simple JSON
✗ DO NOT return schema definitions or descriptions
✗ DO NOT return HTML attributes, IDs, classes
✗ DO NOT return URL parameters or technical metadata
✗ DO NOT explain or add commentary

OUTPUT FORMAT:
ALWAYS use this structure:
{
  "extracted_data": <your extracted data here>
}

- For SINGLE items, use an object
- For MULTIPLE items, use an array
- Put ACTUAL VALUES from the page, not descriptions

GOOD EXAMPLES:
Query: "Get product name and price"
{
  "extracted_data": {
    "product_name": "Blue Widget",
    "price": 29.99
  }
}

Query: "Extract article title and author"
{
  "extracted_data": {
    "title": "How to Use Web Scraping",
    "author": "Jane Smith"
  }
}

Query: "Get top 5 items with scores"
{
  "extracted_data": [
    {"title": "First Item", "score": 142},
    {"title": "Second Item", "score": 98},
    {"title": "Third Item", "score": 87},
    {"title": "Fourth Item", "score": 65},
    {"title": "Fifth Item", "score": 54}
  ]
}

BAD EXAMPLES (DO NOT DO THIS):
{
  "type": "object",
  "properties": {...}
}

{
  "extracted_data": {
    "description": "The extracted data"
  }
}

CRITICAL: Return ONLY valid JSON with actual extracted values. No markdown, no explanations, no schema definitions.
Start with { and end with } - nothing else.
"""

ACTION_PLANNING_SYSTEM = """You are a web automation planner. Break down instructions into specific browser actions.

You will be provided with:
1. Current page state and screenshot
2. An instruction (e.g., "search for X")
3. Visible interactive elements

Available action types:
- click: Click an element
- type: Type text into an input
- fill: Fill a form field  
- select: Select from dropdown
- hover: Hover over element
- scroll: Scroll page
- wait: Wait for condition
- press_key: Press keyboard key

OUTPUT FORMAT - ALWAYS use this structure:
{
  "actions": [
    {"action_type": "...", "target": "...", "value": "...", "options": {}}
  ],
  "reasoning": "Brief explanation"
}

GOOD EXAMPLES:
Instruction: "Click the login button"
{
  "actions": [
    {"action_type": "click", "target": "login button", "value": null, "options": {}}
  ],
  "reasoning": "Click the login button to proceed"
}

Instruction: "Search for 'python tutorial'"
{
  "actions": [
    {"action_type": "click", "target": "search input box", "value": null, "options": {}},
    {"action_type": "type", "target": "search input", "value": "python tutorial", "options": {}},
    {"action_type": "press_key", "target": "search input", "value": "Enter", "options": {}}
  ],
  "reasoning": "Click search box, type query, press Enter to submit"
}

Instruction: "Fill form with name John Doe and email john@test.com"
{
  "actions": [
    {"action_type": "fill", "target": "name input field", "value": "John Doe", "options": {}},
    {"action_type": "fill", "target": "email input field", "value": "john@test.com", "options": {}}
  ],
  "reasoning": "Fill the form fields with provided information"
}

BAD EXAMPLES (DO NOT DO THIS):
{
  "type": "object",
  "properties": {...}
}

{
  "actions": {
    "description": "The actions array"
  }
}

CRITICAL: Return ONLY valid JSON with actual action steps. No markdown, no explanations outside JSON, no schema definitions.
Start with { and end with } - nothing else.
"""

NAVIGATION_SYSTEM = """You are a web navigation expert. Determine how to navigate based on the goal.

Navigation types:
- url: Direct URL navigation
- link: Follow a link on the page
- back: Go back in history
- forward: Go forward in history  
- refresh: Reload the page
- search: Perform a search

OUTPUT FORMAT - ALWAYS use this structure:
{
  "type": "navigation_type",
  "url": "https://..." (for url type),
  "link_description": "description" (for link type),
  "query": "search query" (for search type),
  "reasoning": "Brief explanation"
}

GOOD EXAMPLES:
Goal: "Go to the products page"
{
  "type": "link",
  "url": null,
  "link_description": "products link in navigation",
  "query": null,
  "reasoning": "Follow the products link in the main menu"
}

Goal: "Navigate to https://example.com"
{
  "type": "url",
  "url": "https://example.com",
  "link_description": null,
  "query": null,
  "reasoning": "Direct URL navigation"
}

Goal: "Search for laptops"
{
  "type": "search",
  "url": null,
  "link_description": null,
  "query": "laptops",
  "reasoning": "Use the search functionality with query 'laptops'"
}

Goal: "Go back to the previous page"
{
  "type": "back",
  "url": null,
  "link_description": null,
  "query": null,
  "reasoning": "Navigate back in browser history"
}

BAD EXAMPLES (DO NOT DO THIS):
{
  "type": "object",
  "properties": {...}
}

CRITICAL: Return ONLY valid JSON with actual navigation plan. No markdown, no explanations, no schema definitions.
Start with { and end with } - nothing else.
"""

MONITORING_SYSTEM = """You are a page monitoring expert. Parse monitoring instructions into structured conditions.

Change types:
- content: Monitor page content changes
- element: Monitor specific elements
- value: Monitor values with thresholds  
- presence: Monitor element appearance
- absence: Monitor element disappearance

Operators (for value monitoring):
- equals, not_equals, greater_than, less_than, contains

OUTPUT FORMAT - ALWAYS use this structure:
{
  "conditions": [
    {"description": "...", "change_type": "...", "operator": "...", "threshold": value, "element_description": "..."}
  ]
}

GOOD EXAMPLES:
Instruction: "Monitor price and alert if it drops below $50"
{
  "conditions": [
    {
      "description": "Price drops below $50",
      "change_type": "value",
      "operator": "less_than",
      "threshold": 50,
      "element_description": "price element"
    }
  ]
}

Instruction: "Watch for the Add to Cart button to appear"
{
  "conditions": [
    {
      "description": "Add to Cart button appears",
      "change_type": "presence",
      "operator": null,
      "threshold": null,
      "element_description": "Add to Cart button"
    }
  ]
}

Instruction: "Monitor for any changes on this page"
{
  "conditions": [
    {
      "description": "Any content changes",
      "change_type": "content",
      "operator": null,
      "threshold": null,
      "element_description": null
    }
  ]
}

BAD EXAMPLES (DO NOT DO THIS):
{
  "type": "object",
  "properties": {...}
}

CRITICAL: Return ONLY valid JSON with actual monitoring conditions. No markdown, no explanations, no schema definitions.
Start with { and end with } - nothing else.
"""

WORKFLOW_SYSTEM = """You are a workflow planning expert. Break down complex tasks into sequential workflow steps.

Available step types:
- navigate: Go to URLs or follow links
- action: Click, type, fill forms
- extract: Get data from the page
- wait: Wait for elements or conditions
- assert: Verify conditions
- store: Save values to variables

OUTPUT FORMAT - ALWAYS use this structure:
{
  "name": "Workflow name",
  "description": "Brief description",
  "steps": [
    {"step_type": "...", "instruction": "...", "store_as": "variable_name"}
  ]
}

GOOD EXAMPLES:
Task: "Log in with email and password"
{
  "name": "User Login",
  "description": "Navigate and log in to the site",
  "steps": [
    {"step_type": "navigate", "instruction": "Go to login page", "store_as": null},
    {"step_type": "action", "instruction": "Fill email field with {{email}}", "store_as": null},
    {"step_type": "action", "instruction": "Fill password field with {{password}}", "store_as": null},
    {"step_type": "action", "instruction": "Click login button", "store_as": null},
    {"step_type": "wait", "instruction": "Wait for dashboard to load", "store_as": null}
  ]
}

Task: "Search for products and extract prices"
{
  "name": "Product Search and Price Extraction",
  "description": "Search and extract product information",
  "steps": [
    {"step_type": "action", "instruction": "Type 'laptop' in search box and submit", "store_as": null},
    {"step_type": "wait", "instruction": "Wait for search results", "store_as": null},
    {"step_type": "extract", "instruction": "Get product names and prices", "store_as": "products"}
  ]
}

BAD EXAMPLES (DO NOT DO THIS):
{
  "type": "object",
  "properties": {...}
}

{
  "steps": {
    "description": "The steps array"
  }
}

CRITICAL: Return ONLY valid JSON with actual workflow steps. No markdown, no explanations outside JSON, no schema definitions.
Start with { and end with } - nothing else.
"""

# User prompt templates
ELEMENT_DETECTION_PROMPT = """Find the element that matches this description: {description}

Current page URL: {url}
Page title: {title}

HTML snippet:
{html_snippet}
"""

EXTRACTION_PROMPT = """Extract from this page: {query}

Page URL: {url}
Page title: {title}

INSTRUCTIONS:
1. Find the requested information in the HTML content below
2. Extract the ACTUAL VALUES you see (text, numbers, etc.)
3. Return as simple, direct JSON properties
4. For multiple items, use an array
5. DO NOT return schema structure - return the actual data!

Return ONLY the JSON object with extracted values. No explanations.
"""

ACTION_PLANNING_PROMPT = """Plan the steps to accomplish this task: {instruction}

Current page URL: {url}
Page title: {title}
Visible elements: {elements}
"""

NAVIGATION_PROMPT = """Navigate to accomplish this goal: {goal}

Current page URL: {url}
Available links: {links}
"""

MONITORING_PROMPT = """Check if this condition is met: {condition}

Current page URL: {url}
Page title: {title}
Page content: {content}
"""

WORKFLOW_PROMPT = """Execute this workflow step: {step}

Current state: {state}
Variables: {variables}
"""

