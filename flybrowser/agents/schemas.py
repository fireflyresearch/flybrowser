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
Centralized JSON schemas for structured LLM output.

This module contains all JSON schemas used by FlyBrowser components
for structured LLM responses. Centralizing schemas:
- Ensures consistency across components
- Makes schema maintenance easier
- Provides clear documentation of expected response formats

Usage:
    from flybrowser.agents.schemas import REACT_RESPONSE_SCHEMA, PLAN_SCHEMA
    
    wrapper = StructuredLLMWrapper(llm)
    response = await wrapper.generate_structured(prompt, schema=PLAN_SCHEMA)
"""

from typing import Any, Dict


# =============================================================================
# ReAct Agent Schemas
# =============================================================================

REACT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your reasoning about the current state and what action to take next"
        },
        "action": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for the tool",
                    "additionalProperties": True
                }
            },
            "required": ["tool", "parameters"],
            "additionalProperties": False
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this action (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },
    "required": ["thought", "action"],
    "additionalProperties": False
}


# =============================================================================
# Task Planning Schemas
# =============================================================================

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name for this phase (e.g., 'Navigation', 'Data Extraction')"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this phase accomplishes"
                    },
                    "goals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "What needs to be accomplished"
                                },
                                "success_criteria": {
                                    "type": "string",
                                    "description": "How to determine if goal is met"
                                }
                            },
                            "required": ["description", "success_criteria"]
                        },
                        "description": "Goals to achieve in this phase"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Phase names that must complete before this"
                    }
                },
                "required": ["name", "description", "goals"]
            },
            "description": "Execution phases in order"
        }
    },
    "required": ["phases"]
}

PLAN_ADAPTATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_continue": {
            "type": "boolean",
            "description": "Whether execution should continue"
        },
        "adaptation_type": {
            "type": "string",
            "enum": ["retry", "modify_goal", "skip_goal", "new_phase", "abort"],
            "description": "Type of adaptation to make"
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation for the adaptation decision"
        },
        "modified_phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "goals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "success_criteria": {"type": "string"}
                            },
                            "required": ["description"]
                        }
                    }
                },
                "required": ["name", "goals"]
            },
            "description": "Modified or new phases (if adaptation_type requires)"
        }
    },
    "required": ["should_continue", "adaptation_type", "reasoning"]
}


# =============================================================================
# Obstacle Detection Schemas
# =============================================================================

OBSTACLE_DETECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_blocking": {
            "type": "boolean",
            "description": "Whether any obstacle is blocking page interaction"
        },
        "obstacles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "cookie_banner", "modal", "overlay", "popup",
                            "age_verification", "newsletter", "paywall",
                            "login_prompt", "captcha", "other"
                        ],
                        "description": "Type of obstacle"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the obstacle"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence that this is a real obstacle"
                    },
                    "strategies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "click_button", "click_text", "click_selector",
                                        "press_key", "scroll_away", "wait_for_dismiss",
                                        "click_outside", "click_coordinates", "coordinates",
                                        "css", "text", "aria"
                                    ],
                                    "description": "Strategy type"
                                },
                                "value": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "x": {"type": "integer"},
                                                "y": {"type": "integer"}
                                            },
                                            "required": ["x", "y"]
                                        }
                                    ],
                                    "description": "Value for the strategy (selector, text, key, or {x,y} coordinates)"
                                },
                                "priority": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Priority (1=highest)"
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence this strategy will work"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Why this strategy was chosen"
                                }
                            },
                            "required": ["type", "value", "priority"]
                        },
                        "description": "Dismissal strategies in priority order"
                    }
                },
                "required": ["type", "confidence", "strategies"]
            },
            "description": "Detected obstacles"
        }
    },
    "required": ["is_blocking", "obstacles"]
}


# =============================================================================
# Search and Analysis Schemas
# =============================================================================

SEARCH_RANKING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis_summary": {
            "type": "string",
            "description": "Brief summary of the search results analysis"
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key insights from analyzing the results"
        },
        "ranked_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "relevance_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this result is relevant"
                    }
                },
                "required": ["title", "url", "relevance_score"]
            },
            "description": "Results ranked by relevance"
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recommended next actions"
        }
    },
    "required": ["analysis_summary", "ranked_results"]
}

HTML_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["button", "link", "menu_toggle", "menu", "form_input", "form_submit", "unknown"]
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["navigation", "language_switch", "menu_control", "search", "login", "unknown"]
                    },
                    "text": {"type": "string"},
                    "selector": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning": {"type": "string"},
                    "is_visible": {"type": "boolean"},
                    "href": {"type": "string"},
                    "aria_label": {"type": "string"},
                    "attributes": {"type": "object"}
                },
                "required": ["type", "selector"]
            },
            "description": "Interactive elements found on the page"
        },
        "analysis_summary": {
            "type": "object",
            "properties": {
                "warnings": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    "required": ["elements"]
}


# =============================================================================
# Page Analysis Schemas (Multi-Screenshot)
# =============================================================================

PAGE_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "header", "navigation", "hero", "content",
                            "sidebar", "footer", "form", "unknown"
                        ],
                        "description": "Section type"
                    },
                    "name": {
                        "type": "string",
                        "description": "Section name"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this section contains"
                    },
                    "scroll_range": {
                        "type": "object",
                        "properties": {
                            "start_y": {"type": "integer"},
                            "end_y": {"type": "integer"}
                        },
                        "description": "Y coordinate range in pixels"
                    },
                    "screenshot_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Which screenshot(s) show this section"
                    },
                    "elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key elements in this section"
                    },
                    "navigation_links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        },
                        "description": "Navigation links in this section"
                    }
                },
                "required": ["type", "name"]
            },
            "description": "Page sections identified across screenshots"
        },
        "navigation_structure": {
            "type": "object",
            "properties": {
                "main_menu": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Main navigation menu items"
                },
                "footer_links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Footer navigation links"
                },
                "sidebar_links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sidebar navigation links"
                }
            },
            "description": "Overall navigation structure"
        },
        "main_content_area": {
            "type": "object",
            "properties": {
                "start_y": {"type": "integer"},
                "end_y": {"type": "integer"}
            },
            "description": "Y coordinate range of main content"
        },
        "summary": {
            "type": "string",
            "description": "Overall page summary"
        }
    },
    "required": ["sections", "summary"]
}


# =============================================================================
# Page Exploration Schemas
# =============================================================================

PAGE_EXPLORATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["header", "navigation", "hero", "content", "sidebar", "footer", "form", "other"]
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "scroll_start": {"type": "integer"},
                    "scroll_end": {"type": "integer"},
                    "key_elements": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["type", "scroll_start", "scroll_end"]
            },
            "description": "Page sections identified"
        },
        "navigation_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "likely_url": {"type": "string"},
                    "location": {"type": "string"}
                }
            },
            "description": "Navigation links found"
        },
        "interactive_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "text": {"type": "string"},
                    "selector": {"type": "string"},
                    "location": {"type": "string"}
                }
            },
            "description": "Interactive elements found"
        },
        "summary": {
            "type": "string",
            "description": "Overall page summary"
        }
    },
    "required": ["sections"]
}


# =============================================================================
# Element Detection Schema
# =============================================================================

ELEMENT_DETECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "CSS selector or XPath to locate the element"
        },
        "selector_type": {
            "type": "string",
            "enum": ["css", "xpath"],
            "description": "Type of selector"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in this selector (0.0-1.0)"
        },
        "reasoning": {
            "type": "string",
            "description": "Why this element was selected"
        },
        "alternative_selectors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alternative selectors to try if primary fails"
        }
    },
    "required": ["selector", "selector_type", "confidence"]
}


# =============================================================================
# Response Schemas (for SDK responses)
# =============================================================================

EXTRACTION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "description": "Extracted data matching user's request",
            "additionalProperties": True
        },
        "source": {
            "type": "string",
            "description": "Where the data was extracted from"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the extraction"
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any warnings about the extraction"
        }
    },
    "required": ["data"]
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_schema_for_component(component_name: str) -> Dict[str, Any]:
    """
    Get the appropriate schema for a component.
    
    Args:
        component_name: Name of the component (e.g., 'react', 'planner', 'obstacle')
        
    Returns:
        JSON schema for the component
    """
    schema_map = {
        "react": REACT_RESPONSE_SCHEMA,
        "react_agent": REACT_RESPONSE_SCHEMA,
        "planner": PLAN_SCHEMA,
        "plan": PLAN_SCHEMA,
        "plan_adaptation": PLAN_ADAPTATION_SCHEMA,
        "obstacle": OBSTACLE_DETECTION_SCHEMA,
        "obstacle_detector": OBSTACLE_DETECTION_SCHEMA,
        "search_rank": SEARCH_RANKING_SCHEMA,
        "search_ranking": SEARCH_RANKING_SCHEMA,
        "html_analysis": HTML_ANALYSIS_SCHEMA,
        "page_analysis": PAGE_ANALYSIS_SCHEMA,
        "page_analyzer": PAGE_ANALYSIS_SCHEMA,
        "page_exploration": PAGE_EXPLORATION_SCHEMA,
        "element_detection": ELEMENT_DETECTION_SCHEMA,
        "element_detector": ELEMENT_DETECTION_SCHEMA,
        "extraction": EXTRACTION_RESPONSE_SCHEMA,
    }
    
    return schema_map.get(component_name, {})


def validate_schema_basics(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Basic schema validation without external dependencies.
    
    For more comprehensive validation, use the StructuredLLMWrapper's
    validate_json_schema function.
    
    Args:
        data: Data to validate
        schema: Schema to validate against
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not isinstance(data, dict):
        return False, ["Data must be an object"]
    
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors
