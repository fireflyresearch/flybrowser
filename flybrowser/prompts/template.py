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
Prompt template system with Jinja2 support.

This module provides a professional prompt template system for managing LLM prompts.
It supports:
- Jinja2 templating for dynamic prompts
- Version control for prompts
- Variable validation
- Few-shot examples
- Performance tracking
- A/B testing variants
- Template inheritance

The system enables centralized prompt management, making it easy to:
- Update prompts without code changes
- Test different prompt variations
- Track prompt performance
- Maintain prompt consistency across the application

Example:
    >>> template = PromptTemplate(
    ...     name="data_extraction",
    ...     version="1.0.0",
    ...     user_template="Extract {{ data_type }} from: {{ content }}",
    ...     required_variables=["data_type", "content"]
    ... )
    >>> rendered = template.render(data_type="emails", content="Contact us at...")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from jinja2 import Environment, Template, meta
from pydantic import BaseModel, Field, field_validator


class PromptTemplate(BaseModel):
    """
    A versioned prompt template with Jinja2 support and validation.

    This class represents a single prompt template that can be rendered with
    different variables. It supports system and user prompts, variable validation,
    few-shot examples, and performance tracking.

    Attributes:
        name: Unique template name (e.g., "data_extraction", "element_detection")
        version: Semantic version (e.g., "1.0.0", "2.1.0")
        description: Human-readable description of the template's purpose
        system_template: Optional Jinja2 template for system prompt
        user_template: Jinja2 template for user prompt (required)
        required_variables: List of required variable names
        optional_variables: Dict of optional variables with default values
        examples: List of few-shot examples to include in prompt
        metadata: Additional metadata (tags, author, etc.)
        usage_count: Number of times this template has been used
        success_count: Number of successful uses (for tracking effectiveness)
        variant: Variant identifier for A/B testing (e.g., "A", "B", "control")
        parent_template: Parent template name for inheritance

    Example:
        >>> template = PromptTemplate(
        ...     name="data_extraction",
        ...     version="1.0.0",
        ...     description="Extract structured data from web pages",
        ...     system_template="You are a data extraction expert.",
        ...     user_template="Extract {{ data_type }} from: {{ content }}",
        ...     required_variables=["data_type", "content"],
        ...     optional_variables={"format": "json"}
        ... )
        >>>
        >>> # Render the template
        >>> rendered = template.render(
        ...     data_type="product names",
        ...     content="<html>...</html>"
        ... )
        >>> print(rendered["user"])
    """

    name: str = Field(..., description="Template name")
    version: str = Field(default="1.0.0", description="Template version")
    description: str = Field(default="", description="Template description")

    system_template: Optional[str] = Field(None, description="System prompt template")
    user_template: str = Field(..., description="User prompt template")

    required_variables: List[str] = Field(default_factory=list, description="Required variables")
    optional_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Optional variables with defaults"
    )

    examples: List[Dict[str, str]] = Field(
        default_factory=list, description="Few-shot examples"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Performance tracking
    usage_count: int = Field(default=0, description="Number of times used")
    success_count: int = Field(default=0, description="Number of successful uses")

    # A/B testing
    variant: Optional[str] = Field(None, description="Variant identifier for A/B testing")
    parent_template: Optional[str] = Field(None, description="Parent template for inheritance")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    @field_validator("user_template", "system_template")
    @classmethod
    def validate_template_syntax(cls, v: Optional[str]) -> Optional[str]:
        """Validate Jinja2 template syntax."""
        if v is None:
            return v
        
        try:
            env = Environment()
            env.parse(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid template syntax: {e}")

    def get_required_variables(self) -> List[str]:
        """
        Extract required variables from templates.

        Returns:
            List of required variable names
        """
        env = Environment()
        
        variables = set()
        
        if self.system_template:
            ast = env.parse(self.system_template)
            variables.update(meta.find_undeclared_variables(ast))
        
        if self.user_template:
            ast = env.parse(self.user_template)
            variables.update(meta.find_undeclared_variables(ast))
        
        # Remove optional variables
        variables -= set(self.optional_variables.keys())
        
        return sorted(list(variables))

    def render(self, **variables: Any) -> Dict[str, str]:
        """
        Render the template with provided variables.

        Args:
            **variables: Variables to render the template with

        Returns:
            Dictionary with 'system' and 'user' prompts

        Raises:
            ValueError: If required variables are missing
        """
        # Check required variables
        required = set(self.get_required_variables())
        provided = set(variables.keys())
        missing = required - provided
        
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Merge with optional variables
        all_vars = {**self.optional_variables, **variables}
        
        # Add examples if present
        if self.examples:
            all_vars["examples"] = self.examples
        
        # Render templates
        env = Environment()
        
        result = {}
        
        if self.system_template:
            template = env.from_string(self.system_template)
            result["system"] = template.render(**all_vars)
        
        if self.user_template:
            template = env.from_string(self.user_template)
            result["user"] = template.render(**all_vars)
        
        # Track usage
        self.usage_count += 1
        
        return result

    def record_success(self) -> None:
        """Record a successful use of this template."""
        self.success_count += 1

    def get_success_rate(self) -> float:
        """
        Get the success rate of this template.

        Returns:
            Success rate (0.0 to 1.0)
        """
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def clone(self, variant: str, **modifications: Any) -> "PromptTemplate":
        """
        Create a variant of this template for A/B testing.

        Args:
            variant: Variant identifier
            **modifications: Fields to modify

        Returns:
            New PromptTemplate instance
        """
        data = self.model_dump()
        data["variant"] = variant
        data["parent_template"] = self.name
        data["usage_count"] = 0
        data["success_count"] = 0
        data.update(modifications)
        
        return PromptTemplate(**data)

