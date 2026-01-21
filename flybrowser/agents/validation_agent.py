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
Response validation agent for ensuring LLM responses match expected formats.

This module provides the ResponseValidator class which validates LLM responses
against JSON schemas and attempts to fix malformed responses by asking the LLM
to correct them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from flybrowser.llm.base import BaseLLMProvider
from flybrowser.utils.logger import logger


class ResponseValidator:
    """
    Validates and fixes LLM responses to ensure they match expected formats.
    
    This validator acts as a "judge" that ensures LLM responses are properly
    formatted JSON matching the expected schema. If a response is malformed,
    it will attempt to extract or fix the JSON, or ask the LLM to correct it.
    
    Example:
        >>> validator = ResponseValidator(llm_provider)
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "selector": {"type": "string"},
        ...         "confidence": {"type": "number"}
        ...     },
        ...     "required": ["selector"]
        ... }
        >>> result = await validator.validate_and_fix(
        ...     response_text,
        ...     schema,
        ...     max_attempts=3
        ... )
    """
    
    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """
        Initialize the response validator.
        
        Args:
            llm_provider: LLM provider for fixing invalid responses
        """
        self.llm = llm_provider
        
    async def validate_and_fix(
        self,
        response_text: str,
        schema: Dict[str, Any],
        context: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Validate a response against a schema and fix if needed.
        
        This method will:
        1. Try to parse the response as JSON
        2. If parsing fails, attempt to extract JSON from the text
        3. If extraction fails, ask the LLM to fix the response
        4. Validate the result against the schema
        5. Repeat up to max_attempts times
        
        Args:
            response_text: The LLM response to validate
            schema: JSON schema the response should match
            context: Optional context about what the response should contain
            max_attempts: Maximum validation/fix attempts (default: 3)
            
        Returns:
            Validated and parsed JSON object
            
        Raises:
            ValueError: If validation fails after max_attempts
        """
        current_text = response_text
        
        for attempt in range(max_attempts):
            try:
                # Try to parse as JSON
                parsed = self._try_parse_json(current_text)
                
                if parsed is not None:
                    # Validate against schema
                    if self._validate_schema(parsed, schema):
                        logger.debug(f"Response validated successfully on attempt {attempt + 1}")
                        return parsed
                    else:
                        logger.warning(f"Response doesn't match schema on attempt {attempt + 1}")
                        
                # If we're here, parsing failed or schema didn't match
                if attempt < max_attempts - 1:
                    logger.info(f"Attempting to fix response (attempt {attempt + 1}/{max_attempts})")
                    current_text = await self._ask_llm_to_fix(
                        current_text,
                        schema,
                        context,
                        attempt + 1
                    )
                else:
                    # Last attempt - try best-effort extraction
                    logger.warning("Max attempts reached, trying best-effort extraction")
                    parsed = self._extract_json_best_effort(current_text)
                    if parsed and self._validate_schema(parsed, schema):
                        return parsed
                    raise ValueError(
                        f"Failed to validate response after {max_attempts} attempts. "
                        f"Last response: {current_text[:200]}..."
                    )
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    current_text = await self._ask_llm_to_fix(
                        current_text,
                        schema,
                        context,
                        attempt + 1
                    )
                else:
                    raise ValueError(
                        f"Failed to parse JSON after {max_attempts} attempts: {e}"
                    )
                    
        raise ValueError(f"Validation failed after {max_attempts} attempts")
        
    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse text as JSON, with multiple strategies.
        
        Args:
            text: Text to parse
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Strategy 3: Find first { to last } and try to parse
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
            
        return None
        
    def _extract_json_best_effort(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort JSON extraction from text.
        
        This tries various heuristics to extract valid JSON from
        potentially malformed or verbose responses.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON or None
        """
        # Try all JSON objects in the text
        for match in re.finditer(r'\{[^{}]*\}', text):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
                
        return None
        
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate data against a simple JSON schema.
        
        This is a basic validator that checks:
        - Required properties exist
        - Property types match
        
        Args:
            data: Data to validate
            schema: JSON schema to validate against
            
        Returns:
            True if valid, False otherwise
        """
        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                logger.debug(f"Missing required property: {prop}")
                return False
                
        # Check property types
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in data:
                expected_type = prop_schema.get("type")
                value = data[prop]
                
                if expected_type == "string" and not isinstance(value, str):
                    logger.debug(f"Property {prop} should be string, got {type(value)}")
                    return False
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    logger.debug(f"Property {prop} should be number, got {type(value)}")
                    return False
                elif expected_type == "array" and not isinstance(value, list):
                    logger.debug(f"Property {prop} should be array, got {type(value)}")
                    return False
                elif expected_type == "object" and not isinstance(value, dict):
                    logger.debug(f"Property {prop} should be object, got {type(value)}")
                    return False
                    
        return True
        
    async def _ask_llm_to_fix(
        self,
        malformed_response: str,
        schema: Dict[str, Any],
        context: Optional[str],
        attempt: int
    ) -> str:
        """
        Ask the LLM to fix a malformed response.
        
        Args:
            malformed_response: The malformed response to fix
            schema: Expected JSON schema
            context: Optional context about the task
            attempt: Current attempt number
            
        Returns:
            Fixed response from LLM
        """
        fix_prompt = f"""The previous response was not valid JSON or didn't match the expected format.

Previous response:
{malformed_response[:500]}

Expected JSON schema:
{json.dumps(schema, indent=2)}

{f"Context: {context}" if context else ""}

Please provide ONLY a valid JSON object that matches the schema. Do not include:
- Explanations or reasoning
- Markdown code blocks
- Any text before or after the JSON

Respond with ONLY the JSON object, starting with {{ and ending with }}.

Attempt {attempt}/3 - This is critical. JSON only."""

        try:
            response = await self.llm.generate(
                prompt=fix_prompt,
                system_prompt="You are a JSON formatting expert. Return ONLY valid JSON with no additional text.",
                temperature=0.1,  # Very low temperature for precise formatting
            )
            return response.content
        except Exception as e:
            logger.error(f"Failed to ask LLM to fix response: {e}")
            return malformed_response
