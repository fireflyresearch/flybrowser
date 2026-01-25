# Copyright 2026 Firefly Software Solutions Inc

"""
Centralized Tool Description Manager

Loads tool descriptions from the centralized YAML template and provides
them to tool classes for consistent metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from flybrowser.agents.tools.base import ToolMetadata, ToolParameter
from flybrowser.agents.types import ToolCategory, SafetyLevel
from flybrowser.llm.base import ModelCapability
from flybrowser.utils.logger import logger


class ToolDescriptionManager:
    """Manages centralized tool descriptions from YAML template"""
    
    def __init__(self, descriptions_file: Optional[Path] = None):
        """
        Initialize tool description manager
        
        Args:
            descriptions_file: Path to tool_descriptions.yaml (auto-detected if not provided)
        """
        if descriptions_file is None:
            # Auto-detect path
            descriptions_file = (
                Path(__file__).parent.parent.parent / 
                "prompts" / "metadata" / "tool_descriptions.yaml"
            )
        
        self.descriptions_file = descriptions_file
        self._descriptions: Dict[str, Dict[str, Any]] = {}
        self._load_descriptions()
    
    def _load_descriptions(self):
        """Load tool descriptions from YAML file"""
        if not self.descriptions_file.exists():
            logger.warning(f"Tool descriptions file not found: {self.descriptions_file}")
            return
        
        try:
            with open(self.descriptions_file, 'r') as f:
                data = yaml.safe_load(f)
            
            self._descriptions = data.get('tools', {})
            tool_count = len(self._descriptions)
            logger.info(f"Loaded {tool_count} tool descriptions from centralized template")
            
        except Exception as e:
            logger.error(f"Failed to load tool descriptions: {e}")
    
    def get_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Get ToolMetadata for a tool by name
        
        Args:
            tool_name: Name of the tool (e.g., "complete", "navigate")
            
        Returns:
            ToolMetadata object or None if not found
        """
        if tool_name not in self._descriptions:
            logger.warning(f"Tool description not found: {tool_name}")
            return None
        
        tool_data = self._descriptions[tool_name]
        
        try:
            # Convert parameters
            parameters = []
            for param_data in tool_data.get('parameters', []):
                param = ToolParameter(
                    name=param_data['name'],
                    type=param_data['type'],
                    description=param_data['description'],
                    required=param_data.get('required', False),
                    default=param_data.get('default'),
                    enum=param_data.get('enum'),
                )
                parameters.append(param)
            
            # Convert category and safety level
            category = ToolCategory[tool_data['category']]
            safety_level = SafetyLevel[tool_data['safety_level']]
            
            # Parse capability requirements
            required_caps = []
            if 'required_capabilities' in tool_data:
                required_caps = [
                    ModelCapability[cap]
                    for cap in tool_data['required_capabilities']
                ]
            
            optimal_caps = []
            if 'optimal_capabilities' in tool_data:
                optimal_caps = [
                    ModelCapability[cap]
                    for cap in tool_data['optimal_capabilities']
                ]
            
            # Load examples if provided
            examples = tool_data.get('examples', [])
            
            # Load returns description if provided
            returns_description = tool_data.get('returns_description', '')
            
            # Create metadata
            metadata = ToolMetadata(
                name=tool_data['name'],
                description=tool_data['description'],
                category=category,
                safety_level=safety_level,
                parameters=parameters,
                required_capabilities=required_caps,
                optimal_capabilities=optimal_caps,
                examples=examples,
                returns_description=returns_description,
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to create metadata for {tool_name}: {e}")
            return None
    
    def get_description(self, tool_name: str) -> Optional[str]:
        """
        Get just the description text for a tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Description string or None
        """
        if tool_name in self._descriptions:
            return self._descriptions[tool_name].get('description')
        return None
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List all tool names, optionally filtered by category
        
        Args:
            category: Optional category filter (e.g., "SYSTEM", "NAVIGATION")
            
        Returns:
            List of tool names
        """
        if category:
            return [
                name for name, data in self._descriptions.items()
                if data.get('category') == category
            ]
        return list(self._descriptions.keys())
    
    def get_all_descriptions(self) -> Dict[str, str]:
        """
        Get all tool descriptions as a dictionary
        
        Returns:
            Dictionary of tool_name -> description
        """
        return {
            name: data.get('description', '')
            for name, data in self._descriptions.items()
        }
    
    def validate_tool(self, tool_name: str, tool_metadata: ToolMetadata) -> List[str]:
        """
        Validate that a tool's metadata matches the centralized description
        
        Args:
            tool_name: Name of the tool
            tool_metadata: The tool's current metadata
            
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        if tool_name not in self._descriptions:
            issues.append(f"Tool '{tool_name}' not found in centralized descriptions")
            return issues
        
        central_data = self._descriptions[tool_name]
        
        # Check description
        if tool_metadata.description != central_data['description']:
            issues.append(
                f"Description mismatch for '{tool_name}'. "
                f"Expected: '{central_data['description'][:50]}...'"
            )
        
        # Check parameter count
        central_params = len(central_data.get('parameters', []))
        tool_params = len(tool_metadata.parameters)
        if central_params != tool_params:
            issues.append(
                f"Parameter count mismatch for '{tool_name}'. "
                f"Expected {central_params}, got {tool_params}"
            )
        
        return issues


# Singleton instance
_tool_description_manager: Optional[ToolDescriptionManager] = None


def get_tool_description_manager() -> ToolDescriptionManager:
    """Get or create the global tool description manager singleton"""
    global _tool_description_manager
    if _tool_description_manager is None:
        _tool_description_manager = ToolDescriptionManager()
    return _tool_description_manager


def get_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """
    Convenience function to get tool metadata from centralized descriptions
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        ToolMetadata object or None
        
    Example:
        >>> from flybrowser.agents.tools.descriptions import get_tool_metadata
        >>> metadata = get_tool_metadata("complete")
        >>> print(metadata.description)
    """
    manager = get_tool_description_manager()
    return manager.get_metadata(tool_name)
