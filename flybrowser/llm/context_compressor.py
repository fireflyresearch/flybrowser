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
Context Compressor for Intelligent Memory Management.

This module provides LLM-powered compression of large content into concise
summaries while preserving essential information for web automation tasks.
Used by the ReAct framework to manage token budgets without losing critical
context needed for navigation and interaction.

Key Features:
- Content-type-aware compression (search results, page state, text, errors)
- URL preservation validation - ensures navigation targets are never lost
- Web automation field extraction (selectors, buttons, form fields)
- Hierarchical compression tiers (recent, session, archival)
- Template-based prompts via PromptManager
- Fallback extraction when LLM fails

Content Types:
- SEARCH_RESULTS: Preserves ALL result URLs and snippets
- PAGE_STATE: Preserves navigation links, buttons, selectors
- TEXT_CONTENT: Extracts key facts, preserves embedded URLs
- ERROR_INFO: Preserves error details and suggests recovery
- STRUCTURED_DATA: Preserves field names and values
- FORM_DATA: Preserves field names, types, and selectors
- GENERIC: Default balanced compression

Usage:
    >>> from flybrowser.llm.context_compressor import ContextCompressor, ContentType
    >>> 
    >>> compressor = ContextCompressor(llm_provider)
    >>> 
    >>> # Compress search results (URLs automatically preserved)
    >>> compressed = await compressor.compress_extraction(
    ...     large_data={"results": [{"title": "...", "url": "..."}]},
    ...     content_type=ContentType.SEARCH_RESULTS,
    ...     task_context="Find Python tutorials",
    ...     validate_urls=True,
    ... )
    >>> print(compressed.urls_mentioned)  # All URLs preserved
    >>> print(compressed.format_for_prompt())  # Formatted for LLM
    >>> 
    >>> # Compress page state (navigation preserved)
    >>> compressed = await compressor.compress_extraction(
    ...     large_data={"navigation_links": [...], "buttons": [...]},
    ...     content_type=ContentType.PAGE_STATE,
    ...     page_url="https://example.com",
    ... )
    >>> print(compressed.navigation_targets)  # Clickable elements
    >>> 
    >>> # Compress conversation history
    >>> history = await compressor.compress_history(
    ...     messages=[{"role": "user", "content": "..."}, ...],
    ...     keep_recent=3,
    ... )
    >>> print(history.context_summary)  # Summarized context

Notes:
    - URL-critical data (search, page_state) should NOT be compressed
      by the agent - this module is for compressible content only
    - The ReActAgent handles compression decisions and protects critical data
    - Use estimate_compression_benefit() to check if compression is worthwhile
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

from flybrowser.llm.token_budget import TokenEstimator
from flybrowser.prompts.manager import PromptManager
from flybrowser.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider


# =============================================================================
# COMPRESSION SCHEMAS - State-of-the-art structured compression for web automation
# =============================================================================

# Content type classification for schema selection
class ContentType(str, Enum):
    """Types of content that require different compression strategies."""
    SEARCH_RESULTS = "search_results"  # Search engine results with URLs
    PAGE_STATE = "page_state"          # Page navigation state, links, buttons
    TEXT_CONTENT = "text_content"      # Extracted text, paragraphs
    STRUCTURED_DATA = "structured_data" # Tables, lists, JSON data
    ERROR_INFO = "error_info"          # Errors, failures, exceptions
    FORM_DATA = "form_data"            # Form fields and inputs
    GENERIC = "generic"                # Default fallback


# Base schema with web-automation-specific fields
COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        # Core fields
        "summary": {
            "type": "string",
            "description": "One-line summary of the content (max 100 chars)"
        },
        "key_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet points of essential information (max 10 items, each max 150 chars)"
        },
        "data_values": {
            "type": "object",
            "description": "Key-value pairs of important extracted data (names, prices, counts, dates, etc.)",
            "additionalProperties": True
        },
        # URL preservation (CRITICAL for navigation)
        "urls_mentioned": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ALL URLs found - these are navigation targets the agent needs"
        },
        # Web automation fields
        "selectors_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "CSS/XPath selectors for interactive elements (max 20)"
        },
        "navigation_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "url": {"type": "string"},
                    "type": {"type": "string"}  # link, button, menu
                }
            },
            "description": "Clickable elements with labels for navigation"
        },
        "form_fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},  # text, email, password, select, etc.
                    "selector": {"type": "string"},
                    "required": {"type": "boolean"}
                }
            },
            "description": "Form input fields discovered"
        },
        # Context fields
        "page_context": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "title": {"type": "string"},
                "domain": {"type": "string"}
            },
            "description": "Current page context when data was extracted"
        },
        "task_progress": {
            "type": "object",
            "properties": {
                "accomplished": {"type": "string"},
                "remaining": {"type": "string"},
                "blockers": {"type": "array", "items": {"type": "string"}}
            },
            "description": "What was accomplished and what remains"
        },
        # Learning fields
        "successful_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Action patterns that worked (for future reuse)"
        },
        "error_info": {
            "type": "string",
            "description": "Any error or failure information (if present)"
        },
        # Quality metadata
        "compression_confidence": {
            "type": "number",
            "description": "Self-rated confidence in compression quality (0-1)"
        },
        "content_type": {
            "type": "string",
            "description": "Type of content that was compressed"
        }
    },
    "required": ["summary", "key_facts", "urls_mentioned"]
}

# Specialized schema for search results - prioritizes URL preservation
SEARCH_COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Summary of search query and results quality"
        },
        "total_results": {"type": "integer"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                    "relevance_note": {"type": "string"}
                },
                "required": ["title", "url"]
            },
            "description": "ALL search results with URLs (NEVER omit URLs)"
        },
        "answer_box": {
            "type": "string",
            "description": "Direct answer if present"
        },
        "recommended_result": {
            "type": "integer",
            "description": "Index of most relevant result for the task"
        },
        "compression_confidence": {"type": "number"}
    },
    "required": ["summary", "results"]
}

# Specialized schema for page state - prioritizes navigation elements
PAGE_STATE_COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "page_url": {"type": "string"},
        "page_title": {"type": "string"},
        "navigation_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "href": {"type": "string"},
                    "location": {"type": "string"}  # header, sidebar, footer, main
                },
                "required": ["text", "href"]
            },
            "description": "ALL navigation links (NEVER omit)"
        },
        "buttons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "selector": {"type": "string"},
                    "action_type": {"type": "string"}  # submit, navigation, modal, etc.
                }
            }
        },
        "forms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "submit_selector": {"type": "string"}
                }
            }
        },
        "compression_confidence": {"type": "number"}
    },
    "required": ["summary", "navigation_links"]
}

# Specialized schema for text/content extraction
CONTENT_COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "main_content": {
            "type": "string",
            "description": "Condensed main text content"
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Most important points from the content"
        },
        "data_extracted": {
            "type": "object",
            "description": "Structured data found (prices, dates, names, etc.)",
            "additionalProperties": True
        },
        "headings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Page structure from headings"
        },
        "urls_in_content": {
            "type": "array",
            "items": {"type": "string"}
        },
        "compression_confidence": {"type": "number"}
    },
    "required": ["summary", "main_content"]
}

# Schema for error/failure information
ERROR_COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "error_type": {"type": "string"},
        "error_message": {"type": "string"},
        "failed_action": {"type": "string"},
        "page_context": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "title": {"type": "string"}
            }
        },
        "possible_causes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "suggested_recovery": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alternative approaches to try"
        },
        "compression_confidence": {"type": "number"}
    },
    "required": ["summary", "error_type"]
}

# Schema for history compression - OPTIMIZED FOR BROWSER AUTOMATION
# This schema preserves the critical elements that browser automation agents need:
# - URLs visited (for navigation context)
# - Selectors used (for element interaction)
# - Data extracted (for task progress)
# - Successful patterns (for learning)
HISTORY_COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "context_summary": {
            "type": "string",
            "description": "Summary of what was discussed/done in the compressed turns (include key decisions)"
        },
        "actions_taken": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key actions that were performed (include tool names and parameters)"
        },
        "results_obtained": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important results or findings (include extracted data values)"
        },
        "current_state": {
            "type": "string",
            "description": "Current state after these turns (page URL, data collected, etc.)"
        },
        # CRITICAL FOR BROWSER AUTOMATION:
        "urls_visited": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ALL URLs that were visited or mentioned - agent needs these for navigation"
        },
        "urls_discovered": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs discovered (e.g., from search results) that haven't been visited yet"
        },
        "selectors_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "CSS selectors or element identifiers that were used successfully"
        },
        "data_extracted": {
            "type": "object",
            "description": "Key data values extracted (prices, names, IDs, etc.)",
            "additionalProperties": True
        },
        "successful_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Action patterns that worked well (for reuse)"
        },
        "failed_approaches": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Approaches that failed (to avoid repeating)"
        }
    },
    "required": ["context_summary", "actions_taken", "urls_visited"]
}


@dataclass
class CompressedContent:
    """
    Result of compressing large content - State-of-the-art structure for web automation.
    
    Attributes:
        summary: One-line summary of the content
        key_facts: List of bullet points with essential information
        data_values: Dictionary of important key-value pairs
        urls_mentioned: List of important URLs (CRITICAL - never lose these)
        selectors_found: CSS/XPath selectors discovered
        navigation_targets: Clickable elements with labels/URLs
        form_fields: Form input fields discovered
        page_context: URL/title when data was extracted
        task_progress: What was accomplished vs remaining
        successful_patterns: Action patterns that worked (for learning)
        error_info: Any error information if present
        content_type: Type of content that was compressed
        compression_confidence: LLM's self-rated confidence (0-1)
        original_size_chars: Size of original content in characters
        original_size_tokens: Estimated tokens in original content
        compressed_size_tokens: Tokens in compressed representation
        compression_tier: Which tier this compression belongs to (1=recent, 2=session, 3=archival)
    """
    # Core fields
    summary: str
    key_facts: List[str]
    data_values: Dict[str, Any] = field(default_factory=dict)
    
    # URL preservation (CRITICAL for navigation)
    urls_mentioned: List[str] = field(default_factory=list)
    
    # Web automation fields
    selectors_found: List[str] = field(default_factory=list)
    navigation_targets: List[Dict[str, str]] = field(default_factory=list)  # [{label, url, type}]
    form_fields: List[Dict[str, Any]] = field(default_factory=list)  # [{name, type, selector, required}]
    
    # Context fields
    page_context: Dict[str, str] = field(default_factory=dict)  # {url, title, domain}
    task_progress: Dict[str, Any] = field(default_factory=dict)  # {accomplished, remaining, blockers}
    
    # Learning fields
    successful_patterns: List[str] = field(default_factory=list)
    
    # Error/meta fields
    error_info: Optional[str] = None
    content_type: str = "generic"  # ContentType value
    compression_confidence: float = 0.0  # 0-1 self-rated quality
    
    # Size tracking
    original_size_chars: int = 0
    original_size_tokens: int = 0
    compressed_size_tokens: int = 0
    
    # Hierarchical compression tier
    compression_tier: int = 1  # 1=recent (full), 2=session (summary), 3=archival (facts only)
    
    def format_for_prompt(self) -> str:
        """Format compressed content for inclusion in a prompt."""
        lines = [f"**Summary**: {self.summary}"]
        
        # Key facts
        if self.key_facts:
            lines.append("**Key Facts**:")
            for fact in self.key_facts[:10]:
                lines.append(f"  • {fact}")
        
        # Data values
        if self.data_values:
            lines.append("**Data**:")
            for key, value in list(self.data_values.items())[:10]:
                value_str = str(value)[:100]
                lines.append(f"  • {key}: {value_str}")
        
        # URLs (CRITICAL - show all for navigation)
        if self.urls_mentioned:
            if len(self.urls_mentioned) <= 10:
                lines.append("**URLs (for navigation)**:")
                for url in self.urls_mentioned:
                    lines.append(f"  • {url}")
            else:
                lines.append(f"**URLs ({len(self.urls_mentioned)} total)**:")
                for url in self.urls_mentioned[:10]:
                    lines.append(f"  • {url}")
                lines.append(f"  ... and {len(self.urls_mentioned) - 10} more")
        
        # Navigation targets (if different from URLs)
        if self.navigation_targets:
            lines.append("**Navigation Targets**:")
            for target in self.navigation_targets[:10]:
                label = target.get('label', 'Unknown')
                url = target.get('url', '')
                lines.append(f"  • {label}: {url}")
        
        # Form fields (useful for form filling tasks)
        if self.form_fields:
            lines.append(f"**Form Fields ({len(self.form_fields)})**:")
            for ff in self.form_fields[:8]:
                name = ff.get('name', 'unknown')
                ftype = ff.get('type', 'text')
                lines.append(f"  • {name} ({ftype})")
        
        # Selectors (useful for interaction)
        if self.selectors_found:
            lines.append(f"**Selectors ({len(self.selectors_found)})**: {', '.join(self.selectors_found[:5])}")
        
        # Task progress
        if self.task_progress:
            accomplished = self.task_progress.get('accomplished', '')
            remaining = self.task_progress.get('remaining', '')
            if accomplished:
                lines.append(f"**Done**: {accomplished}")
            if remaining:
                lines.append(f"**Remaining**: {remaining}")
        
        # Successful patterns (for learning)
        if self.successful_patterns:
            lines.append(f"**Patterns that worked**: {', '.join(self.successful_patterns[:3])}")
        
        # Error info
        if self.error_info:
            lines.append(f"**Error**: {self.error_info[:200]}")
        
        # Metadata footer
        confidence_str = f", confidence: {self.compression_confidence:.0%}" if self.compression_confidence > 0 else ""
        lines.append(f"_(Compressed from {self.original_size_tokens:,} to {self.compressed_size_tokens:,} tokens{confidence_str})_")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "summary": self.summary,
            "key_facts": self.key_facts,
            "data_values": self.data_values,
            "urls_mentioned": self.urls_mentioned,
            "selectors_found": self.selectors_found,
            "navigation_targets": self.navigation_targets,
            "form_fields": self.form_fields,
            "page_context": self.page_context,
            "task_progress": self.task_progress,
            "successful_patterns": self.successful_patterns,
            "error_info": self.error_info,
            "content_type": self.content_type,
            "compression_confidence": self.compression_confidence,
            "original_size_chars": self.original_size_chars,
            "original_size_tokens": self.original_size_tokens,
            "compressed_size_tokens": self.compressed_size_tokens,
            "compression_tier": self.compression_tier,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressedContent":
        """Create from dictionary."""
        return cls(
            summary=data.get("summary", ""),
            key_facts=data.get("key_facts", []),
            data_values=data.get("data_values", {}),
            urls_mentioned=data.get("urls_mentioned", []),
            selectors_found=data.get("selectors_found", []),
            navigation_targets=data.get("navigation_targets", []),
            form_fields=data.get("form_fields", []),
            page_context=data.get("page_context", {}),
            task_progress=data.get("task_progress", {}),
            successful_patterns=data.get("successful_patterns", []),
            error_info=data.get("error_info"),
            content_type=data.get("content_type", "generic"),
            compression_confidence=data.get("compression_confidence", 0.0),
            original_size_chars=data.get("original_size_chars", 0),
            original_size_tokens=data.get("original_size_tokens", 0),
            compressed_size_tokens=data.get("compressed_size_tokens", 0),
            compression_tier=data.get("compression_tier", 1),
        )
    
    def get_all_urls(self) -> List[str]:
        """Get all URLs from all fields for validation."""
        urls = set(self.urls_mentioned)
        for target in self.navigation_targets:
            if target.get('url'):
                urls.add(target['url'])
        if self.page_context.get('url'):
            urls.add(self.page_context['url'])
        return list(urls)
    
    def validate_url_preservation(self, original_url_count: int) -> Tuple[bool, str]:
        """
        Validate that compression preserved URLs adequately.
        
        Args:
            original_url_count: Number of URLs in original content
            
        Returns:
            Tuple of (is_valid, message)
        """
        compressed_url_count = len(self.get_all_urls())
        
        # Allow some loss for very large URL sets, but preserve most
        if original_url_count <= 10:
            # Small sets - preserve all
            threshold = original_url_count
        elif original_url_count <= 50:
            # Medium sets - preserve at least 80%
            threshold = int(original_url_count * 0.8)
        else:
            # Large sets - preserve at least 50 or 60%
            threshold = max(50, int(original_url_count * 0.6))
        
        if compressed_url_count >= threshold:
            return True, f"URL preservation OK: {compressed_url_count}/{original_url_count}"
        else:
            return False, f"URL loss detected: {compressed_url_count}/{original_url_count} (need {threshold})"


@dataclass
class CompressedHistory:
    """
    Result of compressing conversation history.
    
    OPTIMIZED FOR BROWSER AUTOMATION - preserves navigation-critical data.
    
    Attributes:
        context_summary: Summary of the compressed turns
        actions_taken: List of key actions performed (with tool names)
        results_obtained: Important results from those turns
        current_state: State after the compressed turns
        urls_visited: URLs that were navigated to (CRITICAL)
        urls_discovered: URLs found but not yet visited (CRITICAL)
        selectors_used: Element selectors that worked successfully
        data_extracted: Key data values extracted
        successful_patterns: Action patterns that worked well
        failed_approaches: Approaches that failed (to avoid repeating)
        turns_compressed: Number of turns that were compressed
        original_tokens: Tokens in original history
        compressed_tokens: Tokens in compressed summary
    """
    context_summary: str
    actions_taken: List[str]
    results_obtained: List[str] = field(default_factory=list)
    current_state: str = ""
    # Browser automation critical fields:
    urls_visited: List[str] = field(default_factory=list)
    urls_discovered: List[str] = field(default_factory=list)
    selectors_used: List[str] = field(default_factory=list)
    data_extracted: Dict[str, Any] = field(default_factory=dict)
    successful_patterns: List[str] = field(default_factory=list)
    failed_approaches: List[str] = field(default_factory=list)
    # Metadata:
    turns_compressed: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    
    def format_as_message(self) -> str:
        """Format as a summary message for conversation history.
        
        IMPORTANT: This format preserves navigation-critical data that the
        browser automation agent needs for continued operation.
        """
        lines = [
            f"[Previous {self.turns_compressed} turns compressed]",
            f"**Summary**: {self.context_summary}",
        ]
        
        # URLs are CRITICAL - show them prominently
        if self.urls_visited:
            lines.append("**URLs Visited**:")
            for url in self.urls_visited[:10]:
                lines.append(f"  • {url}")
            if len(self.urls_visited) > 10:
                lines.append(f"  ... and {len(self.urls_visited) - 10} more")
        
        if self.urls_discovered:
            lines.append("**URLs Discovered (not yet visited)**:")
            for url in self.urls_discovered[:10]:
                lines.append(f"  • {url}")
            if len(self.urls_discovered) > 10:
                lines.append(f"  ... and {len(self.urls_discovered) - 10} more")
        
        if self.actions_taken:
            lines.append("**Actions Taken**:")
            for action in self.actions_taken[:8]:
                lines.append(f"  • {action}")
        
        if self.results_obtained:
            lines.append("**Results**:")
            for result in self.results_obtained[:5]:
                lines.append(f"  • {result}")
        
        if self.data_extracted:
            lines.append("**Data Extracted**:")
            for key, value in list(self.data_extracted.items())[:8]:
                value_str = str(value)[:100]
                lines.append(f"  • {key}: {value_str}")
        
        if self.selectors_used:
            lines.append(f"**Selectors Used**: {', '.join(self.selectors_used[:5])}")
        
        if self.successful_patterns:
            lines.append(f"**Patterns That Worked**: {', '.join(self.successful_patterns[:3])}")
        
        if self.failed_approaches:
            lines.append(f"**Failed Approaches** (avoid repeating): {', '.join(self.failed_approaches[:3])}")
        
        if self.current_state:
            lines.append(f"**Current State**: {self.current_state}")
        
        return "\n".join(lines)


class ContextCompressor:
    """
    LLM-powered context compression for web automation memory management.
    
    Uses structured LLM calls with content-type-specific schemas to compress
    large extraction data while preserving essential information for navigation
    and interaction tasks.
    
    Features:
        - Content-type-aware compression (search, page state, text, errors)
        - URL validation to ensure navigation targets are preserved
        - Template-based prompts via PromptManager
        - Fallback extraction when LLM fails
        - Hierarchical compression tiers
    
    Attributes:
        llm: LLM provider for compression calls
        compression_temperature: Temperature for compression (default 0.1 for consistency)
        max_output_tokens: Maximum tokens for compressed output (default 1000)
    
    Example:
        >>> from flybrowser.llm.context_compressor import ContextCompressor, ContentType
        >>> 
        >>> compressor = ContextCompressor(llm_provider)
        >>> 
        >>> # Compress text extraction (auto-detects content type)
        >>> result = await compressor.compress_extraction(
        ...     large_data={"text": "Long page content...", "title": "Page"},
        ...     task_context="Find product prices",
        ...     page_url="https://example.com/products",
        ... )
        >>> 
        >>> # Access compressed data
        >>> print(result.summary)  # One-line summary
        >>> print(result.key_facts)  # List of bullet points
        >>> print(result.urls_mentioned)  # Preserved URLs
        >>> print(result.compression_confidence)  # 0-1 confidence score
        >>> 
        >>> # Format for LLM prompt
        >>> context = result.format_for_prompt()
        >>> 
        >>> # Compress with specific content type
        >>> result = await compressor.compress_extraction(
        ...     large_data=page_state_data,
        ...     content_type=ContentType.PAGE_STATE,
        ...     validate_urls=True,
        ... )
        >>> print(result.navigation_targets)  # [{"label": ..., "url": ...}]
        >>> print(result.form_fields)  # [{"name": ..., "type": ...}]
    
    Note:
        The ReActAgent decides what to compress - URL-critical data like
        search results and page state should NOT be compressed.
    """
    
    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        compression_temperature: float = 0.1,
        max_output_tokens: int = 1000,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """
        Initialize the context compressor.
        
        Args:
            llm_provider: LLM provider for compression
            compression_temperature: Temperature for compression (low = deterministic)
            max_output_tokens: Maximum tokens for compressed output
            prompt_manager: Optional PromptManager for template-based prompts
        """
        self.llm = llm_provider
        self.compression_temperature = compression_temperature
        self.max_output_tokens = max_output_tokens
        self._prompt_manager = prompt_manager or PromptManager(PromptRegistry())
    
    async def compress_extraction(
        self,
        large_data: Union[Dict[str, Any], str],
        task_context: str = "",
        preserve_keys: Optional[List[str]] = None,
        content_type: Optional[ContentType] = None,
        page_url: str = "",
        page_title: str = "",
        validate_urls: bool = True,
        compression_tier: int = 2,  # Default to session tier
    ) -> CompressedContent:
        """
        Compress large extraction data into a concise summary.
        
        State-of-the-art compression with:
        - Content-type-specific schemas for optimal preservation
        - URL validation to ensure navigation targets are preserved
        - Web automation fields (selectors, forms, navigation targets)
        - Compression confidence scoring
        - Hierarchical tier support
        
        Args:
            large_data: The extraction data (dict or string)
            task_context: Context about what the task is trying to achieve
            preserve_keys: List of keys whose values must be preserved exactly
            content_type: Type of content for schema selection (auto-detected if None)
            page_url: Current page URL for context
            page_title: Current page title for context
            validate_urls: Whether to validate URL preservation after compression
            compression_tier: Hierarchical tier (1=recent/full, 2=session/summary, 3=archival/facts)
            
        Returns:
            CompressedContent with summary, key facts, and web automation fields
        """
        # Convert to dict if string
        data_dict: Dict[str, Any] = {}
        if isinstance(large_data, dict):
            data_dict = large_data
            content_str = json.dumps(large_data, indent=2, default=str)
        else:
            content_str = str(large_data)
            # Try to parse as JSON
            try:
                data_dict = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                data_dict = {"raw_content": content_str}
        
        original_chars = len(content_str)
        original_tokens = TokenEstimator.estimate(content_str).tokens
        
        # Count original URLs for validation
        original_url_count = self._count_urls_in_content(content_str)
        
        # If already small enough, create minimal compression
        if original_tokens < 500:
            return self._create_minimal_compression(
                content_str, original_chars, original_tokens,
                page_url=page_url, page_title=page_title, tier=compression_tier
            )
        
        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(data_dict, content_str)
        
        # Select appropriate schema based on content type
        schema, schema_name = self._get_schema_for_type(content_type)
        
        # Get prompts from template (with fallback to hardcoded)
        system_prompt, user_prompt = self._get_prompts_from_template(
            content_str=content_str,
            content_type=content_type,
            task_context=task_context,
            preserve_keys=preserve_keys,
            page_url=page_url,
            page_title=page_title,
            compression_tier=compression_tier,
        )
        
        try:
            response = await self.llm.generate_structured(
                prompt=user_prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=self.compression_temperature,
                max_tokens=self.max_output_tokens,
            )
            
            # Build CompressedContent from response
            result = self._build_compressed_content(
                response=response,
                content_type=content_type,
                original_chars=original_chars,
                original_tokens=original_tokens,
                page_url=page_url,
                page_title=page_title,
                compression_tier=compression_tier,
            )
            
            # Validate URL preservation if requested
            if validate_urls and original_url_count > 0:
                is_valid, validation_msg = result.validate_url_preservation(original_url_count)
                if not is_valid:
                    logger.warning(f"[ContextCompressor] {validation_msg}. Enhancing with fallback URLs.")
                    # Enhance with fallback URL extraction
                    fallback_urls = self._extract_all_urls(content_str)
                    result.urls_mentioned = list(set(result.urls_mentioned + fallback_urls))
            
            compression_ratio = original_tokens / max(result.compressed_size_tokens, 1)
            logger.info(
                f"[ContextCompressor] Compressed {content_type.value}: "
                f"{original_tokens:,} → {result.compressed_size_tokens:,} tokens "
                f"({compression_ratio:.1f}x, confidence: {result.compression_confidence:.0%})"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"[ContextCompressor] Compression failed: {e}, using fallback")
            return self._create_fallback_compression(
                content_str, original_chars, original_tokens, str(e),
                page_url=page_url, page_title=page_title, tier=compression_tier
            )
    
    def _detect_content_type(self, data: Dict[str, Any], content_str: str) -> ContentType:
        """Auto-detect content type from data structure."""
        # Check for search results patterns
        if any(k in data for k in ['results', 'search_results', 'organic_results', 'ranked_results']):
            return ContentType.SEARCH_RESULTS
        
        # Check for page state patterns
        if any(k in data for k in ['navigation_links', 'buttons', 'page_state', 'links']):
            return ContentType.PAGE_STATE
        
        # Check for form data patterns
        if any(k in data for k in ['form_fields', 'inputs', 'form_data']):
            return ContentType.FORM_DATA
        
        # Check for error patterns
        if any(k in data for k in ['error', 'error_message', 'exception', 'failed']):
            return ContentType.ERROR_INFO
        
        # Check for structured data patterns
        if any(k in data for k in ['items', 'products', 'data', 'table']):
            return ContentType.STRUCTURED_DATA
        
        # Check for text content patterns
        if any(k in data for k in ['text', 'content', 'body', 'main_content', 'paragraphs']):
            return ContentType.TEXT_CONTENT
        
        return ContentType.GENERIC
    
    def _get_schema_for_type(self, content_type: ContentType) -> Tuple[Dict[str, Any], str]:
        """Get appropriate schema for content type."""
        schemas = {
            ContentType.SEARCH_RESULTS: (SEARCH_COMPRESSION_SCHEMA, "search"),
            ContentType.PAGE_STATE: (PAGE_STATE_COMPRESSION_SCHEMA, "page_state"),
            ContentType.TEXT_CONTENT: (CONTENT_COMPRESSION_SCHEMA, "content"),
            ContentType.ERROR_INFO: (ERROR_COMPRESSION_SCHEMA, "error"),
        }
        return schemas.get(content_type, (COMPRESSION_SCHEMA, "generic"))
    
    def _get_prompts_from_template(
        self,
        content_str: str,
        content_type: ContentType,
        task_context: str,
        preserve_keys: Optional[List[str]],
        page_url: str,
        page_title: str,
        compression_tier: int,
    ) -> Tuple[str, str]:
        """
        Get system and user prompts from template.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Truncate content if too large
        if len(content_str) > 25000:
            truncated_content = (
                content_str[:15000] +
                f"\n\n... [{len(content_str) - 20000:,} characters omitted] ...\n\n" +
                content_str[-5000:]
            )
        else:
            truncated_content = content_str
        
        try:
            # Try to use template from PromptManager
            rendered = self._prompt_manager.get_prompt(
                "context_compression",
                enable_ab_testing=False,
                content=truncated_content,
                content_type=content_type.value,
                task_context=task_context,
                preserve_keys=preserve_keys or [],
                page_url=page_url,
                page_title=page_title,
                compression_tier=compression_tier,
            )
            return rendered.get("system", self._get_fallback_system_prompt(content_type)), rendered.get("user", "")
        except Exception as e:
            # Fallback to hardcoded prompts if template not found
            logger.debug(f"Template 'context_compression' not found, using fallback: {e}")
            return self._get_fallback_system_prompt(content_type), self._build_fallback_user_prompt(
                truncated_content, content_type, task_context, preserve_keys, page_url, page_title, compression_tier
            )
    
    def _get_fallback_system_prompt(self, content_type: ContentType) -> str:
        """Fallback system prompt when template is not available."""
        prompts = {
            ContentType.SEARCH_RESULTS: (
                "You are a search result compression specialist. Your PRIMARY goal is to preserve "
                "ALL URLs from search results - these are navigation targets the agent MUST have. "
                "Never omit or summarize URLs. Preserve title + URL + brief snippet for each result."
            ),
            ContentType.PAGE_STATE: (
                "You are a page state compression specialist. Preserve ALL navigation links, "
                "buttons, and interactive elements. The agent needs these to navigate the site. "
                "Include selectors where available."
            ),
            ContentType.TEXT_CONTENT: (
                "You are a content compression specialist. Extract key information, facts, and "
                "data values. Preserve any URLs found in the content. Focus on task-relevant info."
            ),
            ContentType.ERROR_INFO: (
                "You are an error analysis specialist. Preserve the error type, message, and "
                "context. Suggest recovery actions. Note what was being attempted when it failed."
            ),
        }
        return prompts.get(
            content_type,
            "You are a data compression assistant. Extract and preserve essential information concisely. "
            "ALWAYS preserve ALL URLs found - they are critical for navigation."
        )
    
    def _build_fallback_user_prompt(
        self,
        content_str: str,
        content_type: ContentType,
        task_context: str,
        preserve_keys: Optional[List[str]],
        page_url: str,
        page_title: str,
        compression_tier: int,
    ) -> str:
        """Build fallback user prompt when template is not available."""
        # Base rules
        rules = [
            "1. Create a one-line summary (max 100 characters)",
            "2. CRITICAL: Preserve ALL URLs found - these are navigation targets!",
            "3. Preserve key data values (numbers, prices, names, IDs, dates)",
            "4. Rate your compression_confidence from 0.0 to 1.0",
        ]
        
        # Add content-type-specific rules
        if content_type == ContentType.SEARCH_RESULTS:
            rules.extend([
                "5. Preserve title + URL + snippet for EACH search result",
                "6. Note the recommended result index if one stands out",
                "7. Include answer_box content if present",
            ])
        elif content_type == ContentType.PAGE_STATE:
            rules.extend([
                "5. Preserve ALL navigation_links with text and href",
                "6. Include all buttons with text and selector",
                "7. Note any forms and their fields",
            ])
        elif content_type == ContentType.TEXT_CONTENT:
            rules.extend([
                "5. Extract 5-10 key points from the content",
                "6. Preserve the heading structure",
                "7. Include any embedded URLs",
            ])
        elif content_type == ContentType.ERROR_INFO:
            rules.extend([
                "5. Identify the error_type and error_message",
                "6. Note what action failed",
                "7. Suggest 2-3 recovery approaches",
            ])
        else:
            rules.extend([
                "5. Extract 5-10 bullet points of key facts",
                "6. Preserve any selectors or CSS paths found",
                "7. Note form fields and navigation targets",
            ])
        
        # Tier-specific instructions
        tier_instructions = {
            1: "This is RECENT data - preserve maximum detail.",
            2: "This is SESSION data - compress but preserve all URLs and key facts.",
            3: "This is ARCHIVAL data - compress aggressively, keep only essential facts and patterns.",
        }
        tier_note = tier_instructions.get(compression_tier, tier_instructions[2])
        
        # Build prompt
        rules_text = "\n".join(rules)
        
        context_section = ""
        if page_url or page_title:
            context_section = f"\nPage context: {page_title or 'Unknown'} ({page_url or 'Unknown URL'})\n"
        
        preserve_section = ""
        if preserve_keys:
            preserve_section = f"\nIMPORTANT: Preserve exact values for these keys: {', '.join(preserve_keys)}\n"
        
        task_section = ""
        if task_context:
            task_section = f"\nTask context: {task_context}\n"
        
        return f"""Compress the following {content_type.value} data.

{tier_note}
{context_section}{task_section}{preserve_section}
RULES:
{rules_text}

CONTENT TO COMPRESS:
```
{content_str}
```

Compress into the required JSON format. PRESERVE ALL URLs."""
    
    def _build_compressed_content(
        self,
        response: Dict[str, Any],
        content_type: ContentType,
        original_chars: int,
        original_tokens: int,
        page_url: str,
        page_title: str,
        compression_tier: int,
    ) -> CompressedContent:
        """Build CompressedContent from LLM response."""
        # Calculate compressed size
        compressed_text = json.dumps(response)
        compressed_tokens = TokenEstimator.estimate(compressed_text).tokens
        
        # Extract URLs from various response formats
        urls = response.get("urls_mentioned", [])
        
        # For search results, also extract from results array
        if content_type == ContentType.SEARCH_RESULTS:
            for result in response.get("results", []):
                if isinstance(result, dict) and result.get("url"):
                    urls.append(result["url"])
        
        # For page state, extract from navigation_links
        if content_type == ContentType.PAGE_STATE:
            for link in response.get("navigation_links", []):
                if isinstance(link, dict) and link.get("href"):
                    urls.append(link["href"])
        
        # Deduplicate URLs
        urls = list(dict.fromkeys(urls))
        
        # Build navigation targets from response
        navigation_targets = []
        if content_type == ContentType.SEARCH_RESULTS:
            for result in response.get("results", []):
                if isinstance(result, dict):
                    navigation_targets.append({
                        "label": result.get("title", "Unknown"),
                        "url": result.get("url", ""),
                        "type": "search_result"
                    })
        elif content_type == ContentType.PAGE_STATE:
            for link in response.get("navigation_links", []):
                if isinstance(link, dict):
                    navigation_targets.append({
                        "label": link.get("text", "Unknown"),
                        "url": link.get("href", ""),
                        "type": link.get("location", "link")
                    })
        else:
            navigation_targets = response.get("navigation_targets", [])
        
        # Build form fields if present
        form_fields = response.get("form_fields", [])
        if content_type == ContentType.PAGE_STATE:
            for form in response.get("forms", []):
                if isinstance(form, dict):
                    for field_name in form.get("fields", []):
                        form_fields.append({"name": field_name, "type": "unknown"})
        
        return CompressedContent(
            summary=response.get("summary", response.get("context_summary", "Content compressed")),
            key_facts=response.get("key_facts", response.get("key_points", [])),
            data_values=response.get("data_values", response.get("data_extracted", {})),
            urls_mentioned=urls,
            selectors_found=response.get("selectors_found", []),
            navigation_targets=navigation_targets,
            form_fields=form_fields,
            page_context={
                "url": page_url or response.get("page_url", ""),
                "title": page_title or response.get("page_title", ""),
            },
            task_progress=response.get("task_progress", {}),
            successful_patterns=response.get("successful_patterns", []),
            error_info=response.get("error_info", response.get("error_message")),
            content_type=content_type.value,
            compression_confidence=response.get("compression_confidence", 0.7),
            original_size_chars=original_chars,
            original_size_tokens=original_tokens,
            compressed_size_tokens=compressed_tokens,
            compression_tier=compression_tier,
        )
    
    def _count_urls_in_content(self, content: str) -> int:
        """Count URLs in content for validation."""
        url_pattern = r'https?://[^\s<>"\')\]]+'
        urls = set(re.findall(url_pattern, content))
        return len(urls)
    
    def _extract_all_urls(self, content: str) -> List[str]:
        """Extract all URLs from content using multiple patterns."""
        urls = set()
        
        # Full URLs
        url_pattern = r'https?://[^\s<>"\')\]]+'
        for match in re.finditer(url_pattern, content):
            url = match.group().rstrip('.,;:')
            urls.add(url)
        
        # JSON patterns like "url": "..." or "href": "..."
        json_url_pattern = r'["\'](?:url|href|link)["\']\s*:\s*["\']([^"\']*)["\']]'
        for match in re.finditer(json_url_pattern, content, re.IGNORECASE):
            url = match.group(1)
            if url and (url.startswith('http') or url.startswith('/')):
                urls.add(url)
        
        return list(urls)[:100]  # Limit to 100
    
    async def compress_history(
        self,
        messages: List[Dict[str, Any]],
        keep_recent: int = 3,
    ) -> CompressedHistory:
        """
        Compress conversation history messages into a summary.
        
        Takes older messages and creates a summary that preserves
        the key context while reducing token count.
        
        Args:
            messages: List of messages in API format (role, content)
            keep_recent: Number of recent messages to exclude from compression
            
        Returns:
            CompressedHistory with summary of compressed turns
        """
        # Don't compress if too few messages
        if len(messages) <= keep_recent:
            return CompressedHistory(
                context_summary="No compression needed",
                actions_taken=[],
                turns_compressed=0,
            )
        
        # Get messages to compress (exclude recent ones)
        to_compress = messages[:-keep_recent] if keep_recent > 0 else messages
        
        # Calculate original tokens
        original_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_compress
        )
        original_tokens = TokenEstimator.estimate(original_text).tokens
        
        # Build compression prompt
        formatted_messages = []
        for msg in to_compress:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                # Truncate very long individual messages
                if len(content) > 2000:
                    content = content[:1500] + "..." + content[-500:]
                formatted_messages.append(f"[{role.upper()}]: {content}")
        
        history_text = "\n\n".join(formatted_messages)
        
        # Truncate if still too long
        if len(history_text) > 15000:
            history_text = history_text[:12000] + "\n...[truncated]...\n" + history_text[-3000:]
        
        compression_prompt = f"""Compress this BROWSER AUTOMATION conversation history into a summary.

CONVERSATION TO COMPRESS ({len(to_compress)} turns):
{history_text}

## CRITICAL PRESERVATION RULES (Browser Automation)
You MUST preserve these elements - the agent cannot function without them:

1. **URLs** - Preserve ALL URLs mentioned:
   - urls_visited: Pages the agent navigated to
   - urls_discovered: URLs found (from search, links) but not yet visited
   - The agent CANNOT navigate without these URLs!

2. **Actions** - What tools were used and their parameters:
   - Include tool names: navigate, click, type, search, extract_text, etc.
   - Include key parameters: selectors, text entered, URLs

3. **Results** - What was accomplished:
   - Data extracted (prices, names, IDs, counts, dates)
   - Pages successfully loaded
   - Elements found or clicked

4. **Selectors** - CSS selectors or element identifiers that worked

5. **Patterns** - What worked and what failed (to avoid repeating mistakes)

6. **Current State** - Where is the agent now (current URL, current page)

## OUTPUT FORMAT
Return JSON matching the schema with all URL fields populated.
NEVER omit or summarize URLs - they are navigation targets!"""

        try:
            response = await self.llm.generate_structured(
                prompt=compression_prompt,
                schema=HISTORY_COMPRESSION_SCHEMA,
                system_prompt="You are a BROWSER AUTOMATION conversation summarizer. Preserve ALL URLs, selectors, and action results. Never omit URLs!",
                temperature=self.compression_temperature,
                max_tokens=self.max_output_tokens,
            )
            
            compressed_text = json.dumps(response)
            compressed_tokens = TokenEstimator.estimate(compressed_text).tokens
            
            result = CompressedHistory(
                context_summary=response.get("context_summary", "History compressed"),
                actions_taken=response.get("actions_taken", []),
                results_obtained=response.get("results_obtained", []),
                current_state=response.get("current_state", ""),
                # Browser automation critical fields:
                urls_visited=response.get("urls_visited", []),
                urls_discovered=response.get("urls_discovered", []),
                selectors_used=response.get("selectors_used", []),
                data_extracted=response.get("data_extracted", {}),
                successful_patterns=response.get("successful_patterns", []),
                failed_approaches=response.get("failed_approaches", []),
                # Metadata:
                turns_compressed=len(to_compress),
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
            )
            
            compression_ratio = original_tokens / max(compressed_tokens, 1)
            logger.info(
                f"[ContextCompressor] Compressed {len(to_compress)} history turns: "
                f"{original_tokens:,} → {compressed_tokens:,} tokens "
                f"({compression_ratio:.1f}x reduction)"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"[ContextCompressor] History compression failed: {e}")
            
            # FALLBACK: Even if LLM fails, extract URLs - they are critical!
            # The agent cannot navigate without them.
            fallback_urls = self._extract_all_urls(original_text)
            
            # Also try to extract selectors
            selector_pattern = r'[#\.][a-zA-Z][a-zA-Z0-9_-]*'
            fallback_selectors = list(set(re.findall(selector_pattern, original_text)))[:20]
            
            # Extract what looks like actions (tool names in conversation)
            action_pattern = r'(?:navigate|click|type|search|extract|goto|fill|submit|scroll)\s*\([^)]*\)'
            fallback_actions = re.findall(action_pattern, original_text.lower())[:10]
            if not fallback_actions:
                fallback_actions = ["See recent messages for context"]
            
            return CompressedHistory(
                context_summary=f"(Compression failed, URLs extracted: {str(e)[:50]})",
                actions_taken=fallback_actions,
                urls_visited=fallback_urls[:50],  # Keep most important URLs
                urls_discovered=[],  # Can't distinguish without LLM
                selectors_used=fallback_selectors,
                turns_compressed=len(to_compress),
                original_tokens=original_tokens,
                compressed_tokens=TokenEstimator.estimate(
                    f"Summary + {len(fallback_urls)} URLs + {len(fallback_selectors)} selectors"
                ).tokens + len(fallback_urls) * 15,  # ~15 tokens per URL
            )
    
    def _create_minimal_compression(
        self,
        content: str,
        original_chars: int,
        original_tokens: int,
        page_url: str = "",
        page_title: str = "",
        tier: int = 1,
    ) -> CompressedContent:
        """Create minimal compression for already-small content."""
        # Extract some basic info without LLM
        lines = content.split("\n")
        summary = lines[0][:100] if lines else "Small content"
        
        # Simple key facts extraction
        key_facts = []
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 150:
                key_facts.append(line)
        
        # Extract URLs even for small content
        urls = self._extract_all_urls(content)
        
        return CompressedContent(
            summary=summary,
            key_facts=key_facts[:5],
            urls_mentioned=urls,
            page_context={"url": page_url, "title": page_title} if page_url else {},
            content_type="generic",
            compression_confidence=1.0,  # No compression needed
            original_size_chars=original_chars,
            original_size_tokens=original_tokens,
            compressed_size_tokens=original_tokens,  # No reduction
            compression_tier=tier,
        )
    
    def _create_fallback_compression(
        self,
        content: str,
        original_chars: int,
        original_tokens: int,
        error: str,
        page_url: str = "",
        page_title: str = "",
        tier: int = 2,
    ) -> CompressedContent:
        """
        Create fallback compression when LLM fails.
        
        IMPORTANT: This fallback prioritizes URL extraction since URLs
        are critical for navigation. We extract ALL URLs, not just a few.
        """
        # Take first 200 chars as summary
        summary = content[:200].replace("\n", " ").strip()
        if len(content) > 200:
            summary += "..."
        
        # Extract lines that look like key facts
        key_facts = []
        for line in content.split("\n"):
            line = line.strip()
            # Look for lines with colons (key: value patterns)
            if ":" in line and len(line) < 150 and len(line) > 5:
                key_facts.append(line)
                if len(key_facts) >= 10:
                    break
        
        # CRITICAL: Extract ALL URLs using shared method
        urls_list = self._extract_all_urls(content)
        
        # Extract navigation targets from common patterns
        navigation_targets = []
        # Look for patterns like [text](url) or "text": "url"
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            navigation_targets.append({
                "label": match.group(1)[:50],
                "url": match.group(2),
                "type": "markdown_link"
            })
        
        # Also extract data values (prices, numbers with units)
        data_values = {}
        price_pattern = r'[\$€£]\s*[\d,]+\.?\d*'
        for match in re.finditer(price_pattern, content):
            data_values[f"price_{len(data_values)}"] = match.group()
            if len(data_values) >= 5:
                break
        
        # Extract selectors (CSS patterns)
        selectors = []
        selector_pattern = r'[#\.][a-zA-Z][a-zA-Z0-9_-]*(?:\[[^\]]+\])?'
        for match in re.finditer(selector_pattern, content):
            sel = match.group()
            if len(sel) > 2 and sel not in selectors:
                selectors.append(sel)
                if len(selectors) >= 10:
                    break
        
        compressed_tokens = TokenEstimator.estimate(
            summary + "\n".join(key_facts) + "\n".join(urls_list)
        ).tokens
        
        return CompressedContent(
            summary=summary,
            key_facts=key_facts,
            data_values=data_values,
            urls_mentioned=urls_list,
            selectors_found=selectors,
            navigation_targets=navigation_targets[:20],
            page_context={"url": page_url, "title": page_title} if page_url else {},
            error_info=f"Fallback compression (LLM error: {error[:100]})",
            content_type="generic",
            compression_confidence=0.5,  # Lower confidence for fallback
            original_size_chars=original_chars,
            original_size_tokens=original_tokens,
            compressed_size_tokens=compressed_tokens,
            compression_tier=tier,
        )


def estimate_compression_benefit(content: str, threshold_tokens: int = 2000) -> tuple[bool, int]:
    """
    Estimate if compression would be beneficial for given content.
    
    Args:
        content: Content to potentially compress
        threshold_tokens: Minimum tokens before compression is worthwhile
        
    Returns:
        Tuple of (should_compress, estimated_tokens)
    """
    tokens = TokenEstimator.estimate(content).tokens
    should_compress = tokens > threshold_tokens
    return should_compress, tokens
