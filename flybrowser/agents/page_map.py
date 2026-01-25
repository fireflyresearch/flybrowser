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
Page Map - Spatial Understanding of Web Pages

This module provides data structures for representing comprehensive page understanding
including scroll positions, screenshots, detected sections, and navigation elements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SectionType(Enum):
    """Types of page sections."""
    HEADER = "header"
    NAVIGATION = "navigation"
    HERO = "hero"
    CONTENT = "content"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    MODAL = "modal"
    UNKNOWN = "unknown"


class ExplorationScope(str, Enum):
    """
    Scope/depth of page exploration for different operation modes.
    
    Determines how comprehensively a page should be explored:
    - Number of screenshots to capture
    - Scrolling strategy
    - Vision analysis depth
    - Performance vs accuracy trade-off
    """
    
    FULL = "full"
    # Comprehensive exploration of entire page
    # Screenshots: 8-10 (entire scrollable area)
    # Strategy: Systematic scrolling with overlap
    # Use Case: NAVIGATE mode - understanding site structure
    # Performance: Slow (~8-12 seconds), high cost
    
    TARGETED = "targeted"
    # Focus on specific regions (viewport + key areas)
    # Screenshots: 1-2 (viewport + one scroll if needed)
    # Strategy: Viewport + smart target identification
    # Use Case: Intermediate between VIEWPORT and CONTENT
    # Performance: Fast (~2-3 seconds), low-medium cost
    
    CONTENT = "content"
    # Focus on data-rich content areas
    # Screenshots: 3-5 (main content + data regions)
    # Strategy: Identify and capture content sections
    # Use Case: SCRAPE mode - data extraction focus
    # Performance: Medium (~4-6 seconds), medium cost
    
    SMART = "smart"
    # Adaptive exploration based on page characteristics
    # Screenshots: 2-6 (adapts to page height/complexity)
    # Strategy: Intelligent sampling of page sections
    # Use Case: RESEARCH mode - balanced approach, AUTO fallback
    # Performance: Variable (~3-7 seconds), medium cost
    
    VIEWPORT = "viewport"
    # Single screenshot of current viewport only
    # Screenshots: 1 (no scrolling)
    # Strategy: Capture visible area immediately
    # Use Case: EXECUTE mode - fast targeted actions
    # Performance: Very fast (<1 second), minimal cost


@dataclass
class ViewportInfo:
    """Information about browser viewport."""
    width: int  # Viewport width in pixels
    height: int  # Viewport height in pixels
    device_pixel_ratio: float = 1.0  # Device pixel ratio (retina = 2.0)


@dataclass
class ScrollPosition:
    """Represents a scroll position on the page."""
    x: int  # Horizontal scroll position (pixels)
    y: int  # Vertical scroll position (pixels from top)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "timestamp": self.timestamp
        }


@dataclass
class ScreenshotCapture:
    """Represents a captured screenshot with metadata."""
    index: int  # Screenshot index (0, 1, 2...)
    scroll_position: ScrollPosition  # Where page was scrolled when captured
    image_data: bytes  # Raw screenshot bytes (PNG/JPEG)
    image_size_bytes: int  # Size of image data
    visible_area: Dict[str, int]  # {"top": y_start, "bottom": y_end, "left": x_start, "right": x_end}
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding large image data."""
        data = {
            "index": self.index,
            "scroll_position": self.scroll_position.to_dict(),
            "image_size_bytes": self.image_size_bytes,
            "visible_area": self.visible_area,
            "timestamp": self.timestamp,
        }
        if include_image:
            data["image_data"] = self.image_data
        return data


@dataclass
class PageSection:
    """Represents a detected section of the page."""
    type: SectionType  # Type of section
    name: str  # Human-readable name ("Main Navigation", "Hero Banner", etc.)
    description: str  # Description of section content
    scroll_range: Dict[str, int]  # {"start_y": 0, "end_y": 800}
    screenshot_indices: List[int]  # Which screenshots contain this section
    elements: List[str] = field(default_factory=list)  # Key elements in section
    navigation_links: List[Dict[str, str]] = field(default_factory=list)  # Links found
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "scroll_range": self.scroll_range,
            "screenshot_indices": self.screenshot_indices,
            "elements": self.elements,
            "navigation_links": self.navigation_links,
            "metadata": self.metadata,
        }


@dataclass
class PageMap:
    """
    Comprehensive spatial map of a web page.
    
    Represents complete understanding of page structure including:
    - Viewport dimensions
    - Total page height
    - Multiple screenshots at different scroll positions
    - Detected sections with spatial coordinates
    - Navigation elements and links
    - Content summary
    """
    
    # Page identification
    url: str
    title: str
    
    # Spatial information
    viewport: ViewportInfo
    total_height: int  # Total scrollable height in pixels
    total_width: int  # Total scrollable width in pixels
    
    # Visual captures
    screenshots: List[ScreenshotCapture] = field(default_factory=list)
    
    # Structural understanding
    sections: List[PageSection] = field(default_factory=list)
    
    # High-level understanding
    summary: str = ""  # Overall page summary
    main_content_area: Optional[Dict[str, int]] = None  # Primary content scroll range
    navigation_structure: Dict[str, Any] = field(default_factory=dict)  # Nav hierarchy
    
    # DOM-extracted navigation links (actual hrefs from HTML)
    dom_navigation_links: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    analysis_complete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_screenshot_at_position(self, y: int) -> Optional[ScreenshotCapture]:
        """Get screenshot that contains the given Y coordinate."""
        for screenshot in self.screenshots:
            if screenshot.visible_area["top"] <= y <= screenshot.visible_area["bottom"]:
                return screenshot
        return None
    
    def get_section_at_position(self, y: int) -> Optional[PageSection]:
        """Get section that contains the given Y coordinate."""
        for section in self.sections:
            if section.scroll_range["start_y"] <= y <= section.scroll_range["end_y"]:
                return section
        return None
    
    def get_sections_by_type(self, section_type: SectionType) -> List[PageSection]:
        """Get all sections of a specific type."""
        return [s for s in self.sections if s.type == section_type]
    
    def get_all_navigation_links(self) -> List[Dict[str, str]]:
        """
        Get all links from DOM extraction.
        
        Returns raw link data for the LLM to intelligently analyze and categorize.
        No hardcoded assumptions about what constitutes "navigation".
        """
        all_links = []
        seen_hrefs = set()
        
        # Get DOM-extracted links (raw data for LLM to analyze)
        if self.dom_navigation_links:
            for link in self.dom_navigation_links.get('all_links', []):
                href = link.get('href', '')
                if href and href not in seen_hrefs:
                    all_links.append(link)
                    seen_hrefs.add(href)
        
        # Also include vision-extracted links from sections
        for section in self.sections:
            for link in section.navigation_links:
                href = link.get('href', '')
                if href and href not in seen_hrefs:
                    all_links.append({
                        'text': link.get('text', ''),
                        'href': href,
                        'source': 'vision',
                    })
                    seen_hrefs.add(href)
        
        return all_links
    
    def get_interactive_elements(self) -> List[Dict[str, Any]]:
        """Get buttons/toggles that might reveal hidden content (for LLM to analyze)."""
        if self.dom_navigation_links:
            return self.dom_navigation_links.get('interactive_elements', [])
        return []
    
    @property
    def navigation_links(self) -> List[Dict[str, str]]:
        """Property alias for get_all_navigation_links() for convenience."""
        return self.get_all_navigation_links()
    
    def get_coverage_percentage(self) -> float:
        """Calculate what percentage of page has been captured."""
        if not self.screenshots or self.total_height == 0:
            return 0.0
        
        # Calculate unique covered area (handle overlaps)
        covered_ranges = []
        for screenshot in self.screenshots:
            start = screenshot.visible_area["top"]
            end = screenshot.visible_area["bottom"]
            covered_ranges.append((start, end))
        
        # Merge overlapping ranges
        if not covered_ranges:
            return 0.0
        
        covered_ranges.sort()
        merged = [covered_ranges[0]]
        for current in covered_ranges[1:]:
            if current[0] <= merged[-1][1]:
                # Overlapping, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
            else:
                merged.append(current)
        
        # Calculate total covered pixels
        total_covered = sum(end - start for start, end in merged)
        return min(100.0, (total_covered / self.total_height) * 100.0)
    
    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        """
        Convert PageMap to dictionary for serialization.
        
        Args:
            include_images: Whether to include screenshot image data (large!)
        """
        return {
            "url": self.url,
            "title": self.title,
            "viewport": {
                "width": self.viewport.width,
                "height": self.viewport.height,
                "device_pixel_ratio": self.viewport.device_pixel_ratio,
            },
            "total_height": self.total_height,
            "total_width": self.total_width,
            "screenshots": [s.to_dict(include_image=include_images) for s in self.screenshots],
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary,
            "main_content_area": self.main_content_area,
            "navigation_structure": self.navigation_structure,
            "created_at": self.created_at,
            "analysis_complete": self.analysis_complete,
            "coverage_percentage": self.get_coverage_percentage(),
            "metadata": self.metadata,
        }
    
    def format_for_prompt(self, include_screenshots: bool = False, include_navigation: bool = True) -> str:
        """
        Format PageMap for inclusion in LLM prompts.
        
        Args:
            include_screenshots: Whether to include screenshot details
            include_navigation: Whether to include navigation link details (default: True)
        
        Returns:
            Formatted string describing the page structure
        """
        lines = []
        lines.append(f"## Page: {self.title}")
        lines.append(f"URL: {self.url}")
        lines.append(f"Dimensions: {self.total_width}x{self.total_height}px")
        lines.append(f"Coverage: {self.get_coverage_percentage():.1f}%")
        lines.append("")
        
        if self.summary:
            lines.append("### Summary")
            lines.append(self.summary)
            lines.append("")
        
        if self.sections:
            lines.append(f"### Sections ({len(self.sections)})")
            for section in self.sections:
                lines.append(f"- **{section.name}** ({section.type.value}): {section.description}")
                if section.elements:
                    lines.append(f"  Elements: {', '.join(section.elements[:5])}")
            lines.append("")
        
        # Include ALL extracted links for LLM to intelligently analyze
        if include_navigation:
            all_links = self.get_all_navigation_links()
            interactive_elements = self.get_interactive_elements()
            
            if all_links:
                lines.append(f"### Extracted Links ({len(all_links)}) - Analyze these to find navigation")
                lines.append("The following links were extracted from the DOM. Analyze their context to determine:")
                lines.append("- Which are main navigation links to visit")
                lines.append("- Which are anchor links (same page sections)")
                lines.append("- Which are external/irrelevant")
                lines.append("")
                for link in all_links[:25]:  # Show more links for LLM analysis
                    text = link.get("text", "[no text]")
                    href = link.get("href", "")
                    visible = "visible" if link.get("isVisible", True) else "hidden"
                    parent = link.get("parentTag", "")
                    lines.append(f"- [{visible}] {text}: {href} (in <{parent}>)")
                if len(all_links) > 25:
                    lines.append(f"  ... and {len(all_links) - 25} more links")
                lines.append("")
            
            # Show interactive elements that might reveal hidden navigation
            if interactive_elements:
                lines.append(f"### Interactive Elements ({len(interactive_elements)}) - May reveal hidden content")
                lines.append("Buttons/toggles that might expand menus or reveal navigation:")
                for el in interactive_elements[:10]:
                    text = el.get("text", "") or el.get("ariaLabel", "[no label]")
                    expanded = el.get("ariaExpanded", "unknown")
                    lines.append(f"- {text} (expanded: {expanded})")
                lines.append("")
        
        if include_screenshots and self.screenshots:
            lines.append(f"### Screenshots ({len(self.screenshots)})")
            for screenshot in self.screenshots:
                y_pos = screenshot.scroll_position.y
                area = screenshot.visible_area
                lines.append(
                    f"- Screenshot {screenshot.index}: Y={y_pos}px, "
                    f"showing {area['top']}-{area['bottom']}px"
                )
        
        return "\n".join(lines)
