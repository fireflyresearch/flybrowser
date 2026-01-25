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

"""Page utility functions for FlyBrowser.

This module provides helper functions for common page-related operations
used across the FlyBrowser codebase.
"""

from __future__ import annotations


# Markers that identify our custom blank page content
FLYBROWSER_BLANK_MARKERS = [
    "flybrowser-blank-page",  # CSS class marker
    "data-flybrowser-blank",  # Data attribute marker
    "FlyBrowser Blank Page",  # Text content marker
]


def is_blank_page(url: str, html: str = "") -> bool:
    """Check if a URL or page content represents a blank/empty page.
    
    This function identifies both standard blank pages (about:blank) and
    FlyBrowser's custom blank page that displays "Waiting for Agent".
    
    Use this function to avoid wasting LLM calls on blank pages where
    there's no meaningful content to analyze.
    
    Args:
        url: The page URL to check
        html: Optional HTML content of the page for additional detection
        
    Returns:
        True if the page is a blank page, False otherwise
        
    Examples:
        >>> is_blank_page("about:blank")
        True
        >>> is_blank_page("about:srcdoc")
        True
        >>> is_blank_page("https://example.com")
        False
        >>> is_blank_page("", "<div class='flybrowser-blank-page'>...</div>")
        True
    """
    if not url:
        # Empty URL is considered blank
        return True
    
    url_lower = url.lower()
    
    # Check for standard blank page URLs
    if url_lower.startswith("about:"):
        return True
    
    # Check for FlyBrowser blank page URL pattern (if served via API)
    if "/flybrowser/blank" in url_lower:
        return True
    
    # Check for data: URLs that might be our blank page
    if url_lower.startswith("data:"):
        # If it's a data URL with our markers, it's our blank page
        if any(marker.lower() in url_lower for marker in FLYBROWSER_BLANK_MARKERS):
            return True
    
    # Check HTML content for our custom blank page markers
    if html:
        html_lower = html.lower()
        if any(marker.lower() in html_lower for marker in FLYBROWSER_BLANK_MARKERS):
            return True
    
    return False


def is_flybrowser_blank_page(url: str = "", html: str = "") -> bool:
    """Check specifically if this is FlyBrowser's custom blank page.
    
    Unlike is_blank_page(), this only returns True for FlyBrowser's
    custom branded blank page, not for standard about:blank pages.
    
    Args:
        url: The page URL to check
        html: Optional HTML content of the page
        
    Returns:
        True if this is specifically FlyBrowser's custom blank page
    """
    # Check URL for our blank page endpoint
    if url and "/flybrowser/blank" in url.lower():
        return True
    
    # Check HTML content for our markers
    if html:
        html_lower = html.lower()
        return any(marker.lower() in html_lower for marker in FLYBROWSER_BLANK_MARKERS)
    
    return False
