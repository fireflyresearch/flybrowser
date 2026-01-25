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
Text normalization utilities for clean, consistent data output.

This module provides functions to normalize extracted text by converting
fancy Unicode characters to their ASCII equivalents while preserving meaning.

The normalization ensures:
- JSON-safe output (no problematic Unicode that could cause encoding issues)
- Consistent character representation across different data sources
- Clean, readable text without altering semantic content

Example:
    >>> from flybrowser.utils.text_normalizer import normalize_text, normalize_data
    >>> normalize_text("Show HN: ChartGPU – A "smart" library")
    'Show HN: ChartGPU - A "smart" library'
    >>> normalize_data({"title": "EU–INC — A new entity"})
    {'title': 'EU-INC - A new entity'}
"""

import re
import unicodedata
from typing import Any, Dict, List, Union


# Character mappings for normalization
# These preserve meaning while converting to ASCII-safe equivalents
CHAR_MAPPINGS = {
    # Dashes
    '\u2013': '-',  # en-dash
    '\u2014': '-',  # em-dash
    '\u2015': '-',  # horizontal bar
    '\u2212': '-',  # minus sign
    '\u2010': '-',  # hyphen
    '\u2011': '-',  # non-breaking hyphen
    
    # Quotes (double)
    '\u201c': '"',  # left double quotation mark
    '\u201d': '"',  # right double quotation mark
    '\u201e': '"',  # double low-9 quotation mark
    '\u201f': '"',  # double high-reversed-9 quotation mark
    '\u00ab': '"',  # left-pointing double angle quotation
    '\u00bb': '"',  # right-pointing double angle quotation
    
    # Quotes (single)
    '\u2018': "'",  # left single quotation mark
    '\u2019': "'",  # right single quotation mark
    '\u201a': "'",  # single low-9 quotation mark
    '\u201b': "'",  # single high-reversed-9 quotation mark
    '\u2039': "'",  # single left-pointing angle quotation
    '\u203a': "'",  # single right-pointing angle quotation
    '\u02bc': "'",  # modifier letter apostrophe
    '\u0060': "'",  # grave accent (backtick used as apostrophe)
    '\u00b4': "'",  # acute accent
    
    # Ellipsis
    '\u2026': '...',  # horizontal ellipsis
    
    # Spaces
    '\u00a0': ' ',  # non-breaking space
    '\u2002': ' ',  # en space
    '\u2003': ' ',  # em space
    '\u2004': ' ',  # three-per-em space
    '\u2005': ' ',  # four-per-em space
    '\u2006': ' ',  # six-per-em space
    '\u2007': ' ',  # figure space
    '\u2008': ' ',  # punctuation space
    '\u2009': ' ',  # thin space
    '\u200a': ' ',  # hair space
    '\u202f': ' ',  # narrow no-break space
    '\u205f': ' ',  # medium mathematical space
    '\u3000': ' ',  # ideographic space
    
    # Other punctuation
    '\u2022': '*',  # bullet
    '\u2023': '>',  # triangular bullet
    '\u2043': '-',  # hyphen bullet
    '\u25aa': '*',  # black small square (used as bullet)
    '\u25cf': '*',  # black circle (used as bullet)
}

# Zero-width and invisible characters to remove entirely
INVISIBLE_CHARS = {
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u200e',  # left-to-right mark
    '\u200f',  # right-to-left mark
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
    '\u2060',  # word joiner
    '\u2061',  # function application
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # byte order mark / zero-width no-break space
    '\ufffe',  # not a character
    '\uffff',  # not a character
}


def normalize_text(text: str) -> str:
    """
    Normalize text by converting fancy Unicode to ASCII equivalents.
    
    This function:
    1. Converts fancy quotes, dashes, and other typography to ASCII
    2. Removes zero-width and invisible characters
    3. Collapses multiple whitespace to single spaces
    4. Trims leading/trailing whitespace
    
    It preserves:
    - Accented characters (é, ñ, ü, etc.)
    - Currency symbols ($, €, £, ¥)
    - Math symbols (±, ×, ÷, °)
    - Emoji (if present)
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text with ASCII-safe typography
        
    Example:
        >>> normalize_text("Show HN: ChartGPU – A "smart" library")
        'Show HN: ChartGPU - A "smart" library'
        >>> normalize_text("Price: $99.99   (was $149)")
        'Price: $99.99 (was $149)'
    """
    if not isinstance(text, str):
        return text
    
    # Remove invisible characters
    for char in INVISIBLE_CHARS:
        text = text.replace(char, '')
    
    # Apply character mappings
    for src, dst in CHAR_MAPPINGS.items():
        text = text.replace(src, dst)
    
    # Collapse multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_data(data: Any) -> Any:
    """
    Recursively normalize text in any data structure.
    
    This function traverses dictionaries, lists, and nested structures,
    applying text normalization to all string values while preserving
    the original structure.
    
    Args:
        data: Input data (dict, list, str, or other)
        
    Returns:
        Data with all string values normalized
        
    Example:
        >>> data = {
        ...     "title": "EU–INC — A new entity",
        ...     "items": [
        ...         {"name": "Item "One"", "price": "$99"},
        ...         {"name": "Item 'Two'", "price": "$149"}
        ...     ]
        ... }
        >>> normalize_data(data)
        {
            'title': 'EU-INC - A new entity',
            'items': [
                {'name': 'Item "One"', 'price': '$99'},
                {'name': "Item 'Two'", 'price': '$149'}
            ]
        }
    """
    if isinstance(data, str):
        return normalize_text(data)
    elif isinstance(data, dict):
        return {key: normalize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [normalize_data(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(normalize_data(item) for item in data)
    else:
        # Return non-string, non-container types unchanged
        return data


def is_normalized(text: str) -> bool:
    """
    Check if text contains any characters that would be normalized.
    
    Args:
        text: Text to check
        
    Returns:
        True if text is already normalized, False otherwise
        
    Example:
        >>> is_normalized("Normal text here")
        True
        >>> is_normalized("Text with — em-dash")
        False
    """
    if not isinstance(text, str):
        return True
    
    # Check for invisible characters
    for char in INVISIBLE_CHARS:
        if char in text:
            return False
    
    # Check for characters that would be mapped
    for char in CHAR_MAPPINGS:
        if char in text:
            return False
    
    # Check for multiple consecutive whitespace
    if re.search(r'\s{2,}', text):
        return False
    
    # Check for leading/trailing whitespace
    if text != text.strip():
        return False
    
    return True
