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
Content Chunking Strategies for Large Content Handling.

This module provides strategies for splitting large content into
smaller chunks that fit within LLM context windows while preserving
semantic boundaries and structure.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from flybrowser.llm.token_budget import TokenEstimator, ContentType
from flybrowser.utils.logger import logger


@dataclass
class Chunk:
    """A chunk of content with metadata."""
    content: str
    index: int
    total_chunks: int
    estimated_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_first(self) -> bool:
        return self.index == 0
    
    @property
    def is_last(self) -> bool:
        return self.index == self.total_chunks - 1
    
    def format_header(self) -> str:
        """Format chunk header for context."""
        return f"[Chunk {self.index + 1}/{self.total_chunks}]"


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, content: str, max_tokens_per_chunk: int) -> List[Chunk]:
        """
        Split content into chunks.
        
        Args:
            content: Content to split
            max_tokens_per_chunk: Maximum tokens per chunk
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @abstractmethod
    def get_content_type(self) -> ContentType:
        """Get the content type this strategy handles."""
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for text."""
        return TokenEstimator.estimate(text, self.get_content_type()).tokens
    
    def _tokens_to_chars(self, tokens: int) -> int:
        """Convert token count to approximate char count."""
        chars_per_token = TokenEstimator.CHARS_PER_TOKEN.get(self.get_content_type(), 4.0)
        return int(tokens * chars_per_token)


class TextChunker(ChunkingStrategy):
    """
    Chunks plain text content by semantic boundaries.
    
    Prioritizes splitting at:
    1. Paragraph breaks (double newline)
    2. Sentence boundaries (. ! ?)
    3. Clause boundaries (, ; :)
    4. Word boundaries (spaces)
    """
    
    def get_content_type(self) -> ContentType:
        return ContentType.TEXT
    
    def chunk(self, content: str, max_tokens_per_chunk: int) -> List[Chunk]:
        """Split text into chunks at semantic boundaries."""
        if not content:
            return []
        
        total_tokens = self._estimate_tokens(content)
        
        # If content fits, return single chunk
        if total_tokens <= max_tokens_per_chunk:
            return [Chunk(
                content=content,
                index=0,
                total_chunks=1,
                estimated_tokens=total_tokens,
            )]
        
        chunks = []
        max_chars = self._tokens_to_chars(max_tokens_per_chunk)
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed limit
            if len(current_chunk) + len(para) + 2 > max_chars:
                # If current chunk has content, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too large, split it further
                if len(para) > max_chars:
                    sentence_chunks = self._split_by_sentences(para, max_chars)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Convert to Chunk objects
        total = len(chunks)
        return [
            Chunk(
                content=chunk,
                index=i,
                total_chunks=total,
                estimated_tokens=self._estimate_tokens(chunk),
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_by_sentences(self, text: str, max_chars: int) -> List[str]:
        """Split text by sentence boundaries."""
        # Sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars:
                if current:
                    chunks.append(current.strip())
                
                # If sentence itself is too long, split by words
                if len(sentence) > max_chars:
                    word_chunks = self._split_by_words(sentence, max_chars)
                    chunks.extend(word_chunks)
                    current = ""
                else:
                    current = sentence
            else:
                if current:
                    current += " " + sentence
                else:
                    current = sentence
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    def _split_by_words(self, text: str, max_chars: int) -> List[str]:
        """Split text by word boundaries (last resort)."""
        words = text.split()
        chunks = []
        current = ""
        
        for word in words:
            if len(current) + len(word) + 1 > max_chars:
                if current:
                    chunks.append(current.strip())
                current = word
            else:
                if current:
                    current += " " + word
                else:
                    current = word
        
        if current:
            chunks.append(current.strip())
        
        return chunks


class HTMLChunker(ChunkingStrategy):
    """
    Chunks HTML content while preserving structure.
    
    Prioritizes splitting at:
    1. Block-level element boundaries (div, section, article, p)
    2. List items
    3. Table rows
    4. Falls back to text chunking for text nodes
    """
    
    # Block-level elements that are good split points
    BLOCK_ELEMENTS = {
        'div', 'section', 'article', 'main', 'aside', 'header', 'footer',
        'nav', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
        'table', 'tr', 'blockquote', 'pre', 'form', 'fieldset'
    }
    
    def get_content_type(self) -> ContentType:
        return ContentType.HTML
    
    def chunk(self, content: str, max_tokens_per_chunk: int) -> List[Chunk]:
        """Split HTML content preserving structure."""
        if not content:
            return []
        
        total_tokens = self._estimate_tokens(content)
        
        # If content fits, return single chunk
        if total_tokens <= max_tokens_per_chunk:
            return [Chunk(
                content=content,
                index=0,
                total_chunks=1,
                estimated_tokens=total_tokens,
            )]
        
        max_chars = self._tokens_to_chars(max_tokens_per_chunk)
        
        # Try to split at block element boundaries
        chunks = self._split_at_blocks(content, max_chars)
        
        # Convert to Chunk objects
        total = len(chunks)
        return [
            Chunk(
                content=chunk,
                index=i,
                total_chunks=total,
                estimated_tokens=self._estimate_tokens(chunk),
                metadata={"type": "html"}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_at_blocks(self, html: str, max_chars: int) -> List[str]:
        """Split HTML at block element boundaries."""
        # Simple regex-based splitting (for complex HTML, use BeautifulSoup)
        # Pattern matches opening tags of block elements
        block_pattern = r'(<(?:' + '|'.join(self.BLOCK_ELEMENTS) + r')[^>]*>)'
        
        parts = re.split(block_pattern, html, flags=re.IGNORECASE)
        
        chunks = []
        current = ""
        
        for i, part in enumerate(parts):
            if not part:
                continue
            
            # If this part is an opening tag, keep it with next content
            if re.match(block_pattern, part, re.IGNORECASE):
                if len(current) + len(part) > max_chars and current:
                    chunks.append(current.strip())
                    current = part
                else:
                    current += part
            else:
                # Content part
                if len(current) + len(part) > max_chars:
                    if current:
                        chunks.append(current.strip())
                    
                    # If part itself is too large, use text chunker
                    if len(part) > max_chars:
                        text_chunker = TextChunker()
                        sub_chunks = text_chunker._split_by_sentences(part, max_chars)
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part
                else:
                    current += part
        
        if current:
            chunks.append(current.strip())
        
        return [c for c in chunks if c]  # Filter empty


class JSONChunker(ChunkingStrategy):
    """
    Chunks JSON content while maintaining validity.
    
    For arrays: splits into sub-arrays
    For objects: extracts top-level keys into separate chunks
    """
    
    def get_content_type(self) -> ContentType:
        return ContentType.JSON
    
    def chunk(self, content: str, max_tokens_per_chunk: int) -> List[Chunk]:
        """Split JSON content maintaining structure."""
        if not content:
            return []
        
        total_tokens = self._estimate_tokens(content)
        
        # If content fits, return single chunk
        if total_tokens <= max_tokens_per_chunk:
            return [Chunk(
                content=content,
                index=0,
                total_chunks=1,
                estimated_tokens=total_tokens,
            )]
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, fall back to text chunking
            logger.warning("JSONChunker: Invalid JSON, falling back to text chunking")
            text_chunker = TextChunker()
            return text_chunker.chunk(content, max_tokens_per_chunk)
        
        max_chars = self._tokens_to_chars(max_tokens_per_chunk)
        
        if isinstance(data, list):
            chunks = self._chunk_array(data, max_chars)
        elif isinstance(data, dict):
            chunks = self._chunk_object(data, max_chars)
        else:
            # Primitive value, can't chunk
            return [Chunk(
                content=content,
                index=0,
                total_chunks=1,
                estimated_tokens=total_tokens,
            )]
        
        # Convert to Chunk objects
        total = len(chunks)
        return [
            Chunk(
                content=json.dumps(chunk, ensure_ascii=False, indent=2),
                index=i,
                total_chunks=total,
                estimated_tokens=self._estimate_tokens(json.dumps(chunk)),
                metadata={"type": "json", "structure": type(data).__name__}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _chunk_array(self, arr: List, max_chars: int) -> List[List]:
        """Split array into sub-arrays."""
        if not arr:
            return [arr]
        
        chunks = []
        current = []
        current_size = 2  # For "[]"
        
        for item in arr:
            item_str = json.dumps(item, ensure_ascii=False)
            item_size = len(item_str) + 2  # +2 for comma and space
            
            if current_size + item_size > max_chars and current:
                chunks.append(current)
                current = [item]
                current_size = 2 + item_size
            else:
                current.append(item)
                current_size += item_size
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def _chunk_object(self, obj: Dict, max_chars: int) -> List[Dict]:
        """Split object by top-level keys."""
        if not obj:
            return [obj]
        
        chunks = []
        current = {}
        current_size = 2  # For "{}"
        
        for key, value in obj.items():
            item_str = json.dumps({key: value}, ensure_ascii=False)
            item_size = len(item_str)
            
            if current_size + item_size > max_chars and current:
                chunks.append(current)
                current = {key: value}
                current_size = item_size
            else:
                current[key] = value
                current_size += item_size
        
        if current:
            chunks.append(current)
        
        return chunks


class SmartChunker(ChunkingStrategy):
    """
    Auto-detecting chunker that selects the best strategy.
    
    Detects content type and delegates to appropriate chunker.
    """
    
    def __init__(self):
        self._text_chunker = TextChunker()
        self._html_chunker = HTMLChunker()
        self._json_chunker = JSONChunker()
    
    def get_content_type(self) -> ContentType:
        return ContentType.TEXT  # Default
    
    def chunk(self, content: str, max_tokens_per_chunk: int) -> List[Chunk]:
        """Auto-detect content type and chunk accordingly."""
        if not content:
            return []
        
        content_type = self._detect_type(content)
        
        if content_type == ContentType.HTML:
            logger.debug("SmartChunker: Detected HTML content")
            return self._html_chunker.chunk(content, max_tokens_per_chunk)
        elif content_type == ContentType.JSON:
            logger.debug("SmartChunker: Detected JSON content")
            return self._json_chunker.chunk(content, max_tokens_per_chunk)
        else:
            logger.debug("SmartChunker: Using text chunking")
            return self._text_chunker.chunk(content, max_tokens_per_chunk)
    
    def _detect_type(self, content: str) -> ContentType:
        """Detect content type from content."""
        return TokenEstimator._detect_content_type(content)


def get_chunker(content_type: Optional[ContentType] = None) -> ChunkingStrategy:
    """
    Factory function to get appropriate chunker.
    
    Args:
        content_type: Optional content type hint
        
    Returns:
        ChunkingStrategy instance
    """
    if content_type == ContentType.HTML:
        return HTMLChunker()
    elif content_type == ContentType.JSON:
        return JSONChunker()
    elif content_type == ContentType.TEXT:
        return TextChunker()
    else:
        return SmartChunker()
