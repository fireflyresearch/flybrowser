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
Screenshot Cache - Unified Screenshot Management for FlyBrowser.

This module provides intelligent caching and reuse of screenshots across
different components (ObstacleDetector, PageExplorer, React loop) to:
- Eliminate redundant screenshot captures
- Reduce LLM vision API calls
- Maintain full analysis capabilities
- Track page content changes for smart invalidation

Key Features:
- URL normalization for consistent cache keys
- Freshness tracking with configurable TTL
- Page content hashing for change detection
- Scroll-position aware caching
- Domain-level obstacle result caching
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

logger = logging.getLogger(__name__)


# Tracking parameters commonly used for analytics (can be stripped for cache keys)
TRACKING_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'fbclid', 'gclid', 'msclkid', 'ref', 'source', 'mc_cid', 'mc_eid',
    '_ga', '_gl', 'dclid', 'zanpid', 'guccounter', 'guce_referrer',
}


@dataclass
class CachedScreenshot:
    """A cached screenshot with metadata."""
    image_data: bytes
    captured_at: float
    url: str
    scroll_position: Tuple[int, int] = (0, 0)  # (x, y)
    viewport_size: Tuple[int, int] = (0, 0)  # (width, height)
    content_hash: Optional[str] = None
    
    @property
    def age_seconds(self) -> float:
        """Get age of screenshot in seconds."""
        return time.time() - self.captured_at
    
    @property
    def size_kb(self) -> int:
        """Get screenshot size in KB."""
        return len(self.image_data) // 1024
    
    def is_fresh(self, max_age_seconds: float = 30.0) -> bool:
        """Check if screenshot is still fresh."""
        return self.age_seconds < max_age_seconds
    
    def matches_scroll_position(self, x: int, y: int, tolerance: int = 50) -> bool:
        """Check if screenshot matches given scroll position within tolerance."""
        return (
            abs(self.scroll_position[0] - x) <= tolerance and
            abs(self.scroll_position[1] - y) <= tolerance
        )


@dataclass
class CachedObstacleResult:
    """Cached obstacle detection result for a domain."""
    domain: str
    is_blocking: bool
    obstacles_found: int
    obstacles_dismissed: int
    analyzed_at: float
    content_hash: Optional[str] = None
    strategies_used: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def age_seconds(self) -> float:
        """Get age of result in seconds."""
        return time.time() - self.analyzed_at
    
    def is_fresh(self, max_age_seconds: float = 60.0) -> bool:
        """Check if result is still fresh."""
        return self.age_seconds < max_age_seconds


@dataclass 
class CachedPageAnalysis:
    """Cached page analysis result."""
    url: str
    normalized_url: str
    content_hash: str
    analyzed_at: float
    sections: List[Any] = field(default_factory=list)
    navigation_links: List[Any] = field(default_factory=list)
    summary: str = ""
    
    @property
    def age_seconds(self) -> float:
        """Get age of analysis in seconds."""
        return time.time() - self.analyzed_at
    
    def is_fresh(self, max_age_seconds: float = 120.0) -> bool:
        """Check if analysis is still fresh."""
        return self.age_seconds < max_age_seconds


class ScreenshotCache:
    """
    Unified screenshot and analysis cache.
    
    Provides centralized management of screenshots and analysis results
    across ObstacleDetector, PageExplorer, PageAnalyzer, and React loop.
    
    Key optimizations:
    1. Screenshot reuse: Same screenshot used by multiple components
    2. URL normalization: Consistent cache keys across URL variations
    3. Content hashing: Detect actual page changes vs. URL-only changes
    4. Scroll-aware caching: Track screenshots at different scroll positions
    5. Obstacle result caching: Per-domain obstacle detection results
    
    Usage:
        cache = ScreenshotCache()
        
        # Store screenshot
        cache.store_screenshot(url, image_bytes, scroll_pos=(0, 0))
        
        # Retrieve if fresh
        screenshot = cache.get_fresh_screenshot(url, max_age=30.0)
        
        # Check if re-analysis needed
        if cache.needs_reanalysis(url, current_content_hash):
            # Perform analysis
            cache.store_analysis(url, analysis_result, content_hash)
    """
    
    def __init__(
        self,
        screenshot_ttl_seconds: float = 30.0,
        obstacle_ttl_seconds: float = 60.0,
        analysis_ttl_seconds: float = 120.0,
        max_screenshots_per_url: int = 10,
        strip_tracking_params: bool = True,
    ) -> None:
        """
        Initialize screenshot cache.
        
        Args:
            screenshot_ttl_seconds: Time-to-live for screenshots
            obstacle_ttl_seconds: TTL for obstacle detection results
            analysis_ttl_seconds: TTL for page analysis results
            max_screenshots_per_url: Max screenshots to keep per URL (different scroll positions)
            strip_tracking_params: Whether to strip tracking params from URLs for cache keys
        """
        self.screenshot_ttl = screenshot_ttl_seconds
        self.obstacle_ttl = obstacle_ttl_seconds
        self.analysis_ttl = analysis_ttl_seconds
        self.max_screenshots_per_url = max_screenshots_per_url
        self.strip_tracking_params = strip_tracking_params
        
        # Screenshot storage: normalized_url -> list of CachedScreenshot
        self._screenshots: Dict[str, List[CachedScreenshot]] = {}
        
        # Obstacle results: domain -> CachedObstacleResult
        self._obstacle_results: Dict[str, CachedObstacleResult] = {}
        
        # Page analysis: normalized_url -> CachedPageAnalysis
        self._page_analyses: Dict[str, CachedPageAnalysis] = {}
        
        # Content hashes: normalized_url -> hash
        self._content_hashes: Dict[str, str] = {}
        
        # Stats for monitoring
        self._stats = {
            "screenshots_stored": 0,
            "screenshots_reused": 0,
            "screenshots_expired": 0,
            "obstacle_cache_hits": 0,
            "obstacle_cache_misses": 0,
            "analysis_cache_hits": 0,
            "analysis_cache_misses": 0,
        }
        
        logger.info(
            f"[ScreenshotCache] Initialized with TTLs: "
            f"screenshot={screenshot_ttl_seconds}s, "
            f"obstacle={obstacle_ttl_seconds}s, "
            f"analysis={analysis_ttl_seconds}s"
        )
    
    # ==================== URL Normalization ====================
    
    def normalize_url(self, url: str, strip_fragment: bool = True) -> str:
        """
        Normalize URL for consistent cache keys.
        
        Normalizations applied:
        - Lowercase scheme and host
        - Remove default ports (80, 443)
        - Remove trailing slashes from path
        - Optionally strip fragments (#section)
        - Optionally strip tracking parameters
        
        Args:
            url: URL to normalize
            strip_fragment: Whether to remove fragment identifier
            
        Returns:
            Normalized URL string
        """
        try:
            parsed = urlparse(url)
            
            # Lowercase scheme and netloc
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()
            
            # Remove default ports
            if netloc.endswith(':80') and scheme == 'http':
                netloc = netloc[:-3]
            elif netloc.endswith(':443') and scheme == 'https':
                netloc = netloc[:-4]
            
            # Normalize path (remove trailing slash unless root)
            path = parsed.path.rstrip('/') or '/'
            
            # Handle query parameters
            query = parsed.query
            if self.strip_tracking_params and query:
                params = parse_qs(query, keep_blank_values=True)
                # Filter out tracking params
                filtered_params = {
                    k: v for k, v in params.items() 
                    if k.lower() not in TRACKING_PARAMS
                }
                # Sort for consistent ordering
                query = urlencode(sorted(filtered_params.items()), doseq=True)
            
            # Handle fragment
            fragment = '' if strip_fragment else parsed.fragment
            
            return urlunparse((scheme, netloc, path, parsed.params, query, fragment))
            
        except Exception as e:
            logger.warning(f"[ScreenshotCache] URL normalization failed for {url}: {e}")
            return url
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL for domain-level caching."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return url
    
    # ==================== Content Hashing ====================
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """
        Compute hash of page content for change detection.
        
        Uses a fast hash that's good enough for change detection.
        
        Args:
            content: Page content (HTML, text, etc.)
            
        Returns:
            Hex hash string
        """
        # Use MD5 for speed - we don't need cryptographic security
        return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()[:16]
    
    def has_content_changed(self, url: str, new_content_hash: str) -> bool:
        """
        Check if page content has changed since last analysis.
        
        Args:
            url: Page URL
            new_content_hash: Hash of current content
            
        Returns:
            True if content changed or no previous hash exists
        """
        normalized = self.normalize_url(url)
        old_hash = self._content_hashes.get(normalized)
        
        if old_hash is None:
            return True  # No previous hash, treat as changed
        
        return old_hash != new_content_hash
    
    def update_content_hash(self, url: str, content_hash: str) -> None:
        """Update stored content hash for a URL."""
        normalized = self.normalize_url(url)
        self._content_hashes[normalized] = content_hash
    
    # ==================== Screenshot Management ====================
    
    def store_screenshot(
        self,
        url: str,
        image_data: bytes,
        scroll_position: Tuple[int, int] = (0, 0),
        viewport_size: Tuple[int, int] = (0, 0),
        content_hash: Optional[str] = None,
    ) -> CachedScreenshot:
        """
        Store a screenshot in the cache.
        
        Args:
            url: Page URL
            image_data: Screenshot bytes
            scroll_position: (x, y) scroll position
            viewport_size: (width, height) of viewport
            content_hash: Optional content hash for freshness tracking
            
        Returns:
            The stored CachedScreenshot object
        """
        normalized = self.normalize_url(url)
        
        screenshot = CachedScreenshot(
            image_data=image_data,
            captured_at=time.time(),
            url=url,
            scroll_position=scroll_position,
            viewport_size=viewport_size,
            content_hash=content_hash,
        )
        
        # Initialize list if needed
        if normalized not in self._screenshots:
            self._screenshots[normalized] = []
        
        screenshots = self._screenshots[normalized]
        
        # Remove old screenshot at same scroll position (within tolerance)
        screenshots = [
            s for s in screenshots 
            if not s.matches_scroll_position(scroll_position[0], scroll_position[1])
        ]
        
        # Add new screenshot
        screenshots.append(screenshot)
        
        # Limit number of screenshots per URL
        if len(screenshots) > self.max_screenshots_per_url:
            # Keep most recent ones
            screenshots.sort(key=lambda s: s.captured_at, reverse=True)
            screenshots = screenshots[:self.max_screenshots_per_url]
        
        self._screenshots[normalized] = screenshots
        self._stats["screenshots_stored"] += 1
        
        logger.debug(
            f"[ScreenshotCache] Stored screenshot for {normalized}: "
            f"scroll=({scroll_position[0]}, {scroll_position[1]}), "
            f"size={screenshot.size_kb}KB"
        )
        
        return screenshot
    
    def get_fresh_screenshot(
        self,
        url: str,
        scroll_position: Optional[Tuple[int, int]] = None,
        max_age_seconds: Optional[float] = None,
        position_tolerance: int = 50,
    ) -> Optional[CachedScreenshot]:
        """
        Get a fresh screenshot from cache if available.
        
        Args:
            url: Page URL
            scroll_position: Optional (x, y) to match specific scroll position
            max_age_seconds: Override default TTL
            position_tolerance: Tolerance for scroll position matching
            
        Returns:
            CachedScreenshot if found and fresh, None otherwise
        """
        normalized = self.normalize_url(url)
        max_age = max_age_seconds or self.screenshot_ttl
        
        screenshots = self._screenshots.get(normalized, [])
        if not screenshots:
            return None
        
        # Filter by freshness
        fresh_screenshots = [s for s in screenshots if s.is_fresh(max_age)]
        
        if not fresh_screenshots:
            self._stats["screenshots_expired"] += 1
            return None
        
        # If scroll position specified, find matching one
        if scroll_position is not None:
            x, y = scroll_position
            for screenshot in fresh_screenshots:
                if screenshot.matches_scroll_position(x, y, position_tolerance):
                    self._stats["screenshots_reused"] += 1
                    logger.debug(
                        f"[ScreenshotCache] Reusing screenshot for {normalized} "
                        f"at scroll ({x}, {y}), age={screenshot.age_seconds:.1f}s"
                    )
                    return screenshot
            return None
        
        # Return most recent fresh screenshot
        fresh_screenshots.sort(key=lambda s: s.captured_at, reverse=True)
        screenshot = fresh_screenshots[0]
        self._stats["screenshots_reused"] += 1
        
        logger.debug(
            f"[ScreenshotCache] Reusing screenshot for {normalized}, "
            f"age={screenshot.age_seconds:.1f}s"
        )
        
        return screenshot
    
    def get_all_screenshots(self, url: str, max_age_seconds: Optional[float] = None) -> List[CachedScreenshot]:
        """
        Get all fresh screenshots for a URL (all scroll positions).
        
        Args:
            url: Page URL
            max_age_seconds: Override default TTL
            
        Returns:
            List of fresh screenshots sorted by scroll position (top to bottom)
        """
        normalized = self.normalize_url(url)
        max_age = max_age_seconds or self.screenshot_ttl
        
        screenshots = self._screenshots.get(normalized, [])
        fresh = [s for s in screenshots if s.is_fresh(max_age)]
        
        # Sort by scroll Y position
        fresh.sort(key=lambda s: s.scroll_position[1])
        
        return fresh
    
    def clear_screenshots(self, url: Optional[str] = None) -> int:
        """
        Clear screenshots from cache.
        
        Args:
            url: Optional URL to clear. If None, clears all.
            
        Returns:
            Number of screenshots cleared
        """
        if url:
            normalized = self.normalize_url(url)
            count = len(self._screenshots.get(normalized, []))
            self._screenshots.pop(normalized, None)
            return count
        else:
            count = sum(len(v) for v in self._screenshots.values())
            self._screenshots.clear()
            return count
    
    # ==================== Obstacle Result Caching ====================
    
    def store_obstacle_result(
        self,
        url: str,
        is_blocking: bool,
        obstacles_found: int,
        obstacles_dismissed: int,
        strategies_used: Optional[List[Dict[str, Any]]] = None,
        content_hash: Optional[str] = None,
    ) -> CachedObstacleResult:
        """
        Store obstacle detection result for a domain.
        
        Args:
            url: Page URL (domain extracted)
            is_blocking: Whether obstacles are blocking
            obstacles_found: Number of obstacles found
            obstacles_dismissed: Number successfully dismissed
            strategies_used: Strategies that were tried
            content_hash: Page content hash
            
        Returns:
            The stored CachedObstacleResult
        """
        domain = self.extract_domain(url)
        
        result = CachedObstacleResult(
            domain=domain,
            is_blocking=is_blocking,
            obstacles_found=obstacles_found,
            obstacles_dismissed=obstacles_dismissed,
            analyzed_at=time.time(),
            content_hash=content_hash,
            strategies_used=strategies_used or [],
        )
        
        self._obstacle_results[domain] = result
        
        logger.debug(
            f"[ScreenshotCache] Stored obstacle result for {domain}: "
            f"found={obstacles_found}, dismissed={obstacles_dismissed}"
        )
        
        return result
    
    def get_obstacle_result(
        self,
        url: str,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[CachedObstacleResult]:
        """
        Get cached obstacle result for a domain if fresh.
        
        Args:
            url: Page URL (domain extracted)
            max_age_seconds: Override default TTL
            
        Returns:
            CachedObstacleResult if found and fresh, None otherwise
        """
        domain = self.extract_domain(url)
        max_age = max_age_seconds or self.obstacle_ttl
        
        result = self._obstacle_results.get(domain)
        if result and result.is_fresh(max_age):
            self._stats["obstacle_cache_hits"] += 1
            logger.debug(
                f"[ScreenshotCache] Obstacle cache hit for {domain}, "
                f"age={result.age_seconds:.1f}s"
            )
            return result
        
        self._stats["obstacle_cache_misses"] += 1
        return None
    
    # ==================== Page Analysis Caching ====================
    
    def store_page_analysis(
        self,
        url: str,
        content_hash: str,
        sections: Optional[List[Any]] = None,
        navigation_links: Optional[List[Any]] = None,
        summary: str = "",
    ) -> CachedPageAnalysis:
        """
        Store page analysis result.
        
        Args:
            url: Page URL
            content_hash: Hash of analyzed content
            sections: Analyzed page sections
            navigation_links: Extracted navigation links
            summary: Page summary
            
        Returns:
            The stored CachedPageAnalysis
        """
        normalized = self.normalize_url(url)
        
        analysis = CachedPageAnalysis(
            url=url,
            normalized_url=normalized,
            content_hash=content_hash,
            analyzed_at=time.time(),
            sections=sections or [],
            navigation_links=navigation_links or [],
            summary=summary,
        )
        
        self._page_analyses[normalized] = analysis
        self._content_hashes[normalized] = content_hash
        
        logger.debug(
            f"[ScreenshotCache] Stored page analysis for {normalized}: "
            f"{len(sections or [])} sections, {len(navigation_links or [])} links"
        )
        
        return analysis
    
    def get_page_analysis(
        self,
        url: str,
        max_age_seconds: Optional[float] = None,
        content_hash: Optional[str] = None,
    ) -> Optional[CachedPageAnalysis]:
        """
        Get cached page analysis if fresh and content unchanged.
        
        Args:
            url: Page URL
            max_age_seconds: Override default TTL
            content_hash: Current content hash for change detection
            
        Returns:
            CachedPageAnalysis if valid, None otherwise
        """
        normalized = self.normalize_url(url)
        max_age = max_age_seconds or self.analysis_ttl
        
        analysis = self._page_analyses.get(normalized)
        if not analysis:
            self._stats["analysis_cache_misses"] += 1
            return None
        
        if not analysis.is_fresh(max_age):
            self._stats["analysis_cache_misses"] += 1
            return None
        
        # If content hash provided, check if content changed
        if content_hash and analysis.content_hash != content_hash:
            logger.debug(
                f"[ScreenshotCache] Analysis cache invalid for {normalized}: "
                f"content changed"
            )
            self._stats["analysis_cache_misses"] += 1
            return None
        
        self._stats["analysis_cache_hits"] += 1
        logger.debug(
            f"[ScreenshotCache] Analysis cache hit for {normalized}, "
            f"age={analysis.age_seconds:.1f}s"
        )
        return analysis
    
    def needs_reanalysis(
        self,
        url: str,
        current_content_hash: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
    ) -> bool:
        """
        Check if page needs (re)analysis.
        
        Args:
            url: Page URL
            current_content_hash: Hash of current content
            max_age_seconds: Override default TTL
            
        Returns:
            True if analysis needed, False if cached analysis is valid
        """
        analysis = self.get_page_analysis(
            url, 
            max_age_seconds=max_age_seconds,
            content_hash=current_content_hash
        )
        return analysis is None
    
    # ==================== Utility Methods ====================
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self._screenshots.clear()
        self._obstacle_results.clear()
        self._page_analyses.clear()
        self._content_hashes.clear()
        logger.info("[ScreenshotCache] All caches cleared")
    
    def clear_for_url(self, url: str) -> None:
        """Clear all cached data for a specific URL."""
        normalized = self.normalize_url(url)
        domain = self.extract_domain(url)
        
        self._screenshots.pop(normalized, None)
        self._obstacle_results.pop(domain, None)
        self._page_analyses.pop(normalized, None)
        self._content_hashes.pop(normalized, None)
        
        logger.debug(f"[ScreenshotCache] Cleared cache for {normalized}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_screenshots = sum(len(v) for v in self._screenshots.values())
        
        return {
            **self._stats,
            "total_screenshots_cached": total_screenshots,
            "total_urls_cached": len(self._screenshots),
            "obstacle_results_cached": len(self._obstacle_results),
            "page_analyses_cached": len(self._page_analyses),
        }
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        
        # Calculate reuse ratio
        total_stored = stats["screenshots_stored"]
        total_reused = stats["screenshots_reused"]
        reuse_ratio = (total_reused / total_stored * 100) if total_stored > 0 else 0
        
        logger.info(
            f"[ScreenshotCache] Stats: "
            f"screenshots={stats['total_screenshots_cached']} "
            f"(stored={total_stored}, reused={total_reused}, ratio={reuse_ratio:.1f}%), "
            f"obstacle_hits={stats['obstacle_cache_hits']}/{stats['obstacle_cache_hits'] + stats['obstacle_cache_misses']}, "
            f"analysis_hits={stats['analysis_cache_hits']}/{stats['analysis_cache_hits'] + stats['analysis_cache_misses']}"
        )
