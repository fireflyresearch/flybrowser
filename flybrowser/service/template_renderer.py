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
Template Renderer for FlyBrowser.

Provides Jinja2-based template rendering for the web player and other UI components.
Supports both inline (embedded mode) and external (server mode) asset serving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


# Get the directory where this module is located
SERVICE_DIR = Path(__file__).parent
TEMPLATES_DIR = SERVICE_DIR / "templates"
STATIC_DIR = SERVICE_DIR / "static"


class TemplateRenderer:
    """Renders Jinja2 templates for FlyBrowser UI components.
    
    Supports two modes:
    - Server mode: References external CSS/JS files via URLs
    - Embedded mode: Inlines CSS/JS content directly in HTML
    """
    
    _instance: Optional["TemplateRenderer"] = None
    
    def __init__(self) -> None:
        """Initialize the template renderer."""
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        
        # Cache for static file contents (used in embedded mode)
        self._static_cache: Dict[str, str] = {}
    
    @classmethod
    def get_instance(cls) -> "TemplateRenderer":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_static_file(self, relative_path: str, force_reload: bool = False) -> str:
        """Load a static file's content, with optional caching.
        
        Args:
            relative_path: Path relative to STATIC_DIR
            force_reload: If True, bypass cache and reload from disk
        """
        if force_reload or relative_path not in self._static_cache:
            file_path = STATIC_DIR / relative_path
            if file_path.exists():
                self._static_cache[relative_path] = file_path.read_text()
            else:
                self._static_cache[relative_path] = ""
        return self._static_cache[relative_path]
    
    def clear_cache(self) -> None:
        """Clear the static file cache."""
        self._static_cache.clear()
    
    def render_blank(
        self,
        static_url: Optional[str] = None,
        inline_assets: bool = False,
    ) -> str:
        """Render the blank/waiting page template.
        
        Args:
            static_url: Base URL for static assets (server mode)
            inline_assets: Whether to inline CSS/JS (embedded mode)
            
        Returns:
            Rendered HTML string
        """
        template = self.env.get_template("blank.html")
        
        context: Dict[str, Any] = {
            "inline_styles": inline_assets,
            "inline_scripts": inline_assets,
        }
        
        if inline_assets:
            # Load and inline CSS/JS for embedded mode
            context["css_content"] = self._load_static_file("css/blank.css")
            context["js_content"] = self._load_static_file("js/blank.js")
        else:
            # Use external URLs for server mode
            context["static_url"] = static_url or "/static"
        
        return template.render(**context)
    
    def render_player(
        self,
        stream_id: str,
        protocol: str,
        hls_url: str = "",
        dash_url: str = "",
        quality: str = "720p",
        static_url: Optional[str] = None,
        inline_assets: bool = False,
        max_retries: int = 30,
        retry_delay: int = 2000,
    ) -> str:
        """Render the stream player template.
        
        Args:
            stream_id: Unique stream identifier
            protocol: Streaming protocol (hls, dash)
            hls_url: HLS playlist URL
            dash_url: DASH manifest URL
            quality: Video quality label
            static_url: Base URL for static assets (server mode)
            inline_assets: Whether to inline CSS/JS (embedded mode)
            max_retries: Max connection retry attempts
            retry_delay: Delay between retries in ms
            
        Returns:
            Rendered HTML string
        """
        template = self.env.get_template("player.html")
        
        context: Dict[str, Any] = {
            "stream_id": stream_id,
            "stream_id_short": stream_id[:8],
            "protocol": protocol,
            "hls_url": hls_url,
            "dash_url": dash_url,
            "stream_url": hls_url or dash_url,
            "quality": quality,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "inline_styles": inline_assets,
            "inline_scripts": inline_assets,
        }
        
        if inline_assets:
            # Load and inline CSS/JS for embedded mode
            context["css_content"] = self._load_static_file("css/player.css")
            context["js_content"] = self._load_static_file("js/player.js")
        else:
            # Use external URLs for server mode
            context["static_url"] = static_url or "/static"
        
        return template.render(**context)


def render_blank_html(
    static_url: Optional[str] = None,
    inline_assets: bool = False,
) -> str:
    """Convenience function to render blank page HTML.
    
    Args:
        static_url: Base URL for static assets (server mode)
        inline_assets: Whether to inline CSS/JS (embedded mode)
        
    Returns:
        Rendered HTML string
    """
    renderer = TemplateRenderer.get_instance()
    return renderer.render_blank(
        static_url=static_url,
        inline_assets=inline_assets,
    )


def render_player_html(
    stream_id: str,
    protocol: str,
    hls_url: str = "",
    dash_url: str = "",
    quality: str = "720p",
    static_url: Optional[str] = None,
    inline_assets: bool = False,
    max_retries: int = 30,
    retry_delay: int = 2000,
) -> str:
    """Convenience function to render player HTML.
    
    Args:
        stream_id: Unique stream identifier
        protocol: Streaming protocol (hls, dash)
        hls_url: HLS playlist URL
        dash_url: DASH manifest URL
        quality: Video quality label
        static_url: Base URL for static assets (server mode)
        inline_assets: Whether to inline CSS/JS (embedded mode)
        max_retries: Max connection retry attempts
        retry_delay: Delay between retries in ms
        
    Returns:
        Rendered HTML string
    """
    renderer = TemplateRenderer.get_instance()
    return renderer.render_player(
        stream_id=stream_id,
        protocol=protocol,
        hls_url=hls_url,
        dash_url=dash_url,
        quality=quality,
        static_url=static_url,
        inline_assets=inline_assets,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def get_static_dir() -> Path:
    """Get the path to the static files directory."""
    return STATIC_DIR


def get_templates_dir() -> Path:
    """Get the path to the templates directory."""
    return TEMPLATES_DIR
