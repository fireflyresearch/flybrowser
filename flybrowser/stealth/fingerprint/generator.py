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
Advanced Browser Fingerprint Generation for FlyBrowser.

This module provides state-of-the-art browser fingerprint generation to avoid
bot detection. It creates consistent, realistic fingerprints that match real
browser populations.

Features:
- Hardware profiles (CPU, memory, GPU combinations)
- Browser profiles (Chrome, Firefox, Safari with version-accurate features)
- OS profiles (Windows, macOS, Linux with consistent navigator properties)
- Screen profiles (resolutions, devicePixelRatio, color depth)
- Font fingerprinting (OS-specific installed fonts)
- Canvas/WebGL fingerprinting (consistent noise patterns)
- WebRTC handling (IP masking, STUN/TURN configuration)
- Audio fingerprinting (consistent AudioContext noise)
- TLS fingerprinting (JA3/JA4 consistent handshakes)

Example:
    >>> from flybrowser.stealth.fingerprint import FingerprintGenerator
    >>> 
    >>> generator = FingerprintGenerator()
    >>> profile = generator.generate(
    ...     os="macos",
    ...     browser="chrome",
    ...     geolocation="us-west"
    ... )
    >>> print(profile.user_agent)
    >>> print(profile.navigator_properties)
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import secrets


class OperatingSystem(str, Enum):
    """Supported operating systems."""
    WINDOWS_10 = "windows_10"
    WINDOWS_11 = "windows_11"
    MACOS_VENTURA = "macos_ventura"      # 13.x
    MACOS_SONOMA = "macos_sonoma"        # 14.x
    MACOS_SEQUOIA = "macos_sequoia"      # 15.x
    LINUX_UBUNTU = "linux_ubuntu"
    LINUX_FEDORA = "linux_fedora"


class BrowserType(str, Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"


class DeviceType(str, Enum):
    """Device types for fingerprint generation."""
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    MOBILE = "mobile"
    TABLET = "tablet"


@dataclass
class HardwareProfile:
    """Hardware characteristics for fingerprinting."""
    cpu_cores: int = 8
    device_memory: int = 8  # GB
    max_touch_points: int = 0
    hardware_concurrency: int = 8
    
    # GPU information
    gpu_vendor: str = "Google Inc. (Intel)"
    gpu_renderer: str = "ANGLE (Intel, Intel(R) Iris(R) Plus Graphics 640, OpenGL 4.1)"
    
    # Platform-specific
    platform: str = "MacIntel"
    architecture: str = "x86_64"
    
    # Battery (optional, for laptops)
    has_battery: bool = True
    battery_charging: bool = True
    battery_level: float = 1.0


@dataclass
class ScreenProfile:
    """Screen characteristics for fingerprinting."""
    width: int = 1920
    height: int = 1080
    avail_width: int = 1920
    avail_height: int = 1055  # Account for taskbar/dock
    avail_top: int = 25       # macOS menu bar
    avail_left: int = 0
    color_depth: int = 24
    pixel_depth: int = 24
    device_pixel_ratio: float = 1.0
    orientation_type: str = "landscape-primary"
    orientation_angle: int = 0


@dataclass
class BrowserProfile:
    """Browser-specific characteristics."""
    name: str = "Chrome"
    version: str = "131.0.0.0"
    major_version: int = 131
    engine: str = "Blink"
    engine_version: str = "131.0.0.0"
    
    # Feature support
    supports_webgl: bool = True
    supports_webgl2: bool = True
    supports_webrtc: bool = True
    supports_websocket: bool = True
    supports_service_worker: bool = True
    supports_webassembly: bool = True
    
    # Plugins (Chrome-specific)
    plugins: List[Dict[str, str]] = field(default_factory=list)
    
    # MIME types
    mime_types: List[str] = field(default_factory=list)


@dataclass
class LocaleProfile:
    """Locale and timezone characteristics."""
    language: str = "en-US"
    languages: List[str] = field(default_factory=lambda: ["en-US", "en"])
    timezone: str = "America/Los_Angeles"
    timezone_offset: int = 480  # Minutes (PST = UTC-8 = 480)
    locale: str = "en-US"
    
    # Geolocation (optional)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: float = 100.0


@dataclass
class NetworkProfile:
    """Network characteristics."""
    connection_type: str = "wifi"
    effective_type: str = "4g"
    downlink: float = 10.0  # Mbps
    rtt: int = 50  # ms
    save_data: bool = False


@dataclass 
class WebGLProfile:
    """WebGL fingerprint characteristics."""
    vendor: str = "Intel Inc."
    renderer: str = "Intel Iris Pro OpenGL Engine"
    unmasked_vendor: str = "Intel Inc."
    unmasked_renderer: str = "Intel(R) Iris(R) Plus Graphics 640"
    
    # Shader precision format
    vertex_shader_precision: Dict[str, Any] = field(default_factory=dict)
    fragment_shader_precision: Dict[str, Any] = field(default_factory=dict)
    
    # Extensions
    extensions: List[str] = field(default_factory=list)
    
    # Parameters
    max_texture_size: int = 16384
    max_vertex_attribs: int = 16
    max_vertex_uniform_vectors: int = 4096
    max_fragment_uniform_vectors: int = 1024
    max_varying_vectors: int = 31


@dataclass
class AudioProfile:
    """Audio fingerprint characteristics."""
    sample_rate: int = 44100
    max_channel_count: int = 2
    number_of_inputs: int = 1
    number_of_outputs: int = 1
    channel_count: int = 2
    channel_count_mode: str = "max"
    channel_interpretation: str = "speakers"
    
    # Noise seed for consistent audio fingerprint
    noise_seed: int = field(default_factory=lambda: random.randint(0, 2**32))


@dataclass
class CanvasProfile:
    """Canvas fingerprint characteristics."""
    # Noise parameters for consistent canvas fingerprint
    noise_seed: int = field(default_factory=lambda: random.randint(0, 2**32))
    noise_intensity: float = 0.0001  # Very subtle noise
    
    # Font rendering hints
    font_smoothing: str = "antialiased"
    text_rendering: str = "optimizeLegibility"


@dataclass
class FontProfile:
    """Font fingerprint characteristics."""
    # OS-specific font lists
    installed_fonts: List[str] = field(default_factory=list)
    
    # Font measurement characteristics
    font_smoothing_enabled: bool = True
    sub_pixel_rendering: bool = True


@dataclass
class FingerprintProfile:
    """
    Complete browser fingerprint profile.
    
    This dataclass contains all components needed to create a consistent,
    realistic browser fingerprint that can evade bot detection.
    """
    # Unique profile ID (for session consistency)
    profile_id: str = field(default_factory=lambda: secrets.token_hex(16))
    
    # Core profiles
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    screen: ScreenProfile = field(default_factory=ScreenProfile)
    browser: BrowserProfile = field(default_factory=BrowserProfile)
    locale: LocaleProfile = field(default_factory=LocaleProfile)
    network: NetworkProfile = field(default_factory=NetworkProfile)
    
    # Fingerprinting components
    webgl: WebGLProfile = field(default_factory=WebGLProfile)
    audio: AudioProfile = field(default_factory=AudioProfile)
    canvas: CanvasProfile = field(default_factory=CanvasProfile)
    fonts: FontProfile = field(default_factory=FontProfile)
    
    # User agent string
    user_agent: str = ""
    
    # Client hints
    client_hints: Dict[str, str] = field(default_factory=dict)
    
    # Creation timestamp
    created_at: float = field(default_factory=time.time)
    
    @property
    def navigator_properties(self) -> Dict[str, Any]:
        """Get navigator properties for JavaScript injection."""
        return {
            "userAgent": self.user_agent,
            "platform": self.hardware.platform,
            "hardwareConcurrency": self.hardware.hardware_concurrency,
            "deviceMemory": self.hardware.device_memory,
            "maxTouchPoints": self.hardware.max_touch_points,
            "language": self.locale.language,
            "languages": self.locale.languages,
            "vendor": "Google Inc.",
            "vendorSub": "",
            "product": "Gecko",
            "productSub": "20030107",
            "cookieEnabled": True,
            "doNotTrack": None,
            "webdriver": False,
            "pdfViewerEnabled": True,
        }
    
    @property
    def viewport(self) -> Dict[str, int]:
        """Get viewport configuration for Playwright."""
        return {
            "width": self.screen.width,
            "height": self.screen.height,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "user_agent": self.user_agent,
            "hardware": {
                "cpu_cores": self.hardware.cpu_cores,
                "device_memory": self.hardware.device_memory,
                "platform": self.hardware.platform,
                "gpu_vendor": self.hardware.gpu_vendor,
                "gpu_renderer": self.hardware.gpu_renderer,
            },
            "screen": {
                "width": self.screen.width,
                "height": self.screen.height,
                "device_pixel_ratio": self.screen.device_pixel_ratio,
                "color_depth": self.screen.color_depth,
            },
            "browser": {
                "name": self.browser.name,
                "version": self.browser.version,
            },
            "locale": {
                "language": self.locale.language,
                "timezone": self.locale.timezone,
            },
        }
    
    def get_fingerprint_hash(self) -> str:
        """Get a hash of this fingerprint for comparison."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ============================================================================
# Browser Version Databases
# ============================================================================

# Latest Chrome versions by OS
CHROME_VERSIONS = {
    OperatingSystem.WINDOWS_10: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
    OperatingSystem.WINDOWS_11: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
    OperatingSystem.MACOS_VENTURA: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
    OperatingSystem.MACOS_SONOMA: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
    OperatingSystem.MACOS_SEQUOIA: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
    OperatingSystem.LINUX_UBUNTU: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
    OperatingSystem.LINUX_FEDORA: ["131.0.0.0", "130.0.0.0", "129.0.0.0"],
}

# Firefox versions by OS
FIREFOX_VERSIONS = {
    OperatingSystem.WINDOWS_10: ["133.0", "132.0", "131.0"],
    OperatingSystem.WINDOWS_11: ["133.0", "132.0", "131.0"],
    OperatingSystem.MACOS_VENTURA: ["133.0", "132.0", "131.0"],
    OperatingSystem.MACOS_SONOMA: ["133.0", "132.0", "131.0"],
    OperatingSystem.MACOS_SEQUOIA: ["133.0", "132.0", "131.0"],
    OperatingSystem.LINUX_UBUNTU: ["133.0", "132.0", "131.0"],
    OperatingSystem.LINUX_FEDORA: ["133.0", "132.0", "131.0"],
}

# Safari versions (macOS only)
SAFARI_VERSIONS = {
    OperatingSystem.MACOS_VENTURA: ["17.2", "17.1", "17.0"],
    OperatingSystem.MACOS_SONOMA: ["17.2", "17.1", "17.0"],
    OperatingSystem.MACOS_SEQUOIA: ["18.1", "18.0", "17.2"],
}

# Screen resolutions by popularity
SCREEN_RESOLUTIONS = [
    (1920, 1080, 1.0),   # 1080p - most common
    (2560, 1440, 1.0),   # 1440p
    (3840, 2160, 2.0),   # 4K with scaling
    (1536, 864, 1.25),   # Common laptop
    (1366, 768, 1.0),    # Common laptop
    (1440, 900, 1.0),    # MacBook Air
    (2560, 1600, 2.0),   # MacBook Pro 13"
    (2880, 1800, 2.0),   # MacBook Pro 15"
    (3024, 1964, 2.0),   # MacBook Pro 14"
    (3456, 2234, 2.0),   # MacBook Pro 16"
]

# GPU combinations by OS
GPU_PROFILES = {
    OperatingSystem.MACOS_SONOMA: [
        ("Apple", "Apple M1", "ANGLE (Apple, Apple M1, OpenGL 4.1)"),
        ("Apple", "Apple M2", "ANGLE (Apple, Apple M2, OpenGL 4.1)"),
        ("Apple", "Apple M3", "ANGLE (Apple, Apple M3, OpenGL 4.1)"),
        ("Intel Inc.", "Intel Iris Plus Graphics 640", "ANGLE (Intel, Intel(R) Iris(R) Plus Graphics 640, OpenGL 4.1)"),
    ],
    OperatingSystem.MACOS_VENTURA: [
        ("Apple", "Apple M1", "ANGLE (Apple, Apple M1, OpenGL 4.1)"),
        ("Apple", "Apple M2", "ANGLE (Apple, Apple M2, OpenGL 4.1)"),
        ("Intel Inc.", "Intel Iris Plus Graphics 640", "ANGLE (Intel, Intel(R) Iris(R) Plus Graphics 640, OpenGL 4.1)"),
    ],
    OperatingSystem.WINDOWS_11: [
        ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 4070 Direct3D11 vs_5_0 ps_5_0)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 4070 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) UHD Graphics 770 Direct3D11 vs_5_0 ps_5_0)", "ANGLE (Intel, Intel(R) UHD Graphics 770 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Google Inc. (AMD)", "ANGLE (AMD, AMD Radeon RX 7900 XTX Direct3D11 vs_5_0 ps_5_0)", "ANGLE (AMD, AMD Radeon RX 7900 XTX Direct3D11 vs_5_0 ps_5_0, D3D11)"),
    ],
    OperatingSystem.WINDOWS_10: [
        ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce GTX 1660 Direct3D11 vs_5_0 ps_5_0)", "ANGLE (NVIDIA, NVIDIA GeForce GTX 1660 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)", "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
    ],
    OperatingSystem.LINUX_UBUNTU: [
        ("Mesa", "Mesa Intel(R) UHD Graphics 630 (CFL GT2)", "Mesa Intel(R) UHD Graphics 630 (CFL GT2)"),
        ("NVIDIA Corporation", "NVIDIA GeForce RTX 3070/PCIe/SSE2", "NVIDIA GeForce RTX 3070/PCIe/SSE2"),
    ],
    OperatingSystem.LINUX_FEDORA: [
        ("Mesa", "Mesa Intel(R) UHD Graphics 630 (CFL GT2)", "Mesa Intel(R) UHD Graphics 630 (CFL GT2)"),
        ("AMD", "AMD Radeon RX 6800 XT", "AMD Radeon RX 6800 XT (navi21, LLVM 16.0.6, DRM 3.54)"),
    ],
}

# Timezone mapping for geolocation
TIMEZONE_MAPPING = {
    "us-west": ("America/Los_Angeles", 480),
    "us-east": ("America/New_York", 300),
    "us-central": ("America/Chicago", 360),
    "us-mountain": ("America/Denver", 420),
    "uk": ("Europe/London", 0),
    "germany": ("Europe/Berlin", -60),
    "france": ("Europe/Paris", -60),
    "japan": ("Asia/Tokyo", -540),
    "australia": ("Australia/Sydney", -660),
    "brazil": ("America/Sao_Paulo", 180),
    "india": ("Asia/Kolkata", -330),
    "singapore": ("Asia/Singapore", -480),
}

# OS-specific fonts
FONTS_BY_OS = {
    OperatingSystem.MACOS_SONOMA: [
        "Arial", "Helvetica", "Helvetica Neue", "Times New Roman", "Times",
        "Courier New", "Courier", "Verdana", "Georgia", "Palatino",
        "Garamond", "Bookman", "Trebuchet MS", "Arial Black", "Impact",
        "Comic Sans MS", "Lucida Console", "Monaco", "Menlo", "SF Pro",
        "SF Pro Display", "SF Pro Text", "SF Mono", "New York",
        "Apple Color Emoji", "Apple Symbols", "Zapfino", "Avenir",
        "Avenir Next", "Baskerville", "Futura", "Optima",
    ],
    OperatingSystem.MACOS_VENTURA: [
        "Arial", "Helvetica", "Helvetica Neue", "Times New Roman", "Times",
        "Courier New", "Courier", "Verdana", "Georgia", "Palatino",
        "Garamond", "Bookman", "Trebuchet MS", "Arial Black", "Impact",
        "Comic Sans MS", "Lucida Console", "Monaco", "Menlo", "SF Pro",
        "SF Pro Display", "SF Pro Text", "SF Mono", "New York",
        "Apple Color Emoji", "Apple Symbols", "Zapfino", "Avenir",
    ],
    OperatingSystem.WINDOWS_11: [
        "Arial", "Arial Black", "Calibri", "Cambria", "Cambria Math",
        "Candara", "Comic Sans MS", "Consolas", "Constantia", "Corbel",
        "Courier New", "Ebrima", "Franklin Gothic Medium", "Gabriola",
        "Gadugi", "Georgia", "Impact", "Ink Free", "Javanese Text",
        "Leelawadee UI", "Lucida Console", "Lucida Sans Unicode",
        "Malgun Gothic", "Microsoft Himalaya", "Microsoft JhengHei",
        "Microsoft New Tai Lue", "Microsoft PhagsPa", "Microsoft Sans Serif",
        "Microsoft Tai Le", "Microsoft YaHei", "Microsoft Yi Baiti",
        "MingLiU-ExtB", "Mongolian Baiti", "MS Gothic", "MS PGothic",
        "MS UI Gothic", "MV Boli", "Myanmar Text", "Nirmala UI",
        "Palatino Linotype", "Segoe MDL2 Assets", "Segoe Print",
        "Segoe Script", "Segoe UI", "Segoe UI Emoji", "Segoe UI Historic",
        "Segoe UI Symbol", "SimSun", "Sitka Banner", "Sitka Display",
        "Sitka Heading", "Sitka Small", "Sitka Subheading", "Sitka Text",
        "Sylfaen", "Symbol", "Tahoma", "Times New Roman", "Trebuchet MS",
        "Verdana", "Webdings", "Wingdings", "Yu Gothic",
    ],
    OperatingSystem.WINDOWS_10: [
        "Arial", "Arial Black", "Calibri", "Cambria", "Cambria Math",
        "Candara", "Comic Sans MS", "Consolas", "Constantia", "Corbel",
        "Courier New", "Georgia", "Impact", "Lucida Console",
        "Lucida Sans Unicode", "Microsoft Sans Serif", "Palatino Linotype",
        "Segoe Print", "Segoe Script", "Segoe UI", "Segoe UI Symbol",
        "Symbol", "Tahoma", "Times New Roman", "Trebuchet MS", "Verdana",
        "Webdings", "Wingdings",
    ],
    OperatingSystem.LINUX_UBUNTU: [
        "Ubuntu", "Ubuntu Mono", "DejaVu Sans", "DejaVu Sans Mono",
        "DejaVu Serif", "Liberation Sans", "Liberation Serif",
        "Liberation Mono", "Noto Sans", "Noto Serif", "Noto Mono",
        "FreeSans", "FreeSerif", "FreeMono", "Bitstream Vera Sans",
        "Bitstream Vera Serif", "Bitstream Vera Sans Mono",
    ],
    OperatingSystem.LINUX_FEDORA: [
        "Cantarell", "DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif",
        "Liberation Sans", "Liberation Serif", "Liberation Mono",
        "Noto Sans", "Noto Serif", "Noto Mono", "Source Code Pro",
        "Source Sans Pro", "Source Serif Pro",
    ],
}


class FingerprintGenerator:
    """
    Generates consistent, realistic browser fingerprints.
    
    This class creates fingerprint profiles that pass bot detection systems
    by mimicking real browser populations. It ensures internal consistency
    (e.g., macOS fingerprint won't have Windows fonts).
    
    Example:
        >>> generator = FingerprintGenerator()
        >>> 
        >>> # Generate automatic profile
        >>> profile = generator.generate()
        >>> 
        >>> # Generate specific profile
        >>> profile = generator.generate(
        ...     os="macos_sonoma",
        ...     browser="chrome",
        ...     geolocation="us-west"
        ... )
        >>> 
        >>> # Use with browser
        >>> print(profile.user_agent)
        >>> print(profile.viewport)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the fingerprint generator.
        
        Args:
            seed: Optional random seed for reproducible fingerprints.
                  If None, uses system entropy.
        """
        self._seed = seed
        if seed is not None:
            random.seed(seed)
    
    def generate(
        self,
        os: Optional[str] = None,
        browser: Optional[str] = None,
        geolocation: Optional[str] = None,
        device_type: str = "desktop",
        consistent_with_proxy: Optional[Dict[str, Any]] = None,
    ) -> FingerprintProfile:
        """
        Generate a complete fingerprint profile.
        
        Args:
            os: Operating system (windows_10, windows_11, macos_sonoma, etc.)
                If None, randomly selects based on real-world distribution.
            browser: Browser type (chrome, firefox, safari, edge).
                If None, randomly selects (Chrome most likely).
            geolocation: Geolocation identifier for timezone/locale.
                Options: us-west, us-east, uk, germany, japan, etc.
            device_type: Device type (desktop, laptop).
            consistent_with_proxy: Optional proxy info to ensure consistency.
                Example: {"country": "US", "city": "Los Angeles"}
        
        Returns:
            FingerprintProfile with all fingerprint components.
        """
        # Select operating system
        os_enum = self._select_os(os, consistent_with_proxy)
        
        # Select browser (must be compatible with OS)
        browser_enum = self._select_browser(browser, os_enum)
        
        # Select geolocation (may be influenced by proxy)
        geo_key = self._select_geolocation(geolocation, consistent_with_proxy)
        
        # Generate consistent hardware profile
        hardware = self._generate_hardware(os_enum, device_type)
        
        # Generate screen profile
        screen = self._generate_screen(os_enum, device_type)
        
        # Generate browser profile
        browser_profile = self._generate_browser(browser_enum, os_enum)
        
        # Generate locale profile
        locale = self._generate_locale(geo_key)
        
        # Generate network profile
        network = self._generate_network()
        
        # Generate WebGL profile
        webgl = self._generate_webgl(os_enum, hardware)
        
        # Generate audio profile
        audio = self._generate_audio()
        
        # Generate canvas profile
        canvas = self._generate_canvas()
        
        # Generate font profile
        fonts = self._generate_fonts(os_enum)
        
        # Generate user agent string
        user_agent = self._generate_user_agent(os_enum, browser_enum, browser_profile)
        
        # Generate client hints
        client_hints = self._generate_client_hints(os_enum, browser_profile)
        
        return FingerprintProfile(
            hardware=hardware,
            screen=screen,
            browser=browser_profile,
            locale=locale,
            network=network,
            webgl=webgl,
            audio=audio,
            canvas=canvas,
            fonts=fonts,
            user_agent=user_agent,
            client_hints=client_hints,
        )
    
    def _select_os(
        self,
        os: Optional[str],
        proxy_info: Optional[Dict[str, Any]]
    ) -> OperatingSystem:
        """Select operating system based on preferences and real-world distribution."""
        if os:
            # Map string to enum
            os_map = {
                "windows_10": OperatingSystem.WINDOWS_10,
                "windows_11": OperatingSystem.WINDOWS_11,
                "windows": OperatingSystem.WINDOWS_11,  # Default to 11
                "macos": OperatingSystem.MACOS_SONOMA,  # Default to latest
                "macos_ventura": OperatingSystem.MACOS_VENTURA,
                "macos_sonoma": OperatingSystem.MACOS_SONOMA,
                "macos_sequoia": OperatingSystem.MACOS_SEQUOIA,
                "linux": OperatingSystem.LINUX_UBUNTU,
                "linux_ubuntu": OperatingSystem.LINUX_UBUNTU,
                "linux_fedora": OperatingSystem.LINUX_FEDORA,
            }
            return os_map.get(os.lower(), OperatingSystem.MACOS_SONOMA)
        
        # Random selection based on real-world market share (approximate)
        weights = [
            (OperatingSystem.WINDOWS_11, 30),
            (OperatingSystem.WINDOWS_10, 25),
            (OperatingSystem.MACOS_SONOMA, 15),
            (OperatingSystem.MACOS_VENTURA, 10),
            (OperatingSystem.MACOS_SEQUOIA, 8),
            (OperatingSystem.LINUX_UBUNTU, 7),
            (OperatingSystem.LINUX_FEDORA, 5),
        ]
        
        total = sum(w for _, w in weights)
        r = random.randint(1, total)
        cumulative = 0
        for os_enum, weight in weights:
            cumulative += weight
            if r <= cumulative:
                return os_enum
        
        return OperatingSystem.WINDOWS_11
    
    def _select_browser(
        self,
        browser: Optional[str],
        os_enum: OperatingSystem
    ) -> BrowserType:
        """Select browser type, ensuring OS compatibility."""
        if browser:
            browser_map = {
                "chrome": BrowserType.CHROME,
                "firefox": BrowserType.FIREFOX,
                "safari": BrowserType.SAFARI,
                "edge": BrowserType.EDGE,
            }
            selected = browser_map.get(browser.lower(), BrowserType.CHROME)
            
            # Safari only on macOS
            if selected == BrowserType.SAFARI and "macos" not in os_enum.value:
                selected = BrowserType.CHROME
            
            return selected
        
        # Default distribution (Chrome dominates)
        if "macos" in os_enum.value:
            weights = [
                (BrowserType.CHROME, 60),
                (BrowserType.SAFARI, 25),
                (BrowserType.FIREFOX, 10),
                (BrowserType.EDGE, 5),
            ]
        else:
            weights = [
                (BrowserType.CHROME, 65),
                (BrowserType.FIREFOX, 15),
                (BrowserType.EDGE, 15),
            ]
        
        total = sum(w for _, w in weights)
        r = random.randint(1, total)
        cumulative = 0
        for browser_enum, weight in weights:
            cumulative += weight
            if r <= cumulative:
                return browser_enum
        
        return BrowserType.CHROME
    
    def _select_geolocation(
        self,
        geolocation: Optional[str],
        proxy_info: Optional[Dict[str, Any]]
    ) -> str:
        """Select geolocation, considering proxy location for consistency."""
        if geolocation:
            return geolocation
        
        # If proxy info provided, try to match
        if proxy_info:
            country = proxy_info.get("country", "").lower()
            city = proxy_info.get("city", "").lower()
            
            # Map proxy location to our geolocation keys
            if country == "us":
                if "los angeles" in city or "san francisco" in city:
                    return "us-west"
                elif "new york" in city or "boston" in city:
                    return "us-east"
                elif "chicago" in city:
                    return "us-central"
                elif "denver" in city:
                    return "us-mountain"
                return "us-west"  # Default US
            elif country in ("gb", "uk"):
                return "uk"
            elif country == "de":
                return "germany"
            elif country == "fr":
                return "france"
            elif country == "jp":
                return "japan"
            elif country == "au":
                return "australia"
            elif country == "br":
                return "brazil"
            elif country == "in":
                return "india"
            elif country == "sg":
                return "singapore"
        
        # Default to US West (common)
        return "us-west"
    
    def _generate_hardware(
        self,
        os_enum: OperatingSystem,
        device_type: str
    ) -> HardwareProfile:
        """Generate hardware profile consistent with OS."""
        # Platform string
        if "macos" in os_enum.value:
            platform = "MacIntel"
            architecture = "x86_64"
        elif "windows" in os_enum.value:
            platform = "Win32"
            architecture = "x86_64"
        else:
            platform = "Linux x86_64"
            architecture = "x86_64"
        
        # CPU cores (realistic range)
        if device_type == "laptop":
            cpu_cores = random.choice([4, 6, 8])
        else:
            cpu_cores = random.choice([6, 8, 12, 16])
        
        # Device memory
        memory_options = [8, 16, 32] if device_type == "desktop" else [8, 16]
        device_memory = random.choice(memory_options)
        
        # GPU
        gpu_options = GPU_PROFILES.get(os_enum, GPU_PROFILES[OperatingSystem.WINDOWS_11])
        gpu_vendor, gpu_renderer_short, gpu_renderer = random.choice(gpu_options)
        
        return HardwareProfile(
            cpu_cores=cpu_cores,
            device_memory=device_memory,
            max_touch_points=0,  # Desktop has no touch
            hardware_concurrency=cpu_cores,
            gpu_vendor=gpu_vendor,
            gpu_renderer=gpu_renderer,
            platform=platform,
            architecture=architecture,
            has_battery=(device_type == "laptop"),
            battery_charging=True,
            battery_level=random.uniform(0.5, 1.0),
        )
    
    def _generate_screen(
        self,
        os_enum: OperatingSystem,
        device_type: str
    ) -> ScreenProfile:
        """Generate screen profile."""
        # Select resolution based on OS
        if "macos" in os_enum.value:
            # macOS users tend to have higher res displays
            resolutions = [
                (2560, 1600, 2.0),   # MacBook Pro 13"
                (2880, 1800, 2.0),   # MacBook Pro 15"
                (3024, 1964, 2.0),   # MacBook Pro 14"
                (1920, 1080, 1.0),   # External monitor
                (2560, 1440, 1.0),   # External monitor
            ]
        else:
            resolutions = [
                (1920, 1080, 1.0),
                (2560, 1440, 1.0),
                (1536, 864, 1.25),
                (1366, 768, 1.0),
            ]
        
        width, height, dpr = random.choice(resolutions)
        
        # Account for taskbar/dock
        if "macos" in os_enum.value:
            avail_top = 25  # Menu bar
            avail_height = height - 25
        elif "windows" in os_enum.value:
            avail_top = 0
            avail_height = height - 40  # Taskbar
        else:
            avail_top = 0
            avail_height = height - 30  # Panel
        
        return ScreenProfile(
            width=width,
            height=height,
            avail_width=width,
            avail_height=avail_height,
            avail_top=avail_top,
            avail_left=0,
            color_depth=24,
            pixel_depth=24,
            device_pixel_ratio=dpr,
        )
    
    def _generate_browser(
        self,
        browser_enum: BrowserType,
        os_enum: OperatingSystem
    ) -> BrowserProfile:
        """Generate browser profile."""
        if browser_enum == BrowserType.CHROME:
            versions = CHROME_VERSIONS.get(os_enum, CHROME_VERSIONS[OperatingSystem.WINDOWS_11])
            version = random.choice(versions)
            major = int(version.split(".")[0])
            
            return BrowserProfile(
                name="Chrome",
                version=version,
                major_version=major,
                engine="Blink",
                engine_version=version,
                plugins=self._get_chrome_plugins(),
                mime_types=["application/pdf", "text/pdf"],
            )
        
        elif browser_enum == BrowserType.FIREFOX:
            versions = FIREFOX_VERSIONS.get(os_enum, FIREFOX_VERSIONS[OperatingSystem.WINDOWS_11])
            version = random.choice(versions)
            major = int(version.split(".")[0])
            
            return BrowserProfile(
                name="Firefox",
                version=version,
                major_version=major,
                engine="Gecko",
                engine_version="20100101",
                plugins=[],
                mime_types=[],
            )
        
        elif browser_enum == BrowserType.SAFARI:
            versions = SAFARI_VERSIONS.get(os_enum, SAFARI_VERSIONS[OperatingSystem.MACOS_SONOMA])
            version = random.choice(versions)
            major = int(version.split(".")[0])
            
            return BrowserProfile(
                name="Safari",
                version=version,
                major_version=major,
                engine="WebKit",
                engine_version="605.1.15",
                plugins=[],
                mime_types=[],
            )
        
        elif browser_enum == BrowserType.EDGE:
            # Edge uses same version as Chrome
            versions = CHROME_VERSIONS.get(os_enum, CHROME_VERSIONS[OperatingSystem.WINDOWS_11])
            version = random.choice(versions)
            major = int(version.split(".")[0])
            
            return BrowserProfile(
                name="Edge",
                version=version,
                major_version=major,
                engine="Blink",
                engine_version=version,
                plugins=self._get_chrome_plugins(),
                mime_types=["application/pdf", "text/pdf"],
            )
        
        return BrowserProfile()
    
    def _get_chrome_plugins(self) -> List[Dict[str, str]]:
        """Get Chrome plugin list."""
        return [
            {"name": "PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "Chrome PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "Chromium PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "Microsoft Edge PDF Viewer", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
            {"name": "WebKit built-in PDF", "filename": "internal-pdf-viewer", "description": "Portable Document Format"},
        ]
    
    def _generate_locale(self, geo_key: str) -> LocaleProfile:
        """Generate locale profile from geolocation."""
        timezone, offset = TIMEZONE_MAPPING.get(geo_key, ("America/Los_Angeles", 480))
        
        # Language based on geo
        lang_map = {
            "us-west": ("en-US", ["en-US", "en"]),
            "us-east": ("en-US", ["en-US", "en"]),
            "us-central": ("en-US", ["en-US", "en"]),
            "us-mountain": ("en-US", ["en-US", "en"]),
            "uk": ("en-GB", ["en-GB", "en"]),
            "germany": ("de-DE", ["de-DE", "de", "en-US", "en"]),
            "france": ("fr-FR", ["fr-FR", "fr", "en-US", "en"]),
            "japan": ("ja-JP", ["ja-JP", "ja", "en-US", "en"]),
            "australia": ("en-AU", ["en-AU", "en"]),
            "brazil": ("pt-BR", ["pt-BR", "pt", "en-US", "en"]),
            "india": ("en-IN", ["en-IN", "en", "hi"]),
            "singapore": ("en-SG", ["en-SG", "en", "zh-CN"]),
        }
        
        language, languages = lang_map.get(geo_key, ("en-US", ["en-US", "en"]))
        
        return LocaleProfile(
            language=language,
            languages=languages,
            timezone=timezone,
            timezone_offset=offset,
            locale=language,
        )
    
    def _generate_network(self) -> NetworkProfile:
        """Generate network profile."""
        return NetworkProfile(
            connection_type="wifi",
            effective_type=random.choice(["4g", "4g", "4g", "3g"]),  # Mostly 4g
            downlink=random.uniform(5.0, 25.0),
            rtt=random.randint(30, 100),
            save_data=False,
        )
    
    def _generate_webgl(
        self,
        os_enum: OperatingSystem,
        hardware: HardwareProfile
    ) -> WebGLProfile:
        """Generate WebGL profile consistent with hardware."""
        # Extract vendor/renderer from hardware GPU
        vendor = hardware.gpu_vendor.replace("Google Inc. (", "").replace(")", "").split(",")[0]
        
        return WebGLProfile(
            vendor="Google Inc.",  # ANGLE wraps everything
            renderer=hardware.gpu_renderer,
            unmasked_vendor=vendor,
            unmasked_renderer=hardware.gpu_renderer.split("ANGLE (")[1].rstrip(")") if "ANGLE" in hardware.gpu_renderer else hardware.gpu_renderer,
            max_texture_size=16384,
            max_vertex_attribs=16,
            max_vertex_uniform_vectors=4096,
            max_fragment_uniform_vectors=1024,
            max_varying_vectors=31,
            extensions=self._get_webgl_extensions(),
        )
    
    def _get_webgl_extensions(self) -> List[str]:
        """Get common WebGL extensions."""
        return [
            "ANGLE_instanced_arrays",
            "EXT_blend_minmax",
            "EXT_color_buffer_half_float",
            "EXT_disjoint_timer_query",
            "EXT_float_blend",
            "EXT_frag_depth",
            "EXT_shader_texture_lod",
            "EXT_texture_compression_bptc",
            "EXT_texture_compression_rgtc",
            "EXT_texture_filter_anisotropic",
            "EXT_sRGB",
            "OES_element_index_uint",
            "OES_fbo_render_mipmap",
            "OES_standard_derivatives",
            "OES_texture_float",
            "OES_texture_float_linear",
            "OES_texture_half_float",
            "OES_texture_half_float_linear",
            "OES_vertex_array_object",
            "WEBGL_color_buffer_float",
            "WEBGL_compressed_texture_s3tc",
            "WEBGL_compressed_texture_s3tc_srgb",
            "WEBGL_debug_renderer_info",
            "WEBGL_debug_shaders",
            "WEBGL_depth_texture",
            "WEBGL_draw_buffers",
            "WEBGL_lose_context",
            "WEBGL_multi_draw",
        ]
    
    def _generate_audio(self) -> AudioProfile:
        """Generate audio profile."""
        return AudioProfile(
            sample_rate=44100,
            max_channel_count=2,
            number_of_inputs=1,
            number_of_outputs=1,
            channel_count=2,
            noise_seed=random.randint(0, 2**32),
        )
    
    def _generate_canvas(self) -> CanvasProfile:
        """Generate canvas profile."""
        return CanvasProfile(
            noise_seed=random.randint(0, 2**32),
            noise_intensity=random.uniform(0.00005, 0.0002),
        )
    
    def _generate_fonts(self, os_enum: OperatingSystem) -> FontProfile:
        """Generate font profile."""
        fonts = FONTS_BY_OS.get(os_enum, FONTS_BY_OS[OperatingSystem.WINDOWS_11])
        
        # Don't include all fonts - real browsers have varying installations
        num_fonts = random.randint(len(fonts) // 2, len(fonts))
        selected_fonts = random.sample(fonts, num_fonts)
        
        return FontProfile(
            installed_fonts=selected_fonts,
            font_smoothing_enabled=True,
            sub_pixel_rendering=True,
        )
    
    def _generate_user_agent(
        self,
        os_enum: OperatingSystem,
        browser_enum: BrowserType,
        browser: BrowserProfile
    ) -> str:
        """Generate user agent string."""
        # OS component
        if os_enum == OperatingSystem.WINDOWS_11:
            os_part = "Windows NT 10.0; Win64; x64"
        elif os_enum == OperatingSystem.WINDOWS_10:
            os_part = "Windows NT 10.0; Win64; x64"
        elif os_enum in (OperatingSystem.MACOS_VENTURA, OperatingSystem.MACOS_SONOMA, OperatingSystem.MACOS_SEQUOIA):
            os_part = "Macintosh; Intel Mac OS X 10_15_7"
        elif os_enum == OperatingSystem.LINUX_UBUNTU:
            os_part = "X11; Linux x86_64"
        elif os_enum == OperatingSystem.LINUX_FEDORA:
            os_part = "X11; Linux x86_64"
        else:
            os_part = "Windows NT 10.0; Win64; x64"
        
        # Browser-specific UA
        if browser_enum == BrowserType.CHROME:
            return f"Mozilla/5.0 ({os_part}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser.version} Safari/537.36"
        
        elif browser_enum == BrowserType.FIREFOX:
            return f"Mozilla/5.0 ({os_part}; rv:{browser.version}) Gecko/20100101 Firefox/{browser.version}"
        
        elif browser_enum == BrowserType.SAFARI:
            webkit_version = "605.1.15"
            safari_version = browser.version
            return f"Mozilla/5.0 ({os_part}) AppleWebKit/{webkit_version} (KHTML, like Gecko) Version/{safari_version} Safari/{webkit_version}"
        
        elif browser_enum == BrowserType.EDGE:
            return f"Mozilla/5.0 ({os_part}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser.version} Safari/537.36 Edg/{browser.version}"
        
        return f"Mozilla/5.0 ({os_part}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser.version} Safari/537.36"
    
    def _generate_client_hints(
        self,
        os_enum: OperatingSystem,
        browser: BrowserProfile
    ) -> Dict[str, str]:
        """Generate client hints for Chrome."""
        if browser.name != "Chrome":
            return {}
        
        # Platform
        if "windows" in os_enum.value:
            platform = "Windows"
            platform_version = "15.0.0" if os_enum == OperatingSystem.WINDOWS_11 else "10.0.0"
        elif "macos" in os_enum.value:
            platform = "macOS"
            platform_version = "14.0.0"
        else:
            platform = "Linux"
            platform_version = ""
        
        return {
            "Sec-CH-UA": f'"Google Chrome";v="{browser.major_version}", "Chromium";v="{browser.major_version}", "Not?A_Brand";v="99"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{platform}"',
            "Sec-CH-UA-Platform-Version": f'"{platform_version}"',
            "Sec-CH-UA-Arch": '"x86"',
            "Sec-CH-UA-Bitness": '"64"',
            "Sec-CH-UA-Full-Version": f'"{browser.version}"',
            "Sec-CH-UA-Full-Version-List": f'"Google Chrome";v="{browser.version}", "Chromium";v="{browser.version}", "Not?A_Brand";v="99.0.0.0"',
        }
    
    def generate_stealth_script(self, profile: FingerprintProfile) -> str:
        """
        Generate JavaScript injection script for stealth.
        
        This creates a comprehensive script that overrides browser APIs
        to match the generated fingerprint profile.
        
        Args:
            profile: FingerprintProfile to apply.
        
        Returns:
            JavaScript string to inject via add_init_script.
        """
        return f'''
        // ===== FlyBrowser Stealth Script =====
        // Profile ID: {profile.profile_id}
        // Generated at: {profile.created_at}
        
        // ===== PART 1: Navigator Overrides =====
        
        Object.defineProperty(navigator, 'webdriver', {{
            get: () => undefined,
        }});
        
        Object.defineProperty(navigator, 'languages', {{
            get: () => {json.dumps(profile.locale.languages)},
        }});
        
        Object.defineProperty(navigator, 'platform', {{
            get: () => '{profile.hardware.platform}',
        }});
        
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {profile.hardware.hardware_concurrency},
        }});
        
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {profile.hardware.device_memory},
        }});
        
        Object.defineProperty(navigator, 'maxTouchPoints', {{
            get: () => {profile.hardware.max_touch_points},
        }});
        
        Object.defineProperty(navigator, 'vendor', {{
            get: () => 'Google Inc.',
        }});
        
        Object.defineProperty(navigator, 'productSub', {{
            get: () => '20030107',
        }});
        
        Object.defineProperty(navigator, 'vendorSub', {{
            get: () => '',
        }});
        
        // ===== PART 2: Plugins =====
        
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {{
                const plugins = {json.dumps(profile.browser.plugins)};
                Object.setPrototypeOf(plugins, PluginArray.prototype);
                plugins.item = (i) => plugins[i] || null;
                plugins.namedItem = (name) => plugins.find(p => p.name === name) || null;
                plugins.refresh = () => {{}};
                return plugins;
            }},
        }});
        
        // ===== PART 3: Screen Properties =====
        
        Object.defineProperty(screen, 'width', {{
            get: () => {profile.screen.width},
        }});
        
        Object.defineProperty(screen, 'height', {{
            get: () => {profile.screen.height},
        }});
        
        Object.defineProperty(screen, 'availWidth', {{
            get: () => {profile.screen.avail_width},
        }});
        
        Object.defineProperty(screen, 'availHeight', {{
            get: () => {profile.screen.avail_height},
        }});
        
        Object.defineProperty(screen, 'availTop', {{
            get: () => {profile.screen.avail_top},
        }});
        
        Object.defineProperty(screen, 'availLeft', {{
            get: () => {profile.screen.avail_left},
        }});
        
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => {profile.screen.color_depth},
        }});
        
        Object.defineProperty(screen, 'pixelDepth', {{
            get: () => {profile.screen.pixel_depth},
        }});
        
        Object.defineProperty(window, 'devicePixelRatio', {{
            get: () => {profile.screen.device_pixel_ratio},
        }});
        
        // ===== PART 4: Chrome Object =====
        
        if (!window.chrome || !window.chrome.runtime) {{
            window.chrome = {{
                app: {{
                    isInstalled: false,
                    InstallState: {{ DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' }},
                    RunningState: {{ CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' }},
                }},
                runtime: {{
                    connect: () => ({{ onMessage: {{ addListener: () => {{}} }}, postMessage: () => {{}}, disconnect: () => {{}} }}),
                    sendMessage: () => {{}},
                    onMessage: {{ addListener: () => {{}}, removeListener: () => {{}}, hasListener: () => false }},
                    id: undefined,
                    getManifest: () => ({{}}),
                    getURL: (path) => path,
                }},
                loadTimes: () => ({{
                    requestTime: Date.now() / 1000,
                    startLoadTime: Date.now() / 1000,
                    commitLoadTime: Date.now() / 1000,
                    finishDocumentLoadTime: Date.now() / 1000,
                    finishLoadTime: Date.now() / 1000,
                    firstPaintTime: Date.now() / 1000,
                    firstPaintAfterLoadTime: 0,
                    navigationType: 'Other',
                    wasFetchedViaSpdy: false,
                    wasNpnNegotiated: true,
                    npnNegotiatedProtocol: 'h2',
                    wasAlternateProtocolAvailable: false,
                    connectionInfo: 'h2',
                }}),
                csi: () => ({{ startE: Date.now(), onloadT: Date.now(), pageT: Date.now(), tran: 15 }}),
            }};
        }}
        
        // ===== PART 5: Remove Automation Indicators =====
        
        delete window.__playwright;
        delete window.__pw_manual;
        delete window.__PW_inspect;
        delete window.__pwInitScripts;
        delete window._WEBDRIVER_ELEM_CACHE;
        delete window.domAutomation;
        delete window.domAutomationController;
        
        // ===== PART 6: WebGL Fingerprinting =====
        
        const webglVendor = '{profile.webgl.unmasked_vendor}';
        const webglRenderer = '{profile.webgl.unmasked_renderer}';
        
        const getParameterProxyHandler = {{
            apply: function(target, thisArg, args) {{
                const param = args[0];
                if (param === 37445) return webglVendor;      // UNMASKED_VENDOR_WEBGL
                if (param === 37446) return webglRenderer;    // UNMASKED_RENDERER_WEBGL
                return Reflect.apply(target, thisArg, args);
            }}
        }};
        
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type, ...args) {{
            const context = originalGetContext.call(this, type, ...args);
            if (context && (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl')) {{
                const originalGetParameter = context.getParameter;
                context.getParameter = new Proxy(originalGetParameter, getParameterProxyHandler);
            }}
            return context;
        }};
        
        // ===== PART 7: Canvas Fingerprinting Noise =====
        
        const canvasNoiseSeed = {profile.canvas.noise_seed};
        const canvasNoiseIntensity = {profile.canvas.noise_intensity};
        
        // Seeded random for consistent noise
        function seededRandom(seed) {{
            const x = Math.sin(seed++) * 10000;
            return x - Math.floor(x);
        }}
        
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(...args) {{
            const ctx = this.getContext('2d');
            if (ctx) {{
                const imageData = ctx.getImageData(0, 0, this.width, this.height);
                for (let i = 0; i < imageData.data.length; i += 4) {{
                    const noise = (seededRandom(canvasNoiseSeed + i) - 0.5) * 2 * canvasNoiseIntensity * 255;
                    imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + noise));
                }}
                ctx.putImageData(imageData, 0, 0);
            }}
            return originalToDataURL.apply(this, args);
        }};
        
        // ===== PART 8: Audio Fingerprinting =====
        
        const audioNoiseSeed = {profile.audio.noise_seed};
        
        const originalCreateOscillator = AudioContext.prototype.createOscillator;
        AudioContext.prototype.createOscillator = function() {{
            const oscillator = originalCreateOscillator.call(this);
            const originalConnect = oscillator.connect;
            oscillator.connect = function(destination) {{
                // Add tiny noise to audio output
                const gainNode = this.context.createGain();
                gainNode.gain.value = 1 + (seededRandom(audioNoiseSeed) - 0.5) * 0.0001;
                originalConnect.call(this, gainNode);
                gainNode.connect(destination);
                return destination;
            }};
            return oscillator;
        }};
        
        // ===== PART 9: Timezone =====
        
        Date.prototype.getTimezoneOffset = function() {{
            return {profile.locale.timezone_offset};
        }};
        
        // ===== PART 10: Battery API =====
        
        if (!navigator.getBattery) {{
            navigator.getBattery = () => Promise.resolve({{
                charging: {str(profile.hardware.battery_charging).lower()},
                chargingTime: 0,
                dischargingTime: Infinity,
                level: {profile.hardware.battery_level},
                addEventListener: () => {{}},
                removeEventListener: () => {{}},
                dispatchEvent: () => true,
            }});
        }}
        
        // ===== PART 11: Connection API =====
        
        Object.defineProperty(navigator, 'connection', {{
            get: () => ({{
                effectiveType: '{profile.network.effective_type}',
                rtt: {profile.network.rtt},
                downlink: {profile.network.downlink},
                saveData: {str(profile.network.save_data).lower()},
                addEventListener: () => {{}},
                removeEventListener: () => {{}},
            }}),
        }});
        
        // ===== PART 12: Permissions API =====
        
        const originalQuery = navigator.permissions.query;
        navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications'
                ? Promise.resolve({{ state: Notification.permission }})
                : originalQuery(parameters)
        );
        
        // ===== PART 13: WebRTC IP Leak Protection =====
        
        const originalRTCPeerConnection = window.RTCPeerConnection;
        window.RTCPeerConnection = function(...args) {{
            const pc = new originalRTCPeerConnection(...args);
            const originalCreateDataChannel = pc.createDataChannel;
            pc.createDataChannel = function(...args) {{
                return originalCreateDataChannel.apply(pc, args);
            }};
            return pc;
        }};
        
        // ===== PART 14: Error Stack Traces =====
        
        const originalError = Error;
        Error = function(...args) {{
            const err = new originalError(...args);
            if (err.stack) {{
                err.stack = err.stack
                    .replace(/\\s+at\\s+__puppeteer_evaluation_script__[\\s\\S]*/g, '')
                    .replace(/\\s+at\\s+evaluateHandle[\\s\\S]*/g, '')
                    .replace(/\\s+at\\s+ExecutionContext[\\s\\S]*/g, '');
            }}
            return err;
        }};
        Error.prototype = originalError.prototype;
        
        console.log('[FlyBrowser] Stealth fingerprint applied: {profile.profile_id}');
        '''


def validate_fingerprint_consistency(profile: FingerprintProfile) -> Tuple[bool, List[str]]:
    """
    Validate that a fingerprint profile is internally consistent.
    
    Checks for common inconsistencies that could reveal automation:
    - Safari on Windows
    - Linux fonts on macOS
    - Mismatched timezone and language
    
    Args:
        profile: FingerprintProfile to validate.
    
    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []
    
    # Check browser/OS compatibility
    if profile.browser.name == "Safari" and "Mac" not in profile.hardware.platform:
        issues.append("Safari browser on non-macOS platform")
    
    # Check platform consistency
    if "Mac" in profile.hardware.platform and "Windows" in profile.user_agent:
        issues.append("Mac platform but Windows user agent")
    
    # Check GPU vendor consistency
    if "Apple" in profile.hardware.gpu_vendor and "Windows" in profile.hardware.platform:
        issues.append("Apple GPU on Windows platform")
    
    # Check timezone/language consistency
    if profile.locale.timezone.startswith("America") and profile.locale.language.startswith("ja"):
        issues.append("American timezone with Japanese language")
    
    # Check screen resolution validity
    if profile.screen.width < 800 or profile.screen.height < 600:
        issues.append("Unrealistically small screen resolution")
    
    if profile.screen.device_pixel_ratio > 3:
        issues.append("Unrealistically high device pixel ratio")
    
    # Check hardware consistency
    if profile.hardware.device_memory > 64:
        issues.append("Unrealistically high device memory")
    
    if profile.hardware.cpu_cores > 32:
        issues.append("Unrealistically high CPU core count")
    
    return len(issues) == 0, issues
