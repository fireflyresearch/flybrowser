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
FlyBrowser Stealth Package.

This package provides state-of-the-art stealth capabilities for browser automation:

1. **Fingerprint Generation** (`flybrowser.stealth.fingerprint`)
   - Dynamic browser fingerprint profiles
   - OS/Browser/Hardware combinations
   - Canvas, WebGL, Audio fingerprinting
   - Consistent profiles that pass bot detection

2. **CAPTCHA Solving** (`flybrowser.stealth.captcha`)
   - Multi-provider CAPTCHA solving (2Captcha, Anti-Captcha, CapSolver)
   - Auto-detection of reCAPTCHA, hCaptcha, Cloudflare Turnstile
   - Intelligent retry and cost management

3. **Proxy Network** (`flybrowser.stealth.proxy`)
   - Intelligent proxy selection based on target
   - Residential proxy provider integration
   - Fingerprint-proxy-geolocation consistency
   - Health monitoring and automatic rotation

Example - Full Stealth Configuration:
    >>> from flybrowser import FlyBrowser
    >>> from flybrowser.stealth import StealthConfig
    >>> 
    >>> browser = FlyBrowser(
    ...     stealth=StealthConfig(
    ...         fingerprint="auto",
    ...         geolocation="us-west",
    ...         captcha_solver={
    ...             "provider": "2captcha",
    ...             "api_key": "your-key",
    ...             "auto_solve": True,
    ...         },
    ...         proxy_network={
    ...             "providers": [
    ...                 {"provider": "bright_data", "username": "...", "password": "..."},
    ...             ],
    ...             "strategy": "smart",
    ...         },
    ...     ),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Import subpackages
from flybrowser.stealth.fingerprint import (
    FingerprintGenerator,
    FingerprintProfile,
    OperatingSystem,
    BrowserType,
)

from flybrowser.stealth.captcha import (
    CaptchaSolver,
    CaptchaConfig,
    CaptchaType,
    CaptchaSolution,
)

from flybrowser.stealth.proxy import (
    ProxyNetwork,
    ProxyNetworkConfig,
    ProviderConfig,
    ProxyEndpoint,
    ProxyType,
    ProxyProvider,
)


@dataclass
class StealthConfig:
    """
    Unified stealth configuration for FlyBrowser.
    
    This combines fingerprint generation, CAPTCHA solving, and proxy network
    configuration into a single, easy-to-use configuration object.
    
    Simple Example (enabled flags):
        >>> config = StealthConfig(
        ...     fingerprint_enabled=True,   # Enable dynamic fingerprints
        ...     captcha_enabled=True,       # Enable CAPTCHA solving
        ...     captcha_provider="2captcha",
        ...     captcha_api_key="your-key",
        ...     proxy_enabled=True,         # Enable proxy network
        ... )
    
    Advanced Example (full configuration):
        >>> config = StealthConfig(
        ...     fingerprint="auto",  # or "macos_chrome", "windows_firefox", etc.
        ...     geolocation="us-west",
        ...     captcha_solver=CaptchaConfig(
        ...         provider="2captcha",
        ...         api_key="your-key",
        ...         auto_solve=True,
        ...     ),
        ...     proxy_network=ProxyNetworkConfig(
        ...         providers=[...],
        ...         strategy="smart",
        ...     ),
        ... )
    """
    
    # ===== Simple Enabled Flags (for SDK wiring) =====
    # These are the primary way to enable features from the SDK
    
    # Enable fingerprint generation
    fingerprint_enabled: bool = False
    
    # Enable CAPTCHA solving
    captcha_enabled: bool = False
    
    # CAPTCHA provider (2captcha, anticaptcha, capsolver)
    captcha_provider: Optional[str] = None
    
    # CAPTCHA API key
    captcha_api_key: Optional[str] = None
    
    # Auto-solve CAPTCHAs when detected during agent execution
    captcha_auto_solve: bool = True
    
    # Enable proxy network
    proxy_enabled: bool = False
    
    # ===== Advanced Configuration =====
    
    # Fingerprint configuration
    # "auto" = auto-generate, or specific profile like "macos_chrome"
    fingerprint: Union[str, FingerprintProfile] = "auto"
    
    # Operating system override (windows, macos, linux)
    os: Optional[str] = None
    
    # Browser override (chrome, firefox, safari, edge)
    browser: Optional[str] = None
    
    # Geolocation for timezone/language consistency
    # Options: us-west, us-east, uk, germany, france, japan, etc.
    geolocation: str = "us-west"
    
    # CAPTCHA solver configuration (advanced)
    # Can be CaptchaConfig or dict with config values
    captcha_solver: Optional[Union[CaptchaConfig, Dict[str, Any]]] = None
    
    # Proxy network configuration (advanced)
    # Can be ProxyNetworkConfig or dict with config values
    proxy_network: Optional[Union[ProxyNetworkConfig, Dict[str, Any]]] = None
    
    # Enable stealth mode (anti-detection scripts)
    stealth_mode: bool = True
    
    # Human-like behavior simulation
    simulate_human: bool = True
    
    # Typing delay range (ms) for human-like typing
    typing_delay_min: int = 50
    typing_delay_max: int = 150
    
    # Mouse movement simulation
    simulate_mouse_movement: bool = True
    
    # Random delays between actions (ms)
    action_delay_min: int = 100
    action_delay_max: int = 500
    
    def __post_init__(self):
        """Auto-enable features based on configuration."""
        # Auto-enable fingerprint if a specific profile is provided
        if isinstance(self.fingerprint, FingerprintProfile):
            self.fingerprint_enabled = True
        
        # Auto-enable captcha if solver config is provided
        if self.captcha_solver is not None:
            self.captcha_enabled = True
        
        # Auto-enable proxy if network config is provided
        if self.proxy_network is not None:
            self.proxy_enabled = True
    
    def get_fingerprint_profile(self) -> FingerprintProfile:
        """Get or generate fingerprint profile."""
        if isinstance(self.fingerprint, FingerprintProfile):
            return self.fingerprint
        
        generator = FingerprintGenerator()
        
        # Parse fingerprint string
        if self.fingerprint == "auto":
            return generator.generate(
                os=self.os,
                browser=self.browser,
                geolocation=self.geolocation,
            )
        
        # Parse specific profile strings like "macos_chrome"
        parts = self.fingerprint.lower().split("_")
        os_name = parts[0] if parts else None
        browser_name = parts[1] if len(parts) > 1 else None
        
        return generator.generate(
            os=os_name or self.os,
            browser=browser_name or self.browser,
            geolocation=self.geolocation,
        )
    
    def get_captcha_config(self) -> Optional[CaptchaConfig]:
        """Get CAPTCHA solver configuration."""
        if self.captcha_solver is None:
            return None
        
        if isinstance(self.captcha_solver, CaptchaConfig):
            return self.captcha_solver
        
        # Convert dict to CaptchaConfig
        return CaptchaConfig(**self.captcha_solver)
    
    def get_proxy_config(self) -> Optional[ProxyNetworkConfig]:
        """Get proxy network configuration."""
        if self.proxy_network is None:
            return None
        
        if isinstance(self.proxy_network, ProxyNetworkConfig):
            return self.proxy_network
        
        # Convert dict to ProxyNetworkConfig
        # Handle nested provider configs
        config_dict = dict(self.proxy_network)
        if "providers" in config_dict:
            config_dict["providers"] = [
                ProviderConfig(**p) if isinstance(p, dict) else p
                for p in config_dict["providers"]
            ]
        
        return ProxyNetworkConfig(**config_dict)


# Convenience function to create stealth config from dict
def create_stealth_config(config: Optional[Dict[str, Any]] = None) -> StealthConfig:
    """
    Create StealthConfig from dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StealthConfig instance
    """
    if config is None:
        return StealthConfig()
    return StealthConfig(**config)


__all__ = [
    # Main config
    "StealthConfig",
    "create_stealth_config",
    
    # Fingerprint
    "FingerprintGenerator",
    "FingerprintProfile",
    "OperatingSystem",
    "BrowserType",
    
    # CAPTCHA
    "CaptchaSolver",
    "CaptchaConfig",
    "CaptchaType",
    "CaptchaSolution",
    
    # Proxy
    "ProxyNetwork",
    "ProxyNetworkConfig",
    "ProviderConfig",
    "ProxyEndpoint",
    "ProxyType",
    "ProxyProvider",
]
