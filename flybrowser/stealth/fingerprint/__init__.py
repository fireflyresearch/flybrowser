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
Browser Fingerprint Management for FlyBrowser.

This package provides advanced browser fingerprint generation and management:
- Dynamic fingerprint profile generation
- OS/Browser/Hardware profile combinations
- Canvas, WebGL, Audio fingerprint customization
- Fingerprint consistency validation

Fingerprinting is about **browser identity** - creating realistic browser
profiles that match real-world browser populations.

Example:
    >>> from flybrowser.stealth.fingerprint import FingerprintGenerator
    >>> 
    >>> generator = FingerprintGenerator()
    >>> profile = generator.generate(os="macos", browser="chrome")
    >>> print(profile.user_agent)
"""

from flybrowser.stealth.fingerprint.generator import (
    # Main classes
    FingerprintGenerator,
    FingerprintProfile,
    
    # Profile components
    HardwareProfile,
    ScreenProfile,
    BrowserProfile,
    LocaleProfile,
    NetworkProfile,
    WebGLProfile,
    AudioProfile,
    CanvasProfile,
    FontProfile,
    
    # Enums
    OperatingSystem,
    BrowserType,
    DeviceType,
    
    # Validation
    validate_fingerprint_consistency,
)

__all__ = [
    # Main classes
    "FingerprintGenerator",
    "FingerprintProfile",
    
    # Profile components
    "HardwareProfile",
    "ScreenProfile", 
    "BrowserProfile",
    "LocaleProfile",
    "NetworkProfile",
    "WebGLProfile",
    "AudioProfile",
    "CanvasProfile",
    "FontProfile",
    
    # Enums
    "OperatingSystem",
    "BrowserType",
    "DeviceType",
    
    # Validation
    "validate_fingerprint_consistency",
]
