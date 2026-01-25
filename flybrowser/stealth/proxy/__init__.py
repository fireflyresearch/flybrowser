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
Intelligent Proxy Network for FlyBrowser.

This package provides intelligent proxy selection and management:
- Target-aware proxy selection
- Residential proxy provider integration
- Fingerprint-proxy consistency
- Health monitoring and rotation

Example:
    >>> from flybrowser.stealth.proxy import ProxyNetwork, ProxyNetworkConfig
    >>> 
    >>> network = ProxyNetwork(ProxyNetworkConfig(
    ...     providers=[ProviderConfig(provider=ProxyProvider.BRIGHT_DATA, ...)],
    ...     geolocation="us-west",
    ... ))
    >>> proxy = await network.get_proxy("https://example.com")
"""

from flybrowser.stealth.proxy.network import (
    # Main classes
    ProxyNetwork,
    ProxyIntelligence,
    
    # Configuration
    ProxyNetworkConfig,
    ProviderConfig,
    GeoLocation,
    
    # Data classes
    ProxyEndpoint,
    
    # Enums
    ProxyType,
    ProxyProvider,
    SelectionStrategy,
    HealthStatus,
    
    # Provider base class (for custom implementations)
    BaseProxyProvider,
)

__all__ = [
    # Main classes
    "ProxyNetwork",
    "ProxyIntelligence",
    
    # Configuration
    "ProxyNetworkConfig",
    "ProviderConfig",
    "GeoLocation",
    
    # Data classes
    "ProxyEndpoint",
    
    # Enums
    "ProxyType",
    "ProxyProvider",
    "SelectionStrategy",
    "HealthStatus",
    
    # Provider base class
    "BaseProxyProvider",
]
