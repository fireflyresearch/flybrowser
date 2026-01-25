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
Performance configuration for FlyBrowser.

This module provides centralized performance settings to optimize speed and
responsiveness across all components. Settings can be adjusted globally or
per-operation.

Example:
    >>> from flybrowser.core.performance import PerformanceConfig, SpeedPreset
    >>> 
    >>> # Use a speed preset
    >>> config = PerformanceConfig.from_preset(SpeedPreset.FAST)
    >>> 
    >>> # Or customize
    >>> config = PerformanceConfig(
    ...     navigation_timeout_ms=10000,
    ...     action_timeout_ms=5000,
    ... )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SpeedPreset(str, Enum):
    """Speed presets for common use cases."""
    
    FAST = "fast"           # Optimized for speed, may miss slow-loading content
    BALANCED = "balanced"   # Default, good balance of speed and reliability
    THOROUGH = "thorough"   # Slower but more thorough, for complex pages


class WaitStrategy(str, Enum):
    """Wait strategies for navigation."""
    
    COMMIT = "commit"                    # Fastest - just wait for navigation commit
    DOM_CONTENT_LOADED = "domcontentloaded"  # Fast - DOM ready
    LOAD = "load"                        # Standard - all resources loaded
    NETWORK_IDLE = "networkidle"         # Slowest - no network activity


@dataclass
class PerformanceConfig:
    """
    Centralized performance configuration.
    
    All timeouts are in milliseconds unless otherwise noted.
    Delays are in seconds.
    
    Attributes:
        navigation_timeout_ms: Timeout for page navigation
        action_timeout_ms: Timeout for individual actions (click, type, etc.)
        element_timeout_ms: Timeout for element detection
        step_timeout_ms: Timeout for orchestrator steps
        
        retry_delay_seconds: Base delay between retries
        retry_max_delay_seconds: Maximum delay between retries
        max_retries: Maximum number of retries
        
        wait_strategy: Default wait strategy for navigation
        wait_after_action_ms: Wait time after actions (for page to stabilize)
        wait_after_navigation_ms: Wait time after navigation
        
        cache_ttl_ms: Cache TTL for page analysis
        enable_caching: Whether to enable caching
        
        parallel_execution: Whether to enable parallel step execution
        batch_size: Maximum batch size for parallel operations
    """
    
    # Timeouts (milliseconds)
    navigation_timeout_ms: int = 15000      # 15s (was 30s)
    action_timeout_ms: int = 10000          # 10s (was 30s)
    element_timeout_ms: int = 8000          # 8s (was 30s)
    step_timeout_ms: int = 15000            # 15s (was 30s)
    extraction_timeout_ms: int = 20000      # 20s for LLM extraction
    
    # Retry configuration (seconds)
    retry_delay_seconds: float = 0.3        # 300ms (was 1.0s)
    retry_max_delay_seconds: float = 5.0    # 5s (was 10s)
    max_retries: int = 2                    # 2 retries (was 3)
    
    # Wait strategy
    wait_strategy: WaitStrategy = WaitStrategy.DOM_CONTENT_LOADED
    wait_after_action_ms: int = 100         # 100ms (was implicit/varying)
    wait_after_navigation_ms: int = 200     # 200ms settle time
    
    # Caching
    cache_ttl_ms: float = 5000.0            # 5s cache
    enable_caching: bool = True
    
    # Parallel execution
    parallel_execution: bool = True
    batch_size: int = 5
    
    # LLM configuration
    llm_timeout_seconds: float = 30.0       # LLM API timeout
    llm_max_retries: int = 2
    
    @classmethod
    def from_preset(cls, preset: SpeedPreset) -> "PerformanceConfig":
        """
        Create configuration from a speed preset.
        
        Args:
            preset: Speed preset to use
            
        Returns:
            PerformanceConfig with preset values
        """
        if preset == SpeedPreset.FAST:
            return cls(
                navigation_timeout_ms=10000,
                action_timeout_ms=5000,
                element_timeout_ms=5000,
                step_timeout_ms=10000,
                extraction_timeout_ms=15000,
                retry_delay_seconds=0.2,
                retry_max_delay_seconds=3.0,
                max_retries=1,
                wait_strategy=WaitStrategy.DOM_CONTENT_LOADED,
                wait_after_action_ms=50,
                wait_after_navigation_ms=100,
                cache_ttl_ms=3000.0,
                llm_timeout_seconds=20.0,
                llm_max_retries=1,
            )
        elif preset == SpeedPreset.THOROUGH:
            return cls(
                navigation_timeout_ms=30000,
                action_timeout_ms=20000,
                element_timeout_ms=15000,
                step_timeout_ms=30000,
                extraction_timeout_ms=45000,
                retry_delay_seconds=1.0,
                retry_max_delay_seconds=15.0,
                max_retries=3,
                wait_strategy=WaitStrategy.NETWORK_IDLE,
                wait_after_action_ms=300,
                wait_after_navigation_ms=500,
                cache_ttl_ms=10000.0,
                llm_timeout_seconds=60.0,
                llm_max_retries=3,
            )
        else:  # BALANCED (default)
            return cls()
    
    @classmethod
    def from_env(cls) -> "PerformanceConfig":
        """
        Create configuration from environment variables.
        
        Environment variables (prefix FLYBROWSER_PERF_):
            - FLYBROWSER_PERF_PRESET: Speed preset (fast, balanced, thorough)
            - FLYBROWSER_PERF_NAV_TIMEOUT_MS: Navigation timeout
            - FLYBROWSER_PERF_ACTION_TIMEOUT_MS: Action timeout
            - etc.
            
        Returns:
            PerformanceConfig from environment
        """
        # Check for preset first
        preset_name = os.environ.get("FLYBROWSER_PERF_PRESET", "").lower()
        if preset_name in ("fast", "balanced", "thorough"):
            preset = SpeedPreset(preset_name)
            config = cls.from_preset(preset)
        else:
            config = cls()
        
        # Override with specific env vars
        env_mapping = {
            "FLYBROWSER_PERF_NAV_TIMEOUT_MS": ("navigation_timeout_ms", int),
            "FLYBROWSER_PERF_ACTION_TIMEOUT_MS": ("action_timeout_ms", int),
            "FLYBROWSER_PERF_ELEMENT_TIMEOUT_MS": ("element_timeout_ms", int),
            "FLYBROWSER_PERF_STEP_TIMEOUT_MS": ("step_timeout_ms", int),
            "FLYBROWSER_PERF_RETRY_DELAY": ("retry_delay_seconds", float),
            "FLYBROWSER_PERF_MAX_RETRIES": ("max_retries", int),
            "FLYBROWSER_PERF_CACHE_TTL_MS": ("cache_ttl_ms", float),
            "FLYBROWSER_PERF_LLM_TIMEOUT": ("llm_timeout_seconds", float),
        }
        
        for env_var, (attr, type_fn) in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                try:
                    setattr(config, attr, type_fn(value))
                except ValueError:
                    pass  # Ignore invalid values
        
        return config


# Global default configuration
_default_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """
    Get the global performance configuration.
    
    Returns:
        The current PerformanceConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = PerformanceConfig.from_env()
    return _default_config


def set_performance_config(config: PerformanceConfig) -> None:
    """
    Set the global performance configuration.
    
    Args:
        config: New configuration to use
    """
    global _default_config
    _default_config = config


def use_fast_mode() -> None:
    """Switch to fast performance mode globally."""
    set_performance_config(PerformanceConfig.from_preset(SpeedPreset.FAST))


def use_balanced_mode() -> None:
    """Switch to balanced performance mode globally."""
    set_performance_config(PerformanceConfig.from_preset(SpeedPreset.BALANCED))


def use_thorough_mode() -> None:
    """Switch to thorough performance mode globally."""
    set_performance_config(PerformanceConfig.from_preset(SpeedPreset.THOROUGH))
