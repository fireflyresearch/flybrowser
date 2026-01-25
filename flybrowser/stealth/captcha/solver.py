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
Managed CAPTCHA Solving Service for FlyBrowser.

This module provides a pluggable CAPTCHA solving system that integrates with
multiple solving providers to automatically handle CAPTCHAs during automation.

Supported CAPTCHA Types:
- reCAPTCHA v2 (checkbox and invisible)
- reCAPTCHA v3 (score-based)
- hCaptcha
- Cloudflare Turnstile
- FunCaptcha (Arkose Labs)
- Image CAPTCHAs (text recognition)
- GeeTest
- AWS WAF CAPTCHA

Supported Providers:
- 2Captcha
- Anti-Captcha
- CapSolver
- CapMonster Cloud
- DeathByCaptcha

Example:
    >>> from flybrowser.stealth.captcha import CaptchaSolver, CaptchaConfig
    >>> 
    >>> solver = CaptchaSolver(CaptchaConfig(
    ...     provider="2captcha",
    ...     api_key="your-api-key",
    ...     auto_solve=True,
    ... ))
    >>> 
    >>> # Auto-detect and solve
    >>> result = await solver.solve(page)
    >>> print(f"Solved in {result.solve_time_ms}ms, cost: ${result.cost}")
"""

from __future__ import annotations

import asyncio
import base64
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type
import aiohttp

from flybrowser.utils.logger import logger


class CaptchaType(str, Enum):
    """Types of CAPTCHAs that can be detected and solved."""
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V2_INVISIBLE = "recaptcha_v2_invisible"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    CLOUDFLARE_TURNSTILE = "cloudflare_turnstile"
    FUNCAPTCHA = "funcaptcha"
    IMAGE_CAPTCHA = "image_captcha"
    GEETEST = "geetest"
    GEETEST_V4 = "geetest_v4"
    AWS_WAF = "aws_waf"
    TEXT_CAPTCHA = "text_captcha"
    UNKNOWN = "unknown"


class CaptchaProvider(str, Enum):
    """Supported CAPTCHA solving providers."""
    TWO_CAPTCHA = "2captcha"
    ANTI_CAPTCHA = "anti_captcha"
    CAPSOLVER = "capsolver"
    CAPMONSTER = "capmonster"
    DEATHBYCAPTCHA = "deathbycaptcha"


class SolveStatus(str, Enum):
    """Status of a CAPTCHA solve attempt."""
    PENDING = "pending"
    PROCESSING = "processing"
    SOLVED = "solved"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_CAPTCHA = "invalid_captcha"
    UNSUPPORTED = "unsupported"


@dataclass
class CaptchaConfig:
    """Configuration for the CAPTCHA solving service."""
    
    # Primary provider
    provider: str = "2captcha"
    api_key: str = ""
    
    # Fallback providers (in order of preference)
    fallback_providers: List[Dict[str, str]] = field(default_factory=list)
    
    # Behavior settings
    auto_solve: bool = True  # Automatically solve when detected
    auto_detect: bool = True  # Automatically detect CAPTCHAs on pages
    
    # Timeouts
    solve_timeout_seconds: float = 120.0  # Max time to wait for solution
    poll_interval_seconds: float = 2.0  # Interval between status checks
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    
    # Cost management
    max_cost_per_solve: float = 0.01  # Max cost in USD per solve
    daily_budget: Optional[float] = None  # Daily spending limit
    
    # reCAPTCHA v3 specific
    min_score: float = 0.7  # Minimum acceptable score for v3
    
    # Human simulation for better scores
    simulate_human_behavior: bool = True
    
    # Logging
    log_solutions: bool = False  # Log solution tokens (security risk)


@dataclass
class DetectedCaptcha:
    """Information about a detected CAPTCHA on a page."""
    
    captcha_type: CaptchaType
    site_key: Optional[str] = None
    page_url: str = ""
    
    # Element information
    iframe_selector: Optional[str] = None
    container_selector: Optional[str] = None
    submit_selector: Optional[str] = None
    
    # Additional data
    data_s: Optional[str] = None  # For reCAPTCHA
    action: Optional[str] = None  # For reCAPTCHA v3
    enterprise: bool = False
    
    # Image CAPTCHA specific
    image_base64: Optional[str] = None
    image_instructions: Optional[str] = None
    
    # GeeTest specific
    gt: Optional[str] = None
    challenge: Optional[str] = None
    
    # Detection confidence
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "type": self.captcha_type.value,
            "site_key": self.site_key,
            "page_url": self.page_url,
            "enterprise": self.enterprise,
            "action": self.action,
            "data_s": self.data_s,
        }


@dataclass
class CaptchaSolution:
    """Result of a CAPTCHA solve attempt."""
    
    status: SolveStatus
    captcha_type: CaptchaType
    
    # Solution token (for token-based CAPTCHAs)
    token: Optional[str] = None
    
    # For GeeTest
    challenge: Optional[str] = None
    validate: Optional[str] = None
    seccode: Optional[str] = None
    
    # For image CAPTCHAs
    text: Optional[str] = None
    
    # Metadata
    provider_used: Optional[str] = None
    solve_time_ms: float = 0.0
    cost: float = 0.0
    task_id: Optional[str] = None
    
    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # reCAPTCHA v3 score (if applicable)
    score: Optional[float] = None
    
    def is_success(self) -> bool:
        """Check if the solve was successful."""
        return self.status == SolveStatus.SOLVED


# ============================================================================
# Provider Implementations
# ============================================================================

class BaseCaptchaProvider(ABC):
    """Abstract base class for CAPTCHA solving providers."""
    
    def __init__(self, api_key: str, config: CaptchaConfig):
        self.api_key = api_key
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @abstractmethod
    async def create_task(self, captcha: DetectedCaptcha) -> str:
        """Create a solve task and return task ID."""
        pass
    
    @abstractmethod
    async def get_result(self, task_id: str) -> CaptchaSolution:
        """Get the result of a solve task."""
        pass
    
    @abstractmethod
    def get_cost(self, captcha_type: CaptchaType) -> float:
        """Get the cost for solving a specific CAPTCHA type."""
        pass
    
    @abstractmethod
    def supports_captcha(self, captcha_type: CaptchaType) -> bool:
        """Check if this provider supports the given CAPTCHA type."""
        pass
    
    async def solve(self, captcha: DetectedCaptcha) -> CaptchaSolution:
        """Solve a CAPTCHA (convenience method)."""
        start_time = time.time()
        
        try:
            # Create task
            task_id = await self.create_task(captcha)
            
            # Poll for result
            elapsed = 0.0
            while elapsed < self.config.solve_timeout_seconds:
                await asyncio.sleep(self.config.poll_interval_seconds)
                elapsed = time.time() - start_time
                
                result = await self.get_result(task_id)
                
                if result.status == SolveStatus.SOLVED:
                    result.solve_time_ms = (time.time() - start_time) * 1000
                    result.cost = self.get_cost(captcha.captcha_type)
                    return result
                
                if result.status in (SolveStatus.FAILED, SolveStatus.INVALID_CAPTCHA):
                    return result
            
            # Timeout
            return CaptchaSolution(
                status=SolveStatus.TIMEOUT,
                captcha_type=captcha.captcha_type,
                error_message=f"Solve timeout after {self.config.solve_timeout_seconds}s",
            )
            
        except Exception as e:
            return CaptchaSolution(
                status=SolveStatus.FAILED,
                captcha_type=captcha.captcha_type,
                error_message=str(e),
            )


class TwoCaptchaProvider(BaseCaptchaProvider):
    """2Captcha provider implementation."""
    
    BASE_URL = "https://2captcha.com"
    
    # Costs per 1000 solves (approximate)
    COSTS = {
        CaptchaType.RECAPTCHA_V2: 0.00299,
        CaptchaType.RECAPTCHA_V2_INVISIBLE: 0.00299,
        CaptchaType.RECAPTCHA_V3: 0.00299,
        CaptchaType.HCAPTCHA: 0.00299,
        CaptchaType.CLOUDFLARE_TURNSTILE: 0.00299,
        CaptchaType.FUNCAPTCHA: 0.00299,
        CaptchaType.IMAGE_CAPTCHA: 0.001,
        CaptchaType.GEETEST: 0.00299,
        CaptchaType.TEXT_CAPTCHA: 0.001,
    }
    
    async def create_task(self, captcha: DetectedCaptcha) -> str:
        """Create a solve task on 2Captcha."""
        session = await self._get_session()
        
        params = {
            "key": self.api_key,
            "json": 1,
        }
        
        if captcha.captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V2_INVISIBLE):
            params.update({
                "method": "userrecaptcha",
                "googlekey": captcha.site_key,
                "pageurl": captcha.page_url,
                "invisible": 1 if captcha.captcha_type == CaptchaType.RECAPTCHA_V2_INVISIBLE else 0,
            })
            if captcha.data_s:
                params["data-s"] = captcha.data_s
            if captcha.enterprise:
                params["enterprise"] = 1
                
        elif captcha.captcha_type == CaptchaType.RECAPTCHA_V3:
            params.update({
                "method": "userrecaptcha",
                "version": "v3",
                "googlekey": captcha.site_key,
                "pageurl": captcha.page_url,
                "action": captcha.action or "verify",
                "min_score": self.config.min_score,
            })
            if captcha.enterprise:
                params["enterprise"] = 1
                
        elif captcha.captcha_type == CaptchaType.HCAPTCHA:
            params.update({
                "method": "hcaptcha",
                "sitekey": captcha.site_key,
                "pageurl": captcha.page_url,
            })
            
        elif captcha.captcha_type == CaptchaType.CLOUDFLARE_TURNSTILE:
            params.update({
                "method": "turnstile",
                "sitekey": captcha.site_key,
                "pageurl": captcha.page_url,
            })
            
        elif captcha.captcha_type == CaptchaType.FUNCAPTCHA:
            params.update({
                "method": "funcaptcha",
                "publickey": captcha.site_key,
                "pageurl": captcha.page_url,
            })
            
        elif captcha.captcha_type == CaptchaType.IMAGE_CAPTCHA:
            params.update({
                "method": "base64",
                "body": captcha.image_base64,
            })
            if captcha.image_instructions:
                params["textinstructions"] = captcha.image_instructions
                
        elif captcha.captcha_type in (CaptchaType.GEETEST, CaptchaType.GEETEST_V4):
            params.update({
                "method": "geetest" if captcha.captcha_type == CaptchaType.GEETEST else "geetest_v4",
                "gt": captcha.gt,
                "challenge": captcha.challenge,
                "pageurl": captcha.page_url,
            })
        else:
            raise ValueError(f"Unsupported CAPTCHA type: {captcha.captcha_type}")
        
        async with session.post(f"{self.BASE_URL}/in.php", data=params) as resp:
            data = await resp.json()
            
            if data.get("status") != 1:
                raise Exception(f"2Captcha error: {data.get('error_text', 'Unknown error')}")
            
            return data["request"]
    
    async def get_result(self, task_id: str) -> CaptchaSolution:
        """Get result from 2Captcha."""
        session = await self._get_session()
        
        params = {
            "key": self.api_key,
            "action": "get",
            "id": task_id,
            "json": 1,
        }
        
        async with session.get(f"{self.BASE_URL}/res.php", params=params) as resp:
            data = await resp.json()
            
            if data.get("status") == 1:
                return CaptchaSolution(
                    status=SolveStatus.SOLVED,
                    captcha_type=CaptchaType.UNKNOWN,  # Will be set by caller
                    token=data.get("request"),
                    task_id=task_id,
                    provider_used="2captcha",
                )
            
            error = data.get("request", "")
            
            if error == "CAPCHA_NOT_READY":
                return CaptchaSolution(
                    status=SolveStatus.PROCESSING,
                    captcha_type=CaptchaType.UNKNOWN,
                    task_id=task_id,
                )
            
            return CaptchaSolution(
                status=SolveStatus.FAILED,
                captcha_type=CaptchaType.UNKNOWN,
                task_id=task_id,
                error_code=error,
                error_message=data.get("error_text"),
            )
    
    def get_cost(self, captcha_type: CaptchaType) -> float:
        """Get cost for a CAPTCHA type."""
        return self.COSTS.get(captcha_type, 0.003)
    
    def supports_captcha(self, captcha_type: CaptchaType) -> bool:
        """Check if 2Captcha supports this type."""
        return captcha_type in self.COSTS


class AntiCaptchaProvider(BaseCaptchaProvider):
    """Anti-Captcha provider implementation."""
    
    BASE_URL = "https://api.anti-captcha.com"
    
    COSTS = {
        CaptchaType.RECAPTCHA_V2: 0.002,
        CaptchaType.RECAPTCHA_V2_INVISIBLE: 0.002,
        CaptchaType.RECAPTCHA_V3: 0.002,
        CaptchaType.HCAPTCHA: 0.002,
        CaptchaType.FUNCAPTCHA: 0.002,
        CaptchaType.IMAGE_CAPTCHA: 0.0007,
        CaptchaType.GEETEST: 0.002,
        CaptchaType.CLOUDFLARE_TURNSTILE: 0.002,
    }
    
    async def create_task(self, captcha: DetectedCaptcha) -> str:
        """Create a solve task on Anti-Captcha."""
        session = await self._get_session()
        
        task = {}
        
        if captcha.captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V2_INVISIBLE):
            task = {
                "type": "RecaptchaV2TaskProxyless" if not captcha.enterprise else "RecaptchaV2EnterpriseTaskProxyless",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
                "isInvisible": captcha.captcha_type == CaptchaType.RECAPTCHA_V2_INVISIBLE,
            }
            if captcha.data_s:
                task["recaptchaDataSValue"] = captcha.data_s
                
        elif captcha.captcha_type == CaptchaType.RECAPTCHA_V3:
            task = {
                "type": "RecaptchaV3TaskProxyless",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
                "minScore": self.config.min_score,
                "pageAction": captcha.action or "verify",
            }
            
        elif captcha.captcha_type == CaptchaType.HCAPTCHA:
            task = {
                "type": "HCaptchaTaskProxyless",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
            }
            
        elif captcha.captcha_type == CaptchaType.CLOUDFLARE_TURNSTILE:
            task = {
                "type": "TurnstileTaskProxyless",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
            }
            
        elif captcha.captcha_type == CaptchaType.FUNCAPTCHA:
            task = {
                "type": "FunCaptchaTaskProxyless",
                "websiteURL": captcha.page_url,
                "websitePublicKey": captcha.site_key,
            }
            
        elif captcha.captcha_type == CaptchaType.IMAGE_CAPTCHA:
            task = {
                "type": "ImageToTextTask",
                "body": captcha.image_base64,
            }
            if captcha.image_instructions:
                task["comment"] = captcha.image_instructions
                
        elif captcha.captcha_type == CaptchaType.GEETEST:
            task = {
                "type": "GeeTestTaskProxyless",
                "websiteURL": captcha.page_url,
                "gt": captcha.gt,
                "challenge": captcha.challenge,
            }
        else:
            raise ValueError(f"Unsupported CAPTCHA type: {captcha.captcha_type}")
        
        payload = {
            "clientKey": self.api_key,
            "task": task,
        }
        
        async with session.post(f"{self.BASE_URL}/createTask", json=payload) as resp:
            data = await resp.json()
            
            if data.get("errorId", 0) != 0:
                raise Exception(f"Anti-Captcha error: {data.get('errorDescription', 'Unknown error')}")
            
            return str(data["taskId"])
    
    async def get_result(self, task_id: str) -> CaptchaSolution:
        """Get result from Anti-Captcha."""
        session = await self._get_session()
        
        payload = {
            "clientKey": self.api_key,
            "taskId": int(task_id),
        }
        
        async with session.post(f"{self.BASE_URL}/getTaskResult", json=payload) as resp:
            data = await resp.json()
            
            if data.get("errorId", 0) != 0:
                return CaptchaSolution(
                    status=SolveStatus.FAILED,
                    captcha_type=CaptchaType.UNKNOWN,
                    task_id=task_id,
                    error_code=str(data.get("errorCode")),
                    error_message=data.get("errorDescription"),
                )
            
            status = data.get("status")
            
            if status == "ready":
                solution = data.get("solution", {})
                return CaptchaSolution(
                    status=SolveStatus.SOLVED,
                    captcha_type=CaptchaType.UNKNOWN,
                    token=solution.get("gRecaptchaResponse") or solution.get("token") or solution.get("text"),
                    text=solution.get("text"),
                    challenge=solution.get("challenge"),
                    validate=solution.get("validate"),
                    seccode=solution.get("seccode"),
                    task_id=task_id,
                    provider_used="anti_captcha",
                )
            
            return CaptchaSolution(
                status=SolveStatus.PROCESSING,
                captcha_type=CaptchaType.UNKNOWN,
                task_id=task_id,
            )
    
    def get_cost(self, captcha_type: CaptchaType) -> float:
        """Get cost for a CAPTCHA type."""
        return self.COSTS.get(captcha_type, 0.002)
    
    def supports_captcha(self, captcha_type: CaptchaType) -> bool:
        """Check if Anti-Captcha supports this type."""
        return captcha_type in self.COSTS


class CapSolverProvider(BaseCaptchaProvider):
    """CapSolver provider implementation."""
    
    BASE_URL = "https://api.capsolver.com"
    
    COSTS = {
        CaptchaType.RECAPTCHA_V2: 0.001,
        CaptchaType.RECAPTCHA_V2_INVISIBLE: 0.001,
        CaptchaType.RECAPTCHA_V3: 0.001,
        CaptchaType.HCAPTCHA: 0.0008,
        CaptchaType.CLOUDFLARE_TURNSTILE: 0.001,
        CaptchaType.FUNCAPTCHA: 0.001,
        CaptchaType.IMAGE_CAPTCHA: 0.0002,
        CaptchaType.GEETEST: 0.001,
        CaptchaType.GEETEST_V4: 0.001,
        CaptchaType.AWS_WAF: 0.002,
    }
    
    async def create_task(self, captcha: DetectedCaptcha) -> str:
        """Create a solve task on CapSolver."""
        session = await self._get_session()
        
        task = {}
        
        if captcha.captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V2_INVISIBLE):
            task = {
                "type": "ReCaptchaV2TaskProxyLess",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
                "isInvisible": captcha.captcha_type == CaptchaType.RECAPTCHA_V2_INVISIBLE,
            }
            if captcha.enterprise:
                task["type"] = "ReCaptchaV2EnterpriseTaskProxyLess"
                
        elif captcha.captcha_type == CaptchaType.RECAPTCHA_V3:
            task = {
                "type": "ReCaptchaV3TaskProxyLess",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
                "pageAction": captcha.action or "verify",
                "minScore": self.config.min_score,
            }
            if captcha.enterprise:
                task["type"] = "ReCaptchaV3EnterpriseTaskProxyLess"
                
        elif captcha.captcha_type == CaptchaType.HCAPTCHA:
            task = {
                "type": "HCaptchaTaskProxyLess",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
            }
            
        elif captcha.captcha_type == CaptchaType.CLOUDFLARE_TURNSTILE:
            task = {
                "type": "AntiTurnstileTaskProxyLess",
                "websiteURL": captcha.page_url,
                "websiteKey": captcha.site_key,
            }
            
        elif captcha.captcha_type == CaptchaType.FUNCAPTCHA:
            task = {
                "type": "FunCaptchaTaskProxyLess",
                "websiteURL": captcha.page_url,
                "websitePublicKey": captcha.site_key,
            }
            
        elif captcha.captcha_type == CaptchaType.IMAGE_CAPTCHA:
            task = {
                "type": "ImageToTextTask",
                "body": captcha.image_base64,
            }
            
        elif captcha.captcha_type == CaptchaType.GEETEST_V4:
            task = {
                "type": "GeeTestTaskProxyLess",
                "websiteURL": captcha.page_url,
                "captchaId": captcha.gt,
            }
            
        elif captcha.captcha_type == CaptchaType.AWS_WAF:
            task = {
                "type": "AntiAwsWafTaskProxyLess",
                "websiteURL": captcha.page_url,
            }
        else:
            raise ValueError(f"Unsupported CAPTCHA type: {captcha.captcha_type}")
        
        payload = {
            "clientKey": self.api_key,
            "task": task,
        }
        
        async with session.post(f"{self.BASE_URL}/createTask", json=payload) as resp:
            data = await resp.json()
            
            if data.get("errorId", 0) != 0:
                raise Exception(f"CapSolver error: {data.get('errorDescription', 'Unknown error')}")
            
            return data["taskId"]
    
    async def get_result(self, task_id: str) -> CaptchaSolution:
        """Get result from CapSolver."""
        session = await self._get_session()
        
        payload = {
            "clientKey": self.api_key,
            "taskId": task_id,
        }
        
        async with session.post(f"{self.BASE_URL}/getTaskResult", json=payload) as resp:
            data = await resp.json()
            
            if data.get("errorId", 0) != 0:
                return CaptchaSolution(
                    status=SolveStatus.FAILED,
                    captcha_type=CaptchaType.UNKNOWN,
                    task_id=task_id,
                    error_code=str(data.get("errorCode")),
                    error_message=data.get("errorDescription"),
                )
            
            status = data.get("status")
            
            if status == "ready":
                solution = data.get("solution", {})
                return CaptchaSolution(
                    status=SolveStatus.SOLVED,
                    captcha_type=CaptchaType.UNKNOWN,
                    token=solution.get("gRecaptchaResponse") or solution.get("token") or solution.get("text"),
                    text=solution.get("text"),
                    task_id=task_id,
                    provider_used="capsolver",
                )
            
            return CaptchaSolution(
                status=SolveStatus.PROCESSING,
                captcha_type=CaptchaType.UNKNOWN,
                task_id=task_id,
            )
    
    def get_cost(self, captcha_type: CaptchaType) -> float:
        """Get cost for a CAPTCHA type."""
        return self.COSTS.get(captcha_type, 0.001)
    
    def supports_captcha(self, captcha_type: CaptchaType) -> bool:
        """Check if CapSolver supports this type."""
        return captcha_type in self.COSTS


# ============================================================================
# CAPTCHA Detector
# ============================================================================

class CaptchaDetector:
    """
    Detects CAPTCHAs on web pages.
    
    Uses multiple detection strategies:
    - DOM analysis (looking for known CAPTCHA elements)
    - Script analysis (detecting loaded CAPTCHA libraries)
    - Visual analysis (for Cloudflare challenge pages)
    """
    
    # Detection patterns for various CAPTCHA types
    DETECTION_PATTERNS = {
        CaptchaType.RECAPTCHA_V2: {
            "scripts": ["google.com/recaptcha", "gstatic.com/recaptcha"],
            "elements": [
                "iframe[src*='recaptcha']",
                ".g-recaptcha",
                "[data-sitekey]",
                "#g-recaptcha-response",
            ],
            "data_attributes": ["data-sitekey"],
        },
        CaptchaType.RECAPTCHA_V3: {
            "scripts": ["google.com/recaptcha/api.js?render="],
            "elements": ["[data-sitekey]"],
            "data_attributes": ["data-sitekey"],
        },
        CaptchaType.HCAPTCHA: {
            "scripts": ["hcaptcha.com", "js.hcaptcha.com"],
            "elements": [
                "iframe[src*='hcaptcha']",
                ".h-captcha",
                "[data-hcaptcha-sitekey]",
            ],
            "data_attributes": ["data-sitekey", "data-hcaptcha-sitekey"],
        },
        CaptchaType.CLOUDFLARE_TURNSTILE: {
            "scripts": ["challenges.cloudflare.com/turnstile"],
            "elements": [
                "iframe[src*='turnstile']",
                ".cf-turnstile",
                "[data-sitekey]",
            ],
            "data_attributes": ["data-sitekey"],
        },
        CaptchaType.FUNCAPTCHA: {
            "scripts": ["funcaptcha.com", "arkoselabs.com"],
            "elements": [
                "iframe[src*='funcaptcha']",
                "#funcaptcha",
            ],
            "data_attributes": ["data-pkey"],
        },
        CaptchaType.GEETEST: {
            "scripts": ["geetest.com", "gt.js"],
            "elements": [
                ".geetest_holder",
                "#geetest-captcha",
            ],
            "data_attributes": [],
        },
    }
    
    # Cloudflare challenge page indicators
    CLOUDFLARE_INDICATORS = [
        "Just a moment...",
        "Checking your browser",
        "Cloudflare",
        "cf-browser-verification",
        "cf_chl_opt",
    ]
    
    async def detect(self, page) -> Optional[DetectedCaptcha]:
        """
        Detect CAPTCHA on the given page.
        
        Args:
            page: Playwright page object
            
        Returns:
            DetectedCaptcha if found, None otherwise
        """
        try:
            # Check for Cloudflare challenge page first
            cloudflare = await self._detect_cloudflare(page)
            if cloudflare:
                return cloudflare
            
            # Check for specific CAPTCHA types
            for captcha_type, patterns in self.DETECTION_PATTERNS.items():
                detected = await self._detect_type(page, captcha_type, patterns)
                if detected:
                    return detected
            
            return None
            
        except Exception as e:
            logger.error(f"[CAPTCHA] Detection error: {e}")
            return None
    
    async def _detect_cloudflare(self, page) -> Optional[DetectedCaptcha]:
        """Detect Cloudflare challenge page."""
        try:
            # Check page content for Cloudflare indicators
            content = await page.content()
            
            for indicator in self.CLOUDFLARE_INDICATORS:
                if indicator.lower() in content.lower():
                    # Check if it's a Turnstile specifically
                    if "turnstile" in content.lower():
                        # Extract site key
                        site_key = await self._extract_sitekey(
                            page, 
                            self.DETECTION_PATTERNS[CaptchaType.CLOUDFLARE_TURNSTILE]
                        )
                        return DetectedCaptcha(
                            captcha_type=CaptchaType.CLOUDFLARE_TURNSTILE,
                            site_key=site_key,
                            page_url=page.url,
                            confidence=0.9,
                        )
                    
                    # Generic Cloudflare challenge
                    return DetectedCaptcha(
                        captcha_type=CaptchaType.CLOUDFLARE_TURNSTILE,
                        page_url=page.url,
                        confidence=0.7,
                    )
            
            return None
            
        except Exception:
            return None
    
    async def _detect_type(
        self, 
        page, 
        captcha_type: CaptchaType, 
        patterns: Dict
    ) -> Optional[DetectedCaptcha]:
        """Detect a specific CAPTCHA type."""
        try:
            # Check for script sources
            scripts = await page.evaluate("""
                () => Array.from(document.scripts)
                    .map(s => s.src)
                    .filter(s => s)
            """)
            
            for script in scripts:
                for pattern in patterns.get("scripts", []):
                    if pattern in script:
                        site_key = await self._extract_sitekey(page, patterns)
                        return DetectedCaptcha(
                            captcha_type=captcha_type,
                            site_key=site_key,
                            page_url=page.url,
                            confidence=0.95,
                        )
            
            # Check for DOM elements
            for selector in patterns.get("elements", []):
                element = await page.query_selector(selector)
                if element:
                    site_key = await self._extract_sitekey(page, patterns)
                    return DetectedCaptcha(
                        captcha_type=captcha_type,
                        site_key=site_key,
                        page_url=page.url,
                        container_selector=selector,
                        confidence=0.9,
                    )
            
            return None
            
        except Exception:
            return None
    
    async def _extract_sitekey(self, page, patterns: Dict) -> Optional[str]:
        """Extract site key from page."""
        try:
            for attr in patterns.get("data_attributes", []):
                element = await page.query_selector(f"[{attr}]")
                if element:
                    return await element.get_attribute(attr)
            
            # Try to find in script content
            scripts = await page.evaluate("""
                () => Array.from(document.scripts)
                    .map(s => s.textContent)
                    .join(' ')
            """)
            
            # Common patterns for site keys
            import re
            patterns_to_try = [
                r'sitekey["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'data-sitekey["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'render["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            ]
            
            for pattern in patterns_to_try:
                match = re.search(pattern, scripts, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception:
            return None


# ============================================================================
# Main CAPTCHA Solver
# ============================================================================

class CaptchaSolver:
    """
    Main CAPTCHA solving service.
    
    Coordinates detection and solving across multiple providers
    with intelligent fallback and cost management.
    
    Example:
        >>> solver = CaptchaSolver(CaptchaConfig(
        ...     provider="2captcha",
        ...     api_key="your-key",
        ... ))
        >>> 
        >>> # Detect and solve
        >>> result = await solver.solve(page)
        >>> if result.is_success():
        ...     print(f"Token: {result.token}")
    """
    
    PROVIDER_CLASSES: Dict[str, Type[BaseCaptchaProvider]] = {
        "2captcha": TwoCaptchaProvider,
        "anti_captcha": AntiCaptchaProvider,
        "capsolver": CapSolverProvider,
    }
    
    def __init__(self, config: CaptchaConfig):
        """
        Initialize the CAPTCHA solver.
        
        Args:
            config: CaptchaConfig with provider settings
        """
        self.config = config
        self._detector = CaptchaDetector()
        self._providers: List[BaseCaptchaProvider] = []
        self._daily_spend: float = 0.0
        self._daily_spend_reset: float = time.time()
        
        # Initialize primary provider
        self._init_provider(config.provider, config.api_key)
        
        # Initialize fallback providers
        for fallback in config.fallback_providers:
            self._init_provider(fallback["provider"], fallback["api_key"])
    
    def _init_provider(self, provider_name: str, api_key: str) -> None:
        """Initialize a provider."""
        provider_class = self.PROVIDER_CLASSES.get(provider_name)
        if provider_class:
            self._providers.append(provider_class(api_key, self.config))
        else:
            logger.warning(f"[CAPTCHA] Unknown provider: {provider_name}")
    
    async def close(self) -> None:
        """Close all provider sessions."""
        for provider in self._providers:
            await provider.close()
    
    async def detect(self, page) -> Optional[DetectedCaptcha]:
        """
        Detect CAPTCHA on a page.
        
        Args:
            page: Playwright page object
            
        Returns:
            DetectedCaptcha if found, None otherwise
        """
        return await self._detector.detect(page)
    
    async def solve(
        self, 
        page, 
        detected: Optional[DetectedCaptcha] = None
    ) -> CaptchaSolution:
        """
        Detect and solve CAPTCHA on a page.
        
        Args:
            page: Playwright page object
            detected: Optional pre-detected CAPTCHA info
            
        Returns:
            CaptchaSolution with result
        """
        start_time = time.time()
        
        # Detect if not provided
        if detected is None:
            detected = await self.detect(page)
        
        if detected is None:
            return CaptchaSolution(
                status=SolveStatus.INVALID_CAPTCHA,
                captcha_type=CaptchaType.UNKNOWN,
                error_message="No CAPTCHA detected on page",
            )
        
        logger.info(f"[CAPTCHA] Detected {detected.captcha_type.value} on {detected.page_url}")
        
        # Check daily budget
        if self.config.daily_budget and self._daily_spend >= self.config.daily_budget:
            return CaptchaSolution(
                status=SolveStatus.FAILED,
                captcha_type=detected.captcha_type,
                error_message="Daily budget exceeded",
            )
        
        # Try providers in order
        last_error = None
        for provider in self._providers:
            if not provider.supports_captcha(detected.captcha_type):
                continue
            
            # Check cost limit
            cost = provider.get_cost(detected.captcha_type)
            if cost > self.config.max_cost_per_solve:
                continue
            
            for attempt in range(self.config.max_retries):
                try:
                    result = await provider.solve(detected)
                    result.captcha_type = detected.captcha_type
                    
                    if result.is_success():
                        # Track spend
                        self._daily_spend += result.cost
                        
                        # Apply token to page
                        await self._apply_solution(page, detected, result)
                        
                        logger.info(
                            f"[CAPTCHA] Solved {detected.captcha_type.value} "
                            f"in {result.solve_time_ms:.0f}ms, cost: ${result.cost:.4f}"
                        )
                        return result
                    
                    last_error = result.error_message
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[CAPTCHA] Provider error: {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
        
        return CaptchaSolution(
            status=SolveStatus.FAILED,
            captcha_type=detected.captcha_type,
            error_message=last_error or "All providers failed",
            solve_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def _apply_solution(
        self, 
        page, 
        detected: DetectedCaptcha, 
        solution: CaptchaSolution
    ) -> None:
        """Apply the CAPTCHA solution to the page."""
        if not solution.token:
            return
        
        try:
            if detected.captcha_type in (
                CaptchaType.RECAPTCHA_V2, 
                CaptchaType.RECAPTCHA_V2_INVISIBLE,
                CaptchaType.RECAPTCHA_V3
            ):
                # Set the response in the hidden textarea
                await page.evaluate(f"""
                    (token) => {{
                        // Set in textarea
                        const textarea = document.querySelector('#g-recaptcha-response');
                        if (textarea) {{
                            textarea.value = token;
                            textarea.style.display = 'block';
                        }}
                        
                        // Also set in any other response fields
                        document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {{
                            el.value = token;
                        }});
                        
                        // Trigger callback if exists
                        if (typeof ___grecaptcha_cfg !== 'undefined') {{
                            const clients = ___grecaptcha_cfg.clients;
                            if (clients) {{
                                Object.keys(clients).forEach(key => {{
                                    const client = clients[key];
                                    if (client && client.callback) {{
                                        client.callback(token);
                                    }}
                                }});
                            }}
                        }}
                    }}
                """, solution.token)
                
            elif detected.captcha_type == CaptchaType.HCAPTCHA:
                await page.evaluate(f"""
                    (token) => {{
                        const textarea = document.querySelector('[name="h-captcha-response"]');
                        if (textarea) {{
                            textarea.value = token;
                        }}
                        
                        // Trigger hcaptcha callback
                        if (typeof hcaptcha !== 'undefined' && hcaptcha.execute) {{
                            // Signal completion
                        }}
                    }}
                """, solution.token)
                
            elif detected.captcha_type == CaptchaType.CLOUDFLARE_TURNSTILE:
                await page.evaluate(f"""
                    (token) => {{
                        const input = document.querySelector('[name="cf-turnstile-response"]');
                        if (input) {{
                            input.value = token;
                        }}
                    }}
                """, solution.token)
                
            logger.debug(f"[CAPTCHA] Applied solution token to page")
            
        except Exception as e:
            logger.warning(f"[CAPTCHA] Failed to apply solution: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solving statistics."""
        # Reset daily spend if new day
        now = time.time()
        if now - self._daily_spend_reset > 86400:  # 24 hours
            self._daily_spend = 0.0
            self._daily_spend_reset = now
        
        return {
            "daily_spend": self._daily_spend,
            "daily_budget": self.config.daily_budget,
            "providers_count": len(self._providers),
        }
