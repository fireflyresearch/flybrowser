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
CAPTCHA Detection and Solving Tools for ReAct Framework.

This module provides tools for detecting and solving CAPTCHAs during
autonomous browser automation. It integrates with the stealth.captcha module.

Tools:
- DetectCaptchaTool: Detects if a CAPTCHA is present on the page
- SolveCaptchaTool: Solves detected CAPTCHAs using configured providers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from flybrowser.agents.types import SafetyLevel, ToolCategory
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.stealth.captcha import CaptchaSolver


class DetectCaptchaTool(BaseTool):
    """
    Tool for detecting CAPTCHAs on the current page.
    
    This tool analyzes the page to detect various types of CAPTCHAs:
    - reCAPTCHA v2/v3
    - hCaptcha
    - Cloudflare Turnstile
    - FunCaptcha
    - Image CAPTCHAs
    
    Returns information about detected CAPTCHA type and location.
    """
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        self.page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="detect_captcha",
            description=(
                "Detect if a CAPTCHA is present on the current page. "
                "Returns information about any detected CAPTCHA including type "
                "(recaptcha_v2, recaptcha_v3, hcaptcha, turnstile, etc.), site key, "
                "element selector, and whether it's blocking the page. Use this when "
                "the page seems blocked, shows 'I'm not a robot', or form submission fails."
            ),
            category=ToolCategory.UTILITY,
            safety_level=SafetyLevel.SAFE,
            parameters=[],
            returns_description='{"detected": bool, "type": str, "site_key": str, "selector": str, "blocking": bool}',
            requires_page=True,
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute CAPTCHA detection."""
        if not self.page_controller:
            return ToolResult(
                success=False,
                data=None,
                error="Page controller not configured",
            )
        
        try:
            page = self.page_controller.page
            
            # Detection scripts for various CAPTCHA types
            detection_result = await page.evaluate("""() => {
                const result = {
                    detected: false,
                    type: null,
                    site_key: null,
                    selector: null,
                    blocking: false,
                    details: {}
                };
                
                // Check for reCAPTCHA v2
                const recaptchaV2 = document.querySelector('.g-recaptcha, [data-sitekey]');
                if (recaptchaV2) {
                    result.detected = true;
                    result.type = 'recaptcha_v2';
                    result.site_key = recaptchaV2.getAttribute('data-sitekey');
                    result.selector = '.g-recaptcha';
                    result.blocking = true;
                    return result;
                }
                
                // Check for reCAPTCHA v3 (invisible)
                const recaptchaScript = document.querySelector('script[src*="recaptcha/api.js"]');
                if (recaptchaScript && recaptchaScript.src.includes('render=')) {
                    const match = recaptchaScript.src.match(/render=([^&]+)/);
                    if (match && match[1] !== 'explicit') {
                        result.detected = true;
                        result.type = 'recaptcha_v3';
                        result.site_key = match[1];
                        result.blocking = false;
                        return result;
                    }
                }
                
                // Check for hCaptcha
                const hcaptcha = document.querySelector('.h-captcha, [data-hcaptcha-sitekey]');
                if (hcaptcha) {
                    result.detected = true;
                    result.type = 'hcaptcha';
                    result.site_key = hcaptcha.getAttribute('data-sitekey') || 
                                      hcaptcha.getAttribute('data-hcaptcha-sitekey');
                    result.selector = '.h-captcha';
                    result.blocking = true;
                    return result;
                }
                
                // Check for Cloudflare Turnstile
                const turnstile = document.querySelector('.cf-turnstile, [data-turnstile-sitekey]');
                if (turnstile) {
                    result.detected = true;
                    result.type = 'turnstile';
                    result.site_key = turnstile.getAttribute('data-sitekey') ||
                                      turnstile.getAttribute('data-turnstile-sitekey');
                    result.selector = '.cf-turnstile';
                    result.blocking = true;
                    return result;
                }
                
                // Check for Cloudflare challenge page
                const cfChallenge = document.querySelector('#cf-challenge-running, .cf-browser-verification');
                if (cfChallenge || document.title.includes('Just a moment')) {
                    result.detected = true;
                    result.type = 'cloudflare_challenge';
                    result.blocking = true;
                    result.details.message = 'Cloudflare verification page detected';
                    return result;
                }
                
                // Check for FunCaptcha
                const funcaptcha = document.querySelector('#FunCaptcha, [data-pkey]');
                if (funcaptcha) {
                    result.detected = true;
                    result.type = 'funcaptcha';
                    result.site_key = funcaptcha.getAttribute('data-pkey');
                    result.selector = '#FunCaptcha';
                    result.blocking = true;
                    return result;
                }
                
                // Check for generic image CAPTCHA
                const imgCaptcha = document.querySelector(
                    'img[alt*="captcha" i], img[src*="captcha" i], ' +
                    'img[class*="captcha" i], .captcha-image'
                );
                if (imgCaptcha) {
                    result.detected = true;
                    result.type = 'image_captcha';
                    result.selector = 'img[src*="captcha"]';
                    result.blocking = true;
                    result.details.image_url = imgCaptcha.src;
                    return result;
                }
                
                return result;
            }""")
            
            logger.info(f"[CAPTCHA] Detection result: {detection_result}")
            
            return ToolResult(
                success=True,
                data=detection_result,
                error=None,
            )
            
        except Exception as e:
            logger.error(f"[CAPTCHA] Detection failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"CAPTCHA detection failed: {e}",
            )


class SolveCaptchaTool(BaseTool):
    """
    Tool for solving detected CAPTCHAs.
    
    Uses the configured CAPTCHA solving provider to solve CAPTCHAs
    and inject the solution into the page.
    
    Requires a CaptchaSolver instance to be configured in the SDK.
    """
    
    def __init__(
        self,
        page_controller: Optional["PageController"] = None,
        captcha_solver: Optional["CaptchaSolver"] = None,
    ) -> None:
        self.page_controller = page_controller
        self.captcha_solver = captcha_solver
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="solve_captcha",
            description=(
                "Solve a CAPTCHA on the current page. Detects CAPTCHA type if not provided, "
                "sends to solving service, waits for solution, injects it into the page, "
                "and optionally submits the form. Requires CAPTCHA solver in stealth settings."
            ),
            category=ToolCategory.UTILITY,
            safety_level=SafetyLevel.DANGEROUS,
            parameters=[
                ToolParameter(
                    name="captcha_type",
                    type="string",
                    description="CAPTCHA type override (recaptcha_v2, recaptcha_v3, hcaptcha, turnstile)",
                    required=False,
                ),
                ToolParameter(
                    name="site_key",
                    type="string",
                    description="Site key override",
                    required=False,
                ),
                ToolParameter(
                    name="submit_after",
                    type="boolean",
                    description="Submit form after solving (default: true)",
                    required=False,
                    default=True,
                ),
            ],
            returns_description='{"solved": bool, "captcha_type": str, "cost_usd": float, "time_ms": float}',
            requires_page=True,
        )
    
    async def execute(
        self,
        captcha_type: Optional[str] = None,
        site_key: Optional[str] = None,
        submit_after: bool = True,
        **kwargs,
    ) -> ToolResult:
        """Execute CAPTCHA solving."""
        if not self.page_controller:
            return ToolResult(
                success=False,
                data=None,
                error="Page controller not configured",
            )
        
        if not self.captcha_solver:
            return ToolResult(
                success=False,
                data={"solved": False},
                error="CAPTCHA solver not configured. Enable captcha_enabled in StealthConfig.",
            )
        
        try:
            import time
            start_time = time.time()
            page = self.page_controller.page
            current_url = page.url
            
            # Detect CAPTCHA if type not provided
            if not captcha_type or not site_key:
                detect_tool = DetectCaptchaTool(page_controller=self.page_controller)
                detect_result = await detect_tool.execute()
                
                if not detect_result.success or not detect_result.data.get("detected"):
                    return ToolResult(
                        success=True,
                        data={"solved": False, "reason": "No CAPTCHA detected"},
                        error=None,
                    )
                
                captcha_type = captcha_type or detect_result.data.get("type")
                site_key = site_key or detect_result.data.get("site_key")
            
            logger.info(f"[CAPTCHA] Solving {captcha_type} captcha...")
            
            # Solve using the captcha solver
            solution = await self.captcha_solver.solve(
                captcha_type=captcha_type,
                site_key=site_key,
                page_url=current_url,
                page=page,
            )
            
            if not solution.success:
                return ToolResult(
                    success=False,
                    data={"solved": False, "captcha_type": captcha_type},
                    error=f"CAPTCHA solving failed: {solution.error}",
                )
            
            # Inject solution into page
            await self._inject_solution(page, captcha_type, solution.token)
            
            # Submit form if requested
            if submit_after:
                await self._submit_form(page, captcha_type)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"[CAPTCHA] Solved {captcha_type} in {elapsed_ms:.0f}ms")
            
            return ToolResult(
                success=True,
                data={
                    "solved": True,
                    "captcha_type": captcha_type,
                    "cost_usd": solution.cost_usd,
                    "time_ms": elapsed_ms,
                },
                error=None,
            )
            
        except Exception as e:
            logger.error(f"[CAPTCHA] Solving failed: {e}")
            return ToolResult(
                success=False,
                data={"solved": False},
                error=f"CAPTCHA solving failed: {e}",
            )
    
    async def _inject_solution(self, page, captcha_type: str, token: str) -> None:
        """Inject CAPTCHA solution into the page."""
        if captcha_type in ("recaptcha_v2", "recaptcha_v3"):
            await page.evaluate(f"""(token) => {{
                // Find and fill response textareas
                const textareas = document.querySelectorAll(
                    '#g-recaptcha-response, [name="g-recaptcha-response"]'
                );
                textareas.forEach(ta => {{
                    ta.style.display = 'block';
                    ta.value = token;
                }});
                
                // Call callback if available
                if (typeof window.___grecaptcha_cfg !== 'undefined') {{
                    const clients = window.___grecaptcha_cfg.clients;
                    if (clients) {{
                        Object.keys(clients).forEach(key => {{
                            const client = clients[key];
                            if (client && client.callback) {{
                                client.callback(token);
                            }}
                        }});
                    }}
                }}
            }}""", token)
            
        elif captcha_type == "hcaptcha":
            await page.evaluate(f"""(token) => {{
                // Find and fill response textareas
                const textareas = document.querySelectorAll(
                    '[name="h-captcha-response"], [name="g-recaptcha-response"]'
                );
                textareas.forEach(ta => {{
                    ta.value = token;
                }});
                
                // Trigger hcaptcha callback
                if (window.hcaptcha) {{
                    const widget = document.querySelector('.h-captcha');
                    if (widget) {{
                        const widgetId = widget.getAttribute('data-hcaptcha-widget-id');
                        if (widgetId && window.hcaptcha.execute) {{
                            // Token already submitted via callback
                        }}
                    }}
                }}
            }}""", token)
            
        elif captcha_type == "turnstile":
            await page.evaluate(f"""(token) => {{
                // Find and fill turnstile response
                const inputs = document.querySelectorAll(
                    '[name="cf-turnstile-response"], [name="turnstile-response"]'
                );
                inputs.forEach(input => {{
                    input.value = token;
                }});
                
                // Call turnstile callback if available
                if (window.turnstile) {{
                    const widget = document.querySelector('.cf-turnstile');
                    if (widget) {{
                        const callback = widget.getAttribute('data-callback');
                        if (callback && typeof window[callback] === 'function') {{
                            window[callback](token);
                        }}
                    }}
                }}
            }}""", token)
    
    async def _submit_form(self, page, captcha_type: str) -> None:
        """Submit the form containing the CAPTCHA."""
        try:
            # Find and submit the form
            await page.evaluate("""() => {
                // Find the form containing the CAPTCHA
                const captchaElement = document.querySelector(
                    '.g-recaptcha, .h-captcha, .cf-turnstile'
                );
                if (captchaElement) {
                    const form = captchaElement.closest('form');
                    if (form) {
                        // Trigger submit
                        const submitBtn = form.querySelector(
                            'button[type="submit"], input[type="submit"], button:not([type])'
                        );
                        if (submitBtn) {
                            submitBtn.click();
                        } else {
                            form.submit();
                        }
                    }
                }
            }""")
            
            # Wait for navigation or response
            await page.wait_for_load_state("networkidle", timeout=5000)
            
        except Exception as e:
            logger.debug(f"[CAPTCHA] Form submission note: {e}")


class WaitForCaptchaResolvedTool(BaseTool):
    """
    Tool to wait until a CAPTCHA challenge is resolved.
    
    Useful for Cloudflare challenges and other automated checks
    that may resolve on their own after some time.
    """
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        self.page_controller = page_controller
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="wait_for_captcha_resolved",
            description=(
                "Wait for a CAPTCHA challenge page to resolve. Some challenges "
                "(like Cloudflare) may resolve automatically. Waits for the challenge "
                "to complete and the page to navigate away."
            ),
            category=ToolCategory.UTILITY,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="timeout_seconds",
                    type="integer",
                    description="Maximum time to wait (default: 30)",
                    required=False,
                    default=30,
                ),
            ],
            returns_description='{"resolved": bool, "new_url": str, "wait_time_ms": float}',
            requires_page=True,
        )
    
    async def execute(self, timeout_seconds: int = 30, **kwargs) -> ToolResult:
        """Wait for CAPTCHA to resolve."""
        if not self.page_controller:
            return ToolResult(
                success=False,
                data=None,
                error="Page controller not configured",
            )
        
        try:
            import time
            start_time = time.time()
            page = self.page_controller.page
            original_url = page.url
            
            # Wait for either navigation or challenge element to disappear
            try:
                await page.wait_for_function(
                    """() => {
                        // Check if challenge elements are gone
                        const challenge = document.querySelector(
                            '#cf-challenge-running, .cf-browser-verification, ' +
                            '.g-recaptcha, .h-captcha, .cf-turnstile'
                        );
                        return !challenge || document.title !== 'Just a moment...';
                    }""",
                    timeout=timeout_seconds * 1000,
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                new_url = page.url
                
                return ToolResult(
                    success=True,
                    data={
                        "resolved": True,
                        "new_url": new_url,
                        "url_changed": new_url != original_url,
                        "wait_time_ms": elapsed_ms,
                    },
                    error=None,
                )
                
            except Exception as timeout_error:
                elapsed_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    success=True,
                    data={
                        "resolved": False,
                        "new_url": page.url,
                        "wait_time_ms": elapsed_ms,
                        "reason": "Timeout waiting for resolution",
                    },
                    error=None,
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Wait failed: {e}",
            )
