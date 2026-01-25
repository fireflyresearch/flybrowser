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
Browser management for FlyBrowser.

This module provides the BrowserManager class which handles the lifecycle
of Playwright browser instances. It manages browser launching, context creation,
page management, and cleanup.

The BrowserManager supports multiple browser types (Chromium, Firefox, WebKit)
and provides a clean interface for browser operations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from flybrowser.exceptions import BrowserError
from flybrowser.utils.logger import logger

# Import proxy rotation (optional)
try:
    from flybrowser.core.proxy_rotator import ProxyRotator, ProxyConfig
    PROXY_ROTATION_AVAILABLE = True
except ImportError:
    PROXY_ROTATION_AVAILABLE = False
    ProxyRotator = None
    ProxyConfig = None


class BrowserManager:
    """
    Manages Playwright browser instances and their lifecycle.

    This class handles:
    - Browser launching with configurable options
    - Browser context creation
    - Page management
    - Resource cleanup

    Attributes:
        headless: Whether browser runs in headless mode (no visible window)
        browser_type: Type of browser (chromium, firefox, webkit)
        launch_options: Additional Playwright launch options
        page: The current active page instance

    Example:
        >>> manager = BrowserManager(headless=True, browser_type="chromium")
        >>> await manager.start()
        >>> page = manager.page
        >>> await page.goto("https://example.com")
        >>> await manager.stop()
    """

    # Default stealth arguments to avoid bot detection
    STEALTH_ARGS = [
        "--disable-blink-features=AutomationControlled",
        "--disable-features=IsolateOrigins,site-per-process",
        "--disable-site-isolation-trials",
        "--disable-features=BlockInsecurePrivateNetworkRequests",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-breakpad",
        "--disable-component-extensions-with-background-pages",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-hang-monitor",
        "--disable-ipc-flooding-protection",
        "--disable-popup-blocking",
        "--disable-prompt-on-repost",
        "--disable-renderer-backgrounding",
        "--disable-sync",
        "--enable-features=NetworkService,NetworkServiceInProcess",
        "--force-color-profile=srgb",
        "--metrics-recording-only",
        "--password-store=basic",
        "--use-mock-keychain",
        "--exclude-switches=enable-automation",
        "--disable-infobars",
        "--window-size=1920,1080",
        "--start-maximized",
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor",
    ]
    
    # Default user agent (Chrome on macOS) - Updated to latest version
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
    
    # Human-like typing delay range (milliseconds between keystrokes)
    TYPING_DELAY_MIN = 50
    TYPING_DELAY_MAX = 150

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        stealth: bool = True,
        user_agent: Optional[str] = None,
        proxy_rotator: Optional["ProxyRotator"] = None,
        **launch_options: Any,
    ) -> None:
        """
        Initialize the browser manager with configuration.

        Args:
            headless: Whether to run browser in headless mode (no visible window).
                Default: True
            browser_type: Browser type to use. Supported values:
                - "chromium": Chromium-based browser (default)
                - "firefox": Mozilla Firefox
                - "webkit": WebKit (Safari engine)
            stealth: Whether to enable stealth mode to avoid bot detection.
                Default: True
            user_agent: Custom user agent string. If None, uses a realistic default.
            proxy_rotator: Optional ProxyRotator for IP rotation.
                If provided, will rotate proxies for each new context.
            **launch_options: Additional Playwright launch options such as:
                - args: List of command-line arguments
                - downloads_path: Path for downloads
                - proxy: Proxy configuration (if not using proxy_rotator)
                - slow_mo: Slow down operations by specified milliseconds

        Example:
            >>> manager = BrowserManager(
            ...     headless=True,
            ...     browser_type="chromium",
            ...     stealth=True,  # Enable anti-detection
            ... )
            >>> 
            >>> # With proxy rotation
            >>> from flybrowser.core.proxy_rotator import ProxyRotator, ProxyConfig
            >>> proxies = [ProxyConfig(server="http://proxy1.com:8080")]
            >>> rotator = ProxyRotator(proxies)
            >>> manager = BrowserManager(proxy_rotator=rotator)
        """
        self.headless = headless
        self.browser_type = browser_type
        self.stealth = stealth
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.proxy_rotator = proxy_rotator
        self.launch_options = launch_options
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._current_proxy: Optional["ProxyConfig"] = None

    async def start(self) -> None:
        """
        Start the browser and initialize all components.

        This method:
        1. Initializes Playwright
        2. Launches the browser with stealth options (if enabled)
        3. Creates a browser context with realistic settings
        4. Injects anti-detection scripts
        5. Opens an initial page

        Raises:
            BrowserError: If browser fails to start or unsupported browser type

        Example:
            >>> manager = BrowserManager(headless=True, stealth=True)
            >>> await manager.start()
        """
        try:
            logger.info(f"Starting {self.browser_type} browser (headless={self.headless}, stealth={self.stealth})")
            self._playwright = await async_playwright().start()

            # Get the appropriate browser type
            if self.browser_type == "chromium":
                browser_launcher = self._playwright.chromium
            elif self.browser_type == "firefox":
                browser_launcher = self._playwright.firefox
            elif self.browser_type == "webkit":
                browser_launcher = self._playwright.webkit
            else:
                raise BrowserError(f"Unsupported browser type: {self.browser_type}")

            # Build launch options with stealth args
            launch_opts = dict(self.launch_options)
            if self.stealth and self.browser_type == "chromium":
                # Merge stealth args with any user-provided args
                existing_args = launch_opts.get("args", [])
                launch_opts["args"] = list(set(existing_args + self.STEALTH_ARGS))
            
            # Launch browser
            self._browser = await browser_launcher.launch(
                headless=self.headless, **launch_opts
            )

            # Get proxy if rotator is configured
            if self.proxy_rotator:
                self._current_proxy = self.proxy_rotator.get_next_proxy()
                if self._current_proxy:
                    logger.info(f"[PROXY] Using proxy: {self._current_proxy.server}")
            
            # Create browser context with realistic settings
            context_options = {
                "viewport": {"width": 1920, "height": 1080},
                "user_agent": self.user_agent,
                # Realistic browser settings
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "color_scheme": "light",
                "reduced_motion": "no-preference",
                # Permissions that a real browser would have
                "permissions": ["geolocation"],
                # Device scale factor
                "device_scale_factor": 1,
                # Java/Flash disabled (like modern browsers)
                "java_script_enabled": True,
                "has_touch": False,
                "is_mobile": False,
            }
            
            # Add proxy to context if available
            if self._current_proxy:
                context_options["proxy"] = self._current_proxy.to_playwright_format()
            
            self._context = await self._browser.new_context(**context_options)
            
            # Inject stealth scripts to avoid detection
            if self.stealth:
                await self._inject_stealth_scripts()

            # Create initial page
            self._page = await self._context.new_page()
            
            # Load custom blank page instead of about:blank
            await self._load_blank_page()

            logger.info("Browser started successfully")
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise BrowserError(f"Failed to start browser: {e}") from e
    
    async def _inject_stealth_scripts(self) -> None:
        """
        Inject comprehensive JavaScript to mask automation detection.
        
        This removes common fingerprints that websites use to detect Playwright/Puppeteer.
        Uses state-of-the-art evasion techniques.
        """
        stealth_js = """
        // ===== PART 1: Navigator Overrides =====
        
        // Override the navigator.webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Override navigator.plugins to look like a real browser
        Object.defineProperty(navigator, 'plugins', {
            get: () => {
                const plugins = [
                    { 
                        name: 'PDF Viewer', 
                        filename: 'internal-pdf-viewer', 
                        description: 'Portable Document Format',
                        length: 2,
                    },
                    { 
                        name: 'Chrome PDF Viewer', 
                        filename: 'internal-pdf-viewer', 
                        description: 'Portable Document Format',
                        length: 2,
                    },
                    { 
                        name: 'Chromium PDF Viewer', 
                        filename: 'internal-pdf-viewer', 
                        description: 'Portable Document Format',
                        length: 2,
                    },
                    { 
                        name: 'Microsoft Edge PDF Viewer', 
                        filename: 'internal-pdf-viewer', 
                        description: 'Portable Document Format',
                        length: 2,
                    },
                    { 
                        name: 'WebKit built-in PDF', 
                        filename: 'internal-pdf-viewer', 
                        description: 'Portable Document Format',
                        length: 2,
                    },
                ];
                Object.setPrototypeOf(plugins, PluginArray.prototype);
                plugins.item = (i) => plugins[i] || null;
                plugins.namedItem = (name) => plugins.find(p => p.name === name) || null;
                plugins.refresh = () => {};
                plugins.length = plugins.length;
                return plugins;
            },
        });
        
        // Override navigator.languages to be more realistic
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        // Override navigator.platform
        Object.defineProperty(navigator, 'platform', {
            get: () => 'MacIntel',
        });
        
        // Override navigator.hardwareConcurrency
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => 8,
        });
        
        // Override navigator.deviceMemory
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => 8,
        });
        
        // Override navigator.maxTouchPoints
        Object.defineProperty(navigator, 'maxTouchPoints', {
            get: () => 0,
        });
        
        // Override navigator.vendor
        Object.defineProperty(navigator, 'vendor', {
            get: () => 'Google Inc.',
        });
        
        // Override navigator.productSub
        Object.defineProperty(navigator, 'productSub', {
            get: () => '20030107',
        });
        
        // Override navigator.vendorSub
        Object.defineProperty(navigator, 'vendorSub', {
            get: () => '',
        });
        
        // ===== PART 2: Remove Automation Indicators =====
        
        // Remove Playwright/Puppeteer indicators from window
        delete window.__playwright;
        delete window.__pw_manual;
        delete window.__PW_inspect;
        delete window.__pwInitScripts;
        delete window._WEBDRIVER_ELEM_CACHE;
        delete window.domAutomation;
        delete window.domAutomationController;
        
        // ===== PART 3: Chrome Object =====
        
        // Create a realistic chrome object
        if (!window.chrome || !window.chrome.runtime) {
            window.chrome = {
                app: {
                    isInstalled: false,
                    InstallState: {
                        DISABLED: 'disabled',
                        INSTALLED: 'installed',
                        NOT_INSTALLED: 'not_installed',
                    },
                    RunningState: {
                        CANNOT_RUN: 'cannot_run',
                        READY_TO_RUN: 'ready_to_run',
                        RUNNING: 'running',
                    },
                },
                runtime: {
                    connect: () => {
                        return {
                            onMessage: { addListener: () => {} },
                            postMessage: () => {},
                            disconnect: () => {},
                        };
                    },
                    sendMessage: () => {},
                    onMessage: { 
                        addListener: () => {},
                        removeListener: () => {},
                        hasListener: () => false,
                    },
                    id: undefined,
                    getManifest: () => ({}),
                    getURL: (path) => path,
                },
                loadTimes: () => ({
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
                }),
                csi: () => ({
                    startE: Date.now(),
                    onloadT: Date.now(),
                    pageT: Date.now(),
                    tran: 15,
                }),
            };
        }
        
        // ===== PART 4: Permissions API =====
        
        // Override permissions API to avoid detection
        const originalQuery = navigator.permissions.query;
        navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters)
        );
        
        // ===== PART 5: WebGL Fingerprinting =====
        
        // Mask WebGL vendor and renderer to look like real hardware
        const getParameterProxyHandler = {
            apply: function(target, thisArg, args) {
                const param = args[0];
                // UNMASKED_VENDOR_WEBGL
                if (param === 37445) {
                    return 'Intel Inc.';
                }
                // UNMASKED_RENDERER_WEBGL
                if (param === 37446) {
                    return 'Intel Iris Pro OpenGL Engine';
                }
                return Reflect.apply(target, thisArg, args);
            }
        };
        
        // Apply to WebGL contexts
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type, ...args) {
            const context = originalGetContext.call(this, type, ...args);
            if (context && (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl')) {
                const originalGetParameter = context.getParameter;
                context.getParameter = new Proxy(originalGetParameter, getParameterProxyHandler);
            }
            return context;
        };
        
        // ===== PART 6: Canvas Fingerprinting Protection =====
        
        // Add subtle noise to canvas to avoid fingerprinting
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        const originalToBlob = HTMLCanvasElement.prototype.toBlob;
        const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
        
        // Small noise function
        const addCanvasNoise = (canvas) => {
            const context = canvas.getContext('2d');
            if (context) {
                const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    // Add very subtle noise (±1 to rgba values)
                    imageData.data[i] = imageData.data[i] + (Math.random() > 0.5 ? 1 : -1);
                }
                context.putImageData(imageData, 0, 0);
            }
        };
        
        // ===== PART 7: Screen and Window Properties =====
        
        // Make sure screen properties look realistic
        Object.defineProperty(screen, 'availTop', {
            get: () => 23,  // Mac menu bar
        });
        
        Object.defineProperty(screen, 'availLeft', {
            get: () => 0,
        });
        
        // ===== PART 8: WebRTC Leak Protection =====
        
        // Disable WebRTC to prevent IP leaks
        const originalRTCPeerConnection = window.RTCPeerConnection;
        window.RTCPeerConnection = function(...args) {
            const pc = new originalRTCPeerConnection(...args);
            const originalCreateDataChannel = pc.createDataChannel;
            pc.createDataChannel = function(...args) {
                return originalCreateDataChannel.apply(pc, args);
            };
            return pc;
        };
        
        // ===== PART 9: Mouse and Touch Events =====
        
        // Ensure mouse event properties are realistic
        Object.defineProperty(navigator, 'pointerEnabled', {
            get: () => true,
        });
        
        // ===== PART 10: Battery API =====
        
        // Mock battery API
        if (!navigator.getBattery) {
            navigator.getBattery = () => Promise.resolve({
                charging: true,
                chargingTime: 0,
                dischargingTime: Infinity,
                level: 1,
                addEventListener: () => {},
                removeEventListener: () => {},
                dispatchEvent: () => true,
            });
        }
        
        // ===== PART 11: Connection API =====
        
        // Mock connection to look realistic
        Object.defineProperty(navigator, 'connection', {
            get: () => ({
                effectiveType: '4g',
                rtt: 50,
                downlink: 10,
                saveData: false,
                addEventListener: () => {},
                removeEventListener: () => {},
            }),
        });
        
        // ===== PART 12: iframe ContentWindow =====
        
        // Prevent detection through iframe
        Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
            get: function() {
                const win = Reflect.get(this, '_contentWindow');
                if (win) {
                    win.navigator.webdriver = undefined;
                }
                return win;
            },
        });
        
        // ===== PART 13: Error Stack Traces =====
        
        // Clean stack traces to remove automation traces
        const originalError = Error;
        Error = function(...args) {
            const err = new originalError(...args);
            if (err.stack) {
                err.stack = err.stack.replace(/\\s+at\\s+__puppeteer_evaluation_script__[\\s\\S]*/, '');
                err.stack = err.stack.replace(/\\s+at\\s+evaluateHandle[\\s\\S]*/, '');
            }
            return err;
        };
        Error.prototype = originalError.prototype;
        
        // ===== PART 14: Date and Time =====
        
        // Ensure timezone is consistent
        Date.prototype.getTimezoneOffset = function() {
            return 300; // EST (UTC-5)
        };
        
        console.log('[Stealth] Anti-detection scripts loaded successfully');
        """
        
        # Add script to run on every new page
        await self._context.add_init_script(stealth_js)
    
    async def _load_blank_page(self) -> None:
        """
        Load the custom FlyBrowser blank page.
        
        This replaces the default about:blank with a branded 'waiting for agent'
        page that shows the FlyBrowser logo and a visual indicator that the
        browser is ready for automation.
        
        The HTML is injected directly using set_content() with inline CSS/JS
        to avoid any server dependency.
        """
        try:
            from flybrowser.service.template_renderer import render_blank_html
            
            # Render blank page with inline assets (no server required)
            html = render_blank_html(inline_assets=True)
            
            # Inject the HTML directly into the page
            await self._page.set_content(html, wait_until="domcontentloaded")
            
            logger.debug("Custom blank page loaded")
        except Exception as e:
            # If loading fails, just continue with about:blank
            logger.debug(f"Could not load custom blank page: {e}")
    
    async def _load_completion_page(
        self,
        success: bool,
        task: str,
        duration_ms: float,
        iterations: int,
        result_data: Optional[Any] = None,
        error_message: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Load the agent completion page with task results.
        
        This displays a summary page after the agent() method completes execution,
        showing the task status (success/failure), execution metrics, and any
        result data or error messages.
        
        Args:
            success: Whether the agent task completed successfully
            task: The task description that was executed
            duration_ms: Execution duration in milliseconds
            iterations: Number of iterations used
            result_data: Optional result data (for successful extractions)
            error_message: Optional error message (for failures)
            max_iterations: Optional max iterations limit for display
        """
        try:
            from flybrowser.service.template_renderer import render_completion_html
            
            # Render completion page with inline assets (no server required)
            html = render_completion_html(
                success=success,
                task=task,
                duration_ms=duration_ms,
                iterations=iterations,
                result_data=result_data,
                error_message=error_message,
                max_iterations=max_iterations,
                inline_assets=True,
            )
            
            # Inject the HTML directly into the page
            await self._page.set_content(html, wait_until="domcontentloaded")
            
            logger.debug(f"Agent completion page loaded (success={success})")
        except Exception as e:
            # If loading fails, log but don't fail the overall operation
            logger.debug(f"Could not load completion page: {e}")

    async def stop(self) -> None:
        """
        Stop the browser and cleanup all resources.

        This method gracefully closes:
        1. All open pages
        2. The browser context
        3. The browser instance
        4. Playwright resources

        Raises:
            BrowserError: If cleanup fails

        Example:
            >>> await manager.stop()
        """
        try:
            logger.info("Stopping browser")
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            logger.info("Browser stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping browser: {e}")
            raise BrowserError(f"Failed to stop browser: {e}") from e

    async def new_page(self) -> Page:
        """
        Create a new page in the current context.

        Returns:
            New Page instance
        """
        if not self._context:
            raise BrowserError("Browser context not initialized. Call start() first.")
        return await self._context.new_page()

    @property
    def page(self) -> Page:
        """Get the current page."""
        if not self._page:
            raise BrowserError("No active page. Call start() first.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        """Get the browser context."""
        if not self._context:
            raise BrowserError("Browser context not initialized. Call start() first.")
        return self._context

    @property
    def browser(self) -> Browser:
        """Get the browser instance."""
        if not self._browser:
            raise BrowserError("Browser not initialized. Call start() first.")
        return self._browser

    async def __aenter__(self) -> "BrowserManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

