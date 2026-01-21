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
Unified FlyBrowser SDK.

This module provides a single, unified FlyBrowser class that works transparently
in all deployment modes:

1. **Embedded Mode** (no endpoint): Runs browser locally in the same process
2. **Server Mode** (with endpoint): Connects to a FlyBrowser server (standalone or cluster)

The SDK automatically handles:
- Cluster discovery and failover (when connecting to a cluster)
- Session affinity and routing
- Connection pooling and retry logic
- All features work identically in both modes

Example - Embedded Mode (local browser):
    >>> async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    ...     await browser.goto("https://example.com")
    ...     data = await browser.extract("Get the title")

Example - Server Mode (standalone or cluster):
    >>> async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    ...     await browser.goto("https://example.com")
    ...     data = await browser.extract("Get the title")

The same code works regardless of whether the endpoint points to a standalone
server or a multi-node cluster. The SDK handles all complexity internally.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional, Union

from flybrowser.utils.logger import logger


class FlyBrowser:
    """
    Unified FlyBrowser SDK for LLM-powered browser automation.

    This class provides a single, unified interface that works transparently in
    all deployment modes:

    1. **Embedded Mode** (no endpoint): Runs browser locally in the same process
    2. **Server Mode** (with endpoint): Connects to a FlyBrowser server

    The SDK automatically handles cluster discovery, failover, and session routing
    when connecting to a cluster. Developers write the same code regardless of
    deployment mode.

    Example - Embedded Mode:
        >>> async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        ...     await browser.goto("https://example.com")
        ...     data = await browser.extract("Get the title")

    Example - Server Mode (standalone or cluster):
        >>> async with FlyBrowser(endpoint="http://localhost:8000") as browser:
        ...     await browser.goto("https://example.com")
        ...     data = await browser.extract("Get the title")

    The endpoint URL is the ONLY thing that changes between modes. All features
    (screenshots, recordings, extractions, PII masking) work identically.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        headless: bool = True,
        browser_type: str = "chromium",
        recording_enabled: bool = False,
        pii_masking_enabled: bool = True,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize FlyBrowser.

        Args:
            endpoint: Server endpoint URL. If provided, connects to a FlyBrowser
                server (standalone or cluster). If None, runs browser locally.
                Examples:
                - None: Embedded mode (local browser)
                - "http://localhost:8000": Standalone server
                - "http://cluster.example.com:8000": Cluster endpoint
            llm_provider: LLM provider name (openai, anthropic, ollama)
            llm_model: LLM model name (uses provider default if not specified)
            api_key: API key for the LLM provider
            headless: Run browser in headless mode (default: True)
            browser_type: Browser type (chromium, firefox, webkit)
            recording_enabled: Enable session recording (default: False)
            pii_masking_enabled: Enable PII masking (default: True)
            timeout: Request timeout in seconds for server mode (default: 30.0)
            **kwargs: Additional configuration options

        Example - Embedded Mode:
            >>> browser = FlyBrowser(
            ...     llm_provider="openai",
            ...     api_key="sk-...",
            ...     headless=True
            ... )

        Example - Server Mode:
            >>> browser = FlyBrowser(
            ...     endpoint="http://localhost:8000",
            ...     llm_provider="openai",
            ...     api_key="sk-..."
            ... )
        """
        self._endpoint = endpoint
        self._mode = "server" if endpoint else "embedded"
        self._started = False

        # Store configuration for both modes
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._api_key = api_key
        self._headless = headless
        self._browser_type = browser_type
        self._recording_enabled = recording_enabled
        self._pii_masking_enabled = pii_masking_enabled
        self._timeout = timeout
        self._kwargs = kwargs

        # Mode-specific components (initialized in start())
        self._client = None  # HTTP client for server mode
        self._session_id = None  # Session ID for server mode

        # Embedded mode components
        self.llm = None
        self.browser_manager = None
        self.page_controller = None
        self.element_detector = None
        self.extraction_agent = None
        self.pii_handler = None
        self._screenshot_capture = None
        self._recording_manager = None

        logger.info(f"Initializing FlyBrowser in {self._mode} mode")

    async def start(self) -> None:
        """
        Start the browser session.

        In embedded mode: Launches a local Playwright browser.
        In server mode: Creates a session on the FlyBrowser server.

        This method is called automatically when using the async context manager.

        Raises:
            RuntimeError: If already started or connection fails
        """
        if self._started:
            logger.warning("FlyBrowser already started")
            return

        if self._mode == "server":
            await self._start_server_mode()
        else:
            await self._start_embedded_mode()

        self._started = True
        logger.info(f"FlyBrowser started in {self._mode} mode")

    async def _start_server_mode(self) -> None:
        """Start in server mode - connect to FlyBrowser server."""
        from flybrowser.client import FlyBrowserClient

        self._client = FlyBrowserClient(
            endpoint=self._endpoint,
            timeout=self._timeout,
        )
        await self._client.start()

        # Create a session on the server
        session_data = await self._client.create_session(
            llm_provider=self._llm_provider,
            llm_model=self._llm_model,
            api_key=self._api_key,
            headless=self._headless,
        )
        self._session_id = session_data.get("session_id")

        if not self._session_id:
            raise RuntimeError("Failed to create session on server")

        logger.info(f"Created server session: {self._session_id}")

    async def _start_embedded_mode(self) -> None:
        """Start in embedded mode - run browser locally."""
        # Import components only when needed (embedded mode)
        from flybrowser.agents.action_agent import ActionAgent
        from flybrowser.agents.extraction_agent import ExtractionAgent
        from flybrowser.agents.monitoring_agent import MonitoringAgent
        from flybrowser.agents.navigation_agent import NavigationAgent
        from flybrowser.agents.workflow_agent import WorkflowAgent
        from flybrowser.core.browser import BrowserManager
        from flybrowser.core.element import ElementDetector
        from flybrowser.core.page import PageController
        from flybrowser.core.recording import (
            RecordingConfig,
            RecordingManager,
            ScreenshotCapture,
        )
        from flybrowser.llm.factory import LLMProviderFactory
        from flybrowser.security.pii_handler import PIIConfig, PIIHandler

        # Initialize LLM provider
        self.llm = LLMProviderFactory.create(
            provider=self._llm_provider,
            model=self._llm_model,
            api_key=self._api_key,
        )

        # Initialize browser manager
        self.browser_manager = BrowserManager(
            headless=self._headless,
            browser_type=self._browser_type,
        )
        await self.browser_manager.start()

        # Initialize PII handler first (needed by agents)
        pii_config = PIIConfig(enabled=self._pii_masking_enabled)
        self.pii_handler = PIIHandler(pii_config)

        # Initialize components
        self.page_controller = PageController(self.browser_manager.page)
        self.element_detector = ElementDetector(self.browser_manager.page, self.llm)

        # Initialize all agents with PII handler
        self.extraction_agent = ExtractionAgent(
            self.page_controller, self.element_detector, self.llm,
            pii_handler=self.pii_handler
        )
        self.action_agent = ActionAgent(
            self.browser_manager.page, self.element_detector, self.llm,
            pii_handler=self.pii_handler
        )
        self.navigation_agent = NavigationAgent(
            self.browser_manager.page, self.page_controller, self.llm,
            pii_handler=self.pii_handler
        )
        self.workflow_agent = WorkflowAgent(
            self.browser_manager.page, self.element_detector, self.llm,
            pii_handler=self.pii_handler
        )
        self.monitoring_agent = MonitoringAgent(
            self.browser_manager.page, self.llm,
            pii_handler=self.pii_handler
        )

        # Initialize recording components if enabled
        if self._recording_enabled:
            recording_config = RecordingConfig(enabled=True)
            self._screenshot_capture = ScreenshotCapture(recording_config)
            self._recording_manager = RecordingManager(recording_config)
            logger.info("Recording components initialized")

    async def stop(self) -> None:
        """
        Stop the browser session and cleanup resources.

        In embedded mode: Closes the browser and cleans up Playwright.
        In server mode: Closes the session on the server.
        """
        if not self._started:
            logger.warning("FlyBrowser not started")
            return

        logger.info("Stopping FlyBrowser")

        if self._mode == "server":
            if self._session_id and self._client:
                try:
                    await self._client.close_session(self._session_id)
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
            if self._client:
                await self._client.stop()
        else:
            if self.browser_manager:
                await self.browser_manager.stop()

        self._started = False
        logger.info("FlyBrowser stopped successfully")

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to (must include protocol, e.g., https://)
            wait_until: When to consider navigation succeeded.

        Example:
            >>> await browser.goto("https://example.com")
        """
        self._ensure_started()

        if self._mode == "server":
            await self._client.navigate(self._session_id, url)
        else:
            await self.page_controller.goto(url, wait_until=wait_until)

    async def navigate(self, instruction: str, use_vision: bool = True) -> Dict[str, Any]:
        """
        Navigate using natural language instructions.

        Uses the NavigationAgent for intelligent navigation with:
        - Natural language understanding ("go to the login page")
        - Smart waiting for page loads
        - Automatic retry on navigation failures
        - Link and button detection

        Args:
            instruction: Natural language navigation instruction
            use_vision: Use vision for element detection (default: True)

        Returns:
            NavigationResult with details

        Example:
            >>> await browser.navigate("go to the login page")
            >>> await browser.navigate("click the 'Products' menu item")
        """
        self._ensure_started()

        if self._mode == "server":
            return await self._client.navigate(self._session_id, instruction)

        # Embedded mode: Use NavigationAgent
        result = await self.navigation_agent.execute(instruction, use_vision=use_vision)
        return {
            "success": result.success,
            "url": result.url,
            "title": result.title,
            "navigation_type": result.navigation_type.value if result.navigation_type else None,
            "error": result.error,
        }

    async def extract(
        self, query: str, use_vision: bool = False, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract data from the current page using natural language.

        Args:
            query: Natural language query describing what to extract.
            use_vision: Use vision-based extraction (default: False)
            schema: Optional JSON schema for structured extraction.

        Returns:
            Dictionary containing the extracted data.

        Example:
            >>> data = await browser.extract("What is the page title?")
        """
        self._ensure_started()

        if self._mode == "server":
            result = await self._client.extract(self._session_id, query, schema)
            return result.get("data", result)
        else:
            return await self.extraction_agent.execute(query, use_vision=use_vision, schema=schema)

    async def act(self, instruction: str, use_vision: bool = True) -> Dict[str, Any]:
        """
        Perform an action on the page based on natural language instruction.

        Uses the ActionAgent for intelligent action execution with:
        - Natural language understanding
        - Multi-step action planning
        - Automatic retry on failure
        - PII-safe credential handling

        Args:
            instruction: Natural language instruction (e.g., "click the login button")
            use_vision: Use vision for element detection (default: True)

        Returns:
            ActionResult with execution details

        Example:
            >>> await browser.act("click the login button")
            >>> await browser.act("type 'hello' into the search box")
        """
        self._ensure_started()

        if self._mode == "server":
            result = await self._client.action(self._session_id, instruction)
            return result

        # Embedded mode: Use ActionAgent for intelligent action execution
        result = await self.action_agent.execute(instruction, use_vision=use_vision)
        return {
            "success": result.success,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "error": result.error,
            "details": result.details,
        }

    async def screenshot(self, full_page: bool = False, mask_pii: bool = True) -> Dict[str, Any]:
        """
        Take a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page (default: False)
            mask_pii: Apply PII masking (default: True)

        Returns:
            Dictionary with screenshot data including base64-encoded image.

        Example:
            >>> screenshot = await browser.screenshot(full_page=True)
            >>> print(screenshot["data_base64"][:50])
        """
        self._ensure_started()

        if self._mode == "server":
            return await self._client.screenshot(self._session_id, full_page)
        else:
            from flybrowser.core.recording import RecordingConfig, ScreenshotCapture

            if not self._screenshot_capture:
                config = RecordingConfig(enabled=True)
                self._screenshot_capture = ScreenshotCapture(config)

            screenshot = await self._screenshot_capture.take(
                self.browser_manager.page,
                full_page=full_page,
                save_to_file=False,
            )

            return {
                "screenshot_id": screenshot.id,
                "format": screenshot.format.value,
                "width": screenshot.width,
                "height": screenshot.height,
                "data_base64": screenshot.to_base64(),
                "url": screenshot.url,
                "timestamp": screenshot.timestamp,
            }

    async def get_screenshot_base64(self, full_page: bool = False) -> str:
        """Take a screenshot and return as base64 string."""
        result = await self.screenshot(full_page=full_page)
        return result.get("data_base64", "")

    async def start_recording(self) -> Dict[str, Any]:
        """
        Start recording the browser session.

        Returns:
            Dictionary with recording_id.

        Example:
            >>> await browser.start_recording()
            >>> await browser.goto("https://example.com")
            >>> recording = await browser.stop_recording()
        """
        self._ensure_started()

        if self._mode == "server":
            # Server mode: call recording API
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/recording/start",
                json={"video_enabled": True},
            )
            return response or {}
        else:
            from flybrowser.core.recording import RecordingConfig, RecordingManager

            if not self._recording_manager:
                config = RecordingConfig(enabled=True)
                self._recording_manager = RecordingManager(config)

            await self._recording_manager.start_session(self.browser_manager.browser)
            logger.info("Recording started")
            return {"recording_id": "local", "status": "started"}

    async def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and return recording data.

        Returns:
            Dictionary with recording info including screenshots and video.
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/recording/stop",
            )
            return response or {}
        else:
            if not self._recording_manager:
                return {"session_id": None, "screenshots": [], "video": None}

            result = await self._recording_manager.stop_session()
            logger.info(f"Recording stopped: {result.get('session_id')}")
            return result

    # PII handling methods
    def store_credential(self, name: str, value: str, pii_type: str = "password") -> str:
        """
        Store a credential securely for later use.

        Args:
            name: Name/identifier for the credential
            value: The sensitive value to store
            pii_type: Type of PII (password, email, phone, etc.)

        Returns:
            Credential ID for later retrieval

        Example:
            >>> pwd_id = browser.store_credential("login_password", "secret123")
            >>> await browser.secure_fill("#password", pwd_id)
        """
        if self._mode == "server":
            # Server mode: credentials stored on server
            raise NotImplementedError("Credential storage in server mode - use secure_fill API directly")
        else:
            from flybrowser.security.pii_handler import PIIType
            pii_type_enum = PIIType(pii_type)
            return self.pii_handler.store_credential(name, value, pii_type_enum)

    async def secure_fill(self, selector: str, credential_id: str, clear_first: bool = True) -> bool:
        """
        Securely fill a form field with a stored credential.

        Args:
            selector: CSS selector for the input field
            credential_id: ID of the stored credential
            clear_first: Clear the field before filling (default: True)

        Returns:
            True if successful
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/secure-fill",
                json={"selector": selector, "credential_id": credential_id, "clear_first": clear_first},
            )
            return response.get("success", False) if response else False
        else:
            return await self.pii_handler.secure_fill(
                self.browser_manager.page,
                selector,
                credential_id,
                clear_first,
            )

    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text for safe logging or display.

        Args:
            text: Text that may contain PII

        Returns:
            Text with PII masked
        """
        if self._mode == "server":
            # For server mode, use local masker
            from flybrowser.security.pii_handler import PIIMasker
            masker = PIIMasker()
            return masker.mask_text(text)
        else:
            return self.pii_handler.mask_for_llm(text)

    # Workflow methods
    async def run_workflow(
        self,
        workflow_definition: Union[str, Dict[str, Any]],
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a multi-step workflow.

        Uses the WorkflowAgent for complex automation tasks with:
        - Multi-step execution with dependencies
        - Variable substitution and state management
        - Conditional logic and loops
        - Error handling and recovery

        Args:
            workflow_definition: Workflow as YAML/JSON string or dict
            variables: Variables to substitute in the workflow

        Returns:
            WorkflowResult with execution details

        Example:
            >>> workflow = '''
            ... name: login_workflow
            ... steps:
            ...   - name: navigate
            ...     action: goto
            ...     url: https://example.com/login
            ...   - name: fill_email
            ...     action: type
            ...     selector: "#email"
            ...     value: "{{email}}"
            ...   - name: submit
            ...     action: click
            ...     selector: "#submit"
            ... '''
            >>> result = await browser.run_workflow(workflow, {"email": "user@example.com"})
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/workflow",
                json={"workflow": workflow_definition, "variables": variables or {}},
            )
            return response or {}

        # Embedded mode: Use WorkflowAgent
        result = await self.workflow_agent.execute(workflow_definition, variables=variables)
        return {
            "success": result.success,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "error": result.error,
            "step_results": result.step_results,
            "variables": result.variables,
        }

    async def monitor(
        self,
        condition: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Monitor the page for a condition to be met.

        Uses the MonitoringAgent for intelligent page monitoring with:
        - Natural language condition specification
        - Configurable timeout and polling
        - Element appearance/disappearance detection
        - Content change detection

        Args:
            condition: Natural language condition to wait for
            timeout: Maximum time to wait in seconds (default: 30)
            poll_interval: Time between checks in seconds (default: 0.5)

        Returns:
            MonitoringResult with details

        Example:
            >>> await browser.monitor("wait for the loading spinner to disappear")
            >>> await browser.monitor("wait for 'Success' message to appear")
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/monitor",
                json={"condition": condition, "timeout": timeout, "poll_interval": poll_interval},
            )
            return response or {}

        # Embedded mode: Use MonitoringAgent
        result = await self.monitoring_agent.execute(
            condition, timeout=timeout, poll_interval=poll_interval
        )
        return {
            "success": result.success,
            "condition_met": result.condition_met,
            "elapsed_time": result.elapsed_time,
            "error": result.error,
            "details": result.details,
        }

    # Utility methods
    @property
    def mode(self) -> str:
        """Get the current mode (embedded or server)."""
        return self._mode

    @property
    def session_id(self) -> Optional[str]:
        """Get the session ID (server mode only)."""
        return self._session_id

    @property
    def endpoint(self) -> Optional[str]:
        """Get the server endpoint (server mode only)."""
        return self._endpoint

    def _ensure_started(self) -> None:
        """Ensure FlyBrowser has been started."""
        if not self._started:
            raise RuntimeError("FlyBrowser not started. Call start() first or use async context manager.")

    async def __aenter__(self) -> "FlyBrowser":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
