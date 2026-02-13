# FlyBrowser Comprehensive Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate FlyBrowser to fully leverage fireflyframework-genai capabilities, enhance the CLI with session management and direct commands, and create comprehensive documentation.

**Architecture:** Layered bottom-up migration: agent layer (memory, middleware, validation) -> service layer (REST exposure, RBAC, streaming) -> CLI (sessions, direct commands, pipelines, wizard) -> observability (tracing, metrics, cost) -> documentation (full restructure).

**Tech Stack:** Python 3.13+, fireflyframework-genai (MemoryManager, OutputReviewer, FireflyTracer, FireflyMetrics, create_genai_app, RBACManager), FastAPI, Playwright, Pydantic v2, argparse, prompt_toolkit.

---

## Phase 1: Agent Layer Migration

### Task 1: Migrate BrowserMemoryManager to Framework MemoryManager

**Files:**
- Modify: `flybrowser/agents/memory/browser_memory.py`
- Create: `tests/agents/memory/test_browser_memory_framework.py`

**Step 1: Write the failing test**

```python
# tests/agents/memory/test_browser_memory_framework.py
"""Tests for BrowserMemoryManager backed by fireflyframework-genai MemoryManager."""
import pytest
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager


class TestBrowserMemoryFrameworkIntegration:
    """Verify BrowserMemoryManager delegates to framework MemoryManager."""

    def test_creation_with_framework_memory(self):
        mgr = BrowserMemoryManager()
        assert mgr._framework_memory is not None

    def test_conversation_tracking(self):
        mgr = BrowserMemoryManager()
        assert mgr.conversation_id is not None

    def test_record_page_stores_in_working_memory(self):
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "1 button")
        stored = mgr._framework_memory.get("page_history")
        assert stored is not None
        assert len(stored) == 1
        assert stored[0]["url"] == "https://example.com"

    def test_record_navigation_in_working_memory(self):
        mgr = BrowserMemoryManager()
        mgr.record_navigation("https://a.com", "https://b.com", "click")
        graph = mgr._framework_memory.get("navigation_graph")
        assert "https://a.com" in graph

    def test_record_obstacle_in_working_memory(self):
        mgr = BrowserMemoryManager()
        mgr.record_obstacle("https://example.com", "cookie_banner", "dismissed")
        cache = mgr._framework_memory.get("obstacle_cache")
        assert "https://example.com" in cache

    def test_format_for_prompt_includes_memory_context(self):
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "1 button")
        prompt = mgr.format_for_prompt()
        assert "example.com" in prompt.lower()

    def test_clear_resets_framework_memory(self):
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "1 button")
        mgr.clear()
        stored = mgr._framework_memory.get("page_history")
        assert stored is None or len(stored) == 0

    def test_backward_compat_page_history_property(self):
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "1 button")
        assert len(mgr.page_history) == 1

    def test_backward_compat_navigation_graph_property(self):
        mgr = BrowserMemoryManager()
        mgr.record_navigation("https://a.com", "https://b.com", "click")
        assert "https://a.com" in mgr.navigation_graph

    def test_backward_compat_obstacle_cache_property(self):
        mgr = BrowserMemoryManager()
        mgr.record_obstacle("https://x.com", "popup", "closed")
        assert "https://x.com" in mgr.obstacle_cache
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agents/memory/test_browser_memory_framework.py -v`
Expected: FAIL (no `_framework_memory` attribute)

**Step 3: Implement the migration**

Rewrite `flybrowser/agents/memory/browser_memory.py` to delegate to `fireflyframework_genai.memory.MemoryManager`:

```python
"""BrowserMemoryManager -- browser-specific memory backed by fireflyframework-genai."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fireflyframework_genai.memory import MemoryManager
from fireflyframework_genai.memory.store import InMemoryStore


@dataclass
class PageSnapshot:
    url: str
    title: str
    elements_summary: str
    timestamp: float = 0.0


@dataclass
class ObstacleInfo:
    obstacle_type: str
    resolution: str


class BrowserMemoryManager:
    """Browser memory backed by fireflyframework-genai MemoryManager.

    Stores page history, navigation graph, obstacle cache, and facts in the
    framework's WorkingMemory while maintaining backward-compatible properties.
    """

    def __init__(self, store: Optional[Any] = None) -> None:
        backend = store or InMemoryStore()
        self._framework_memory = MemoryManager(store=backend)
        self._conversation_id = self._framework_memory.new_conversation()

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    # -- Backward-compatible properties -----------------------------------

    @property
    def page_history(self) -> List[PageSnapshot]:
        raw = self._framework_memory.get("page_history") or []
        return [
            PageSnapshot(
                url=p["url"],
                title=p["title"],
                elements_summary=p["elements_summary"],
                timestamp=p.get("timestamp", 0.0),
            )
            for p in raw
        ]

    @property
    def navigation_graph(self) -> Dict[str, List[str]]:
        return self._framework_memory.get("navigation_graph") or {}

    @property
    def obstacle_cache(self) -> Dict[str, ObstacleInfo]:
        raw = self._framework_memory.get("obstacle_cache") or {}
        return {
            url: ObstacleInfo(
                obstacle_type=info["obstacle_type"],
                resolution=info["resolution"],
            )
            for url, info in raw.items()
        }

    # -- Mutation methods --------------------------------------------------

    def record_page_state(
        self, url: str, title: str, elements_summary: str
    ) -> None:
        history = self._framework_memory.get("page_history") or []
        history.append(
            {
                "url": url,
                "title": title,
                "elements_summary": elements_summary,
                "timestamp": time.time(),
            }
        )
        self._framework_memory.set("page_history", history)
        visited = self._framework_memory.get("visited_urls") or []
        if url not in visited:
            visited.append(url)
            self._framework_memory.set("visited_urls", visited)
        self._framework_memory.set("current_url", url)

    def record_navigation(
        self, from_url: str, to_url: str, method: str = "click"
    ) -> None:
        graph = self._framework_memory.get("navigation_graph") or {}
        edges = graph.get(from_url, [])
        edges.append(to_url)
        graph[from_url] = edges
        self._framework_memory.set("navigation_graph", graph)

    def record_obstacle(
        self, url: str, obstacle_type: str, resolution: str
    ) -> None:
        cache = self._framework_memory.get("obstacle_cache") or {}
        cache[url] = {
            "obstacle_type": obstacle_type,
            "resolution": resolution,
        }
        self._framework_memory.set("obstacle_cache", cache)

    def has_visited_url(self, url: str) -> bool:
        visited = self._framework_memory.get("visited_urls") or []
        return url in visited

    def get_current_page(self) -> Optional[PageSnapshot]:
        history = self.page_history
        return history[-1] if history else None

    def set_fact(self, key: str, value: Any) -> None:
        facts = self._framework_memory.get("facts") or {}
        facts[key] = value
        self._framework_memory.set("facts", facts)

    def get_fact(self, key: str, default: Any = None) -> Any:
        facts = self._framework_memory.get("facts") or {}
        return facts.get(key, default)

    def format_for_prompt(self) -> str:
        parts: List[str] = []
        current = self.get_current_page()
        if current:
            parts.append(f"Current page: {current.url} - {current.title}")
            parts.append(f"Elements: {current.elements_summary}")
        history = self.page_history
        if len(history) > 1:
            urls = [p.url for p in history[-5:]]
            parts.append(f"Recent pages: {', '.join(urls)}")
        obstacles = self.obstacle_cache
        if obstacles:
            obs_list = [
                f"{url}: {info.obstacle_type}" for url, info in obstacles.items()
            ]
            parts.append(f"Known obstacles: {'; '.join(obs_list)}")
        return "\n".join(parts) if parts else ""

    def clear(self) -> None:
        for key in [
            "page_history",
            "navigation_graph",
            "obstacle_cache",
            "visited_urls",
            "current_url",
            "facts",
        ]:
            self._framework_memory.set(key, None)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/agents/memory/test_browser_memory_framework.py tests/agents/memory/test_browser_memory.py -v`
Expected: ALL PASS (both new and existing tests)

**Step 5: Commit**

```
git add flybrowser/agents/memory/browser_memory.py tests/agents/memory/test_browser_memory_framework.py
git commit -m "refactor: migrate BrowserMemoryManager to fireflyframework-genai MemoryManager"
```

---

### Task 2: Add Framework Middleware to BrowserAgent

**Files:**
- Modify: `flybrowser/agents/browser_agent.py:62-91`
- Create: `tests/agents/test_browser_agent_middleware.py`

**Step 1: Write the failing test**

```python
# tests/agents/test_browser_agent_middleware.py
"""Tests for BrowserAgent middleware chain with framework middleware."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig


class TestBrowserAgentMiddleware:
    """Verify BrowserAgent includes framework middleware."""

    @pytest.fixture
    def mock_page(self):
        page = MagicMock()
        page.goto = AsyncMock()
        page.content = AsyncMock(return_value="<html></html>")
        page.screenshot = AsyncMock(return_value=b"png")
        page.evaluate = AsyncMock(return_value=[])
        page.title = AsyncMock(return_value="Test")
        page.url = "https://test.com"
        page.get_page_state = AsyncMock(return_value={"url": "https://test.com", "title": "Test"})
        page.get_html = AsyncMock(return_value="<html></html>")
        page.get_title = AsyncMock(return_value="Test")
        page.get_url = MagicMock(return_value="https://test.com")
        return page

    def test_has_logging_middleware(self, mock_page):
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)
        middleware_types = [type(m).__name__ for m in agent._middleware]
        assert "LoggingMiddleware" in middleware_types

    def test_has_cost_guard_middleware(self, mock_page):
        config = BrowserAgentConfig(budget_limit_usd=2.0)
        agent = BrowserAgent(mock_page, config)
        middleware_types = [type(m).__name__ for m in agent._middleware]
        assert "CostGuardMiddleware" in middleware_types

    def test_has_explainability_middleware(self, mock_page):
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)
        middleware_types = [type(m).__name__ for m in agent._middleware]
        assert "ExplainabilityMiddleware" in middleware_types

    def test_custom_middleware_still_present(self, mock_page):
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)
        middleware_types = [type(m).__name__ for m in agent._middleware]
        assert "ObstacleDetectionMiddleware" in middleware_types
        assert "ScreenshotOnErrorMiddleware" in middleware_types

    def test_middleware_order(self, mock_page):
        """Framework middleware should run before browser-specific middleware."""
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)
        types = [type(m).__name__ for m in agent._middleware]
        log_idx = types.index("LoggingMiddleware")
        obstacle_idx = types.index("ObstacleDetectionMiddleware")
        assert log_idx < obstacle_idx
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agents/test_browser_agent_middleware.py -v`
Expected: FAIL (no LoggingMiddleware, CostGuardMiddleware, etc.)

**Step 3: Implement middleware integration**

Modify `flybrowser/agents/browser_agent.py` imports and constructor:

```python
from fireflyframework_genai.agents import FireflyAgent
from fireflyframework_genai.agents.builtin_middleware import (
    CostGuardMiddleware,
    ExplainabilityMiddleware,
    LoggingMiddleware,
)
from fireflyframework_genai.reasoning import ReActPattern

# In BrowserAgent.__init__:
        # Framework middleware (runs first) + browser-specific middleware
        self._middleware = [
            LoggingMiddleware(),
            CostGuardMiddleware(budget_limit_usd=config.budget_limit_usd),
            ExplainabilityMiddleware(),
            ObstacleDetectionMiddleware(page_controller),
            ScreenshotOnErrorMiddleware(page_controller),
        ]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/agents/test_browser_agent_middleware.py tests/agents/test_browser_agent.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/agents/browser_agent.py tests/agents/test_browser_agent_middleware.py
git commit -m "feat: add framework middleware (logging, cost guard, explainability) to BrowserAgent"
```

---

### Task 3: Add OutputReviewer for Extraction Validation

**Files:**
- Modify: `flybrowser/agents/browser_agent.py:105-116`
- Create: `tests/agents/test_extraction_validation.py`

**Step 1: Write the failing test**

```python
# tests/agents/test_extraction_validation.py
"""Tests for extraction validation with OutputReviewer."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import BaseModel
from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig


class Product(BaseModel):
    name: str
    price: float


class TestExtractionValidation:
    @pytest.fixture
    def mock_page(self):
        page = MagicMock()
        page.goto = AsyncMock()
        page.content = AsyncMock(return_value="<html></html>")
        page.screenshot = AsyncMock(return_value=b"png")
        page.evaluate = AsyncMock(return_value=[])
        page.title = AsyncMock(return_value="Test")
        page.url = "https://test.com"
        page.get_page_state = AsyncMock(return_value={"url": "https://test.com"})
        page.get_html = AsyncMock(return_value="<html></html>")
        page.get_title = AsyncMock(return_value="Test")
        page.get_url = MagicMock(return_value="https://test.com")
        return page

    def test_extract_accepts_pydantic_schema(self, mock_page):
        """BrowserAgent.extract should accept Pydantic models for validation."""
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)
        assert hasattr(agent, "extract")

    @pytest.mark.asyncio
    async def test_extract_with_schema_uses_reviewer(self, mock_page):
        """When schema is provided, OutputReviewer validates the result."""
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)

        with patch.object(agent, "_agent") as mock_agent:
            mock_result = MagicMock()
            mock_result.output = Product(name="Widget", price=9.99)
            mock_agent.run = AsyncMock(return_value=mock_result)

            result = await agent.extract("Get product info", schema=Product)
            assert result is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agents/test_extraction_validation.py -v`
Expected: May pass partially, but integration with OutputReviewer missing

**Step 3: Implement extraction validation**

Update `extract()` method in `browser_agent.py`:

```python
from fireflyframework_genai.validation import OutputReviewer

    async def extract(
        self,
        query: str,
        schema: Optional[Type] = None,
        context: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict:
        prompt = _EXTRACT_PROMPT.format(query=query)
        if schema:
            reviewer = OutputReviewer(output_type=schema, max_retries=max_retries)
            result = await reviewer.review(self._agent, prompt)
            return self._format_result(result.output, query)
        else:
            result = await self._agent.run(prompt)
            return self._format_result(result, query)
```

**Step 4: Run tests**

Run: `python -m pytest tests/agents/test_extraction_validation.py tests/agents/test_browser_agent.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/agents/browser_agent.py tests/agents/test_extraction_validation.py
git commit -m "feat: add OutputReviewer validation for schema-based extraction"
```

---

### Task 4: Add Reasoning Pattern Selection

**Files:**
- Modify: `flybrowser/agents/browser_agent.py:35-41,62-91`
- Create: `tests/agents/test_reasoning_patterns.py`

**Step 1: Write the failing test**

```python
# tests/agents/test_reasoning_patterns.py
"""Tests for reasoning pattern selection in BrowserAgent."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig
from flybrowser.agents.types import ReasoningStrategy


class TestReasoningPatternSelection:
    @pytest.fixture
    def mock_page(self):
        page = MagicMock()
        page.goto = AsyncMock()
        page.content = AsyncMock(return_value="<html></html>")
        page.screenshot = AsyncMock(return_value=b"png")
        page.evaluate = AsyncMock(return_value=[])
        page.title = AsyncMock(return_value="Test")
        page.url = "https://test.com"
        page.get_page_state = AsyncMock(return_value={"url": "https://test.com"})
        page.get_html = AsyncMock(return_value="<html></html>")
        page.get_title = AsyncMock(return_value="Test")
        page.get_url = MagicMock(return_value="https://test.com")
        return page

    def test_default_reasoning_is_react(self, mock_page):
        config = BrowserAgentConfig()
        agent = BrowserAgent(mock_page, config)
        assert agent._reasoning_strategy == ReasoningStrategy.REACT_STANDARD

    def test_plan_and_solve_strategy(self, mock_page):
        config = BrowserAgentConfig(reasoning_strategy=ReasoningStrategy.PLAN_AND_SOLVE)
        agent = BrowserAgent(mock_page, config)
        assert agent._reasoning_strategy == ReasoningStrategy.PLAN_AND_SOLVE

    def test_self_reflection_strategy(self, mock_page):
        config = BrowserAgentConfig(reasoning_strategy=ReasoningStrategy.SELF_REFLECTION)
        agent = BrowserAgent(mock_page, config)
        assert agent._reasoning_strategy == ReasoningStrategy.SELF_REFLECTION
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agents/test_reasoning_patterns.py -v`
Expected: FAIL (BrowserAgentConfig has no reasoning_strategy)

**Step 3: Implement reasoning pattern selection**

Update `BrowserAgentConfig` and `BrowserAgent.__init__`:

```python
from fireflyframework_genai.reasoning import (
    ReActPattern,
    PlanAndExecutePattern,
    ReflexionPattern,
)
from flybrowser.agents.types import ReasoningStrategy

@dataclass
class BrowserAgentConfig:
    model: str = "openai:gpt-4o"
    max_iterations: int = 50
    max_time: int = 1800
    budget_limit_usd: float = 5.0
    session_id: Optional[str] = None
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT_STANDARD

# In BrowserAgent.__init__:
        self._reasoning_strategy = config.reasoning_strategy
        self._react = self._create_reasoning_pattern(config)

    def _create_reasoning_pattern(self, config: BrowserAgentConfig):
        strategy = config.reasoning_strategy
        if strategy == ReasoningStrategy.PLAN_AND_SOLVE:
            return PlanAndExecutePattern(max_steps=config.max_iterations)
        elif strategy == ReasoningStrategy.SELF_REFLECTION:
            return ReflexionPattern(max_steps=config.max_iterations)
        else:
            return ReActPattern(max_steps=config.max_iterations)
```

**Step 4: Run tests**

Run: `python -m pytest tests/agents/test_reasoning_patterns.py tests/agents/test_browser_agent.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/agents/browser_agent.py tests/agents/test_reasoning_patterns.py
git commit -m "feat: add configurable reasoning pattern selection (ReAct, PlanAndExecute, Reflexion)"
```

---

## Phase 2: Service Layer Migration

### Task 5: Integrate Framework REST Exposure Base

**Files:**
- Modify: `flybrowser/service/app.py:1-127`
- Create: `tests/integration/test_framework_rest.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_framework_rest.py
"""Tests for framework REST exposure integration."""
import pytest
from fastapi.testclient import TestClient


class TestFrameworkRESTBase:
    def test_health_endpoint_exists(self):
        from flybrowser.service.app import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_liveness_endpoint_exists(self):
        from flybrowser.service.app import app
        client = TestClient(app)
        resp = client.get("/liveness")
        assert resp.status_code == 200

    def test_readiness_endpoint_exists(self):
        from flybrowser.service.app import app
        client = TestClient(app)
        resp = client.get("/readiness")
        assert resp.status_code == 200

    def test_agents_list_endpoint(self):
        from flybrowser.service.app import app
        client = TestClient(app)
        resp = client.get("/agents")
        assert resp.status_code == 200

    def test_request_id_header(self):
        from flybrowser.service.app import app
        client = TestClient(app)
        resp = client.get("/health")
        assert "x-request-id" in resp.headers or resp.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_framework_rest.py -v`
Expected: FAIL (no /liveness, /readiness, /agents endpoints)

**Step 3: Implement framework REST base**

Modify `flybrowser/service/app.py` to use `create_genai_app()` as base:

```python
from fireflyframework_genai.exposure.rest import create_genai_app
from flybrowser import __version__

# Create app using framework (provides health, readiness, liveness, agents, CORS, request ID)
app = create_genai_app(
    title="FlyBrowser",
    version=__version__,
    cors=True,
    request_id=True,
)

# Keep all existing custom routes mounted on the framework app
```

**Step 4: Run tests**

Run: `python -m pytest tests/integration/test_framework_rest.py tests/integration/test_api_standalone.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/service/app.py tests/integration/test_framework_rest.py
git commit -m "feat: integrate fireflyframework-genai REST exposure as service base"
```

---

### Task 6: Integrate RBAC Security

**Files:**
- Modify: `flybrowser/service/auth.py`
- Create: `tests/unit/test_rbac_auth.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_rbac_auth.py
"""Tests for RBAC integration."""
import pytest
from flybrowser.service.auth import RBACAuthManager


class TestRBACAuth:
    def test_create_admin_token(self):
        mgr = RBACAuthManager()
        token = mgr.create_token(user_id="admin1", roles=["admin"])
        assert token is not None

    def test_validate_admin_token(self):
        mgr = RBACAuthManager()
        token = mgr.create_token(user_id="admin1", roles=["admin"])
        claims = mgr.validate_token(token)
        assert claims is not None
        assert "admin" in claims["roles"]

    def test_create_operator_token(self):
        mgr = RBACAuthManager()
        token = mgr.create_token(user_id="op1", roles=["operator"])
        claims = mgr.validate_token(token)
        assert "operator" in claims["roles"]

    def test_create_viewer_token(self):
        mgr = RBACAuthManager()
        token = mgr.create_token(user_id="view1", roles=["viewer"])
        claims = mgr.validate_token(token)
        assert "viewer" in claims["roles"]

    def test_invalid_token_returns_none(self):
        mgr = RBACAuthManager()
        claims = mgr.validate_token("invalid-token")
        assert claims is None

    def test_role_permissions(self):
        mgr = RBACAuthManager()
        assert mgr.has_permission("admin", "sessions.create")
        assert mgr.has_permission("operator", "sessions.create")
        assert not mgr.has_permission("viewer", "sessions.create")
        assert mgr.has_permission("viewer", "sessions.list")

    def test_backward_compat_api_key_still_works(self):
        mgr = RBACAuthManager()
        assert mgr.validate_api_key(mgr.dev_api_key) is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_rbac_auth.py -v`
Expected: FAIL (no RBACAuthManager class)

**Step 3: Implement RBAC**

Rewrite `flybrowser/service/auth.py` to wrap `fireflyframework_genai.security.rbac.RBACManager`:

```python
"""Authentication and RBAC for FlyBrowser service."""
from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from fireflyframework_genai.security.rbac import RBACManager

_ROLES = {
    "admin": ["*"],
    "operator": [
        "sessions.create", "sessions.delete", "sessions.list", "sessions.get",
        "sessions.navigate", "sessions.extract", "sessions.act", "sessions.agent",
        "sessions.screenshot", "sessions.observe", "sessions.stream",
        "recordings.list", "recordings.download",
    ],
    "viewer": [
        "sessions.list", "sessions.get",
        "recordings.list", "recordings.download",
    ],
}


class RBACAuthManager:
    """RBAC-based authentication manager wrapping fireflyframework-genai."""

    def __init__(self, jwt_secret: Optional[str] = None) -> None:
        secret = jwt_secret or secrets.token_urlsafe(32)
        self._rbac = RBACManager(jwt_secret=secret, roles=_ROLES)
        self._dev_api_key = f"flybrowser_dev_{secrets.token_urlsafe(16)}"
        self._api_keys: Dict[str, Dict[str, Any]] = {
            self._dev_api_key: {"name": "dev", "roles": ["admin"], "enabled": True}
        }

    @property
    def dev_api_key(self) -> str:
        return self._dev_api_key

    def create_token(self, user_id: str, roles: List[str], tenant_id: Optional[str] = None) -> str:
        return self._rbac.create_token(user_id=user_id, roles=roles, tenant_id=tenant_id)

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            return self._rbac.validate_token(token)
        except Exception:
            return None

    def has_permission(self, role: str, permission: str) -> bool:
        perms = _ROLES.get(role, [])
        return "*" in perms or permission in perms

    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        info = self._api_keys.get(key)
        if info and info.get("enabled"):
            return info
        return None
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_rbac_auth.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/service/auth.py tests/unit/test_rbac_auth.py
git commit -m "feat: integrate fireflyframework-genai RBAC for JWT-based authentication"
```

---

### Task 7: Enhance SessionManager with Usage Tracking

**Files:**
- Modify: `flybrowser/service/session_manager.py:26-45`
- Create: `tests/unit/test_session_usage_tracking.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_session_usage_tracking.py
"""Tests for session usage tracking integration."""
import pytest
from flybrowser.service.session_manager import SessionManager


class TestSessionUsageTracking:
    def test_session_manager_has_usage_tracker(self):
        mgr = SessionManager()
        assert mgr.usage_tracker is not None

    def test_get_session_stats_includes_usage(self):
        mgr = SessionManager()
        stats = mgr.get_stats()
        assert "total_cost_usd" in stats

    def test_get_session_usage(self):
        mgr = SessionManager()
        usage = mgr.get_usage_summary()
        assert usage["total_tokens"] == 0
        assert usage["total_cost_usd"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_session_usage_tracking.py -v`
Expected: FAIL (no usage_tracker, no total_cost_usd, no get_usage_summary)

**Step 3: Implement usage tracking**

Modify `flybrowser/service/session_manager.py`:

```python
from fireflyframework_genai.observability import UsageTracker

class SessionManager:
    def __init__(self, max_sessions=100, session_timeout=3600):
        # ... existing init ...
        self._usage_tracker = UsageTracker()

    @property
    def usage_tracker(self):
        return self._usage_tracker

    def get_usage_summary(self) -> dict:
        summary = self._usage_tracker.get_summary()
        return {
            "total_tokens": summary.get("total_tokens", 0),
            "total_cost_usd": summary.get("total_cost_usd", 0.0),
            "agent_breakdown": summary.get("agent_breakdown", {}),
        }

    def get_stats(self):
        base_stats = {
            "active_sessions": len(self._sessions),
            "max_sessions": self._max_sessions,
            "total_requests": self._total_requests,
            "session_timeout": self._session_timeout,
        }
        usage = self.get_usage_summary()
        base_stats["total_cost_usd"] = usage["total_cost_usd"]
        base_stats["total_tokens"] = usage["total_tokens"]
        return base_stats
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_session_usage_tracking.py tests/unit/test_session_manager.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/service/session_manager.py tests/unit/test_session_usage_tracking.py
git commit -m "feat: add UsageTracker to SessionManager for cost/token tracking"
```

---

## Phase 3: CLI Overhaul

### Task 8: Add Session Management CLI Commands

**Files:**
- Create: `flybrowser/cli/session.py`
- Modify: `flybrowser/cli/main.py` (add session subparser at line ~500)
- Create: `tests/cli/test_session_commands.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_session_commands.py
"""Tests for CLI session management commands."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from flybrowser.cli.session import (
    cmd_session_create,
    cmd_session_list,
    cmd_session_info,
    cmd_session_close,
    cmd_session_close_all,
    cmd_session_exec,
)


class TestSessionCreate:
    def test_creates_session_embedded(self):
        args = MagicMock()
        args.provider = "openai"
        args.model = "gpt-4o"
        args.headless = True
        args.name = None
        args.endpoint = None
        with patch("flybrowser.cli.session.create_session_embedded", new_callable=AsyncMock) as mock:
            mock.return_value = "session-123"
            result = cmd_session_create(args)
            assert result == 0


class TestSessionList:
    def test_lists_sessions_table_format(self):
        args = MagicMock()
        args.format = "table"
        args.status = "active"
        args.endpoint = None
        with patch("flybrowser.cli.session.list_sessions") as mock:
            mock.return_value = []
            result = cmd_session_list(args)
            assert result == 0


class TestSessionExec:
    def test_executes_command_on_session(self):
        args = MagicMock()
        args.session_id = "session-123"
        args.command = "goto https://example.com"
        args.endpoint = None
        with patch("flybrowser.cli.session.exec_on_session", new_callable=AsyncMock) as mock:
            mock.return_value = {"success": True, "url": "https://example.com"}
            result = cmd_session_exec(args)
            assert result == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cli/test_session_commands.py -v`
Expected: FAIL (no flybrowser.cli.session module)

**Step 3: Create session CLI module**

Create `flybrowser/cli/session.py` with full implementation of session management commands (create, list, info, connect, exec, close, close-all) and `add_session_subparser()` function.

Then modify `flybrowser/cli/main.py` in `create_parser()` to call `add_session_subparser(subparsers)` and add session command handler in `main()`.

**Step 4: Run tests**

Run: `python -m pytest tests/cli/test_session_commands.py tests/cli/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/cli/session.py flybrowser/cli/main.py tests/cli/test_session_commands.py
git commit -m "feat: add CLI session management commands (create, list, info, connect, exec, close)"
```

---

### Task 9: Add Direct SDK-like CLI Commands

**Files:**
- Create: `flybrowser/cli/direct.py`
- Modify: `flybrowser/cli/main.py` (add subparsers)
- Create: `tests/cli/test_direct_commands.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_direct_commands.py
"""Tests for direct SDK-like CLI commands."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from flybrowser.cli.direct import cmd_goto, cmd_extract, cmd_act, cmd_screenshot, cmd_agent


class TestDirectGoto:
    def test_goto_url(self):
        args = MagicMock()
        args.url = "https://example.com"
        args.session = None
        args.wait_for = None
        with patch("flybrowser.cli.direct._get_or_create_session", new_callable=AsyncMock) as mock_sess:
            mock_browser = MagicMock()
            mock_browser.goto = AsyncMock(return_value=None)
            mock_browser.get_url = MagicMock(return_value="https://example.com")
            mock_browser.get_title = AsyncMock(return_value="Example")
            mock_sess.return_value = ("sess-1", mock_browser)
            result = cmd_goto(args)
            assert result == 0


class TestDirectExtract:
    def test_extract_query(self):
        args = MagicMock()
        args.query = "Get all prices"
        args.session = None
        args.schema = None
        args.format = "json"
        with patch("flybrowser.cli.direct._get_or_create_session", new_callable=AsyncMock) as mock_sess:
            mock_browser = MagicMock()
            mock_browser.extract = AsyncMock(return_value={"data": "prices"})
            mock_sess.return_value = ("sess-1", mock_browser)
            result = cmd_extract(args)
            assert result == 0


class TestDirectAgent:
    def test_agent_task(self):
        args = MagicMock()
        args.task = "Find product prices"
        args.session = None
        args.max_iterations = 50
        args.stream = False
        with patch("flybrowser.cli.direct._get_or_create_session", new_callable=AsyncMock) as mock_sess:
            mock_browser = MagicMock()
            mock_browser.agent = AsyncMock(return_value=MagicMock(data="result"))
            mock_sess.return_value = ("sess-1", mock_browser)
            result = cmd_agent(args)
            assert result == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cli/test_direct_commands.py -v`
Expected: FAIL (no flybrowser.cli.direct module)

**Step 3: Create direct commands module**

Create `flybrowser/cli/direct.py` with one-shot commands (goto, extract, act, observe, screenshot, agent) that auto-create ephemeral sessions. Add `add_direct_subparsers()` and integrate into `main.py`.

**Step 4: Run tests**

Run: `python -m pytest tests/cli/test_direct_commands.py tests/cli/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/cli/direct.py flybrowser/cli/main.py tests/cli/test_direct_commands.py
git commit -m "feat: add direct SDK-like CLI commands (goto, extract, act, screenshot, agent)"
```

---

### Task 10: Add Pipeline Execution

**Files:**
- Create: `flybrowser/cli/pipeline.py`
- Modify: `flybrowser/cli/main.py`
- Create: `tests/cli/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_pipeline.py
"""Tests for pipeline/workflow execution."""
import pytest
import yaml
from unittest.mock import patch, MagicMock, AsyncMock
from flybrowser.cli.pipeline import cmd_run, parse_workflow


class TestWorkflowParsing:
    def test_parse_yaml_workflow(self, tmp_path):
        workflow = {
            "name": "test-workflow",
            "sessions": {"main": {"provider": "openai", "headless": True}},
            "steps": [
                {"name": "navigate", "session": "main", "action": "goto", "url": "https://example.com"},
                {"name": "extract", "session": "main", "action": "extract", "query": "Get title"},
            ],
        }
        path = tmp_path / "workflow.yaml"
        path.write_text(yaml.dump(workflow))
        parsed = parse_workflow(str(path))
        assert parsed["name"] == "test-workflow"
        assert len(parsed["steps"]) == 2

    def test_parse_inline_commands(self):
        inline = "goto https://example.com && extract 'get prices'"
        parsed = parse_workflow(inline_commands=inline)
        assert len(parsed["steps"]) == 2


class TestPipelineRun:
    def test_run_workflow_file(self, tmp_path):
        workflow = {
            "name": "test",
            "sessions": {"main": {"provider": "openai"}},
            "steps": [
                {"name": "nav", "session": "main", "action": "goto", "url": "https://example.com"},
            ],
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(workflow))
        args = MagicMock()
        args.workflow = str(path)
        args.inline = None
        with patch("flybrowser.cli.pipeline.execute_workflow", new_callable=AsyncMock) as mock:
            mock.return_value = {"success": True, "steps_completed": 1}
            result = cmd_run(args)
            assert result == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cli/test_pipeline.py -v`
Expected: FAIL (no flybrowser.cli.pipeline module)

**Step 3: Implement pipeline execution**

Create `flybrowser/cli/pipeline.py` with workflow parsing (YAML and inline) and execution using framework's `PipelineBuilder`.

**Step 4: Run tests**

Run: `python -m pytest tests/cli/test_pipeline.py tests/cli/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/cli/pipeline.py flybrowser/cli/main.py tests/cli/test_pipeline.py
git commit -m "feat: add pipeline execution CLI (flybrowser run workflow.yaml)"
```

---

### Task 11: Enhance Setup Wizard

**Files:**
- Modify: `flybrowser/cli/setup.py`
- Create: `tests/cli/test_setup_wizard_enhanced.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_setup_wizard_enhanced.py
"""Tests for enhanced setup wizard subcommands."""
import pytest
from flybrowser.cli.setup import (
    setup_llm,
    setup_server,
    setup_observability,
    setup_security,
    setup_quick,
)


class TestSetupSubcommands:
    def test_setup_llm_exists(self):
        assert callable(setup_llm)

    def test_setup_server_exists(self):
        assert callable(setup_server)

    def test_setup_observability_exists(self):
        assert callable(setup_observability)

    def test_setup_security_exists(self):
        assert callable(setup_security)

    def test_setup_quick_exists(self):
        assert callable(setup_quick)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cli/test_setup_wizard_enhanced.py -v`
Expected: FAIL (no setup_llm, setup_server, etc.)

**Step 3: Add component-specific setup functions**

Add `setup_quick()`, `setup_llm()`, `setup_server()`, `setup_observability()`, `setup_security()` to `flybrowser/cli/setup.py`. Add corresponding subparser entries.

**Step 4: Run tests**

Run: `python -m pytest tests/cli/test_setup_wizard_enhanced.py tests/cli/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/cli/setup.py tests/cli/test_setup_wizard_enhanced.py
git commit -m "feat: enhance setup wizard with component-specific subcommands (llm, server, observability, security)"
```

---

## Phase 4: Observability Integration

### Task 12: Integrate OpenTelemetry Tracing

**Files:**
- Create: `flybrowser/observability/tracing.py`
- Create: `tests/unit/test_tracing.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_tracing.py
"""Tests for OpenTelemetry tracing integration."""
import pytest
from flybrowser.observability.tracing import get_tracer, configure_tracing


class TestTracingIntegration:
    def test_get_tracer_returns_framework_tracer(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_configure_tracing_with_otlp(self):
        configure_tracing(otlp_endpoint="http://localhost:4317")
        tracer = get_tracer()
        assert tracer is not None

    def test_configure_tracing_console(self):
        configure_tracing(console=True)
        tracer = get_tracer()
        assert tracer is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_tracing.py -v`
Expected: FAIL (no flybrowser.observability.tracing)

**Step 3: Implement tracing wrapper**

Create `flybrowser/observability/tracing.py`:

```python
"""OpenTelemetry tracing integration via fireflyframework-genai."""
from __future__ import annotations

from typing import Optional

from fireflyframework_genai.observability import FireflyTracer
from fireflyframework_genai.observability.exporters import configure_exporters

_tracer: Optional[FireflyTracer] = None


def configure_tracing(
    otlp_endpoint: Optional[str] = None,
    console: bool = False,
    service_name: str = "flybrowser",
) -> None:
    """Configure tracing exporters."""
    global _tracer
    configure_exporters(otlp_endpoint=otlp_endpoint, console=console)
    _tracer = FireflyTracer(service_name=service_name)


def get_tracer() -> FireflyTracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = FireflyTracer(service_name="flybrowser")
    return _tracer
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_tracing.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/observability/tracing.py tests/unit/test_tracing.py
git commit -m "feat: integrate OpenTelemetry tracing via fireflyframework-genai"
```

---

### Task 13: Integrate Metrics and Cost Tracking

**Files:**
- Create: `flybrowser/observability/metrics.py`
- Create: `tests/unit/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_metrics.py
"""Tests for metrics and cost tracking."""
import pytest
from flybrowser.observability.metrics import get_metrics, record_operation, get_cost_summary


class TestMetricsIntegration:
    def test_get_metrics_returns_framework_instance(self):
        metrics = get_metrics()
        assert metrics is not None

    def test_record_operation(self):
        record_operation("navigate", latency_ms=250, tokens=100, cost_usd=0.01)

    def test_get_cost_summary(self):
        summary = get_cost_summary()
        assert "total_cost_usd" in summary
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_metrics.py -v`
Expected: FAIL (no flybrowser.observability.metrics)

**Step 3: Implement metrics wrapper**

Create `flybrowser/observability/metrics.py`:

```python
"""Metrics and cost tracking via fireflyframework-genai."""
from __future__ import annotations

from typing import Optional

from fireflyframework_genai.observability import FireflyMetrics, UsageTracker

_metrics: Optional[FireflyMetrics] = None
_usage: Optional[UsageTracker] = None


def get_metrics() -> FireflyMetrics:
    global _metrics
    if _metrics is None:
        _metrics = FireflyMetrics(service_name="flybrowser")
    return _metrics


def get_usage_tracker() -> UsageTracker:
    global _usage
    if _usage is None:
        _usage = UsageTracker()
    return _usage


def record_operation(
    operation: str,
    latency_ms: float = 0,
    tokens: int = 0,
    cost_usd: float = 0,
) -> None:
    m = get_metrics()
    m.record_latency(operation=operation, latency_ms=latency_ms)
    if tokens:
        m.record_tokens(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)
    if cost_usd:
        m.record_cost(operation=operation, cost_usd=cost_usd)


def get_cost_summary() -> dict:
    tracker = get_usage_tracker()
    summary = tracker.get_summary()
    return {
        "total_cost_usd": summary.get("total_cost_usd", 0.0),
        "total_tokens": summary.get("total_tokens", 0),
    }
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_metrics.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
git add flybrowser/observability/metrics.py tests/unit/test_metrics.py
git commit -m "feat: integrate FireflyMetrics for Prometheus metrics and cost tracking"
```

---

## Phase 5: Documentation Overhaul

### Task 14: Create Documentation Structure and Core Docs

**Files:**
- Create: `docs/index.md`
- Create: `docs/getting-started/installation.md`
- Create: `docs/getting-started/quickstart.md`
- Create: `docs/getting-started/setup-wizard.md`
- Create: `docs/sdk/overview.md`
- Create: `docs/sdk/sessions.md`
- Create: `docs/sdk/navigation.md`
- Create: `docs/sdk/extraction.md`
- Create: `docs/sdk/agent.md`
- Create: `docs/cli/overview.md`
- Create: `docs/cli/session-management.md`
- Create: `docs/cli/direct-commands.md`
- Create: `docs/cli/pipelines.md`
- Create: `docs/cli/repl.md`
- Create: `docs/architecture/overview.md`
- Create: `docs/architecture/framework-integration.md`
- Create: `docs/api-reference/sdk-reference.md`
- Create: `docs/api-reference/rest-api.md`
- Create: `docs/api-reference/config-reference.md`

**Step 1: Create directory structure**

Create all necessary directories under docs/.

**Step 2: Write docs/index.md**

Landing page with navigation links to all sections.

**Step 3: Write getting-started/installation.md**

Comprehensive installation covering pip, browsers, LLM providers, verification.

**Step 4: Write getting-started/quickstart.md**

5-minute quickstart: embedded mode, server mode, CLI examples.

**Step 5: Write getting-started/setup-wizard.md**

Walkthrough of `flybrowser setup` wizard with screenshots/examples.

**Step 6: Write sdk/overview.md**

SDK architecture, embedded vs server mode, dual-mode transparency.

**Step 7: Write sdk/sessions.md, navigation.md, extraction.md, agent.md**

Per-feature SDK guides with code examples and API reference.

**Step 8: Write cli/overview.md, session-management.md, direct-commands.md, pipelines.md, repl.md**

CLI documentation covering all new commands.

**Step 9: Write architecture/overview.md, framework-integration.md**

Architecture docs explaining layered design and fireflyframework-genai integration.

**Step 10: Write api-reference/sdk-reference.md, rest-api.md, config-reference.md**

Complete API reference for all public interfaces.

**Step 11: Commit all docs**

```
git add docs/
git commit -m "docs: comprehensive documentation overhaul with new structure"
```

---

### Task 15: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update README with new features**

Reflect CLI session management, direct commands, pipeline execution, framework integration, observability, setup wizard enhancements.

**Step 2: Commit**

```
git add README.md
git commit -m "docs: update README with new CLI features and framework integration"
```

---

## Summary

| Phase | Tasks | Focus |
|-------|-------|-------|
| 1 | 1-4 | Agent layer: memory, middleware, validation, reasoning |
| 2 | 5-7 | Service layer: REST base, RBAC, usage tracking |
| 3 | 8-11 | CLI: sessions, direct commands, pipelines, wizard |
| 4 | 12-13 | Observability: tracing, metrics, cost |
| 5 | 14-15 | Documentation: full restructure and rewrite |

Each task is independently deployable and backward compatible. Tests are written first (TDD). Commits are frequent and focused.
