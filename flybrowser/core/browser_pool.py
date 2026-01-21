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
Browser Pool Management for FlyBrowser.

This module provides a browser pool management system for handling multiple
concurrent browser sessions. It includes:
- BrowserPool: Manages a pool of browser instances
- JobQueue: Distributes workloads across browser instances
- Resource management and lifecycle handling
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from flybrowser.core.browser import BrowserManager
from flybrowser.exceptions import BrowserError
from flybrowser.utils.logger import logger

T = TypeVar("T")


class BrowserSessionState(str, Enum):
    """State of a browser session in the pool."""

    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class JobState(str, Enum):
    """State of a job in the queue."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PoolConfig:
    """Configuration for the browser pool.

    Attributes:
        min_size: Minimum number of browser instances to maintain in the pool.
            These browsers are pre-warmed and ready for immediate use.
            Default: 1
        max_size: Maximum number of browser instances allowed in the pool.
            The pool will scale up to this limit based on demand.
            Default: 10
        idle_timeout_seconds: Time in seconds before an idle browser is closed.
            Only applies when pool size is above min_size.
            Default: 300 (5 minutes)
        max_session_age_seconds: Maximum lifetime of a browser session.
            Sessions older than this are recycled to prevent memory leaks.
            Default: 3600 (1 hour)
        startup_timeout_seconds: Timeout for browser startup operations.
            Default: 30
        shutdown_timeout_seconds: Timeout for browser shutdown operations.
            Default: 10
        health_check_interval_seconds: Interval between pool maintenance checks.
            Default: 60 (1 minute)
        headless: Whether to run browsers in headless mode (no visible window).
            Default: True
        browser_type: Type of browser to use (chromium, firefox, webkit).
            Default: "chromium"
        browser_options: Additional options passed to the browser manager.
            Default: {}

    Example:
        >>> config = PoolConfig(min_size=2, max_size=20, headless=True)
        >>> pool = BrowserPool(config)
    """

    min_size: int = 1
    max_size: int = 10
    idle_timeout_seconds: float = 300.0  # 5 minutes
    max_session_age_seconds: float = 3600.0  # 1 hour
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    health_check_interval_seconds: float = 60.0
    headless: bool = True
    browser_type: str = "chromium"
    browser_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "PoolConfig":
        """Create PoolConfig from environment variables.

        Environment variables (all optional, with FLYBROWSER_ prefix):
            FLYBROWSER_POOL_MIN_SIZE: Minimum pool size
            FLYBROWSER_POOL_MAX_SIZE: Maximum pool size
            FLYBROWSER_POOL_IDLE_TIMEOUT: Idle timeout in seconds
            FLYBROWSER_POOL_MAX_SESSION_AGE: Max session age in seconds
            FLYBROWSER_POOL_HEADLESS: Run in headless mode (true/false)
            FLYBROWSER_POOL_BROWSER_TYPE: Browser type

        Returns:
            PoolConfig with values from environment or defaults
        """
        import os

        def get_float(key: str, default: float) -> float:
            val = os.environ.get(key)
            return float(val) if val else default

        def get_int(key: str, default: int) -> int:
            val = os.environ.get(key)
            return int(val) if val else default

        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        return cls(
            min_size=get_int("FLYBROWSER_POOL_MIN_SIZE", 1),
            max_size=get_int("FLYBROWSER_POOL_MAX_SIZE", 10),
            idle_timeout_seconds=get_float("FLYBROWSER_POOL_IDLE_TIMEOUT", 300.0),
            max_session_age_seconds=get_float("FLYBROWSER_POOL_MAX_SESSION_AGE", 3600.0),
            startup_timeout_seconds=get_float("FLYBROWSER_POOL_STARTUP_TIMEOUT", 30.0),
            shutdown_timeout_seconds=get_float("FLYBROWSER_POOL_SHUTDOWN_TIMEOUT", 10.0),
            health_check_interval_seconds=get_float("FLYBROWSER_POOL_HEALTH_CHECK_INTERVAL", 60.0),
            headless=get_bool("FLYBROWSER_POOL_HEADLESS", True),
            browser_type=os.environ.get("FLYBROWSER_POOL_BROWSER_TYPE", "chromium"),
        )


@dataclass
class BrowserSession:
    """Represents a browser session in the pool."""

    id: str
    manager: BrowserManager
    state: BrowserSessionState = BrowserSessionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    job_count: int = 0
    current_job_id: Optional[str] = None
    error_count: int = 0

    def is_expired(self, max_age: float) -> bool:
        """Check if session has exceeded max age."""
        return (time.time() - self.created_at) > max_age

    def is_idle_timeout(self, timeout: float) -> bool:
        """Check if session has been idle too long."""
        return self.state == BrowserSessionState.IDLE and (time.time() - self.last_used_at) > timeout


@dataclass
class Job:
    """Represents a job to be executed in the browser pool."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.PENDING
    priority: int = 0  # Higher = more priority
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    timeout_seconds: float = 300.0  # 5 minutes default


class JobQueue:
    """
    Priority-based job queue for browser tasks.

    Jobs are executed in priority order (higher priority first),
    with FIFO ordering for jobs with the same priority.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the job queue."""
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def put(self, job: Job) -> str:
        """Add a job to the queue."""
        async with self._lock:
            self._jobs[job.id] = job
            # Use negative priority for max-heap behavior
            await self._queue.put((-job.priority, job.created_at, job.id))
            logger.debug(f"Job {job.id} added to queue (priority={job.priority})")
            return job.id

    async def get(self) -> Optional[Job]:
        """Get the next job from the queue."""
        try:
            _, _, job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            async with self._lock:
                job = self._jobs.get(job_id)
                if job and job.state == JobState.PENDING:
                    return job
                return None
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job and job.state == JobState.PENDING:
                job.state = JobState.CANCELLED
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = sum(1 for j in self._jobs.values() if j.state == JobState.PENDING)
        running = sum(1 for j in self._jobs.values() if j.state == JobState.RUNNING)
        completed = sum(1 for j in self._jobs.values() if j.state == JobState.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.state == JobState.FAILED)

        return {
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": len(self._jobs),
        }


class BrowserPool:
    """
    Manages a pool of browser instances for concurrent task execution.

    The pool automatically scales between min_size and max_size based on demand,
    handles session lifecycle, and distributes jobs across available browsers.

    Example:
        >>> config = PoolConfig(min_size=2, max_size=10)
        >>> pool = BrowserPool(config)
        >>> await pool.start()
        >>>
        >>> async def my_task(browser: BrowserManager):
        ...     page = browser.page
        ...     await page.goto("https://example.com")
        ...     return await page.title()
        >>>
        >>> job_id = await pool.submit(my_task)
        >>> result = await pool.wait_for_result(job_id)
        >>>
        >>> await pool.stop()
    """

    def __init__(self, config: Optional[PoolConfig] = None) -> None:
        """Initialize the browser pool."""
        self.config = config or PoolConfig()
        self._sessions: Dict[str, BrowserSession] = {}
        self._job_queue = JobQueue()
        self._lock = asyncio.Lock()
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._maintenance_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the browser pool and initialize minimum sessions."""
        if self._running:
            logger.warning("Browser pool already running")
            return

        logger.info(f"Starting browser pool (min={self.config.min_size}, max={self.config.max_size})")
        self._running = True

        # Start minimum number of sessions
        for _ in range(self.config.min_size):
            await self._create_session()

        # Start worker tasks
        for _ in range(self.config.max_size):
            task = asyncio.create_task(self._worker_loop())
            self._worker_tasks.append(task)

        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        logger.info(f"Browser pool started with {len(self._sessions)} sessions")

    async def stop(self) -> None:
        """Stop the browser pool and cleanup all sessions."""
        if not self._running:
            return

        logger.info("Stopping browser pool")
        self._running = False

        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()

        if self._maintenance_task:
            self._maintenance_task.cancel()

        # Stop all sessions
        async with self._lock:
            for session in list(self._sessions.values()):
                await self._destroy_session(session.id)

        logger.info("Browser pool stopped")

    async def submit(
        self,
        func: Callable[[BrowserManager], T],
        *args: Any,
        priority: int = 0,
        timeout_seconds: float = 300.0,
        **kwargs: Any,
    ) -> str:
        """
        Submit a job to the pool.

        Args:
            func: Async function that takes a BrowserManager as first argument
            *args: Additional positional arguments for the function
            priority: Job priority (higher = more priority)
            timeout_seconds: Maximum time for job execution
            **kwargs: Additional keyword arguments for the function

        Returns:
            Job ID for tracking
        """
        job = Job(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout_seconds=timeout_seconds,
        )

        await self._job_queue.put(job)

        # Scale up if needed
        await self._maybe_scale_up()

        return job.id

    async def wait_for_result(self, job_id: str, timeout: float = 300.0) -> Any:
        """Wait for a job to complete and return its result."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            job = self._job_queue.get_job(job_id)
            if not job:
                raise BrowserError(f"Job {job_id} not found")

            if job.state == JobState.COMPLETED:
                return job.result
            elif job.state == JobState.FAILED:
                raise BrowserError(f"Job {job_id} failed: {job.error}")
            elif job.state == JobState.CANCELLED:
                raise BrowserError(f"Job {job_id} was cancelled")

            await asyncio.sleep(0.1)

        raise BrowserError(f"Timeout waiting for job {job_id}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        return await self._job_queue.cancel(job_id)

    async def _create_session(self) -> Optional[BrowserSession]:
        """Create a new browser session."""
        async with self._lock:
            if len(self._sessions) >= self.config.max_size:
                return None

            session_id = str(uuid.uuid4())
            manager = BrowserManager(
                headless=self.config.headless,
                browser_type=self.config.browser_type,
                **self.config.browser_options,
            )

            session = BrowserSession(
                id=session_id,
                manager=manager,
                state=BrowserSessionState.STARTING,
            )
            self._sessions[session_id] = session

        try:
            await asyncio.wait_for(
                manager.start(),
                timeout=self.config.startup_timeout_seconds,
            )
            session.state = BrowserSessionState.IDLE
            logger.info(f"Browser session {session_id} created")
            return session
        except Exception as e:
            logger.error(f"Failed to create browser session: {e}")
            async with self._lock:
                if session_id in self._sessions:
                    del self._sessions[session_id]
            return None

    async def _destroy_session(self, session_id: str) -> None:
        """Destroy a browser session."""
        session = self._sessions.get(session_id)
        if not session:
            return

        session.state = BrowserSessionState.STOPPING
        try:
            await asyncio.wait_for(
                session.manager.stop(),
                timeout=self.config.shutdown_timeout_seconds,
            )
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")
        finally:
            if session_id in self._sessions:
                del self._sessions[session_id]
            logger.info(f"Browser session {session_id} destroyed")

    async def _get_idle_session(self) -> Optional[BrowserSession]:
        """Get an idle session from the pool."""
        async with self._lock:
            for session in self._sessions.values():
                if session.state == BrowserSessionState.IDLE:
                    session.state = BrowserSessionState.BUSY
                    return session
        return None

    async def _release_session(self, session: BrowserSession) -> None:
        """Release a session back to the pool."""
        async with self._lock:
            if session.id in self._sessions:
                session.state = BrowserSessionState.IDLE
                session.last_used_at = time.time()
                session.current_job_id = None

    async def _maybe_scale_up(self) -> None:
        """Scale up the pool if needed."""
        async with self._lock:
            idle_count = sum(
                1 for s in self._sessions.values()
                if s.state == BrowserSessionState.IDLE
            )
            pending_jobs = self._job_queue.size()

            if idle_count == 0 and pending_jobs > 0 and len(self._sessions) < self.config.max_size:
                # Create a new session
                asyncio.create_task(self._create_session())

    async def _worker_loop(self) -> None:
        """Worker loop that processes jobs from the queue."""
        while self._running:
            try:
                job = await self._job_queue.get()
                if not job:
                    continue

                session = await self._get_idle_session()
                if not session:
                    # No idle session, try to create one
                    session = await self._create_session()
                    if not session:
                        # Put job back in queue
                        await self._job_queue.put(job)
                        await asyncio.sleep(0.5)
                        continue

                # Execute the job
                job.state = JobState.RUNNING
                job.started_at = time.time()
                job.session_id = session.id
                session.current_job_id = job.id
                session.job_count += 1

                try:
                    result = await asyncio.wait_for(
                        job.func(session.manager, *job.args, **job.kwargs),
                        timeout=job.timeout_seconds,
                    )
                    job.result = result
                    job.state = JobState.COMPLETED
                except Exception as e:
                    job.error = str(e)
                    job.state = JobState.FAILED
                    session.error_count += 1
                    logger.error(f"Job {job.id} failed: {e}")
                finally:
                    job.completed_at = time.time()
                    await self._release_session(session)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1.0)

    async def _maintenance_loop(self) -> None:
        """Maintenance loop for session cleanup and health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                async with self._lock:
                    sessions_to_remove = []

                    for session in self._sessions.values():
                        # Check for expired sessions
                        if session.is_expired(self.config.max_session_age_seconds):
                            if session.state == BrowserSessionState.IDLE:
                                sessions_to_remove.append(session.id)

                        # Check for idle timeout (only if above min_size)
                        elif len(self._sessions) > self.config.min_size:
                            if session.is_idle_timeout(self.config.idle_timeout_seconds):
                                sessions_to_remove.append(session.id)

                # Remove sessions outside the lock
                for session_id in sessions_to_remove:
                    await self._destroy_session(session_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        idle = sum(1 for s in self._sessions.values() if s.state == BrowserSessionState.IDLE)
        busy = sum(1 for s in self._sessions.values() if s.state == BrowserSessionState.BUSY)

        return {
            "total_sessions": len(self._sessions),
            "idle_sessions": idle,
            "busy_sessions": busy,
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "job_queue": self._job_queue.get_stats(),
        }

    async def __aenter__(self) -> "BrowserPool":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

